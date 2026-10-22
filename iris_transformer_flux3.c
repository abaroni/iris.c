/*
 * FLUX.3 Image Klein Transformer Implementation
 *
 * Implements Flux3Transformer2DModel (5.6B for Klein):
 * - Three token modes: noisy image, reference images (image_cond) and text.
 * - Per-mode blocks: each mode first goes through its own blocks (5 for
 *   Klein, separate weights per mode), self-attending within the mode only.
 * - Joint blocks (28): text, image and reference tokens are concatenated
 *   into one [TEXT, IMAGE, REFS] sequence.
 * - Every block is the same: LayerNorm (no affine) followed by AdaLN
 *   shift/scale, then attention and a SwiGLU FFN computed in parallel on
 *   the same input, summed and added back through one gate.
 * - Modulation is per mode: image tokens see timestep t, text and reference
 *   tokens timestep 0. Each mode has one modulation projection shared by
 *   all its per-mode blocks and one shared by all the joint blocks, so the
 *   modulation vectors are computed once per forward pass.
 * - Step-distilled checkpoints are flow maps: the embedding of the timestep
 *   the step jumps to is added to every mode's timestep embedding.
 * - 4-axis RoPE (t, h, w, l), 32 dims per axis, theta 10000, consecutive
 *   pair rotation. Image tokens sit at (0, row/2, col/2, 0), text tokens at
 *   (0, 0, 0, index) and reference i at (0, row/2, col/2, -(i+1)).
 *
 * Weights stay as bf16 in the mmapped safetensors files. The CPU path
 * converts each matrix to f32 right before its matmul. With a device
 * backend (CUDA / ROCm, see iris_dev.h) the weights are uploaded once and
 * the whole forward pass runs on the GPU.
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define F3_MAX_SHARDS   8
#define F3_ROPE_AXES    4
#define F3_ATTN_QCHUNK  256   /* Queries per attention chunk (CPU path) */

/* Token modes, in the order of the joint sequence. */
enum { F3_TXT = 0, F3_IMG = 1, F3_COND = 2, F3_NUM_MODES = 3 };

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/* One transformer block. Per-mode and joint blocks share the layout. */
typedef struct {
    const uint16_t *to_q, *to_k, *to_v;  /* [dim, dim] */
    const uint16_t *to_out;              /* [dim, dim] */
    float *norm_q, *norm_k;              /* [head_dim] */
    const uint16_t *ff_in;               /* [2*mlp, dim]: gate rows, then up rows */
    const uint16_t *ff_out;              /* [dim, mlp] */
} f3_block_t;

typedef struct iris_transformer_flux3 {
    /* Architecture (from transformer/config.json) */
    int dim;                /* 3072 */
    int heads;              /* 24 */
    int head_dim;           /* 128 */
    int mlp;                /* 9216 (dim * mlp_ratio) */
    int num_mode_layers;    /* 5 */
    int num_single_layers;  /* 28 */
    int in_channels;        /* 128 */
    int cond_in_channels;   /* 128 */
    int text_dim;           /* 20480 */
    int t_channels;         /* 256 */
    int axes_dim[F3_ROPE_AXES]; /* 32, 32, 32, 32 */
    float rope_theta;       /* 10000 */
    float eps;              /* 1e-6 */
    int flow_map;           /* 1 for step-distilled checkpoints */

    /* Embedders */
    const uint16_t *t_lin1, *t_lin2;     /* timestep MLP: [dim, t_channels], [dim, dim] */
    const uint16_t *tt_lin1, *tt_lin2;   /* target timestep MLP (flow map only) */
    const uint16_t *x_emb[F3_NUM_MODES]; /* [dim, in] for IMG and COND, NULL for TXT */
    const uint16_t *ctx_emb;             /* [dim, text_dim] */

    /* Modulation projections [3*dim, dim]: shift, scale, gate */
    const uint16_t *early_mod[F3_NUM_MODES];
    const uint16_t *single_mod[F3_NUM_MODES];

    /* Blocks */
    f3_block_t *mode_blocks[F3_NUM_MODES]; /* [num_mode_layers] each */
    f3_block_t *single_blocks;             /* [num_single_layers] */

    /* Output layers (image tokens only) */
    const uint16_t *norm_out;   /* [2*dim, dim]: scale, shift */
    const uint16_t *proj_out;   /* [in_channels, dim] */

    safetensors_file_t *sf[F3_MAX_SHARDS];
    int num_sf;

    /* Device copy of the model (GPU path), NULL when running on the CPU */
    struct f3_dev *dev;

    /* CPU work memory, grown on demand */
    float *wbuf;            /* f32 copy of the current weight matrix */
    size_t wbuf_elems;
    float *work;            /* activations, see ensure_work() */
    int work_seq;
} iris_transformer_flux3_t;

/* A run of consecutive tokens sharing one modulation vector. */
typedef struct {
    int start, len;
    const float *mod;       /* [3*dim]: shift, scale, gate */
} f3_segment_t;

void iris_transformer_free_flux3(iris_transformer_flux3_t *tf);

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

/* y[rows, out] = x[rows, in] @ W[out, in]^T with bf16 W. */
static void linear_bf16(iris_transformer_flux3_t *tf, float *y, const float *x,
                        const uint16_t *W, int rows, int in, int out) {
#ifdef USE_METAL
    /* W points into the mmapped weights, so it is a valid key for the
     * Metal weight cache (the f32 scratch below would not be). */
    (void)tf;
    iris_linear_nobias_bf16(y, x, W, rows, in, out);
#else
    size_t n = (size_t)in * out;
    if (n > tf->wbuf_elems) {
        free(tf->wbuf);
        tf->wbuf = malloc(n * sizeof(float));
        tf->wbuf_elems = tf->wbuf ? n : 0;
        if (!tf->wbuf) return;
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)W[i] << 16;
        memcpy(tf->wbuf + i, &bits, sizeof(float));
    }
    iris_linear_nobias(y, x, tf->wbuf, rows, in, out);
#endif
}

/* LayerNorm without affine parameters. */
static void layer_norm(float *out, const float *x, int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *o = out + (size_t)r * dim;
        double mean = 0, var = 0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) {
            double d = xr[i] - mean;
            var += d * d;
        }
        float inv = (float)(1.0 / sqrt(var / dim + eps));
        float m = (float)mean;
        for (int i = 0; i < dim; i++) o[i] = (xr[i] - m) * inv;
    }
}

/* Per-head RMSNorm with weight, in place. x: [seq, heads * head_dim] */
static void head_rms_norm(float *x, const float *w, int seq, int heads,
                          int head_dim, float eps) {
    for (int s = 0; s < seq * heads; s++) {
        float *v = x + (size_t)s * head_dim;
        float ss = 0;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        float inv = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * inv * w[i];
    }
}

/* Rotate consecutive pairs (x0, x1) of every head by the token's angles.
 * cos/sin: [seq, head_dim/2] */
static void apply_rope(float *x, const float *cos_t, const float *sin_t,
                       int seq, int heads, int head_dim) {
    int half = head_dim / 2;
    for (int s = 0; s < seq; s++) {
        const float *c = cos_t + (size_t)s * half;
        const float *sn = sin_t + (size_t)s * half;
        for (int h = 0; h < heads; h++) {
            float *v = x + ((size_t)s * heads + h) * head_dim;
            for (int i = 0; i < half; i++) {
                float x0 = v[2 * i], x1 = v[2 * i + 1];
                v[2 * i]     = x0 * c[i] - x1 * sn[i];
                v[2 * i + 1] = x1 * c[i] + x0 * sn[i];
            }
        }
    }
}

/* Full (unmasked) attention over [seq, heads * head_dim] tensors, processed
 * per head in chunks of queries to bound the score buffer. */
static void attention(float *out, const float *q, const float *k, const float *v,
                      float *scores, int seq, int heads, int head_dim) {
    int dim = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < heads; h++) {
        const float *qh = q + h * head_dim;
        const float *kh = k + h * head_dim;
        const float *vh = v + h * head_dim;
        float *oh = out + h * head_dim;
        for (int q0 = 0; q0 < seq; q0 += F3_ATTN_QCHUNK) {
            int nq = seq - q0 < F3_ATTN_QCHUNK ? seq - q0 : F3_ATTN_QCHUNK;
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, seq, head_dim,
                        scale, qh + (size_t)q0 * dim, dim, kh, dim, 0.0f, scores, seq);
#else
            for (int i = 0; i < nq; i++) {
                const float *qi = qh + (size_t)(q0 + i) * dim;
                for (int j = 0; j < seq; j++) {
                    const float *kj = kh + (size_t)j * dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                    scores[(size_t)i * seq + j] = dot * scale;
                }
            }
#endif
            iris_softmax(scores, nq, seq);
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nq, head_dim, seq,
                        1.0f, scores, seq, vh, dim, 0.0f, oh + (size_t)q0 * dim, dim);
#else
            for (int i = 0; i < nq; i++) {
                float *oi = oh + (size_t)(q0 + i) * dim;
                memset(oi, 0, head_dim * sizeof(float));
                for (int j = 0; j < seq; j++) {
                    float p = scores[(size_t)i * seq + j];
                    const float *vj = vh + (size_t)j * dim;
                    for (int d = 0; d < head_dim; d++) oi[d] += p * vj[d];
                }
            }
#endif
        }
    }
}

/* ========================================================================
 * Timestep Embedding and Modulation
 * ======================================================================== */

/* Sinusoidal embedding of t (already scaled by 1000), [cos | sin] halves
 * like diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0). */
static void sinusoidal_embedding(float *out, float t, int channels) {
    int half = channels / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        out[i] = cosf(t * freq);
        out[half + i] = sinf(t * freq);
    }
}

/* out[dim] = lin2(silu(lin1(sinusoidal(t * 1000)))) */
static void timestep_embed(iris_transformer_flux3_t *tf, float *out, float t,
                           const uint16_t *lin1, const uint16_t *lin2) {
    float sin_emb[512];
    float *hidden = malloc(tf->dim * sizeof(float));
    sinusoidal_embedding(sin_emb, t * 1000.0f, tf->t_channels);
    linear_bf16(tf, hidden, sin_emb, lin1, 1, tf->t_channels, tf->dim);
    iris_silu(hidden, tf->dim);
    linear_bf16(tf, out, hidden, lin2, 1, tf->dim, tf->dim);
    free(hidden);
}

/* ========================================================================
 * RoPE Tables
 * ======================================================================== */

/* Fill cos/sin rows [seq, head_dim/2] for tokens at 4-axis positions
 * pos[seq][4]. Frequencies are computed in double like the reference. */
static void rope_rows(iris_transformer_flux3_t *tf, float *cos_t, float *sin_t,
                      const float (*pos)[F3_ROPE_AXES], int seq) {
    int half = tf->head_dim / 2;
    for (int s = 0; s < seq; s++) {
        int p = 0;
        for (int ax = 0; ax < F3_ROPE_AXES; ax++) {
            int d = tf->axes_dim[ax];
            for (int j = 0; j < d / 2; j++, p++) {
                double freq = 1.0 / pow(tf->rope_theta, (double)(2 * j) / d);
                double angle = pos[s][ax] * freq;
                cos_t[(size_t)s * half + p] = (float)cos(angle);
                sin_t[(size_t)s * half + p] = (float)sin(angle);
            }
        }
    }
}

/* Positions of an h x w image token grid (row-major), l = l_coord. */
static void image_positions(float (*pos)[F3_ROPE_AXES], int h, int w, float l_coord) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            float *p = pos[r * w + c];
            p[0] = 0.0f;
            p[1] = 0.5f * r;
            p[2] = 0.5f * c;
            p[3] = l_coord;
        }
    }
}

/* RoPE tables [total, head_dim/2] for the joint sequence [TXT, IMG, REFS]
 * (caller frees). */
static void joint_rope(iris_transformer_flux3_t *tf, int txt_seq, int img_h, int img_w,
                       const int *ref_h, const int *ref_w, int num_refs,
                       float **rope_cos, float **rope_sin) {
    int total = txt_seq + img_h * img_w;
    for (int i = 0; i < num_refs; i++) total += ref_h[i] * ref_w[i];
    int half = tf->head_dim / 2;

    float (*pos)[F3_ROPE_AXES] = calloc(total, sizeof(*pos));
    for (int s = 0; s < txt_seq; s++) pos[s][3] = (float)s;
    image_positions(pos + txt_seq, img_h, img_w, 0.0f);
    int off = txt_seq + img_h * img_w;
    for (int i = 0; i < num_refs; i++) {
        image_positions(pos + off, ref_h[i], ref_w[i], -(float)(i + 1));
        off += ref_h[i] * ref_w[i];
    }
    *rope_cos = malloc((size_t)total * half * sizeof(float));
    *rope_sin = malloc((size_t)total * half * sizeof(float));
    rope_rows(tf, *rope_cos, *rope_sin, (const float (*)[F3_ROPE_AXES])pos, total);
    free(pos);
}

/* ========================================================================
 * Block Forward (CPU)
 * ======================================================================== */

/* Work memory layout for a sequence of seq tokens (offsets in floats). */
typedef struct {
    float *norm, *q, *k, *v, *attn, *proj, *ff_hidden, *ff_act, *scores;
} f3_work_t;

static int ensure_work(iris_transformer_flux3_t *tf, int seq, f3_work_t *w) {
    size_t sd = (size_t)seq * tf->dim;
    size_t sm = (size_t)seq * tf->mlp;
    size_t total = 6 * sd + 3 * sm + (size_t)F3_ATTN_QCHUNK * seq;
    if (seq > tf->work_seq) {
        free(tf->work);
        tf->work = malloc(total * sizeof(float));
        tf->work_seq = tf->work ? seq : 0;
        if (!tf->work) return 0;
    }
    float *p = tf->work;
    w->norm = p;       p += sd;
    w->q = p;          p += sd;
    w->k = p;          p += sd;
    w->v = p;          p += sd;
    w->attn = p;       p += sd;
    w->proj = p;       p += sd;
    w->ff_hidden = p;  p += 2 * sm;
    w->ff_act = p;     p += sm;
    w->scores = p;
    return 1;
}

/* One block over x[seq, dim] (in place). Each segment of tokens gets its
 * own shift/scale/gate; per-mode blocks pass a single segment.
 *   n = LayerNorm(x) * (1 + scale) + shift
 *   x = x + gate * (attention(n) + ffn(n)) */
static void block_forward(iris_transformer_flux3_t *tf, const f3_block_t *b,
                          float *x, int seq, const f3_segment_t *segs, int nsegs,
                          const float *rope_cos, const float *rope_sin,
                          f3_work_t *w) {
    int dim = tf->dim, mlp = tf->mlp;

    layer_norm(w->norm, x, seq, dim, tf->eps);
    for (int g = 0; g < nsegs; g++) {
        const float *shift = segs[g].mod, *scale = shift + dim;
        for (int s = segs[g].start; s < segs[g].start + segs[g].len; s++) {
            float *n = w->norm + (size_t)s * dim;
            for (int i = 0; i < dim; i++) n[i] = n[i] * (1.0f + scale[i]) + shift[i];
        }
    }

    /* Attention */
    linear_bf16(tf, w->q, w->norm, b->to_q, seq, dim, dim);
    linear_bf16(tf, w->k, w->norm, b->to_k, seq, dim, dim);
    linear_bf16(tf, w->v, w->norm, b->to_v, seq, dim, dim);
    head_rms_norm(w->q, b->norm_q, seq, tf->heads, tf->head_dim, tf->eps);
    head_rms_norm(w->k, b->norm_k, seq, tf->heads, tf->head_dim, tf->eps);
    apply_rope(w->q, rope_cos, rope_sin, seq, tf->heads, tf->head_dim);
    apply_rope(w->k, rope_cos, rope_sin, seq, tf->heads, tf->head_dim);
    attention(w->attn, w->q, w->k, w->v, w->scores, seq, tf->heads, tf->head_dim);
    linear_bf16(tf, w->proj, w->attn, b->to_out, seq, dim, dim);

    /* SwiGLU FFN on the same normalized input */
    linear_bf16(tf, w->ff_hidden, w->norm, b->ff_in, seq, dim, 2 * mlp);
    for (int s = 0; s < seq; s++) {
        const float *gate = w->ff_hidden + (size_t)s * 2 * mlp;
        const float *up = gate + mlp;
        float *a = w->ff_act + (size_t)s * mlp;
        for (int i = 0; i < mlp; i++) a[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
    }
    linear_bf16(tf, w->attn, w->ff_act, b->ff_out, seq, mlp, dim);

    /* Gated residual */
    for (int g = 0; g < nsegs; g++) {
        const float *gate = segs[g].mod + 2 * dim;
        for (int s = segs[g].start; s < segs[g].start + segs[g].len; s++) {
            float *xr = x + (size_t)s * dim;
            const float *pa = w->proj + (size_t)s * dim;
            const float *pf = w->attn + (size_t)s * dim;
            for (int i = 0; i < dim; i++) xr[i] += gate[i] * (pa[i] + pf[i]);
        }
    }
}

/* Transpose a channel-first latent [ch, n] into tokens [n, ch]. */
static void latent_to_tokens(float *tokens, const float *latent, int ch, int n) {
    for (int c = 0; c < ch; c++)
        for (int i = 0; i < n; i++)
            tokens[(size_t)i * ch + c] = latent[(size_t)c * n + i];
}

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Forward (CUDA / ROCm)
 * ======================================================================== */

/* Device copy of a block. The projections are fused into two GEMMs:
 * w_in = [to_q; to_k; to_v; ff_in] maps the normalized input to q, k, v and
 * the FFN gate/up halves; w_out = [to_out | ff_out] maps the concatenated
 * [attention output | FFN activation] back to dim, summing both terms. */
typedef struct {
    uint16_t *w_in;          /* [3*dim + 2*mlp, dim] */
    uint16_t *w_out;         /* [dim, dim + mlp] */
    float *norm_q, *norm_k;  /* [head_dim] */
} f3_dev_block_t;

struct f3_dev {
    f3_dev_block_t *mode_blocks[F3_NUM_MODES];
    f3_dev_block_t *single_blocks;
    uint16_t *t_lin1, *t_lin2, *tt_lin1, *tt_lin2;
    uint16_t *x_emb[F3_NUM_MODES], *ctx_emb;
    uint16_t *early_mod[F3_NUM_MODES], *single_mod[F3_NUM_MODES];
    uint16_t *norm_out, *proj_out;

    /* Activations for up to seq tokens */
    int seq;
    float *x;                /* [seq, dim] residual stream */
    uint16_t *norm;          /* [seq, dim] modulated LayerNorm output */
    uint16_t *h;             /* [seq, 3*dim + 2*mlp] q, k, v, gate, up */
    uint16_t *cat;           /* [seq, dim + mlp] attention out, FFN activation */
    float *out;              /* [seq, dim] block output */
    float *rope_cos, *rope_sin;  /* [seq, head_dim/2] */

    /* Model inputs (f32 upload + bf16 copy), grown on demand */
    size_t in_elems;
    float *in_f32;
    uint16_t *in_bf16;

    /* Timestep embeddings and modulation vectors, see vec_layout() */
    float *vec;
    uint16_t *vec_bf16;
};

/* Offsets (in floats) inside dev->vec. */
typedef struct {
    size_t sin, hidden, temb, target, mod, fin, total;
} f3_vec_layout_t;

static f3_vec_layout_t vec_layout(const iris_transformer_flux3_t *tf) {
    f3_vec_layout_t v;
    size_t dim = tf->dim;
    v.sin = 0;                               /* 3 sinusoidal embeddings */
    v.hidden = v.sin + 3 * tf->t_channels;   /* timestep MLP hidden */
    v.temb = v.hidden + dim;                 /* image / zero embeddings */
    v.target = v.temb + 2 * dim;             /* target timestep embedding */
    v.mod = v.target + dim;                  /* early/single mods per mode */
    v.fin = v.mod + (size_t)2 * F3_NUM_MODES * 3 * dim;  /* output scale, shift */
    v.total = v.fin + 2 * dim;
    return v;
}

static uint16_t *dev_bf16(const uint16_t *src, size_t n) {
    uint16_t *d = iris_dev_alloc(n * sizeof(uint16_t));
    if (d) iris_dev_upload(d, src, n * sizeof(uint16_t));
    return d;
}

static float *dev_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

static int dev_load_block(iris_transformer_flux3_t *tf, f3_dev_block_t *db, const f3_block_t *b) {
    size_t dim = tf->dim, mlp = tf->mlp, es = sizeof(uint16_t);
    size_t dd = dim * dim;

    /* w_in: vertical concatenation, the parts are contiguous rows. */
    db->w_in = iris_dev_alloc((3 * dim + 2 * mlp) * dim * es);
    if (db->w_in) {
        iris_dev_upload(db->w_in, b->to_q, dd * es);
        iris_dev_upload(db->w_in + dd, b->to_k, dd * es);
        iris_dev_upload(db->w_in + 2 * dd, b->to_v, dd * es);
        iris_dev_upload(db->w_in + 3 * dd, b->ff_in, 2 * mlp * dim * es);
    }

    /* w_out: each row is the to_out row followed by the ff_out row. */
    db->w_out = iris_dev_alloc(dim * (dim + mlp) * es);
    if (db->w_out) {
        iris_dev_upload_2d(db->w_out, (dim + mlp) * es, b->to_out, dim * es, dim * es, dim);
        iris_dev_upload_2d(db->w_out + dim, (dim + mlp) * es, b->ff_out, mlp * es, mlp * es, dim);
    }

    db->norm_q = dev_f32(b->norm_q, tf->head_dim);
    db->norm_k = dev_f32(b->norm_k, tf->head_dim);
    return db->w_in && db->w_out && db->norm_q && db->norm_k;
}

static void dev_free_block(f3_dev_block_t *db) {
    iris_dev_free(db->w_in);
    iris_dev_free(db->w_out);
    iris_dev_free(db->norm_q);
    iris_dev_free(db->norm_k);
}

static void dev_free_activations(struct f3_dev *d) {
    iris_dev_free(d->x);
    iris_dev_free(d->norm);
    iris_dev_free(d->h);
    iris_dev_free(d->cat);
    iris_dev_free(d->out);
    iris_dev_free(d->rope_cos);
    iris_dev_free(d->rope_sin);
    d->x = d->out = d->rope_cos = d->rope_sin = NULL;
    d->norm = d->h = d->cat = NULL;
    d->seq = 0;
}

static void dev_free(iris_transformer_flux3_t *tf) {
    struct f3_dev *d = tf->dev;
    if (!d) return;
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (!d->mode_blocks[m]) continue;
        for (int l = 0; l < tf->num_mode_layers; l++) dev_free_block(&d->mode_blocks[m][l]);
        free(d->mode_blocks[m]);
        iris_dev_free(d->x_emb[m]);
        iris_dev_free(d->early_mod[m]);
        iris_dev_free(d->single_mod[m]);
    }
    if (d->single_blocks) {
        for (int l = 0; l < tf->num_single_layers; l++) dev_free_block(&d->single_blocks[l]);
        free(d->single_blocks);
    }
    iris_dev_free(d->t_lin1);
    iris_dev_free(d->t_lin2);
    iris_dev_free(d->tt_lin1);
    iris_dev_free(d->tt_lin2);
    iris_dev_free(d->ctx_emb);
    iris_dev_free(d->norm_out);
    iris_dev_free(d->proj_out);
    dev_free_activations(d);
    iris_dev_free(d->in_f32);
    iris_dev_free(d->in_bf16);
    iris_dev_free(d->vec);
    iris_dev_free(d->vec_bf16);
    free(d);
    tf->dev = NULL;
}

/* Upload the model to the device. Returns 0 (leaving tf->dev NULL) if the
 * device runs out of memory, so the CPU path is used instead. */
static int dev_load(iris_transformer_flux3_t *tf) {
    struct f3_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    tf->dev = d;
    size_t dim = tf->dim, ok = 1;

    d->t_lin1 = dev_bf16(tf->t_lin1, dim * tf->t_channels);
    d->t_lin2 = dev_bf16(tf->t_lin2, dim * dim);
    ok &= d->t_lin1 && d->t_lin2;
    if (tf->flow_map) {
        d->tt_lin1 = dev_bf16(tf->tt_lin1, dim * tf->t_channels);
        d->tt_lin2 = dev_bf16(tf->tt_lin2, dim * dim);
        ok &= d->tt_lin1 && d->tt_lin2;
    }
    d->ctx_emb = dev_bf16(tf->ctx_emb, dim * tf->text_dim);
    d->x_emb[F3_IMG] = dev_bf16(tf->x_emb[F3_IMG], dim * tf->in_channels);
    d->x_emb[F3_COND] = dev_bf16(tf->x_emb[F3_COND], dim * tf->cond_in_channels);
    d->norm_out = dev_bf16(tf->norm_out, 2 * dim * dim);
    d->proj_out = dev_bf16(tf->proj_out, (size_t)tf->in_channels * dim);
    ok &= d->ctx_emb && d->x_emb[F3_IMG] && d->x_emb[F3_COND] && d->norm_out && d->proj_out;

    for (int m = 0; ok && m < F3_NUM_MODES; m++) {
        d->early_mod[m] = dev_bf16(tf->early_mod[m], 3 * dim * dim);
        d->single_mod[m] = dev_bf16(tf->single_mod[m], 3 * dim * dim);
        d->mode_blocks[m] = calloc(tf->num_mode_layers, sizeof(f3_dev_block_t));
        ok &= d->early_mod[m] && d->single_mod[m] && d->mode_blocks[m];
        for (int l = 0; ok && l < tf->num_mode_layers; l++)
            ok &= dev_load_block(tf, &d->mode_blocks[m][l], &tf->mode_blocks[m][l]);
    }
    d->single_blocks = ok ? calloc(tf->num_single_layers, sizeof(f3_dev_block_t)) : NULL;
    ok &= d->single_blocks != NULL;
    for (int l = 0; ok && l < tf->num_single_layers; l++)
        ok &= dev_load_block(tf, &d->single_blocks[l], &tf->single_blocks[l]);

    f3_vec_layout_t vl = vec_layout(tf);
    d->vec = ok ? iris_dev_alloc(vl.total * sizeof(float)) : NULL;
    d->vec_bf16 = ok ? iris_dev_alloc(vl.total * sizeof(uint16_t)) : NULL;
    ok &= d->vec && d->vec_bf16;

    if (!ok) {
        fprintf(stderr, "FLUX.3: not enough GPU memory, using the CPU\n");
        dev_free(tf);
        return 0;
    }
    iris_dev_sync();
    return 1;
}

static int dev_ensure(iris_transformer_flux3_t *tf, int seq, size_t in_elems) {
    struct f3_dev *d = tf->dev;
    size_t dim = tf->dim, mlp = tf->mlp, half = tf->head_dim / 2;
    if (seq > d->seq) {
        dev_free_activations(d);
        d->x = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->norm = iris_dev_alloc((size_t)seq * dim * sizeof(uint16_t));
        d->h = iris_dev_alloc((size_t)seq * (3 * dim + 2 * mlp) * sizeof(uint16_t));
        d->cat = iris_dev_alloc((size_t)seq * (dim + mlp) * sizeof(uint16_t));
        d->out = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->rope_cos = iris_dev_alloc((size_t)seq * half * sizeof(float));
        d->rope_sin = iris_dev_alloc((size_t)seq * half * sizeof(float));
        if (!d->x || !d->norm || !d->h || !d->cat || !d->out || !d->rope_cos || !d->rope_sin) {
            dev_free_activations(d);
            return 0;
        }
        d->seq = seq;
    }
    if (in_elems > d->in_elems) {
        iris_dev_free(d->in_f32);
        iris_dev_free(d->in_bf16);
        d->in_f32 = iris_dev_alloc(in_elems * sizeof(float));
        d->in_bf16 = iris_dev_alloc(in_elems * sizeof(uint16_t));
        d->in_elems = (d->in_f32 && d->in_bf16) ? in_elems : 0;
        if (!d->in_elems) return 0;
    }
    return 1;
}

/* out[dim] = lin2(silu(lin1(sinusoidal))) for a sinusoidal embedding
 * already on the device. */
static void dev_timestep_embed(iris_transformer_flux3_t *tf, float *out, const float *sin_emb,
                               const uint16_t *lin1, const uint16_t *lin2) {
    struct f3_dev *d = tf->dev;
    f3_vec_layout_t vl = vec_layout(tf);
    float *hidden = d->vec + vl.hidden;
    uint16_t *sin_b = d->vec_bf16 + vl.sin, *hidden_b = d->vec_bf16 + vl.hidden;
    iris_dev_f32_to_bf16(sin_b, sin_emb, tf->t_channels);
    iris_dev_linear_f32(hidden, sin_b, lin1, 1, tf->t_channels, tf->dim);
    iris_dev_silu(hidden, tf->dim);
    iris_dev_f32_to_bf16(hidden_b, hidden, tf->dim);
    iris_dev_linear_f32(out, hidden_b, lin2, 1, tf->dim, tf->dim);
}

/* Project input rows (f32 on the host) into the residual stream. */
static void dev_embed(iris_transformer_flux3_t *tf, float *x, const float *in, int rows,
                      int in_dim, const uint16_t *W) {
    struct f3_dev *d = tf->dev;
    size_t n = (size_t)rows * in_dim;
    iris_dev_upload(d->in_f32, in, n * sizeof(float));
    iris_dev_f32_to_bf16(d->in_bf16, d->in_f32, n);
    iris_dev_linear_f32(x, d->in_bf16, W, rows, in_dim, tf->dim);
}

/* One block on the device over x[seq, dim], same math as block_forward().
 * Segment modulation vectors are device pointers. */
static void dev_block_forward(iris_transformer_flux3_t *tf, const f3_dev_block_t *b,
                              float *x, int seq, const f3_segment_t *segs, int nsegs,
                              const float *rope_cos, const float *rope_sin) {
    struct f3_dev *d = tf->dev;
    int dim = tf->dim, mlp = tf->mlp, hd = tf->head_dim;
    int ldh = 3 * dim + 2 * mlp, ldc = dim + mlp;

    for (int g = 0; g < nsegs; g++) {
        size_t off = (size_t)segs[g].start * dim;
        iris_dev_layer_norm_mod(d->norm + off, x + off, segs[g].mod, segs[g].mod + dim,
                                segs[g].len, dim, tf->eps);
    }
    iris_dev_linear_bf16(d->h, d->norm, b->w_in, seq, dim, ldh);
    iris_dev_head_norm_rope(d->h, ldh, b->norm_q, rope_cos, rope_sin, seq, tf->heads, hd, tf->eps, 0);
    iris_dev_head_norm_rope(d->h + dim, ldh, b->norm_k, rope_cos, rope_sin, seq, tf->heads, hd, tf->eps, 0);
    iris_dev_attention(d->cat, ldc, d->h, d->h + dim, d->h + 2 * dim, ldh, seq, seq,
                       tf->heads, tf->heads, hd, 1.0f / sqrtf((float)hd), 0, NULL);
    iris_dev_swiglu(d->cat + dim, ldc, d->h + 3 * dim, d->h + 3 * dim + mlp, ldh, seq, mlp);
    iris_dev_linear_f32(d->out, d->cat, b->w_out, seq, dim + mlp, dim);
    for (int g = 0; g < nsegs; g++) {
        size_t off = (size_t)segs[g].start * dim;
        iris_dev_residual(x + off, d->out + off, segs[g].mod + 2 * dim, segs[g].len, dim);
    }
}

static float *forward_dev(iris_transformer_flux3_t *tf,
                          const float *img, int img_h, int img_w,
                          const float *txt, int txt_seq,
                          const float *const *refs, const int *ref_h,
                          const int *ref_w, int num_refs,
                          float t, float t_target) {
    struct f3_dev *d = tf->dev;
    int dim = tf->dim, half = tf->head_dim / 2;
    int img_seq = img_h * img_w;
    int cond_seq = 0;
    for (int i = 0; i < num_refs; i++) cond_seq += ref_h[i] * ref_w[i];
    int total = txt_seq + img_seq + cond_seq;
    size_t in_elems = (size_t)txt_seq * tf->text_dim;
    if ((size_t)img_seq * tf->in_channels > in_elems) in_elems = (size_t)img_seq * tf->in_channels;
    if ((size_t)cond_seq * tf->cond_in_channels > in_elems)
        in_elems = (size_t)cond_seq * tf->cond_in_channels;
    if (!dev_ensure(tf, total, in_elems)) {
        fprintf(stderr, "FLUX.3: GPU allocation failed for %d tokens\n", total);
        return NULL;
    }
    f3_vec_layout_t vl = vec_layout(tf);

    /* 1. Timestep embeddings (image at t, text and references at 0), plus
     *    the target timestep for flow maps, then SiLU. */
    int tc = tf->t_channels;
    float sin_host[3 * 512];
    sinusoidal_embedding(sin_host, t * 1000.0f, tc);
    sinusoidal_embedding(sin_host + tc, 0.0f, tc);
    sinusoidal_embedding(sin_host + 2 * tc, t_target * 1000.0f, tc);
    iris_dev_upload(d->vec + vl.sin, sin_host, 3 * tc * sizeof(float));
    float *temb_img = d->vec + vl.temb, *temb_zero = temb_img + dim;
    dev_timestep_embed(tf, temb_img, d->vec + vl.sin, d->t_lin1, d->t_lin2);
    dev_timestep_embed(tf, temb_zero, d->vec + vl.sin + tc, d->t_lin1, d->t_lin2);
    if (tf->flow_map) {
        float *tt = d->vec + vl.target;
        dev_timestep_embed(tf, tt, d->vec + vl.sin + 2 * tc, d->tt_lin1, d->tt_lin2);
        iris_dev_add(temb_img, tt, dim);
        iris_dev_add(temb_zero, tt, dim);
    }
    iris_dev_silu(temb_img, 2 * dim);
    uint16_t *act_img = d->vec_bf16 + vl.temb, *act_zero = act_img + dim;
    iris_dev_f32_to_bf16(act_img, temb_img, 2 * dim);

    /* 2. Modulation vectors and output layer scale/shift. */
    float *early[F3_NUM_MODES], *single[F3_NUM_MODES];
    for (int m = 0; m < F3_NUM_MODES; m++) {
        const uint16_t *act = (m == F3_IMG) ? act_img : act_zero;
        early[m] = d->vec + vl.mod + (size_t)(2 * m) * 3 * dim;
        single[m] = d->vec + vl.mod + (size_t)(2 * m + 1) * 3 * dim;
        if (m == F3_COND && num_refs == 0) continue;
        iris_dev_linear_f32(early[m], act, d->early_mod[m], 1, dim, 3 * dim);
        iris_dev_linear_f32(single[m], act, d->single_mod[m], 1, dim, 3 * dim);
    }
    float *fin = d->vec + vl.fin;
    iris_dev_linear_f32(fin, act_img, d->norm_out, 1, dim, 2 * dim);

    /* 3. Input projections into the joint sequence [TXT, IMG, COND]. */
    float *xi = d->x + (size_t)txt_seq * dim;
    dev_embed(tf, d->x, txt, txt_seq, tf->text_dim, d->ctx_emb);
    float *tokens = malloc(in_elems * sizeof(float));
    latent_to_tokens(tokens, img, tf->in_channels, img_seq);
    dev_embed(tf, xi, tokens, img_seq, tf->in_channels, d->x_emb[F3_IMG]);
    float *xr = xi + (size_t)img_seq * dim;
    for (int i = 0; i < num_refs; i++) {
        int n = ref_h[i] * ref_w[i];
        latent_to_tokens(tokens, refs[i], tf->cond_in_channels, n);
        dev_embed(tf, xr, tokens, n, tf->cond_in_channels, d->x_emb[F3_COND]);
        xr += (size_t)n * dim;
    }
    free(tokens);

    /* 4. RoPE tables. */
    float *rope_cos, *rope_sin;
    joint_rope(tf, txt_seq, img_h, img_w, ref_h, ref_w, num_refs, &rope_cos, &rope_sin);
    iris_dev_upload(d->rope_cos, rope_cos, (size_t)total * half * sizeof(float));
    iris_dev_upload(d->rope_sin, rope_sin, (size_t)total * half * sizeof(float));
    free(rope_cos);
    free(rope_sin);

    /* 5. Per-mode blocks. */
    int mode_start[F3_NUM_MODES] = {0, txt_seq, txt_seq + img_seq};
    int mode_len[F3_NUM_MODES] = {txt_seq, img_seq, cond_seq};
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (mode_len[m] == 0) continue;
        f3_segment_t seg = {0, mode_len[m], early[m]};
        float *xm = d->x + (size_t)mode_start[m] * dim;
        for (int l = 0; l < tf->num_mode_layers; l++)
            dev_block_forward(tf, &d->mode_blocks[m][l], xm, mode_len[m], &seg, 1,
                              d->rope_cos + (size_t)mode_start[m] * half,
                              d->rope_sin + (size_t)mode_start[m] * half);
    }

    /* 6. Joint blocks. */
    f3_segment_t segs[F3_NUM_MODES];
    int nsegs = 0;
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (mode_len[m] == 0) continue;
        segs[nsegs].start = mode_start[m];
        segs[nsegs].len = mode_len[m];
        segs[nsegs].mod = single[m];
        nsegs++;
    }
    for (int l = 0; l < tf->num_single_layers; l++)
        dev_block_forward(tf, &d->single_blocks[l], d->x, total, segs, nsegs,
                          d->rope_cos, d->rope_sin);

    /* 7. Output layer on the image tokens. */
    iris_dev_layer_norm_mod(d->norm, xi, fin + dim, fin, img_seq, dim, tf->eps);
    iris_dev_linear_f32(d->out, d->norm, d->proj_out, img_seq, dim, tf->in_channels);
    float *vel_tokens = malloc((size_t)img_seq * tf->in_channels * sizeof(float));
    iris_dev_download(vel_tokens, d->out, (size_t)img_seq * tf->in_channels * sizeof(float));

    float *out = malloc((size_t)tf->in_channels * img_seq * sizeof(float));
    for (int s = 0; s < img_seq; s++)
        for (int c = 0; c < tf->in_channels; c++)
            out[(size_t)c * img_seq + s] = vel_tokens[(size_t)s * tf->in_channels + c];
    free(vel_tokens);
    return out;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* Predict the velocity of the image latent.
 *
 * img:      noisy latent [in_channels, img_h * img_w] (channel-first)
 * txt:      text embeddings [txt_seq, text_dim]
 * refs:     optional reference latents, refs[i] is [cond_in_channels,
 *           ref_h[i] * ref_w[i]] (VAE-encoded, BN-normalized, patchified)
 * t:        flow timestep (sigma) in [0, 1]
 * t_target: timestep the step jumps to (used by flow-map models only)
 *
 * Returns the velocity [in_channels, img_h * img_w] (caller frees). */
float *iris_transformer_forward_flux3(iris_transformer_flux3_t *tf,
                                      const float *img, int img_h, int img_w,
                                      const float *txt, int txt_seq,
                                      const float *const *refs, const int *ref_h,
                                      const int *ref_w, int num_refs,
                                      float t, float t_target) {
#ifdef IRIS_DEV
    if (tf->dev)
        return forward_dev(tf, img, img_h, img_w, txt, txt_seq, refs, ref_h, ref_w,
                           num_refs, t, t_target);
#endif
    int dim = tf->dim;
    int img_seq = img_h * img_w;
    int cond_seq = 0;
    for (int i = 0; i < num_refs; i++) cond_seq += ref_h[i] * ref_w[i];
    int total = txt_seq + img_seq + cond_seq;
    int half = tf->head_dim / 2;

    f3_work_t w;
    if (!ensure_work(tf, total, &w)) return NULL;

    /* 1. Timestep embeddings: image tokens at t, text and references at 0. */
    float *temb_img = malloc(dim * sizeof(float));
    float *temb_zero = malloc(dim * sizeof(float));
    timestep_embed(tf, temb_img, t, tf->t_lin1, tf->t_lin2);
    timestep_embed(tf, temb_zero, 0.0f, tf->t_lin1, tf->t_lin2);
    if (tf->flow_map) {
        float *tt = malloc(dim * sizeof(float));
        timestep_embed(tf, tt, t_target, tf->tt_lin1, tf->tt_lin2);
        for (int i = 0; i < dim; i++) {
            temb_img[i] += tt[i];
            temb_zero[i] += tt[i];
        }
        free(tt);
    }
    iris_silu(temb_img, dim);
    iris_silu(temb_zero, dim);

    /* 2. Modulation vectors, one per mode for each block family. */
    float *mods = malloc((size_t)F3_NUM_MODES * 2 * 3 * dim * sizeof(float));
    float *early[F3_NUM_MODES], *single[F3_NUM_MODES];
    for (int m = 0; m < F3_NUM_MODES; m++) {
        const float *temb = (m == F3_IMG) ? temb_img : temb_zero;
        early[m] = mods + (size_t)(2 * m) * 3 * dim;
        single[m] = mods + (size_t)(2 * m + 1) * 3 * dim;
        if (m == F3_COND && num_refs == 0) continue;
        linear_bf16(tf, early[m], temb, tf->early_mod[m], 1, dim, 3 * dim);
        linear_bf16(tf, single[m], temb, tf->single_mod[m], 1, dim, 3 * dim);
    }

    /* 3. Input projections into the joint sequence [TXT, IMG, COND]. */
    float *x = malloc((size_t)total * dim * sizeof(float));
    float *xt = x;
    float *xi = x + (size_t)txt_seq * dim;
    float *xc = xi + (size_t)img_seq * dim;
    float *tokens = malloc((size_t)(img_seq > cond_seq ? img_seq : cond_seq) *
                           tf->in_channels * sizeof(float));
    linear_bf16(tf, xt, txt, tf->ctx_emb, txt_seq, tf->text_dim, dim);
    latent_to_tokens(tokens, img, tf->in_channels, img_seq);
    linear_bf16(tf, xi, tokens, tf->x_emb[F3_IMG], img_seq, tf->in_channels, dim);
    float *xr = xc;
    for (int i = 0; i < num_refs; i++) {
        int n = ref_h[i] * ref_w[i];
        latent_to_tokens(tokens, refs[i], tf->cond_in_channels, n);
        linear_bf16(tf, xr, tokens, tf->x_emb[F3_COND], n, tf->cond_in_channels, dim);
        xr += (size_t)n * dim;
    }
    free(tokens);

    /* 4. RoPE tables for the joint sequence. */
    float *rope_cos, *rope_sin;
    joint_rope(tf, txt_seq, img_h, img_w, ref_h, ref_w, num_refs, &rope_cos, &rope_sin);

    /* 5. Per-mode blocks, each mode attending within itself. */
    int mode_start[F3_NUM_MODES] = {0, txt_seq, txt_seq + img_seq};
    int mode_len[F3_NUM_MODES] = {txt_seq, img_seq, cond_seq};
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (mode_len[m] == 0) continue;
        f3_segment_t seg = {0, mode_len[m], early[m]};
        float *xm = x + (size_t)mode_start[m] * dim;
        const float *mc = rope_cos + (size_t)mode_start[m] * half;
        const float *ms = rope_sin + (size_t)mode_start[m] * half;
        for (int l = 0; l < tf->num_mode_layers; l++)
            block_forward(tf, &tf->mode_blocks[m][l], xm, mode_len[m], &seg, 1, mc, ms, &w);
    }

    /* 6. Joint blocks over the whole sequence. */
    f3_segment_t segs[F3_NUM_MODES];
    int nsegs = 0;
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (mode_len[m] == 0) continue;
        segs[nsegs].start = mode_start[m];
        segs[nsegs].len = mode_len[m];
        segs[nsegs].mod = single[m];
        nsegs++;
    }
    for (int l = 0; l < tf->num_single_layers; l++)
        block_forward(tf, &tf->single_blocks[l], x, total, segs, nsegs, rope_cos, rope_sin, &w);

    /* 7. Output layer on the image tokens: LN * (1 + scale) + shift, proj. */
    float *fin = malloc(2 * dim * sizeof(float));
    linear_bf16(tf, fin, temb_img, tf->norm_out, 1, dim, 2 * dim);
    layer_norm(w.norm, xi, img_seq, dim, tf->eps);
    for (int s = 0; s < img_seq; s++) {
        float *n = w.norm + (size_t)s * dim;
        for (int i = 0; i < dim; i++) n[i] = n[i] * (1.0f + fin[i]) + fin[dim + i];
    }
    float *vel_tokens = w.q;  /* [img_seq, in_channels] scratch */
    linear_bf16(tf, vel_tokens, w.norm, tf->proj_out, img_seq, dim, tf->in_channels);

    float *out = malloc((size_t)tf->in_channels * img_seq * sizeof(float));
    for (int s = 0; s < img_seq; s++)
        for (int c = 0; c < tf->in_channels; c++)
            out[(size_t)c * img_seq + s] = vel_tokens[(size_t)s * tf->in_channels + c];

    free(fin);
    free(rope_cos);
    free(rope_sin);
    free(x);
    free(mods);
    free(temb_img);
    free(temb_zero);
    return out;
}

/* ========================================================================
 * Loading
 * ======================================================================== */

static const safetensor_t *find_tensor(iris_transformer_flux3_t *tf, const char *name,
                                       safetensors_file_t **file) {
    for (int f = 0; f < tf->num_sf; f++) {
        const safetensor_t *t = safetensors_find(tf->sf[f], name);
        if (t) {
            *file = tf->sf[f];
            return t;
        }
    }
    return NULL;
}

/* Zero-copy pointer to a bf16 weight inside the mmapped shard. */
static const uint16_t *get_bf16(iris_transformer_flux3_t *tf, const char *name) {
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(tf, name, &sf);
    if (!t || !safetensor_is_bf16(t)) {
        fprintf(stderr, "Flux3: missing bf16 tensor %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

/* Small tensors are converted to f32 (caller frees). */
static float *get_f32(iris_transformer_flux3_t *tf, const char *name) {
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(tf, name, &sf);
    if (!t) {
        fprintf(stderr, "Flux3: missing tensor %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static int load_block(iris_transformer_flux3_t *tf, f3_block_t *b, const char *prefix) {
    char name[256];
#define F3_BF16(field, suffix) \
    snprintf(name, sizeof(name), "%s.%s", prefix, suffix); \
    b->field = get_bf16(tf, name);
    F3_BF16(to_q, "attn.to_q.weight");
    F3_BF16(to_k, "attn.to_k.weight");
    F3_BF16(to_v, "attn.to_v.weight");
    F3_BF16(to_out, "attn.to_out.0.weight");
    F3_BF16(ff_in, "ff.linear_in.weight");
    F3_BF16(ff_out, "ff.linear_out.weight");
#undef F3_BF16
    snprintf(name, sizeof(name), "%s.attn.norm_q.weight", prefix);
    b->norm_q = get_f32(tf, name);
    snprintf(name, sizeof(name), "%s.attn.norm_k.weight", prefix);
    b->norm_k = get_f32(tf, name);
    return b->to_q && b->to_k && b->to_v && b->to_out && b->ff_in && b->ff_out &&
           b->norm_q && b->norm_k;
}

/* Read an integer / float / bool config value, keeping def if absent. */
static double config_num(const char *json, const char *key, double def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p, ':'))) return def;
    p++;
    while (*p == ' ') p++;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return atof(p);
}

static int open_shards(iris_transformer_flux3_t *tf, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json",
             model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
                 model_dir);
        tf->sf[0] = safetensors_open(path);
        tf->num_sf = tf->sf[0] ? 1 : 0;
        return tf->num_sf;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    if (!json) { fclose(f); return 0; }
    size_t n = fread(json, 1, size, f);
    json[n] = '\0';
    fclose(f);

    /* Unique shard file names among the weight_map values. */
    char names[F3_MAX_SHARDS][128];
    int count = 0;
    for (char *p = json; (p = strstr(p, ".safetensors\"")) != NULL; p++) {
        char *start = p;
        while (start > json && start[-1] != '"') start--;
        int len = (int)(p - start) + (int)strlen(".safetensors");
        if (len >= 128) continue;
        char fname[128];
        memcpy(fname, start, len);
        fname[len] = '\0';
        int seen = 0;
        for (int i = 0; i < count; i++) if (!strcmp(names[i], fname)) seen = 1;
        if (!seen && count < F3_MAX_SHARDS) strcpy(names[count++], fname);
    }
    free(json);

    for (int i = 0; i < count; i++) {
        snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, names[i]);
        tf->sf[i] = safetensors_open(path);
        if (!tf->sf[i]) {
            fprintf(stderr, "Flux3: failed to open %s\n", path);
            return 0;
        }
        tf->num_sf = i + 1;
    }
    return tf->num_sf;
}

/* Load the transformer from <model_dir>/transformer. Weights are not
 * copied: they are used straight from the mmapped safetensors shards. */
iris_transformer_flux3_t *iris_transformer_load_flux3(const char *model_dir) {
    iris_transformer_flux3_t *tf = calloc(1, sizeof(*tf));
    if (!tf) return NULL;

    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Flux3: missing %s\n", path);
        free(tf);
        return NULL;
    }
    char cfg[8192];
    size_t n = fread(cfg, 1, sizeof(cfg) - 1, f);
    cfg[n] = '\0';
    fclose(f);

    tf->heads = (int)config_num(cfg, "num_attention_heads", 24);
    tf->head_dim = (int)config_num(cfg, "attention_head_dim", 128);
    tf->dim = tf->heads * tf->head_dim;
    tf->mlp = (int)(tf->dim * config_num(cfg, "mlp_ratio", 3.0));
    tf->num_mode_layers = (int)config_num(cfg, "num_layers", 5);
    tf->num_single_layers = (int)config_num(cfg, "num_single_layers", 28);
    tf->in_channels = (int)config_num(cfg, "in_channels", 128);
    tf->cond_in_channels = (int)config_num(cfg, "cond_in_channels", 128);
    tf->text_dim = (int)config_num(cfg, "joint_attention_dim", 20480);
    tf->t_channels = (int)config_num(cfg, "timestep_channels", 256);
    tf->rope_theta = (float)config_num(cfg, "rope_theta", 10000);
    tf->eps = (float)config_num(cfg, "eps", 1e-6);
    tf->flow_map = (int)config_num(cfg, "flow_map", 0);
    for (int i = 0; i < F3_ROPE_AXES; i++) tf->axes_dim[i] = tf->head_dim / F3_ROPE_AXES;
    const char *axes = strstr(cfg, "\"axes_dims_rope\"");
    if (axes && (axes = strchr(axes, '['))) {
        for (int i = 0; i < F3_ROPE_AXES && axes; i++) {
            tf->axes_dim[i] = atoi(axes + 1);
            axes = strchr(axes + 1, ',');
        }
    }
    if (tf->t_channels > 512) goto error;

    if (!open_shards(tf, model_dir)) goto error;

    tf->t_lin1 = get_bf16(tf, "timestep_embedder.linear_1.weight");
    tf->t_lin2 = get_bf16(tf, "timestep_embedder.linear_2.weight");
    if (!tf->t_lin1 || !tf->t_lin2) goto error;
    if (tf->flow_map) {
        tf->tt_lin1 = get_bf16(tf, "target_timestep_embedder.linear_1.weight");
        tf->tt_lin2 = get_bf16(tf, "target_timestep_embedder.linear_2.weight");
        if (!tf->tt_lin1 || !tf->tt_lin2) goto error;
    }
    tf->ctx_emb = get_bf16(tf, "context_embedder.weight");
    tf->x_emb[F3_IMG] = get_bf16(tf, "x_embedders.image.weight");
    tf->x_emb[F3_COND] = get_bf16(tf, "x_embedders.image_cond.weight");
    tf->norm_out = get_bf16(tf, "norm_out.linear.weight");
    tf->proj_out = get_bf16(tf, "proj_out.weight");
    if (!tf->ctx_emb || !tf->x_emb[F3_IMG] || !tf->x_emb[F3_COND] ||
        !tf->norm_out || !tf->proj_out) goto error;

    static const char *mode_names[F3_NUM_MODES] = {"txt", "image", "image_cond"};
    char name[256];
    for (int m = 0; m < F3_NUM_MODES; m++) {
        snprintf(name, sizeof(name), "early_stream_modulations.%s.linear.weight", mode_names[m]);
        tf->early_mod[m] = get_bf16(tf, name);
        snprintf(name, sizeof(name), "single_stream_modulations.%s.linear.weight", mode_names[m]);
        tf->single_mod[m] = get_bf16(tf, name);
        if (!tf->early_mod[m] || !tf->single_mod[m]) goto error;

        tf->mode_blocks[m] = calloc(tf->num_mode_layers, sizeof(f3_block_t));
        if (!tf->mode_blocks[m]) goto error;
        for (int l = 0; l < tf->num_mode_layers; l++) {
            if (m == F3_TXT) snprintf(name, sizeof(name), "txt_mode_blocks.%d", l);
            else snprintf(name, sizeof(name), "mode_blocks.%s.%d", mode_names[m], l);
            if (!load_block(tf, &tf->mode_blocks[m][l], name)) goto error;
        }
    }
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(f3_block_t));
    if (!tf->single_blocks) goto error;
    for (int l = 0; l < tf->num_single_layers; l++) {
        snprintf(name, sizeof(name), "single_transformer_blocks.%d", l);
        if (!load_block(tf, &tf->single_blocks[l], name)) goto error;
    }

#ifdef IRIS_DEV
    if (iris_dev_available()) dev_load(tf);
#endif
    if (iris_verbose)
        fprintf(stderr, "  FLUX.3 transformer: dim=%d heads=%d blocks=%dx3+%d mlp=%d%s%s\n",
                tf->dim, tf->heads, tf->num_mode_layers, tf->num_single_layers,
                tf->mlp, tf->flow_map ? " (flow map)" : "", tf->dev ? " on GPU" : "");
    return tf;

error:
    iris_transformer_free_flux3(tf);
    return NULL;
}

static void free_block(f3_block_t *b) {
    free(b->norm_q);
    free(b->norm_k);
}

void iris_transformer_free_flux3(iris_transformer_flux3_t *tf) {
    if (!tf) return;
#ifdef IRIS_DEV
    dev_free(tf);
#endif
    for (int m = 0; m < F3_NUM_MODES; m++) {
        if (!tf->mode_blocks[m]) continue;
        for (int l = 0; l < tf->num_mode_layers; l++) free_block(&tf->mode_blocks[m][l]);
        free(tf->mode_blocks[m]);
    }
    if (tf->single_blocks) {
        for (int l = 0; l < tf->num_single_layers; l++) free_block(&tf->single_blocks[l]);
        free(tf->single_blocks);
    }
    for (int i = 0; i < tf->num_sf; i++) safetensors_close(tf->sf[i]);
    free(tf->wbuf);
    free(tf->work);
    free(tf);
}
