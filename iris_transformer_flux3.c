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
 * converts each matrix to f32 right before its matmul.
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* Transpose a channel-first latent [ch, n] into tokens [n, ch]. */
static void latent_to_tokens(float *tokens, const float *latent, int ch, int n) {
    for (int c = 0; c < ch; c++)
        for (int i = 0; i < n; i++)
            tokens[(size_t)i * ch + c] = latent[(size_t)c * n + i];
}

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
    float (*pos)[F3_ROPE_AXES] = calloc(total, sizeof(*pos));
    for (int s = 0; s < txt_seq; s++) pos[s][3] = (float)s;
    image_positions(pos + txt_seq, img_h, img_w, 0.0f);
    int off = txt_seq + img_seq;
    for (int i = 0; i < num_refs; i++) {
        image_positions(pos + off, ref_h[i], ref_w[i], -(float)(i + 1));
        off += ref_h[i] * ref_w[i];
    }
    float *rope_cos = malloc((size_t)total * half * sizeof(float));
    float *rope_sin = malloc((size_t)total * half * sizeof(float));
    rope_rows(tf, rope_cos, rope_sin, (const float (*)[F3_ROPE_AXES])pos, total);
    free(pos);

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

    if (iris_verbose)
        fprintf(stderr, "  FLUX.3 transformer: dim=%d heads=%d blocks=%dx3+%d mlp=%d%s\n",
                tf->dim, tf->heads, tf->num_mode_layers, tf->num_single_layers,
                tf->mlp, tf->flow_map ? " (flow map)" : "");
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
