/*
 * Z-Image S3-DiT Transformer Implementation
 *
 * Implements the Z-Image-Turbo (6B) Scalable Single-Stream DiT architecture.
 *
 * Architecture:
 * - 2 noise_refiner blocks (modulated, image-only self-attention)
 * - 2 context_refiner blocks (unmodulated, text-only self-attention)
 * - 30 main transformer blocks (modulated, full self-attention)
 * - 30 heads, 128 dim per head (3840 hidden)
 * - 3-axis RoPE (32+48+48 = 128 dims, theta=256)
 * - SwiGLU activation (8/3 expansion)
 * - AdaLN modulation: scale + tanh(gate) only (no shift)
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
/* Device backend (CUDA / ROCm / Metal) */
#ifdef IRIS_DEV
#define ZI_DEV
#include "iris_dev.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* ========================================================================
 * Constants
 * ======================================================================== */

#define ZI_SEQ_MULTI_OF     32      /* Pad sequences to multiples of 32 */
#define ZI_NORM_EPS         1e-5f   /* RMSNorm epsilon */
#define ZI_BF16_SDPA_SEQ    1024    /* Prefer bf16 SDPA at large sequence lengths */
#define ZI_MAX_SHARDS       32

/* Cumulative zImage timing counters (defined in iris_sample.c). */
extern double iris_timing_zi_total;
extern double iris_timing_zi_embeddings;
extern double iris_timing_zi_noise_refiner;
extern double iris_timing_zi_context_refiner;
extern double iris_timing_zi_main_blocks;
extern double iris_timing_zi_final;

#if defined(USE_METAL) || defined(ZI_DEV)
static inline double zi_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
#endif

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/* Single transformer block weights */
typedef struct {
    /* Attention */
    float *attn_q_weight;       /* [dim, dim] */
    float *attn_k_weight;       /* [dim, dim] */
    float *attn_v_weight;       /* [dim, dim] */
    float *attn_out_weight;     /* [dim, dim] */
    float *attn_norm_q;         /* [n_heads, head_dim] for QK norm */
    float *attn_norm_k;         /* [n_heads, head_dim] */
    float *attn_norm1;          /* [dim] RMSNorm before attention */
    float *attn_norm2;          /* [dim] RMSNorm after attention */

    /* FFN (SwiGLU) */
    float *ffn_w1;              /* [ffn_dim, dim] gate projection */
    float *ffn_w2;              /* [dim, ffn_dim] down projection */
    float *ffn_w3;              /* [ffn_dim, dim] up projection */
    float *ffn_norm1;           /* [dim] RMSNorm before FFN */
    float *ffn_norm2;           /* [dim] RMSNorm after FFN */

    /* AdaLN modulation (NULL for context_refiner blocks) */
    float *adaln_weight;        /* [4*dim, adaln_dim] */
    float *adaln_bias;          /* [4*dim] */

#ifdef USE_METAL
    /* BF16 weight pointers for GPU path (converted from f32 at load time) */
    uint16_t *attn_q_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_k_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_v_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_qkv_weight_bf16; /* [3*dim, dim] fused [q;k;v] */
    uint16_t *attn_out_weight_bf16; /* [dim, dim] */
    uint16_t *ffn_w1_bf16;          /* [ffn_dim, dim] */
    uint16_t *ffn_w2_bf16;          /* [dim, ffn_dim] */
    uint16_t *ffn_w3_bf16;          /* [ffn_dim, dim] */
    uint16_t *ffn_w13_weight_bf16;  /* [2*ffn_dim, dim] fused [w1;w3] */
#endif
} zi_block_t;

/* Final layer weights */
typedef struct {
    float *adaln_weight;        /* [dim, adaln_dim] */
    float *adaln_bias;          /* [dim] */
    float *norm_weight;         /* NULL (no affine) or [dim] */
    float *linear_weight;       /* [out_ch, dim] */
    float *linear_bias;         /* [out_ch] */
} zi_final_t;

/* Z-Image transformer context */
typedef struct zi_transformer {
#ifdef ZI_DEV
    struct zi_dev *dev;
#endif
    /* Architecture config */
    int dim;                    /* 3840 */
    int n_heads;                /* 30 */
    int head_dim;               /* 128 */
    int n_layers;               /* 30 */
    int n_refiner;              /* 2 */
    int ffn_dim;                /* 8*dim/3 = 10240 */
    int in_channels;            /* 16 */
    int patch_size;             /* 2 */
    int adaln_dim;              /* min(dim, 256) = 256 */
    float rope_theta;           /* 256.0 */
    int axes_dims[3];           /* [32, 48, 48] */
    int axes_lens[3];           /* [1024, 512, 512] */

    /* Embedders */
    float *t_emb_mlp0_weight;   /* [mid_size, 256] */
    float *t_emb_mlp0_bias;     /* [mid_size] */
    float *t_emb_mlp2_weight;   /* [adaln_dim, mid_size] */
    float *t_emb_mlp2_bias;     /* [adaln_dim] */
    int t_emb_mid_size;         /* intermediate timestep MLP size */

    float *cap_emb_norm;        /* [cap_feat_dim] RMSNorm weight */
    float *cap_emb_linear_w;    /* [dim, cap_feat_dim] */
    float *cap_emb_linear_b;    /* [dim] */
    int cap_feat_dim;           /* 2560 */

    float *x_emb_weight;        /* [dim, patch_feat] where patch_feat = ps*ps*in_ch */
    float *x_emb_bias;          /* [dim] */

    float *x_pad_token;         /* [dim] */
    float *cap_pad_token;       /* [dim] */

    /* Transformer blocks */
    zi_block_t *noise_refiner;  /* [n_refiner] */
    zi_block_t *context_refiner;/* [n_refiner] */
    zi_block_t *layers;         /* [n_layers] */

    /* Final layer */
    zi_final_t final_layer;

    /* CPU mmap mode: keep shard files open and use direct f32 pointers. */
    int mmap_f32_weights;
    safetensors_file_t *sf_files[ZI_MAX_SHARDS];
    int num_sf_files;

    /* Precomputed RoPE frequencies (complex pairs) */
    float *rope_cos[3];         /* [axes_lens[i], axes_dims[i]/2] */
    float *rope_sin[3];         /* [axes_lens[i], axes_dims[i]/2] */

    /* Working memory */
    float *work_x;              /* Main token buffer */
    float *work_tmp;            /* Temporary buffer */
    float *work_qkv;            /* Q, K, V buffers */
    float *work_attn;           /* Attention scores */
    float *work_ffn;            /* FFN intermediate */
    size_t work_alloc;          /* Total allocated */
    int max_seq;                /* Max sequence length allocated for */

#ifdef USE_METAL
    int use_gpu;                /* 1 if GPU path available */
    /* Cached preassembled RoPE tables for GPU path (reused across steps) */
    int gpu_rope_img_seq;
    int gpu_rope_cap_seq;
    int gpu_rope_uni_seq;
    int gpu_rope_h_tokens;
    int gpu_rope_w_tokens;
    float *gpu_img_rope_cos;
    float *gpu_img_rope_sin;
    float *gpu_cap_rope_cos;
    float *gpu_cap_rope_sin;
    float *gpu_uni_rope_cos;
    float *gpu_uni_rope_sin;
#endif
} zi_transformer_t;

void iris_transformer_free_zimage(zi_transformer_t *tf);

#ifdef USE_METAL
/* GPU scratch buffers for block forward pass.
 * Pre-allocated once for max sequence length, reused across all blocks. */
typedef struct {
    int seq, dim, ffn_dim;
    iris_gpu_tensor_t norm;     /* [seq, dim] */
    iris_gpu_tensor_t fused;    /* [seq, max(3*dim, 2*ffn_dim)] */
    iris_gpu_tensor_t q;        /* [seq, dim] */
    iris_gpu_tensor_t k;        /* [seq, dim] */
    iris_gpu_tensor_t v;        /* [seq, dim] */
    iris_gpu_tensor_t attn_out; /* [seq, dim] */
    iris_gpu_tensor_t proj;     /* [seq, dim] */
    iris_gpu_tensor_t norm2;    /* [seq, dim] */
    iris_gpu_tensor_t gate_up;  /* [seq, ffn_dim] */
    iris_gpu_tensor_t up;       /* [seq, ffn_dim] */
    iris_gpu_tensor_t down;     /* [seq, dim] */
    /* BF16 attention scratch (for SDPA path via iris_gpu_attention_fused_bf16) */
    iris_gpu_tensor_t q_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t k_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t v_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t attn_out_bf16;/* [seq, dim] bf16 */
    float *mod;                     /* [4*dim] CPU modulation scratch */
    float *fused_attn_norm;         /* [dim] CPU fused RMS weight scratch */
    float *fused_ffn_norm;          /* [dim] CPU fused RMS weight scratch */
} zi_gpu_scratch_t;

static void zi_gpu_scratch_free(zi_gpu_scratch_t *s) {
    if (!s) return;
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->fused) iris_gpu_tensor_free(s->fused);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->proj) iris_gpu_tensor_free(s->proj);
    if (s->norm2) iris_gpu_tensor_free(s->norm2);
    if (s->gate_up) iris_gpu_tensor_free(s->gate_up);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->down) iris_gpu_tensor_free(s->down);
    if (s->q_bf16) iris_gpu_tensor_free(s->q_bf16);
    if (s->k_bf16) iris_gpu_tensor_free(s->k_bf16);
    if (s->v_bf16) iris_gpu_tensor_free(s->v_bf16);
    if (s->attn_out_bf16) iris_gpu_tensor_free(s->attn_out_bf16);
    if (s->mod) free(s->mod);
    if (s->fused_attn_norm) free(s->fused_attn_norm);
    if (s->fused_ffn_norm) free(s->fused_ffn_norm);
    memset(s, 0, sizeof(*s));
}

static int zi_gpu_scratch_init(zi_gpu_scratch_t *s, int seq, int dim, int ffn_dim) {
    memset(s, 0, sizeof(*s));
    s->seq = seq;
    s->dim = dim;
    s->ffn_dim = ffn_dim;
    int fused_dim = 3 * dim;
    if (2 * ffn_dim > fused_dim) fused_dim = 2 * ffn_dim;

    s->norm = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->fused = iris_gpu_tensor_alloc((size_t)seq * fused_dim);
    s->q = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->k = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->v = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->attn_out = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->proj = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->norm2 = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->gate_up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->down = iris_gpu_tensor_alloc((size_t)seq * dim);

    if (!s->norm || !s->fused || !s->q || !s->k || !s->v || !s->attn_out ||
        !s->proj || !s->norm2 || !s->gate_up || !s->up || !s->down) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    {
        size_t qkv_elems = (size_t)seq * dim;
        s->q_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->k_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->v_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->attn_out_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        if (!s->q_bf16 || !s->k_bf16 || !s->v_bf16 || !s->attn_out_bf16) {
            zi_gpu_scratch_free(s);
            return 0;
        }
    }

    s->mod = (float *)malloc(4 * (size_t)dim * sizeof(float));
    s->fused_attn_norm = (float *)malloc((size_t)dim * sizeof(float));
    s->fused_ffn_norm = (float *)malloc((size_t)dim * sizeof(float));
    if (!s->mod || !s->fused_attn_norm || !s->fused_ffn_norm) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    return 1;
}

static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf);

static uint16_t *zi_concat_bf16(const uint16_t *a, size_t na,
                                 const uint16_t *b, size_t nb) {
    if (!a || !b || na == 0 || nb == 0) return NULL;
    uint16_t *out = (uint16_t *)malloc((na + nb) * sizeof(uint16_t));
    if (!out) return NULL;
    memcpy(out, a, na * sizeof(uint16_t));
    memcpy(out + na, b, nb * sizeof(uint16_t));
    return out;
}

static uint16_t *zi_concat3_bf16(const uint16_t *a, size_t na,
                                  const uint16_t *b, size_t nb,
                                  const uint16_t *c, size_t nc) {
    if (!a || !b || !c || na == 0 || nb == 0 || nc == 0) return NULL;
    uint16_t *out = (uint16_t *)malloc((na + nb + nc) * sizeof(uint16_t));
    if (!out) return NULL;
    memcpy(out, a, na * sizeof(uint16_t));
    memcpy(out + na, b, nb * sizeof(uint16_t));
    memcpy(out + na + nb, c, nc * sizeof(uint16_t));
    return out;
}

/* GPU linear projection writing into a preallocated f32 output tensor.
 * Tries bf16 weight path first (fast), falls back to f32 weights. The "into"
 * variant avoids allocating a new tensor each call, which matters when
 * running 30+ blocks per step. */
static int zi_gpu_linear_into_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                                   const uint16_t *W_bf16, const float *W_f32,
                                   int seq_len, int in_dim, int out_dim) {
    size_t n = (size_t)seq_len * (size_t)out_dim;

    if (W_bf16) {
        if (iris_gpu_linear_bf16_into(out, x, W_bf16, seq_len, in_dim, out_dim)) {
            return 1;
        }
        iris_gpu_tensor_t tmp_bf16 = iris_gpu_linear_bf16(x, W_bf16, seq_len, in_dim, out_dim);
        if (tmp_bf16) {
            iris_gpu_copy_f32(out, tmp_bf16, n);
            iris_gpu_tensor_free(tmp_bf16);
            return 1;
        }
    }

    if (W_f32) {
        iris_gpu_tensor_t tmp_f32 = iris_gpu_linear(x, W_f32, NULL, seq_len, in_dim, out_dim);
        if (tmp_f32) {
            iris_gpu_copy_f32(out, tmp_f32, n);
            iris_gpu_tensor_free(tmp_f32);
            return 1;
        }
    }

    return 0;
}

/* Self-attention dispatcher for GPU. Tries bf16 SDPA first for large
 * sequences (>= 1024 tokens) since bf16 attention fits in memory better,
 * then falls back to f32 fused attention, then tries the other precision,
 * and finally falls back to the legacy f32->f16->f32 path. This cascading
 * fallback ensures attention works at any sequence length. */
static int zi_gpu_attention(iris_gpu_tensor_t out_f32,
                             iris_gpu_tensor_t q_f32, iris_gpu_tensor_t k_f32, iris_gpu_tensor_t v_f32,
                             int seq, int n_heads, int head_dim, float attn_scale,
                             zi_gpu_scratch_t *scratch) {
    int prefer_bf16 = (seq >= ZI_BF16_SDPA_SEQ);

    if (prefer_bf16) {
        if (iris_gpu_convert_f32_to_bf16_into(scratch->q_bf16, q_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->k_bf16, k_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->v_bf16, v_f32) &&
            iris_gpu_attention_fused_bf16(scratch->attn_out_bf16,
                                          scratch->q_bf16, scratch->k_bf16, scratch->v_bf16,
                                          seq, seq, n_heads, head_dim, attn_scale) &&
            iris_gpu_convert_bf16_to_f32_into(out_f32, scratch->attn_out_bf16)) {
            return 1;
        }
    }

    if (iris_gpu_attention_fused(out_f32, q_f32, k_f32, v_f32,
                                 seq, seq, n_heads, head_dim, attn_scale)) {
        return 1;
    }

    if (!prefer_bf16) {
        if (iris_gpu_convert_f32_to_bf16_into(scratch->q_bf16, q_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->k_bf16, k_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->v_bf16, v_f32) &&
            iris_gpu_attention_fused_bf16(scratch->attn_out_bf16,
                                          scratch->q_bf16, scratch->k_bf16, scratch->v_bf16,
                                          seq, seq, n_heads, head_dim, attn_scale) &&
            iris_gpu_convert_bf16_to_f32_into(out_f32, scratch->attn_out_bf16)) {
            return 1;
        }
    }

    return iris_gpu_attention_bf16(out_f32, q_f32, k_f32, v_f32,
                                    seq, seq, n_heads, head_dim, attn_scale);
}

static void zi_gpu_rope_cache_clear(zi_transformer_t *tf) {
    free(tf->gpu_img_rope_cos); tf->gpu_img_rope_cos = NULL;
    free(tf->gpu_img_rope_sin); tf->gpu_img_rope_sin = NULL;
    free(tf->gpu_cap_rope_cos); tf->gpu_cap_rope_cos = NULL;
    free(tf->gpu_cap_rope_sin); tf->gpu_cap_rope_sin = NULL;
    free(tf->gpu_uni_rope_cos); tf->gpu_uni_rope_cos = NULL;
    free(tf->gpu_uni_rope_sin); tf->gpu_uni_rope_sin = NULL;
    tf->gpu_rope_img_seq = 0;
    tf->gpu_rope_cap_seq = 0;
    tf->gpu_rope_uni_seq = 0;
    tf->gpu_rope_h_tokens = 0;
    tf->gpu_rope_w_tokens = 0;
}

/* Preassembles and caches RoPE cos/sin tables for the current image geometry
 * (H_tokens, W_tokens, cap_seq_len). The geometry is stable across denoising
 * steps, so this avoids rebuilding tables every transformer call. Invalidated
 * when dimensions change (e.g., different image size). Builds separate tables
 * for noise refiner (image-only), context refiner (caption-only), and main
 * blocks (unified [img, cap] sequence). */
static int zi_gpu_rope_cache_prepare(zi_transformer_t *tf,
                                      int cap_seq_len, int H_tokens, int W_tokens) {
    int img_seq = H_tokens * W_tokens;
    int uni_seq = img_seq + cap_seq_len;

    if (tf->gpu_img_rope_cos &&
        tf->gpu_rope_img_seq == img_seq &&
        tf->gpu_rope_cap_seq == cap_seq_len &&
        tf->gpu_rope_uni_seq == uni_seq &&
        tf->gpu_rope_h_tokens == H_tokens &&
        tf->gpu_rope_w_tokens == W_tokens) {
        return 1;
    }

    zi_gpu_rope_cache_clear(tf);

    int head_dim = tf->head_dim;
    tf->gpu_img_rope_cos = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_img_rope_sin = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_cap_rope_cos = (float *)malloc((size_t)cap_seq_len * head_dim * sizeof(float));
    tf->gpu_cap_rope_sin = (float *)malloc((size_t)cap_seq_len * head_dim * sizeof(float));
    tf->gpu_uni_rope_cos = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));
    tf->gpu_uni_rope_sin = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));

    if (!tf->gpu_img_rope_cos || !tf->gpu_img_rope_sin ||
        !tf->gpu_cap_rope_cos || !tf->gpu_cap_rope_sin ||
        !tf->gpu_uni_rope_cos || !tf->gpu_uni_rope_sin) {
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    int cap_padded_for_pos = ((cap_seq_len + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF)
                              * ZI_SEQ_MULTI_OF;
    int *img_pos = (int *)calloc((size_t)img_seq * 3, sizeof(int));
    int *cap_pos = (int *)calloc((size_t)cap_seq_len * 3, sizeof(int));
    int *uni_pos = (int *)malloc((size_t)uni_seq * 3 * sizeof(int));
    if (!img_pos || !cap_pos || !uni_pos) {
        free(img_pos);
        free(cap_pos);
        free(uni_pos);
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int idx = h * W_tokens + w;
            img_pos[idx * 3 + 0] = cap_padded_for_pos + 1;
            img_pos[idx * 3 + 1] = h;
            img_pos[idx * 3 + 2] = w;
        }
    }

    for (int s = 0; s < cap_seq_len; s++) {
        cap_pos[s * 3 + 0] = 1 + s;
        cap_pos[s * 3 + 1] = 0;
        cap_pos[s * 3 + 2] = 0;
    }

    memcpy(uni_pos, img_pos, (size_t)img_seq * 3 * sizeof(int));
    memcpy(uni_pos + (size_t)img_seq * 3, cap_pos, (size_t)cap_seq_len * 3 * sizeof(int));

    zi_build_rope_table(tf->gpu_img_rope_cos, tf->gpu_img_rope_sin, img_pos, img_seq, tf);
    zi_build_rope_table(tf->gpu_cap_rope_cos, tf->gpu_cap_rope_sin, cap_pos, cap_seq_len, tf);
    zi_build_rope_table(tf->gpu_uni_rope_cos, tf->gpu_uni_rope_sin, uni_pos, uni_seq, tf);

    free(img_pos);
    free(cap_pos);
    free(uni_pos);

    tf->gpu_rope_img_seq = img_seq;
    tf->gpu_rope_cap_seq = cap_seq_len;
    tf->gpu_rope_uni_seq = uni_seq;
    tf->gpu_rope_h_tokens = H_tokens;
    tf->gpu_rope_w_tokens = W_tokens;
    return 1;
}

static void iris_warmup_bf16_zimage(zi_transformer_t *tf) {
    if (!tf || !tf->use_gpu) return;
    if (!iris_metal_available()) return;

    size_t attn_elems = (size_t)tf->dim * tf->dim;
    size_t ffn_up_elems = (size_t)tf->ffn_dim * tf->dim;
    size_t ffn_down_elems = (size_t)tf->dim * tf->ffn_dim;

    zi_block_t *groups[3] = { tf->noise_refiner, tf->context_refiner, tf->layers };
    int counts[3] = { tf->n_refiner, tf->n_refiner, tf->n_layers };

    for (int g = 0; g < 3; g++) {
        zi_block_t *blocks = groups[g];
        int n = counts[g];
        if (!blocks) continue;

        for (int i = 0; i < n; i++) {
            zi_block_t *b = &blocks[i];

            if (b->attn_q_weight_bf16) iris_metal_warmup_bf16(b->attn_q_weight_bf16, attn_elems);
            if (b->attn_k_weight_bf16) iris_metal_warmup_bf16(b->attn_k_weight_bf16, attn_elems);
            if (b->attn_v_weight_bf16) iris_metal_warmup_bf16(b->attn_v_weight_bf16, attn_elems);
            if (b->attn_out_weight_bf16) iris_metal_warmup_bf16(b->attn_out_weight_bf16, attn_elems);
            if (b->attn_qkv_weight_bf16) iris_metal_warmup_bf16(b->attn_qkv_weight_bf16, attn_elems * 3);

            if (b->ffn_w1_bf16) iris_metal_warmup_bf16(b->ffn_w1_bf16, ffn_up_elems);
            if (b->ffn_w2_bf16) iris_metal_warmup_bf16(b->ffn_w2_bf16, ffn_down_elems);
            if (b->ffn_w3_bf16) iris_metal_warmup_bf16(b->ffn_w3_bf16, ffn_up_elems);
            if (b->ffn_w13_weight_bf16) iris_metal_warmup_bf16(b->ffn_w13_weight_bf16, ffn_up_elems * 2);
        }
    }
}
#endif /* USE_METAL */

/* ========================================================================
 * Forward declarations
 * ======================================================================== */

void iris_transformer_free_zimage(zi_transformer_t *tf);

/* Forward declarations for functions used by GPU path */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps);
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps);
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf);
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf);
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps);

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. */
static void zi_sinusoidal_embedding(float *out, float t, int dim) {
    int half = dim / 2;
    float log_max_period = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float freq = expf(-log_max_period * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);
        out[i + half] = sinf(angle);
    }
}

/* Projects the sinusoidal timestep embedding through an MLP
 * (Linear -> SiLU -> Linear) to produce the adaln_dim-sized conditioning
 * vector. This drives all AdaLN modulation in the transformer -- it is how
 * every block knows which denoising step it is operating on. */
static void zi_timestep_embed(zi_transformer_t *tf, float *out, float t) {
    float sin_emb[256];
    zi_sinusoidal_embedding(sin_emb, t * 1000.0f, 256);

    /* MLP: Linear(256 -> mid) + SiLU + Linear(mid -> adaln_dim) */
    int mid = tf->t_emb_mid_size;
    float *hidden = (float *)malloc(mid * sizeof(float));

    /* Linear 0 */
    iris_matmul_t(hidden, sin_emb, tf->t_emb_mlp0_weight, 1, 256, mid);
    for (int i = 0; i < mid; i++) hidden[i] += tf->t_emb_mlp0_bias[i];

    /* SiLU */
    iris_silu(hidden, mid);

    /* Linear 2 */
    iris_matmul_t(out, hidden, tf->t_emb_mlp2_weight, 1, mid, tf->adaln_dim);
    for (int i = 0; i < tf->adaln_dim; i++) out[i] += tf->t_emb_mlp2_bias[i];

    free(hidden);
}

/* ========================================================================
 * RoPE
 * ======================================================================== */

/* Precomputes cos/sin frequency tables for all 3 RoPE axes
 * (T=32 dims, H=48 dims, W=48 dims) up to max_pos=1024 per axis.
 * Uses theta=256.0, much smaller than the usual 10000, giving shorter-range
 * position sensitivity suited to Z-Image's spatial layout. Tables are
 * allocated once at load time and reused across all denoising steps. */
static void zi_precompute_rope(zi_transformer_t *tf) {
    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;
        int max_pos = tf->axes_lens[ax];

        tf->rope_cos[ax] = (float *)malloc(max_pos * half_d * sizeof(float));
        tf->rope_sin[ax] = (float *)malloc(max_pos * half_d * sizeof(float));

        for (int pos = 0; pos < max_pos; pos++) {
            for (int i = 0; i < half_d; i++) {
                float freq = 1.0f / powf(tf->rope_theta, (float)(2 * i) / (float)d);
                float angle = (float)pos * freq;
                tf->rope_cos[ax][pos * half_d + i] = cosf(angle);
                tf->rope_sin[ax][pos * half_d + i] = sinf(angle);
            }
        }
    }
}

/* Applies 3-axis RoPE to Q or K in-place using consecutive-pair rotation:
 * (x0*cos - x1*sin, x1*cos + x0*sin) on elements (d, d+1).
 * Each axis section of head_dim (T=32, H=48, W=48) gets its own position
 * from pos_ids[s,3]. This differs from Flux's split-half convention --
 * Z-Image pairs adjacent elements (d, d+1) rather than (d, d+half). */
static void zi_apply_rope(float *x, const int *pos_ids, int seq, int n_heads,
                           zi_transformer_t *tf) {
    int head_dim = tf->head_dim;
    int offset = 0;

    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;

        for (int s = 0; s < seq; s++) {
            int pos = pos_ids[s * 3 + ax];
            if (pos < 0 || pos >= tf->axes_lens[ax]) continue;

            const float *cos_tab = tf->rope_cos[ax] + pos * half_d;
            const float *sin_tab = tf->rope_sin[ax] + pos * half_d;

            for (int h = 0; h < n_heads; h++) {
                float *head = x + (s * n_heads + h) * head_dim + offset;
                for (int i = 0; i < half_d; i++) {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    float c = cos_tab[i];
                    float sn = sin_tab[i];
                    head[2 * i]     = x0 * c - x1 * sn;
                    head[2 * i + 1] = x1 * c + x0 * sn;
                }
            }
        }
        offset += d;
    }
}

/* ========================================================================
 * Block Forward Pass (BLAS)
 * ======================================================================== */

/* RMSNorm: out = x * weight / sqrt(mean(x^2) + eps) */
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * dim;
        float *or_ = out + r * dim;
        float sum_sq = 0;
        for (int i = 0; i < dim; i++) sum_sq += xr[i] * xr[i];
        float rms = 1.0f / sqrtf(sum_sq / dim + eps);
        for (int i = 0; i < dim; i++) or_[i] = xr[i] * rms * weight[i];
    }
}

/* Per-head RMSNorm for QK normalization.
 * x: [seq, n_heads * head_dim], norm_weight: [head_dim] (shared across heads) */
static void zi_qk_norm(float *x, const float *norm_weight, int seq,
                         int n_heads, int head_dim, float eps) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *ptr = x + s * n_heads * head_dim + h * head_dim;
            float sum_sq = 0;
            for (int i = 0; i < head_dim; i++) sum_sq += ptr[i] * ptr[i];
            float rms = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int i = 0; i < head_dim; i++) ptr[i] = ptr[i] * rms * norm_weight[i];
        }
    }
}

/* Scaled dot-product self-attention on the CPU path.
 * Computes Q@K^T per head, applies padding mask (sets masked positions to
 * -1e9 so softmax zeros them out), then scores@V. The mask distinguishes
 * real tokens from padding in the sequence. This is the slow reference path;
 * the GPU path uses fused SDPA kernels instead. */
static void zi_attention(float *out, const float *x,
                          const zi_block_t *block, const int *pos_ids,
                          const int *mask, int seq,
                          zi_transformer_t *tf) {
    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;

    float *q = tf->work_qkv;
    float *k = q + seq * dim;
    float *v = k + seq * dim;

    /* Q, K, V projections */
    iris_matmul_t(q, x, block->attn_q_weight, seq, dim, dim);
    iris_matmul_t(k, x, block->attn_k_weight, seq, dim, dim);
    iris_matmul_t(v, x, block->attn_v_weight, seq, dim, dim);

    /* QK normalization */
    zi_qk_norm(q, block->attn_norm_q, seq, n_heads, head_dim, ZI_NORM_EPS);
    zi_qk_norm(k, block->attn_norm_k, seq, n_heads, head_dim, ZI_NORM_EPS);

    /* Apply RoPE */
    zi_apply_rope(q, pos_ids, seq, n_heads, tf);
    zi_apply_rope(k, pos_ids, seq, n_heads, tf);

    /* Scaled dot-product attention per head. The output of a head overwrites
     * its queries, no longer needed once the scores are computed: out is
     * work_tmp in the callers, so it must not hold the projection input. */
    float scale = 1.0f / sqrtf((float)head_dim);
    float *attn_out = q;

    for (int h = 0; h < n_heads; h++) {
        float *scores = tf->work_attn;

        /* Compute Q @ K^T for this head */
        for (int i = 0; i < seq; i++) {
            const float *qi = q + i * dim + h * head_dim;
            for (int j = 0; j < seq; j++) {
                const float *kj = k + j * dim + h * head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += qi[d] * kj[d];
                scores[i * seq + j] = dot * scale;
            }
        }

        /* Apply mask: set padding positions to -inf */
        if (mask) {
            for (int i = 0; i < seq; i++) {
                for (int j = 0; j < seq; j++) {
                    if (!mask[j])
                        scores[i * seq + j] = -1e9f;
                }
            }
        }

        /* Softmax */
        iris_softmax(scores, seq, seq);

        /* Scores @ V */
        for (int i = 0; i < seq; i++) {
            float *oi = attn_out + i * dim + h * head_dim;
            memset(oi, 0, head_dim * sizeof(float));
            for (int j = 0; j < seq; j++) {
                float s = scores[i * seq + j];
                const float *vj = v + j * dim + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    oi[d] += s * vj[d];
            }
        }
    }

    /* Output projection */
    iris_matmul_t(out, attn_out, block->attn_out_weight, seq, dim, dim);
}

/* SwiGLU FFN: silu(W1 @ x) * (W3 @ x) then W2 */
static void zi_ffn(float *out, const float *x, const zi_block_t *block,
                    int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int ffn_dim = tf->ffn_dim;
    float *gate = tf->work_ffn;
    float *up = gate + seq * ffn_dim;

    /* W1 (gate) and W3 (up) projections */
    iris_matmul_t(gate, x, block->ffn_w1, seq, dim, ffn_dim);
    iris_matmul_t(up, x, block->ffn_w3, seq, dim, ffn_dim);

    /* SiLU(gate) * up */
    int n = seq * ffn_dim;
    iris_silu(gate, n);
    for (int i = 0; i < n; i++) gate[i] *= up[i];

    /* W2 (down) projection */
    iris_matmul_t(out, gate, block->ffn_w2, seq, ffn_dim, dim);
}

/* One S3-DiT block on CPU. Two modes: modulated (noise_refiner + main layers)
 * applies AdaLN with scale and tanh-gated residuals; unmodulated
 * (context_refiner) is a plain pre-norm attention + FFN block. The modulation
 * uses 4 parameters per block: scale_msa, gate_msa, scale_mlp, gate_mlp.
 * Note: no additive shift in Z-Image's block modulation (unlike Flux's
 * AdaLN which has shift). */
static void zi_block_forward(float *x, const zi_block_t *block,
                              const int *pos_ids, const int *mask,
                              const float *t_emb, int seq,
                              zi_transformer_t *tf) {
    int dim = tf->dim;
    int n = seq * dim;
    float *attn_out = tf->work_tmp;
    float *norm_out = tf->work_tmp + n;
    float *scaled = tf->work_tmp + 2 * n;
    float *ffn_out = tf->work_tmp + 3 * n;

    if (!tf->work_tmp || tf->max_seq < seq) return;

    if (block->adaln_weight) {
        /* Modulated block: extract scale_msa, gate_msa, scale_mlp, gate_mlp */
        float mod[4 * dim];
        iris_matmul_t(mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
        for (int i = 0; i < 4 * dim; i++) mod[i] += block->adaln_bias[i];

        float *scale_msa = mod;
        float *gate_msa  = mod + dim;
        float *scale_mlp = mod + 2 * dim;
        float *gate_mlp  = mod + 3 * dim;

        /* Apply tanh to gates, 1+scale */
        for (int i = 0; i < dim; i++) {
            scale_msa[i] = 1.0f + scale_msa[i];
            gate_msa[i] = tanhf(gate_msa[i]);
            scale_mlp[i] = 1.0f + scale_mlp[i];
            gate_mlp[i] = tanhf(gate_mlp[i]);
        }

        /* Attention: h = attention(norm1(x) * scale_msa) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                scaled[s * dim + i] = norm_out[s * dim + i] * scale_msa[i];

        zi_attention(attn_out, scaled, block, pos_ids, mask, seq, tf);

        /* x = x + gate_msa * norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);

        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                x[s * dim + i] += gate_msa[i] * norm_out[s * dim + i];

        /* FFN: h = ffn(norm1(x) * scale_mlp) */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                scaled[s * dim + i] = norm_out[s * dim + i] * scale_mlp[i];

        zi_ffn(ffn_out, scaled, block, seq, tf);

        /* x = x + gate_mlp * norm2(ffn_out) */
        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                x[s * dim + i] += gate_mlp[i] * norm_out[s * dim + i];

    } else {
        /* Unmodulated block (context_refiner): no scale/gate */

        /* Attention: h = attention(norm1(x)) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);

        zi_attention(attn_out, norm_out, block, pos_ids, mask, seq, tf);

        /* x = x + norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
        /* FFN */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);

        zi_ffn(ffn_out, norm_out, block, seq, tf);

        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
    }
}

/* ========================================================================
 * GPU Forward Pass (Metal)
 * ======================================================================== */

#ifdef USE_METAL

/* Convert f32 array to bf16 (CPU-side, for weight conversion at load time).
 * Uses round-to-nearest-even for best accuracy.
 * Caller owns the returned buffer. */
static uint16_t *zi_f32_to_bf16(const float *src, size_t n) {
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst) return NULL;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        /* Round to nearest even: add rounding bias, handle tie-breaking */
        uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding_bias;
        dst[i] = (uint16_t)(bits >> 16);
    }
    return dst;
}

/* Build a pre-assembled [seq, head_dim] RoPE cos/sin table by merging 3 axes.
 * pos_ids: [seq, 3] with (T, H, W) per position.
 * The table has cos/sin values laid out as consecutive pairs so that
 * iris_gpu_rope_2d (axis_dim=head_dim) applies rotation across the full head. */
static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf) {
    int head_dim = tf->head_dim;

    for (int s = 0; s < seq; s++) {
        int offset = 0;
        for (int ax = 0; ax < 3; ax++) {
            int d = tf->axes_dims[ax];
            int half_d = d / 2;
            int pos = pos_ids[s * 3 + ax];

            /* Clamp pos to valid range */
            if (pos < 0) pos = 0;
            if (pos >= tf->axes_lens[ax]) pos = tf->axes_lens[ax] - 1;

            const float *ax_cos = tf->rope_cos[ax] + pos * half_d;
            const float *ax_sin = tf->rope_sin[ax] + pos * half_d;

            /* Write as consecutive pairs [cos_0, cos_0, cos_1, cos_1, ...] */
            for (int i = 0; i < half_d; i++) {
                cos_out[s * head_dim + offset + 2 * i]     = ax_cos[i];
                cos_out[s * head_dim + offset + 2 * i + 1] = ax_cos[i];
                sin_out[s * head_dim + offset + 2 * i]     = ax_sin[i];
                sin_out[s * head_dim + offset + 2 * i + 1] = ax_sin[i];
            }
            offset += d;
        }
    }
}

/* GPU-accelerated block forward. Fuses norm weight with modulation scale on
 * CPU (one multiply per dim), then passes the fused weight to GPU RMSNorm to
 * avoid an extra GPU kernel. Uses fused QKV and W1/W3 matmuls when available.
 * Returns 0 on any failure so the caller can fall back to CPU. */
static int zi_block_forward_gpu(iris_gpu_tensor_t hidden_gpu,
                                 const zi_block_t *block,
                                 const float *rope_cos, const float *rope_sin,
                                 const float *t_emb, const float *precomputed_mod, int seq,
                                 zi_transformer_t *tf,
                                 zi_gpu_scratch_t *scratch) {
    if (!iris_metal_available() || !iris_metal_shaders_available()) return 0;
    if (!hidden_gpu || !scratch) return 0;

    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;
    int ffn_dim = tf->ffn_dim;

    if (block->adaln_weight) {
        /* ---- Modulated block ---- */

        const float *mod = precomputed_mod;
        if (!mod) {
            /* Fallback path: compute modulation on the fly. */
            float *scratch_mod = scratch->mod;
            iris_matmul_t(scratch_mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
            for (int i = 0; i < 4 * dim; i++) scratch_mod[i] += block->adaln_bias[i];

            /* Apply 1+scale to scales, tanh to gates */
            for (int i = 0; i < dim; i++) {
                scratch_mod[i] = 1.0f + scratch_mod[i];
                scratch_mod[dim + i] = tanhf(scratch_mod[dim + i]);
                scratch_mod[2 * dim + i] = 1.0f + scratch_mod[2 * dim + i];
                scratch_mod[3 * dim + i] = tanhf(scratch_mod[3 * dim + i]);
            }
            mod = scratch_mod;
        }

        const float *scale_msa = mod;
        const float *gate_msa  = mod + dim;
        const float *scale_mlp = mod + 2 * dim;
        const float *gate_mlp  = mod + 3 * dim;

        /* CPU: fuse norm_weight * scale into a single weight for RMSNorm */
        float *fused_attn_norm = scratch->fused_attn_norm;
        for (int i = 0; i < dim; i++)
            fused_attn_norm[i] = block->attn_norm1[i] * scale_msa[i];

        /* GPU: RMSNorm with fused weight (= rms_norm(x) * attn_norm1 * scale_msa) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_attn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Q, K, V projections (fused when available). */
        if (block->attn_qkv_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->attn_qkv_weight_bf16, NULL,
                                       seq, dim, 3 * dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, dim, 0);
            } else {
                /* Fallback to unfused projections on GPU if fused path fails. */
                if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                            block->attn_q_weight_bf16, block->attn_q_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                            block->attn_k_weight_bf16, block->attn_k_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                            block->attn_v_weight_bf16, block->attn_v_weight,
                                            seq, dim, dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                        block->attn_q_weight_bf16, block->attn_q_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                        block->attn_k_weight_bf16, block->attn_k_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                        block->attn_v_weight_bf16, block->attn_v_weight,
                                        seq, dim, dim)) return 0;
        }

        /* GPU: Fused QK normalization + RoPE */
        iris_gpu_qk_norm_rope_single(scratch->q, scratch->k,
                                      block->attn_norm_q, block->attn_norm_k,
                                      rope_cos, rope_sin,
                                      seq, n_heads, head_dim, ZI_NORM_EPS);

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16, block->attn_out_weight,
                                    seq, dim, dim)) return 0;

        /* GPU: Fused attn_norm2 + gated residual: x += gate_msa * norm2(proj) */
        iris_gpu_rms_norm_gated_add(hidden_gpu, scratch->proj,
                                     block->attn_norm2, gate_msa,
                                     seq, dim, ZI_NORM_EPS);

        /* CPU: fuse FFN norm weight * scale_mlp */
        float *fused_ffn_norm = scratch->fused_ffn_norm;
        for (int i = 0; i < dim; i++)
            fused_ffn_norm[i] = block->ffn_norm1[i] * scale_mlp[i];

        /* GPU: FFN input norm with fused weight */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_ffn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: SwiGLU FFN (fused w1/w3 when available). */
        if (block->ffn_w13_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->ffn_w13_weight_bf16, NULL,
                                       seq, dim, 2 * ffn_dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, 0, ffn_dim);
            } else {
                if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                            block->ffn_w1_bf16, block->ffn_w1,
                                            seq, dim, ffn_dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                            block->ffn_w3_bf16, block->ffn_w3,
                                            seq, dim, ffn_dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                        block->ffn_w1_bf16, block->ffn_w1,
                                        seq, dim, ffn_dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                        block->ffn_w3_bf16, block->ffn_w3,
                                        seq, dim, ffn_dim)) return 0;
        }
        iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);

        /* GPU: FFN down projection */
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16, block->ffn_w2,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: Fused ffn_norm2 + gated residual: x += gate_mlp * norm2(ffn_out) */
        iris_gpu_rms_norm_gated_add(hidden_gpu, scratch->down,
                                     block->ffn_norm2, gate_mlp,
                                     seq, dim, ZI_NORM_EPS);

    } else {
        /* ---- Unmodulated block (context_refiner) ---- */

        /* GPU: RMSNorm (plain weight, no scale) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->attn_norm1,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Q, K, V projections (fused when available). */
        if (block->attn_qkv_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->attn_qkv_weight_bf16, NULL,
                                       seq, dim, 3 * dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, dim, 0);
            } else {
                if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                            block->attn_q_weight_bf16, block->attn_q_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                            block->attn_k_weight_bf16, block->attn_k_weight,
                                            seq, dim, dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                            block->attn_v_weight_bf16, block->attn_v_weight,
                                            seq, dim, dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                        block->attn_q_weight_bf16, block->attn_q_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                        block->attn_k_weight_bf16, block->attn_k_weight,
                                        seq, dim, dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                        block->attn_v_weight_bf16, block->attn_v_weight,
                                        seq, dim, dim)) return 0;
        }

        /* GPU: Fused QK normalization + RoPE */
        iris_gpu_qk_norm_rope_single(scratch->q, scratch->k,
                                      block->attn_norm_q, block->attn_norm_k,
                                      rope_cos, rope_sin,
                                      seq, n_heads, head_dim, ZI_NORM_EPS);

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16, block->attn_out_weight,
                                    seq, dim, dim)) return 0;

        /* GPU: norm2(proj) + residual: x += norm2(attn_out) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->proj, block->attn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);

        /* GPU: FFN */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->ffn_norm1,
                               seq, dim, ZI_NORM_EPS);
        if (block->ffn_w13_weight_bf16) {
            if (zi_gpu_linear_into_f32(scratch->fused, scratch->norm,
                                       block->ffn_w13_weight_bf16, NULL,
                                       seq, dim, 2 * ffn_dim)) {
                iris_gpu_split_qkv_mlp(scratch->fused,
                                       scratch->q, scratch->k, scratch->v,
                                       scratch->gate_up, scratch->up,
                                       seq, 0, ffn_dim);
            } else {
                if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                            block->ffn_w1_bf16, block->ffn_w1,
                                            seq, dim, ffn_dim)) return 0;
                if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                            block->ffn_w3_bf16, block->ffn_w3,
                                            seq, dim, ffn_dim)) return 0;
            }
        } else {
            if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                        block->ffn_w1_bf16, block->ffn_w1,
                                        seq, dim, ffn_dim)) return 0;
            if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                        block->ffn_w3_bf16, block->ffn_w3,
                                        seq, dim, ffn_dim)) return 0;
        }
        iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16, block->ffn_w2,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: ffn_norm2 + residual: x += norm2(ffn_out) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->down, block->ffn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);
    }

    return 1;
}

/* Precompute modulation for one block:
 * mod_out layout = [scale_msa, gate_msa, scale_mlp, gate_mlp], each dim.
 * Scales are stored as (1 + scale), gates are tanh(gate). */
static int zi_precompute_block_modulation(float *mod_out, const zi_block_t *block,
                                          const float *t_emb, int adaln_dim, int dim) {
    if (!mod_out || !block || !block->adaln_weight || !block->adaln_bias || !t_emb) return 0;

    iris_matmul_t(mod_out, t_emb, block->adaln_weight, 1, adaln_dim, 4 * dim);
    for (int i = 0; i < 4 * dim; i++) mod_out[i] += block->adaln_bias[i];

    for (int i = 0; i < dim; i++) {
        mod_out[i] = 1.0f + mod_out[i];
        mod_out[dim + i] = tanhf(mod_out[dim + i]);
        mod_out[2 * dim + i] = 1.0f + mod_out[2 * dim + i];
        mod_out[3 * dim + i] = tanhf(mod_out[3 * dim + i]);
    }

    return 1;
}

/* Full GPU-accelerated Z-Image transformer forward pass. Pipeline:
 * CPU timestep embed + patchify + caption norm -> GPU embedding projections ->
 * GPU noise refiner (2 blocks, image only) -> GPU context refiner (2 blocks,
 * caption only) -> GPU concat [img, cap] -> GPU main blocks (30, unified) ->
 * GPU final layer -> CPU unpatchify. Pre-computes all block modulations once
 * per step. Uses batch mode to submit all GPU work in one command buffer.
 * Returns NULL on failure (caller falls back to CPU). */
static float *zi_transformer_forward_gpu(zi_transformer_t *tf,
                                          const float *latent,
                                          int latent_h, int latent_w,
                                          float timestep,
                                          const float *cap_feats,
                                          int cap_seq_len) {
    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* No padding for GPU path — GPU attention handles arbitrary seq lengths */
    int cap_padded = cap_seq_len;
    int unified_seq = img_seq + cap_padded;
    double t_embed_ms = 0.0, t_noise_ms = 0.0, t_context_ms = 0.0;
    double t_main_ms = 0.0, t_final_ms = 0.0;
    double stage_start = zi_time_ms();

    /* === CPU: Timestep embedding === */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* === CPU: Patchify image === */
    float *img_patches = (float *)malloc(img_seq * patch_feat * sizeof(float));
    if (!img_patches) return NULL;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* === CPU: Caption RMSNorm === */
    float *cap_normed = (float *)malloc(cap_seq_len * tf->cap_feat_dim * sizeof(float));
    if (!cap_normed) {
        free(img_patches);
        return NULL;
    }
    zi_rms_norm(cap_normed, cap_feats, tf->cap_emb_norm,
                cap_seq_len, tf->cap_feat_dim, ZI_NORM_EPS);

    /* === Embed image/caption (prefer GPU linear, fall back to CPU) === */
    iris_gpu_tensor_t img_gpu = NULL;
    iris_gpu_tensor_t cap_gpu = NULL;

    /* Image projection on GPU */
    iris_gpu_tensor_t img_patch_gpu = iris_gpu_tensor_create(img_patches, (size_t)img_seq * patch_feat);
    if (img_patch_gpu) {
        img_gpu = iris_gpu_linear(img_patch_gpu, tf->x_emb_weight, tf->x_emb_bias,
                                  img_seq, patch_feat, dim);
        iris_gpu_tensor_free(img_patch_gpu);
    }

    /* Caption projection on GPU */
    iris_gpu_tensor_t cap_norm_gpu = iris_gpu_tensor_create(cap_normed, (size_t)cap_seq_len * tf->cap_feat_dim);
    if (cap_norm_gpu) {
        cap_gpu = iris_gpu_linear(cap_norm_gpu, tf->cap_emb_linear_w, tf->cap_emb_linear_b,
                                  cap_seq_len, tf->cap_feat_dim, dim);
        iris_gpu_tensor_free(cap_norm_gpu);
    }

    /* CPU fallback if either embedding projection failed */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) {
            iris_gpu_tensor_free(img_gpu);
            img_gpu = NULL;
        }
        if (cap_gpu) {
            iris_gpu_tensor_free(cap_gpu);
            cap_gpu = NULL;
        }

        float *img_emb = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        float *cap_emb = (float *)malloc((size_t)cap_seq_len * dim * sizeof(float));
        if (!img_emb || !cap_emb) {
            free(img_emb);
            free(cap_emb);
            free(img_patches);
            free(cap_normed);
            return NULL;
        }

        iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_seq, patch_feat, dim);
        for (int s = 0; s < img_seq; s++) {
            for (int i = 0; i < dim; i++) {
                img_emb[s * dim + i] += tf->x_emb_bias[i];
            }
        }

        iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                      cap_seq_len, tf->cap_feat_dim, dim);
        for (int s = 0; s < cap_seq_len; s++) {
            for (int i = 0; i < dim; i++) {
                cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
            }
        }

        img_gpu = iris_gpu_tensor_create(img_emb, (size_t)img_seq * dim);
        cap_gpu = iris_gpu_tensor_create(cap_emb, (size_t)cap_seq_len * dim);
        free(img_emb);
        free(cap_emb);
    }

    free(img_patches);
    free(cap_normed);

    /* === CPU: Pre-assemble RoPE tables (cached across steps) === */
    if (!zi_gpu_rope_cache_prepare(tf, cap_seq_len, H_tokens, W_tokens)) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    const float *img_rope_cos = tf->gpu_img_rope_cos;
    const float *img_rope_sin = tf->gpu_img_rope_sin;
    const float *cap_rope_cos = tf->gpu_cap_rope_cos;
    const float *cap_rope_sin = tf->gpu_cap_rope_sin;
    const float *uni_rope_cos = tf->gpu_uni_rope_cos;
    const float *uni_rope_sin = tf->gpu_uni_rope_sin;
    t_embed_ms = zi_time_ms() - stage_start;

    /* === GPU: Process embedded tokens === */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    iris_gpu_tensor_set_persistent(img_gpu, 1);
    iris_gpu_tensor_set_persistent(cap_gpu, 1);

    /* Allocate scratch for max sequence length (unified_seq) */
    zi_gpu_scratch_t scratch;
    if (!zi_gpu_scratch_init(&scratch, unified_seq, dim, tf->ffn_dim)) {
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* Precompute modulation once per step for all modulated blocks. */
    int n_mod_blocks = tf->n_refiner + tf->n_layers;
    float *step_mod = NULL;
    if (n_mod_blocks > 0) {
        step_mod = (float *)malloc((size_t)n_mod_blocks * 4 * dim * sizeof(float));
        if (step_mod) {
            int mod_idx = 0;
            int mod_ok = 1;
            for (int i = 0; i < tf->n_refiner && mod_ok; i++) {
                mod_ok = zi_precompute_block_modulation(
                    step_mod + (size_t)mod_idx * 4 * dim,
                    &tf->noise_refiner[i], t_emb, tf->adaln_dim, dim);
                mod_idx++;
            }
            for (int i = 0; i < tf->n_layers && mod_ok; i++) {
                mod_ok = zi_precompute_block_modulation(
                    step_mod + (size_t)mod_idx * 4 * dim,
                    &tf->layers[i], t_emb, tf->adaln_dim, dim);
                mod_idx++;
            }
            if (!mod_ok) {
                free(step_mod);
                step_mod = NULL;
            }
        }
    }

    iris_gpu_batch_begin();

    /* === Noise refiner: 2 modulated blocks on image tokens === */
    int gpu_ok = 1;
    int mod_idx = 0;
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu(img_gpu, &tf->noise_refiner[i],
                                       img_rope_cos, img_rope_sin,
                                       t_emb, block_mod, img_seq, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }
    t_noise_ms = zi_time_ms() - stage_start;

    /* === Context refiner: 2 unmodulated blocks on caption tokens === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        gpu_ok = zi_block_forward_gpu(cap_gpu, &tf->context_refiner[i],
                                       cap_rope_cos, cap_rope_sin,
                                       NULL, NULL, cap_seq_len, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }
    t_context_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* === Concatenate: unified = [img, cap] === */
    iris_gpu_tensor_t unified_gpu = iris_gpu_tensor_alloc(unified_seq * dim);
    if (!unified_gpu) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    iris_gpu_tensor_set_persistent(unified_gpu, 1);

    /* Copy img then cap into unified entirely on GPU (no CPU sync). */
    size_t img_elems = (size_t)img_seq * dim;
    size_t cap_elems = (size_t)cap_seq_len * dim;
    iris_gpu_copy_region_f32(unified_gpu, 0, img_gpu, 0, img_elems);
    iris_gpu_copy_region_f32(unified_gpu, img_elems, cap_gpu, 0, cap_elems);

    iris_gpu_tensor_free(img_gpu);
    iris_gpu_tensor_free(cap_gpu);

    /* === Main transformer: 30 modulated blocks on unified sequence === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_layers && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu(unified_gpu, &tf->layers[i],
                                       uni_rope_cos, uni_rope_sin,
                                       t_emb, block_mod, unified_seq, tf, &scratch);
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }
    t_main_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(unified_gpu);
        return NULL;
    }

    /* === Final layer on GPU: slice image tokens -> LayerNorm+scale -> Linear === */
    stage_start = zi_time_ms();
    int out_ch = ps * ps * in_ch;
    iris_gpu_tensor_t img_hidden_gpu = iris_gpu_tensor_alloc((size_t)img_seq * dim);
    iris_gpu_tensor_t final_norm_gpu = iris_gpu_tensor_alloc((size_t)img_seq * dim);
    iris_gpu_tensor_t final_out_gpu = NULL;

    /* Prepare final AdaLN parameters on CPU once per step. */
    float *final_scale = (float *)malloc(dim * sizeof(float));
    float *final_shift = (float *)calloc(dim, sizeof(float));  /* zero shift */
    float *final_scale_param = (float *)malloc(dim * sizeof(float)); /* adaln expects (1+scale) */
    if (!img_hidden_gpu || !final_norm_gpu || !final_scale || !final_shift ||
        !final_scale_param ||
        !zi_final_compute_scale(final_scale, &tf->final_layer, t_emb, tf)) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(unified_gpu);
        if (img_hidden_gpu) iris_gpu_tensor_free(img_hidden_gpu);
        if (final_norm_gpu) iris_gpu_tensor_free(final_norm_gpu);
        free(final_scale);
        free(final_shift);
        free(final_scale_param);
        return NULL;
    }
    for (int i = 0; i < dim; i++) final_scale_param[i] = final_scale[i] - 1.0f;

    /* Slice first img_seq tokens from unified hidden. */
    iris_gpu_copy_region_f32(img_hidden_gpu, 0, unified_gpu, 0, (size_t)img_seq * dim);
    iris_gpu_adaln_norm(final_norm_gpu, img_hidden_gpu,
                        final_shift, final_scale_param, img_seq, dim, 1e-6f);
    final_out_gpu = iris_gpu_linear(final_norm_gpu, tf->final_layer.linear_weight, NULL,
                                    img_seq, dim, out_ch);
    if (!final_out_gpu) {
        iris_gpu_batch_end();
        zi_gpu_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(unified_gpu);
        iris_gpu_tensor_free(img_hidden_gpu);
        iris_gpu_tensor_free(final_norm_gpu);
        free(final_scale);
        free(final_shift);
        free(final_scale_param);
        return NULL;
    }

    iris_gpu_batch_end();
    zi_gpu_scratch_free(&scratch);
    free(step_mod);
    iris_gpu_tensor_free(unified_gpu);
    iris_gpu_tensor_free(img_hidden_gpu);
    iris_gpu_tensor_free(final_norm_gpu);
    free(final_shift);
    free(final_scale_param);

    /* Read back final projected patches and add bias on CPU. */
    float *final_out = (float *)malloc((size_t)img_seq * out_ch * sizeof(float));
    if (!final_out) {
        iris_gpu_tensor_free(final_out_gpu);
        free(final_scale);
        return NULL;
    }
    float *final_out_data = iris_gpu_tensor_data(final_out_gpu);
    memcpy(final_out, final_out_data, (size_t)img_seq * out_ch * sizeof(float));
    iris_gpu_tensor_free(final_out_gpu);
    for (int s = 0; s < img_seq; s++) {
        for (int i = 0; i < out_ch; i++) {
            final_out[s * out_ch + i] += tf->final_layer.linear_bias[i];
        }
    }
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);
    free(final_scale);

    /* === CPU: Unpatchify === */
    float *output = (float *)calloc(in_ch * latent_h * latent_w, sizeof(float));
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    free(final_out);
    t_final_ms = zi_time_ms() - stage_start;

    /* Accumulate per-step zImage GPU timing. */
    iris_timing_zi_embeddings += t_embed_ms;
    iris_timing_zi_noise_refiner += t_noise_ms;
    iris_timing_zi_context_refiner += t_context_ms;
    iris_timing_zi_main_blocks += t_main_ms;
    iris_timing_zi_final += t_final_ms;
    iris_timing_zi_total += t_embed_ms + t_noise_ms + t_context_ms + t_main_ms + t_final_ms;

    return output;
}

#endif /* USE_METAL */

/* ========================================================================
 * Final Layer
 * ======================================================================== */

/* Final layer AdaLN modulation: scale = 1 + Linear(SiLU(t_emb)) */
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf) {
    if (!scale || !fl || !t_emb || !tf) return 0;

    float silu_emb[256];
    memcpy(silu_emb, t_emb, tf->adaln_dim * sizeof(float));
    iris_silu(silu_emb, tf->adaln_dim);

    iris_matmul_t(scale, silu_emb, fl->adaln_weight, 1, tf->adaln_dim, tf->dim);
    for (int i = 0; i < tf->dim; i++) scale[i] = 1.0f + scale[i] + fl->adaln_bias[i];
    return 1;
}

/* Z-Image final layer: LayerNorm (no affine) -> scale by
 * (1 + SiLU(Linear(t_emb))) -> Linear projection to patch channels.
 * Note the SiLU activation in the final layer's modulation -- this differs
 * from the block modulation which has no activation. Output shape is
 * [img_seq, patch_size^2 * in_channels]. */
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int out_dim = tf->patch_size * tf->patch_size * tf->in_channels;

    float *scale = (float *)malloc(dim * sizeof(float));
    if (!scale || !zi_final_compute_scale(scale, fl, t_emb, tf)) {
        free(scale);
        return;
    }

    /* LayerNorm (no affine) -> scale */
    float *normed = (float *)malloc(seq * dim * sizeof(float));
    for (int s = 0; s < seq; s++) {
        const float *xr = x + s * dim;
        float *nr = normed + s * dim;

        /* Compute mean and variance */
        float mean = 0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= dim;

        float var = 0;
        for (int i = 0; i < dim; i++) {
            float d = xr[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / sqrtf(var + 1e-6f); /* Final LayerNorm uses 1e-6 */

        for (int i = 0; i < dim; i++)
            nr[i] = (xr[i] - mean) * inv_std * scale[i];
    }

    /* Linear projection: dim -> out_dim */
    iris_matmul_t(out, normed, fl->linear_weight, seq, dim, out_dim);
    for (int s = 0; s < seq; s++)
        for (int i = 0; i < out_dim; i++)
            out[s * out_dim + i] += fl->linear_bias[i];

    free(scale);
    free(normed);
}

/* ========================================================================
 * Patchify / Unpatchify
 * ======================================================================== */

/* Converts latent [in_ch, H, W] to patch sequence [n_patches, ps*ps*in_ch].
 * Gathers each ps x ps spatial block into a flat vector, ordering as
 * (ph, pw, channel). This is the inverse of unpatchify and creates the
 * token sequence the transformer operates on. */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            float *dst = out + patch_idx * patch_feat;
            int di = 0;

            /* Gather patch: iterate (ph, pw, c) */
            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        dst[di++] = latent[c * H * W + sy * W + sx];
                    }
                }
            }
        }
    }
}

/* Unpatchify: [n_patches, patch_feat_dim] -> [in_ch, H, W] */
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            const float *src = patches + patch_idx * patch_feat;
            int si = 0;

            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        latent[c * H * W + sy * W + sx] = src[si++];
                    }
                }
            }
        }
    }
}

#ifdef ZI_DEV
/* ========================================================================
 * GPU Forward (CUDA / ROCm / Metal)
 * ======================================================================== */

typedef struct {
    uint16_t *qkv, *proj, *ff_in, *ff_out, *mod;
    float *norm_q, *norm_k, *attn_norm1, *attn_norm2, *ffn_norm1, *ffn_norm2;
    float *mod_bias;
} zi_dev_block_t;

struct zi_dev {
    zi_dev_block_t *noise, *context, *layers;
    uint16_t *t_lin1, *t_lin2, *img_in, *cap_in, *mod_final, *proj_final;
    float *t_bias1, *t_bias2, *img_bias, *cap_bias, *cap_norm;
    float *img_pad, *cap_pad, *mod_final_bias, *proj_final_bias;
    float *vec, *mods;          /* timestep scratch and per-block modulation */
    uint16_t *vec_bf16;

    int seq;
    float *x, *out;             /* f32 residual and projected output */
    uint16_t *norm, *h, *act;   /* normalized input, QKV / gate-up, attention / FFN */
    float *rope_cos, *rope_sin; /* [seq, head_dim/2] */
    int *mask;
    size_t in_elems;
    float *in_f32;
    uint16_t *in_bf16;
};

/* Convert the mmapped F32 weights in bounded chunks (2 MB of host scratch),
 * rounding to nearest even like the device conversion. Concatenation is on
 * the device, so neither QKV nor the gate/up weights need a host-sized copy. */
static uint16_t *zi_dev_rows(const float *const *parts, const size_t *counts, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        if (!parts[i]) return NULL;
        total += counts[i];
    }
    size_t chunk = total < (1u << 20) ? total : (1u << 20);
    uint16_t *tmp = malloc(chunk * sizeof(uint16_t));
    if (!tmp) return NULL;
    uint16_t *out = iris_dev_alloc(total * sizeof(uint16_t));
    if (!out) { free(tmp); return NULL; }
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        for (size_t j = 0; j < counts[i]; j += chunk) {
            size_t len = counts[i] - j;
            if (len > chunk) len = chunk;
            for (size_t k = 0; k < len; k++) {
                uint32_t bits;
                memcpy(&bits, parts[i] + j + k, sizeof(bits));
                bits += 0x7fff + ((bits >> 16) & 1);
                tmp[k] = (uint16_t)(bits >> 16);
            }
            iris_dev_upload(out + off + j, tmp, len * sizeof(uint16_t));
        }
        off += counts[i];
    }
    free(tmp);
    return out;
}

static uint16_t *zi_dev_bf16(const float *p, size_t n) {
    return zi_dev_rows(&p, &n, 1);
}

static float *zi_dev_f32(const float *p, size_t n) {
    if (!p) return NULL;
    float *out = iris_dev_alloc(n * sizeof(float));
    if (out) iris_dev_upload(out, p, n * sizeof(float));
    return out;
}

static int zi_dev_load_block(zi_transformer_t *tf, zi_dev_block_t *d, const zi_block_t *b) {
    size_t dim = tf->dim, dd = dim * dim, md = dim * tf->ffn_dim;
    const float *qkv[] = {b->attn_q_weight, b->attn_k_weight, b->attn_v_weight};
    size_t qkv_n[] = {dd, dd, dd};
    const float *ff[] = {b->ffn_w1, b->ffn_w3};
    size_t ff_n[] = {md, md};
    d->qkv = zi_dev_rows(qkv, qkv_n, 3);
    d->proj = zi_dev_bf16(b->attn_out_weight, dd);
    d->ff_in = zi_dev_rows(ff, ff_n, 2);
    d->ff_out = zi_dev_bf16(b->ffn_w2, md);
    d->norm_q = zi_dev_f32(b->attn_norm_q, tf->head_dim);
    d->norm_k = zi_dev_f32(b->attn_norm_k, tf->head_dim);
    d->attn_norm1 = zi_dev_f32(b->attn_norm1, dim);
    d->attn_norm2 = zi_dev_f32(b->attn_norm2, dim);
    d->ffn_norm1 = zi_dev_f32(b->ffn_norm1, dim);
    d->ffn_norm2 = zi_dev_f32(b->ffn_norm2, dim);
    if (b->adaln_weight) {
        d->mod = zi_dev_bf16(b->adaln_weight, 4 * dim * tf->adaln_dim);
        d->mod_bias = zi_dev_f32(b->adaln_bias, 4 * dim);
        if (!d->mod || !d->mod_bias) return 0;
    }
    return d->qkv && d->proj && d->ff_in && d->ff_out && d->norm_q && d->norm_k &&
           d->attn_norm1 && d->attn_norm2 && d->ffn_norm1 && d->ffn_norm2;
}

static void zi_dev_free_blocks(zi_dev_block_t *b, int n) {
    if (!b) return;
    for (int i = 0; i < n; i++) {
        iris_dev_free(b[i].qkv);
        iris_dev_free(b[i].proj);
        iris_dev_free(b[i].ff_in);
        iris_dev_free(b[i].ff_out);
        iris_dev_free(b[i].mod);
        iris_dev_free(b[i].mod_bias);
        iris_dev_free(b[i].norm_q);
        iris_dev_free(b[i].norm_k);
        iris_dev_free(b[i].attn_norm1);
        iris_dev_free(b[i].attn_norm2);
        iris_dev_free(b[i].ffn_norm1);
        iris_dev_free(b[i].ffn_norm2);
    }
    free(b);
}

static void zi_dev_free_activations(struct zi_dev *d) {
    iris_dev_free(d->x); d->x = NULL;
    iris_dev_free(d->out); d->out = NULL;
    iris_dev_free(d->norm); d->norm = NULL;
    iris_dev_free(d->h); d->h = NULL;
    iris_dev_free(d->act); d->act = NULL;
    iris_dev_free(d->rope_cos); d->rope_cos = NULL;
    iris_dev_free(d->rope_sin); d->rope_sin = NULL;
    iris_dev_free(d->mask); d->mask = NULL;
    d->seq = 0;
}

static void zi_dev_free(zi_transformer_t *tf) {
    struct zi_dev *d = tf->dev;
    if (!d) return;
    zi_dev_free_blocks(d->noise, tf->n_refiner);
    zi_dev_free_blocks(d->context, tf->n_refiner);
    zi_dev_free_blocks(d->layers, tf->n_layers);
    iris_dev_free(d->t_lin1);
    iris_dev_free(d->t_lin2);
    iris_dev_free(d->t_bias1);
    iris_dev_free(d->t_bias2);
    iris_dev_free(d->img_in);
    iris_dev_free(d->cap_in);
    iris_dev_free(d->img_bias);
    iris_dev_free(d->cap_bias);
    iris_dev_free(d->cap_norm);
    iris_dev_free(d->img_pad);
    iris_dev_free(d->cap_pad);
    iris_dev_free(d->mod_final);
    iris_dev_free(d->mod_final_bias);
    iris_dev_free(d->proj_final);
    iris_dev_free(d->proj_final_bias);
    iris_dev_free(d->vec);
    iris_dev_free(d->vec_bf16);
    iris_dev_free(d->mods);
    zi_dev_free_activations(d);
    iris_dev_free(d->in_f32);
    iris_dev_free(d->in_bf16);
    free(d);
    tf->dev = NULL;
}

static int zi_dev_load(zi_transformer_t *tf) {
    struct zi_dev *d = calloc(1, sizeof(*d));
    tf->dev = d;
    if (!d) goto fail;
    size_t dim = tf->dim, mid = tf->t_emb_mid_size, ad = tf->adaln_dim;
    int patch = tf->patch_size * tf->patch_size * tf->in_channels;
    d->t_lin1 = zi_dev_bf16(tf->t_emb_mlp0_weight, mid * 256);
    d->t_lin2 = zi_dev_bf16(tf->t_emb_mlp2_weight, ad * mid);
    d->t_bias1 = zi_dev_f32(tf->t_emb_mlp0_bias, mid);
    d->t_bias2 = zi_dev_f32(tf->t_emb_mlp2_bias, ad);
    d->img_in = zi_dev_bf16(tf->x_emb_weight, dim * patch);
    d->img_bias = zi_dev_f32(tf->x_emb_bias, dim);
    d->img_pad = zi_dev_f32(tf->x_pad_token, dim);
    d->cap_in = zi_dev_bf16(tf->cap_emb_linear_w, dim * tf->cap_feat_dim);
    d->cap_bias = zi_dev_f32(tf->cap_emb_linear_b, dim);
    d->cap_norm = zi_dev_f32(tf->cap_emb_norm, tf->cap_feat_dim);
    d->cap_pad = zi_dev_f32(tf->cap_pad_token, dim);
    d->mod_final = zi_dev_bf16(tf->final_layer.adaln_weight, dim * ad);
    d->mod_final_bias = zi_dev_f32(tf->final_layer.adaln_bias, dim);
    d->proj_final = zi_dev_bf16(tf->final_layer.linear_weight, patch * dim);
    d->proj_final_bias = zi_dev_f32(tf->final_layer.linear_bias, patch);
    if (!d->t_lin1 || !d->t_lin2 || !d->t_bias1 || !d->t_bias2 || !d->img_in ||
        !d->img_bias || !d->img_pad || !d->cap_in || !d->cap_bias || !d->cap_norm ||
        !d->cap_pad || !d->mod_final || !d->mod_final_bias || !d->proj_final || !d->proj_final_bias) goto fail;
    d->noise = calloc(tf->n_refiner, sizeof(*d->noise));
    d->context = calloc(tf->n_refiner, sizeof(*d->context));
    d->layers = calloc(tf->n_layers, sizeof(*d->layers));
    if ((tf->n_refiner && (!d->noise || !d->context)) || (tf->n_layers && !d->layers)) goto fail;
    for (int i = 0; i < tf->n_refiner; i++) {
        if (!zi_dev_load_block(tf, &d->noise[i], &tf->noise_refiner[i]) ||
            !zi_dev_load_block(tf, &d->context[i], &tf->context_refiner[i])) goto fail;
    }
    for (int i = 0; i < tf->n_layers; i++)
        if (!zi_dev_load_block(tf, &d->layers[i], &tf->layers[i])) goto fail;
    d->vec = iris_dev_alloc((256 + mid + ad) * sizeof(float));
    d->vec_bf16 = iris_dev_alloc((256 + mid + ad) * sizeof(uint16_t));
    d->mods = iris_dev_alloc(((size_t)(tf->n_refiner + tf->n_layers) * 4 + 1) * dim * sizeof(float));
    if (!d->vec || !d->vec_bf16 || !d->mods) goto fail;
    iris_dev_sync();
    return 1;
fail:
    fprintf(stderr, "Z-Image: GPU weight upload failed, using the CPU\n");
    zi_dev_free(tf);
    return 0;
}

static int zi_dev_ensure(zi_transformer_t *tf, int seq, size_t in_elems) {
    struct zi_dev *d = tf->dev;
    size_t dim = tf->dim, ff = tf->ffn_dim, half = tf->head_dim / 2;
    size_t patch = tf->patch_size * tf->patch_size * tf->in_channels;
    if (seq > d->seq) {
        zi_dev_free_activations(d);
        d->x = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->out = iris_dev_alloc((size_t)seq * (dim > patch ? dim : patch) * sizeof(float));
        d->norm = iris_dev_alloc((size_t)seq * dim * sizeof(uint16_t));
        d->h = iris_dev_alloc((size_t)seq * (3 * dim > 2 * ff ? 3 * dim : 2 * ff) * sizeof(uint16_t));
        d->act = iris_dev_alloc((size_t)seq * (dim > ff ? dim : ff) * sizeof(uint16_t));
        d->rope_cos = iris_dev_alloc((size_t)seq * half * sizeof(float));
        d->rope_sin = iris_dev_alloc((size_t)seq * half * sizeof(float));
        d->mask = iris_dev_alloc((size_t)seq * sizeof(int));
        if (!d->x || !d->out || !d->norm || !d->h || !d->act || !d->rope_cos || !d->rope_sin || !d->mask) return 0;
        d->seq = seq;
    }
    if (in_elems > d->in_elems) {
        iris_dev_free(d->in_f32);
        iris_dev_free(d->in_bf16);
        d->in_f32 = iris_dev_alloc(in_elems * sizeof(float));
        d->in_bf16 = iris_dev_alloc(in_elems * sizeof(uint16_t));
        if (!d->in_f32 || !d->in_bf16) return 0;
        d->in_elems = in_elems;
    }
    return 1;
}

/* Gather the tables made by zi_precompute_rope. Invalid positions leave an
 * axis unrotated, exactly as zi_apply_rope does on the CPU. */
static void zi_dev_rope(float *cos, float *sin, const int *pos, int seq, zi_transformer_t *tf) {
    int half = tf->head_dim / 2;
    for (int s = 0; s < seq; s++) {
        int off = 0;
        for (int ax = 0; ax < 3; ax++) {
            int n = tf->axes_dims[ax] / 2, p = pos[s * 3 + ax];
            int valid = p >= 0 && p < tf->axes_lens[ax];
            for (int i = 0; i < n; i++) {
                cos[(size_t)s * half + off + i] = valid ? tf->rope_cos[ax][p * n + i] : 1.0f;
                sin[(size_t)s * half + off + i] = valid ? tf->rope_sin[ax][p * n + i] : 0.0f;
            }
            off += n;
        }
    }
}

static void zi_dev_block(zi_transformer_t *tf, const zi_dev_block_t *b, float *x,
                          int seq, int first, const float *mod, const int *mask) {
    struct zi_dev *d = tf->dev;
    int dim = tf->dim, ff = tf->ffn_dim, hd = tf->head_dim;
    const float *cos = d->rope_cos + (size_t)first * (hd / 2);
    const float *sin = d->rope_sin + (size_t)first * (hd / 2);
    iris_dev_rms_norm_mod(d->norm, x, b->attn_norm1, mod, seq, dim, ZI_NORM_EPS);
    iris_dev_linear_bf16(d->h, d->norm, b->qkv, seq, dim, 3 * dim);
    iris_dev_head_norm_rope(d->h, 3 * dim, b->norm_q, cos, sin, seq, tf->n_heads, hd, ZI_NORM_EPS, 0);
    iris_dev_head_norm_rope(d->h + dim, 3 * dim, b->norm_k, cos, sin, seq, tf->n_heads, hd, ZI_NORM_EPS, 0);
    iris_dev_attention(d->act, dim, d->h, d->h + dim, d->h + 2 * dim, 3 * dim,
        seq, seq, tf->n_heads, tf->n_heads, hd, 1.0f / sqrtf(hd), 0, mask);
    iris_dev_linear_f32(d->out, d->act, b->proj, seq, dim, dim);
    iris_dev_rms_norm_residual(x, d->out, b->attn_norm2, mod ? mod + dim : NULL, seq, dim, ZI_NORM_EPS);
    iris_dev_rms_norm_mod(d->norm, x, b->ffn_norm1, mod ? mod + 2 * dim : NULL, seq, dim, ZI_NORM_EPS);
    iris_dev_linear_bf16(d->h, d->norm, b->ff_in, seq, dim, 2 * ff);
    iris_dev_swiglu(d->act, ff, d->h, d->h + ff, 2 * ff, seq, ff);
    iris_dev_linear_f32(d->out, d->act, b->ff_out, seq, ff, dim);
    iris_dev_rms_norm_residual(x, d->out, b->ffn_norm2, mod ? mod + 3 * dim : NULL, seq, dim, ZI_NORM_EPS);
}

static float *zi_forward_dev(zi_transformer_t *tf, const float *latent, int latent_h, int latent_w,
                              float timestep, const float *cap_feats, int cap_seq) {
    struct zi_dev *d = tf->dev;
    int dim = tf->dim, ps = tf->patch_size, patch = ps * ps * tf->in_channels;
    int ht = latent_h / ps, wt = latent_w / ps, img_seq = ht * wt;
    int ip = (img_seq + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF * ZI_SEQ_MULTI_OF;
    int cp = (cap_seq + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF * ZI_SEQ_MULTI_OF;
    int seq = ip + cp, half = tf->head_dim / 2;
    size_t in_elems = (size_t)img_seq * patch;
    if (in_elems < (size_t)cap_seq * tf->cap_feat_dim) in_elems = (size_t)cap_seq * tf->cap_feat_dim;
    if (!zi_dev_ensure(tf, seq, in_elems)) goto fail;
    float *tokens = malloc((size_t)img_seq * patch * sizeof(float));
    float *output = calloc((size_t)tf->in_channels * latent_h * latent_w, sizeof(float));
    int *pos = calloc((size_t)seq * 3, sizeof(int));
    int *mask = malloc((size_t)seq * sizeof(int));
    float *cos = malloc((size_t)seq * half * sizeof(float));
    float *sin = malloc((size_t)seq * half * sizeof(float));
    if (!tokens || !output || !pos || !mask || !cos || !sin) {
        free(tokens); free(output); free(pos); free(mask); free(cos); free(sin);
        goto fail;
    }
    double start = zi_time_ms();
    for (int s = 0; s < ip; s++) {
        mask[s] = s < img_seq;
        if (s < img_seq) {
            pos[3*s] = cp + 1;
            pos[3*s+1] = s / wt;
            pos[3*s+2] = s % wt;
        }
    }
    for (int s = 0; s < cp; s++) {
        mask[ip+s] = s < cap_seq;
        pos[3*(ip+s)] = 1 + s;
    }
    zi_dev_rope(cos, sin, pos, seq, tf);
    iris_dev_upload(d->rope_cos, cos, (size_t)seq * half * sizeof(float));
    iris_dev_upload(d->rope_sin, sin, (size_t)seq * half * sizeof(float));
    iris_dev_upload(d->mask, mask, (size_t)seq * sizeof(int));
    free(pos); free(mask); free(cos); free(sin);

    zi_patchify(tokens, latent, tf->in_channels, latent_h, latent_w, ps);
    iris_dev_upload(d->in_f32, tokens, (size_t)img_seq * patch * sizeof(float));
    iris_dev_f32_to_bf16(d->in_bf16, d->in_f32, (size_t)img_seq * patch);
    iris_dev_linear_f32(d->x, d->in_bf16, d->img_in, img_seq, patch, dim);
    iris_dev_add_bias(d->x, d->img_bias, img_seq, dim);
    iris_dev_fill_rows(d->x + (size_t)img_seq * dim, d->img_pad, ip - img_seq, dim);
    float *cap = d->x + (size_t)ip * dim;
    iris_dev_upload(d->in_f32, cap_feats, (size_t)cap_seq * tf->cap_feat_dim * sizeof(float));
    iris_dev_rms_norm(d->in_bf16, d->in_f32, d->cap_norm, cap_seq, tf->cap_feat_dim, ZI_NORM_EPS);
    iris_dev_linear_f32(cap, d->in_bf16, d->cap_in, cap_seq, tf->cap_feat_dim, dim);
    iris_dev_add_bias(cap, d->cap_bias, cap_seq, dim);
    iris_dev_fill_rows(cap + (size_t)cap_seq * dim, d->cap_pad, cp - cap_seq, dim);

    float sin_emb[256];
    zi_sinusoidal_embedding(sin_emb, timestep * 1000.0f, 256);
    iris_dev_upload(d->vec, sin_emb, sizeof(sin_emb));
    iris_dev_f32_to_bf16(d->vec_bf16, d->vec, 256);
    float *hidden = d->vec + 256, *temb = hidden + tf->t_emb_mid_size;
    iris_dev_linear_f32(hidden, d->vec_bf16, d->t_lin1, 1, 256, tf->t_emb_mid_size);
    iris_dev_add_bias(hidden, d->t_bias1, 1, tf->t_emb_mid_size);
    iris_dev_silu(hidden, tf->t_emb_mid_size);
    iris_dev_f32_to_bf16(d->vec_bf16, hidden, tf->t_emb_mid_size);
    iris_dev_linear_f32(temb, d->vec_bf16, d->t_lin2, 1, tf->t_emb_mid_size, tf->adaln_dim);
    iris_dev_add_bias(temb, d->t_bias2, 1, tf->adaln_dim);
    iris_dev_f32_to_bf16(d->vec_bf16, temb, tf->adaln_dim);
    int nmod = tf->n_refiner + tf->n_layers;
    for (int i = 0; i < nmod; i++) {
        zi_dev_block_t *b = i < tf->n_refiner ? &d->noise[i] : &d->layers[i - tf->n_refiner];
        float *mod = d->mods + (size_t)i * 4 * dim;
        iris_dev_linear_f32(mod, d->vec_bf16, b->mod, 1, tf->adaln_dim, 4 * dim);
        iris_dev_add_bias(mod, b->mod_bias, 1, 4 * dim);
    }
    /* Only the final modulation applies SiLU to the timestep embedding. */
    float *fin = d->mods + (size_t)nmod * 4 * dim;
    iris_dev_silu(temb, tf->adaln_dim);
    iris_dev_f32_to_bf16(d->vec_bf16, temb, tf->adaln_dim);
    iris_dev_linear_f32(fin, d->vec_bf16, d->mod_final, 1, tf->adaln_dim, dim);
    iris_dev_add_bias(fin, d->mod_final_bias, 1, dim);
    iris_dev_sync();
    double embed_time = zi_time_ms() - start;
    start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_dev_block(tf, &d->noise[i], d->x, ip, 0, d->mods + (size_t)i * 4 * dim,
                     ip == img_seq ? NULL : d->mask);
        if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, 2 * tf->n_refiner);
    }
    iris_dev_sync();
    double noise_time = zi_time_ms() - start;
    start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_dev_block(tf, &d->context[i], cap, cp, ip, NULL, cp == cap_seq ? NULL : d->mask + ip);
        if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, 2 * tf->n_refiner);
    }
    iris_dev_sync();
    double context_time = zi_time_ms() - start;
    start = zi_time_ms();
    for (int i = 0; i < tf->n_layers; i++) {
        zi_dev_block(tf, &d->layers[i], d->x, seq, 0, d->mods + (size_t)(tf->n_refiner + i) * 4 * dim,
                     ip == img_seq && cp == cap_seq ? NULL : d->mask);
        if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }
    iris_dev_sync();
    double main_time = zi_time_ms() - start;
    start = zi_time_ms();
    iris_dev_layer_norm_mod(d->norm, d->x, NULL, fin, img_seq, dim, 1e-6f);
    iris_dev_linear_f32(d->out, d->norm, d->proj_final, img_seq, dim, patch);
    iris_dev_add_bias(d->out, d->proj_final_bias, img_seq, patch);
    iris_dev_download(tokens, d->out, (size_t)img_seq * patch * sizeof(float));
    zi_unpatchify(output, tokens, tf->in_channels, latent_h, latent_w, ps);
    free(tokens);
    double final_time = zi_time_ms() - start;
    iris_timing_zi_embeddings += embed_time;
    iris_timing_zi_noise_refiner += noise_time;
    iris_timing_zi_context_refiner += context_time;
    iris_timing_zi_main_blocks += main_time;
    iris_timing_zi_final += final_time;
    iris_timing_zi_total += embed_time + noise_time + context_time + main_time + final_time;
    if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);
    return output;
fail:
    fprintf(stderr, "Z-Image: GPU workspace allocation failed, using the CPU\n");
    zi_dev_free(tf);
    return NULL;
}
#endif /* ZI_DEV */

/* ========================================================================
 * Main Forward Pass
 * ======================================================================== */

/* Top-level Z-Image transformer entry point. Tries GPU path first, falls
 * back to CPU on failure. CPU path pads sequences to multiples of 32 and
 * uses padding masks. Pipeline: patchify -> embed image/caption ->
 * noise refiner (image self-attention) -> context refiner (caption
 * self-attention) -> concatenate [image, caption] -> main blocks (full
 * self-attention) -> final layer -> unpatchify. */
float *iris_transformer_forward_zimage(zi_transformer_t *tf,
                                const float *latent,
                                int latent_h, int latent_w,
                                float timestep,
                                const float *cap_feats,
                                int cap_seq_len) {
#ifdef ZI_DEV
    if (tf->dev) {
        float *out = zi_forward_dev(tf, latent, latent_h, latent_w, timestep, cap_feats, cap_seq_len);
        if (out) return out;
    }
#endif
#ifdef USE_METAL
    /* Try GPU-accelerated path first */
    if (tf->use_gpu) {
        float *result = zi_transformer_forward_gpu(tf, latent, latent_h, latent_w,
                                                    timestep, cap_feats, cap_seq_len);
        if (result) return result;
        /* Fall back to CPU on GPU failure */
        fprintf(stderr, "Z-Image GPU path failed, falling back to CPU\n");
    }
#endif

    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;  /* 64 */

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* Pad sequences to multiples of ZI_SEQ_MULTI_OF */
    int img_pad = (ZI_SEQ_MULTI_OF - (img_seq % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int cap_pad = (ZI_SEQ_MULTI_OF - (cap_seq_len % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int img_padded = img_seq + img_pad;
    int cap_padded = cap_seq_len + cap_pad;
    int unified_seq = img_padded + cap_padded;

    /* Ensure working memory is sufficient */
    size_t needed = (size_t)unified_seq * dim * 4 +
                    (size_t)unified_seq * dim * 3 +  /* QKV */
                    (size_t)unified_seq * unified_seq + /* attention scores */
                    (size_t)unified_seq * tf->ffn_dim * 2;
    if (needed > tf->work_alloc) {
        free(tf->work_x);
        free(tf->work_tmp);
        free(tf->work_qkv);
        free(tf->work_attn);
        free(tf->work_ffn);
        tf->work_x = (float *)malloc(unified_seq * dim * sizeof(float));
        tf->work_tmp = (float *)malloc(unified_seq * dim * 4 * sizeof(float));
        tf->work_qkv = (float *)malloc(unified_seq * dim * 3 * sizeof(float));
        tf->work_attn = (float *)malloc((size_t)unified_seq * unified_seq * sizeof(float));
        tf->work_ffn = (float *)malloc((size_t)unified_seq * tf->ffn_dim * 2 * sizeof(float));
        if (!tf->work_x || !tf->work_tmp || !tf->work_qkv || !tf->work_attn || !tf->work_ffn) {
            free(tf->work_x); tf->work_x = NULL;
            free(tf->work_tmp); tf->work_tmp = NULL;
            free(tf->work_qkv); tf->work_qkv = NULL;
            free(tf->work_attn); tf->work_attn = NULL;
            free(tf->work_ffn); tf->work_ffn = NULL;
            tf->work_alloc = 0;
            tf->max_seq = 0;
            return NULL;
        }
        tf->work_alloc = needed;
        tf->max_seq = unified_seq;
    }

    /* 1. Timestep embedding */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* 2. Patchify image -> [img_seq, patch_feat] */
    float *img_patches = (float *)malloc(img_padded * patch_feat * sizeof(float));
    if (!img_patches) return NULL;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* Pad image patches (repeat last token) */
    for (int i = img_seq; i < img_padded; i++)
        memcpy(img_patches + i * patch_feat,
               img_patches + (img_seq - 1) * patch_feat,
               patch_feat * sizeof(float));

    /* Embed image: [img_padded, patch_feat] -> [img_padded, dim] */
    float *img_emb = (float *)malloc(img_padded * dim * sizeof(float));
    if (!img_emb) {
        free(img_patches);
        return NULL;
    }
    iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_padded, patch_feat, dim);
    for (int s = 0; s < img_padded; s++)
        for (int i = 0; i < dim; i++)
            img_emb[s * dim + i] += tf->x_emb_bias[i];
    free(img_patches);

    /* Apply pad token to image padding positions */
    for (int s = img_seq; s < img_padded; s++)
        memcpy(img_emb + s * dim, tf->x_pad_token, dim * sizeof(float));

    /* 3. Caption embedding: RMSNorm -> Linear */
    float *cap_emb = (float *)malloc(cap_padded * dim * sizeof(float));
    float *cap_normed = (float *)malloc(cap_padded * tf->cap_feat_dim * sizeof(float));
    if (!cap_emb || !cap_normed) {
        free(img_emb);
        free(cap_emb);
        free(cap_normed);
        return NULL;
    }

    /* Pad caption features (repeat last token) */
    float *cap_padded_feats = (float *)malloc(cap_padded * tf->cap_feat_dim * sizeof(float));
    if (!cap_padded_feats) {
        free(img_emb);
        free(cap_emb);
        free(cap_normed);
        return NULL;
    }
    memcpy(cap_padded_feats, cap_feats, cap_seq_len * tf->cap_feat_dim * sizeof(float));
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_padded_feats + s * tf->cap_feat_dim,
               cap_feats + (cap_seq_len - 1) * tf->cap_feat_dim,
               tf->cap_feat_dim * sizeof(float));

    zi_rms_norm(cap_normed, cap_padded_feats, tf->cap_emb_norm,
                cap_padded, tf->cap_feat_dim, ZI_NORM_EPS);
    free(cap_padded_feats);

    iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                  cap_padded, tf->cap_feat_dim, dim);
    for (int s = 0; s < cap_padded; s++)
        for (int i = 0; i < dim; i++)
            cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
    free(cap_normed);

    /* Apply pad token to caption padding positions */
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_emb + s * dim, tf->cap_pad_token, dim * sizeof(float));

    /* 4. Build position IDs */

    /* Image position IDs: (T=cap_padded+1, H=h_idx, W=w_idx)
     * All image tokens share the same T position (one frame). */
    int *img_pos = (int *)calloc(img_padded * 3, sizeof(int));
    if (!img_pos) {
        free(img_emb);
        free(cap_emb);
        return NULL;
    }
    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int idx = h * W_tokens + w;
            img_pos[idx * 3 + 0] = cap_padded + 1;  /* T (same for all) */
            img_pos[idx * 3 + 1] = h;                /* H */
            img_pos[idx * 3 + 2] = w;                /* W */
        }
    }
    /* Padding tokens get (0, 0, 0) */

    /* Caption position IDs: (T=1+seq_idx, H=0, W=0) */
    int *cap_pos = (int *)calloc(cap_padded * 3, sizeof(int));
    if (!cap_pos) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        return NULL;
    }
    for (int s = 0; s < cap_padded; s++) {
        cap_pos[s * 3 + 0] = 1 + s;  /* T */
        cap_pos[s * 3 + 1] = 0;       /* H */
        cap_pos[s * 3 + 2] = 0;       /* W */
    }

    /* Image and caption masks */
    int *img_mask = (int *)malloc(img_padded * sizeof(int));
    if (!img_mask) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        free(cap_pos);
        return NULL;
    }
    for (int i = 0; i < img_seq; i++) img_mask[i] = 1;
    for (int i = img_seq; i < img_padded; i++) img_mask[i] = 0;

    int *cap_mask = (int *)malloc(cap_padded * sizeof(int));
    if (!cap_mask) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        free(cap_pos);
        free(img_mask);
        return NULL;
    }
    for (int i = 0; i < cap_seq_len; i++) cap_mask[i] = 1;
    for (int i = cap_seq_len; i < cap_padded; i++) cap_mask[i] = 0;

    /* 5. Noise refiner: image-only self-attention with modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(img_emb, &tf->noise_refiner[i], img_pos, img_mask,
                          t_emb, img_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }

    /* 6. Context refiner: caption-only self-attention without modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(cap_emb, &tf->context_refiner[i], cap_pos, cap_mask,
                          NULL, cap_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }

    /* 7. Build unified sequence: [image_tokens, caption_tokens] */
    float *unified = tf->work_x;
    memcpy(unified, img_emb, img_padded * dim * sizeof(float));
    memcpy(unified + img_padded * dim, cap_emb, cap_padded * dim * sizeof(float));
    free(img_emb);
    free(cap_emb);

    /* Unified position IDs */
    int *unified_pos = (int *)malloc(unified_seq * 3 * sizeof(int));
    if (!unified_pos) {
        free(img_pos);
        free(cap_pos);
        free(img_mask);
        free(cap_mask);
        return NULL;
    }
    memcpy(unified_pos, img_pos, img_padded * 3 * sizeof(int));
    memcpy(unified_pos + img_padded * 3, cap_pos, cap_padded * 3 * sizeof(int));
    free(img_pos);
    free(cap_pos);

    /* Unified mask */
    int *unified_mask = (int *)malloc(unified_seq * sizeof(int));
    if (!unified_mask) {
        free(unified_pos);
        free(img_mask);
        free(cap_mask);
        return NULL;
    }
    memcpy(unified_mask, img_mask, img_padded * sizeof(int));
    memcpy(unified_mask + img_padded, cap_mask, cap_padded * sizeof(int));
    free(img_mask);
    free(cap_mask);

    /* 8. Main transformer layers */
    for (int i = 0; i < tf->n_layers; i++) {
        zi_block_forward(unified, &tf->layers[i], unified_pos, unified_mask,
                          t_emb, unified_seq, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }

    free(unified_pos);
    free(unified_mask);

    /* 9. Final layer: extract image tokens only, then project */
    float *img_out = (float *)malloc(img_seq * dim * sizeof(float));
    if (!img_out) return NULL;
    memcpy(img_out, unified, img_seq * dim * sizeof(float));

    int out_ch = ps * ps * in_ch;  /* 64 */
    float *final_out = (float *)malloc(img_seq * out_ch * sizeof(float));
    if (!final_out) {
        free(img_out);
        return NULL;
    }
    zi_final_forward(final_out, img_out, &tf->final_layer, t_emb, img_seq, tf);
    free(img_out);
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

    /* 10. Unpatchify: [n_patches, 64] -> [16, latent_h, latent_w] */
    float *output = (float *)calloc(in_ch * latent_h * latent_w, sizeof(float));
    if (!output) {
        free(final_out);
        return NULL;
    }
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    free(final_out);

    return output;
}

/* ========================================================================
 * Weight Loading (Safetensors)
 * ======================================================================== */

static float *zi_get_tensor(safetensors_file_t **files, int n_files,
                              const char *name, int mmap_f32_weights) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) {
                fprintf(stderr, "Error: Z-Image tensor '%s' is not F32 in mmap mode\n", name);
                return NULL;
            }
            return (float *)safetensors_data(files[f], t);
        }
        return safetensors_get_f32(files[f], t);
    }
    fprintf(stderr, "Warning: Z-Image tensor '%s' not found\n", name);
    return NULL;
}

static float *zi_get_tensor_optional(safetensors_file_t **files, int n_files,
                                       const char *name, int mmap_f32_weights) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) return NULL;
            return (float *)safetensors_data(files[f], t);
        }
        return safetensors_get_f32(files[f], t);
    }
    return NULL;
}

static int zi_all_tensors_f32(safetensors_file_t **files, int n_files) {
    for (int f = 0; f < n_files; f++) {
        safetensors_file_t *sf = files[f];
        if (!sf) return 0;
        for (int i = 0; i < sf->num_tensors; i++) {
            if (sf->tensors[i].dtype != DTYPE_F32) return 0;
        }
    }
    return 1;
}

static int zi_load_block(zi_block_t *block, safetensors_file_t **files,
                          int n_files, const char *prefix, int has_modulation,
                          int dim, int ffn_dim, int use_gpu,
                          int mmap_f32_weights) {
    char name[256];

    /* Attention weights */
    snprintf(name, sizeof(name), "%s.attention.to_q.weight", prefix);
    block->attn_q_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.to_k.weight", prefix);
    block->attn_k_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.to_v.weight", prefix);
    block->attn_v_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.to_out.0.weight", prefix);
    block->attn_out_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* QK norm */
    snprintf(name, sizeof(name), "%s.attention.norm_q.weight", prefix);
    block->attn_norm_q = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.norm_k.weight", prefix);
    block->attn_norm_k = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* Pre/post attention norms */
    snprintf(name, sizeof(name), "%s.attention_norm1.weight", prefix);
    block->attn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention_norm2.weight", prefix);
    block->attn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* FFN weights */
    snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
    block->ffn_w1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
    block->ffn_w2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
    block->ffn_w3 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* FFN norms */
    snprintf(name, sizeof(name), "%s.ffn_norm1.weight", prefix);
    block->ffn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.ffn_norm2.weight", prefix);
    block->ffn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* AdaLN modulation (only for modulated blocks) */
    if (has_modulation) {
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.weight", prefix);
        block->adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.bias", prefix);
        block->adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    } else {
        block->adaln_weight = NULL;
        block->adaln_bias = NULL;
    }

    if (!block->attn_q_weight || !block->attn_k_weight || !block->attn_v_weight ||
        !block->attn_out_weight || !block->attn_norm_q || !block->attn_norm_k ||
        !block->attn_norm1 || !block->attn_norm2 || !block->ffn_w1 ||
        !block->ffn_w2 || !block->ffn_w3 || !block->ffn_norm1 ||
        !block->ffn_norm2) {
        return 0;
    }
    if (has_modulation && (!block->adaln_weight || !block->adaln_bias)) {
        return 0;
    }

#ifdef USE_METAL
    /* Convert large weight matrices to bf16 for GPU path */
    if (use_gpu) {
        size_t attn_mat_elems = (size_t)dim * dim;
        size_t ffn_mat_elems = (size_t)ffn_dim * dim;

        block->attn_q_weight_bf16 = zi_f32_to_bf16(block->attn_q_weight, (size_t)dim * dim);
        block->attn_k_weight_bf16 = zi_f32_to_bf16(block->attn_k_weight, (size_t)dim * dim);
        block->attn_v_weight_bf16 = zi_f32_to_bf16(block->attn_v_weight, (size_t)dim * dim);
        block->attn_out_weight_bf16 = zi_f32_to_bf16(block->attn_out_weight, (size_t)dim * dim);
        block->ffn_w1_bf16 = zi_f32_to_bf16(block->ffn_w1, (size_t)ffn_dim * dim);
        block->ffn_w2_bf16 = zi_f32_to_bf16(block->ffn_w2, (size_t)dim * ffn_dim);
        block->ffn_w3_bf16 = zi_f32_to_bf16(block->ffn_w3, (size_t)ffn_dim * dim);
        if (!block->attn_q_weight_bf16 || !block->attn_k_weight_bf16 ||
            !block->attn_v_weight_bf16 || !block->attn_out_weight_bf16 ||
            !block->ffn_w1_bf16 || !block->ffn_w2_bf16 || !block->ffn_w3_bf16) {
            return 0;
        }
        block->attn_qkv_weight_bf16 = zi_concat3_bf16(block->attn_q_weight_bf16, attn_mat_elems,
                                                       block->attn_k_weight_bf16, attn_mat_elems,
                                                       block->attn_v_weight_bf16, attn_mat_elems);
        block->ffn_w13_weight_bf16 = zi_concat_bf16(block->ffn_w1_bf16, ffn_mat_elems,
                                                    block->ffn_w3_bf16, ffn_mat_elems);

        /* Free f32 copies of large weights (keep small norm/adaln weights as f32) */
        free(block->attn_q_weight); block->attn_q_weight = NULL;
        free(block->attn_k_weight); block->attn_k_weight = NULL;
        free(block->attn_v_weight); block->attn_v_weight = NULL;
        free(block->attn_out_weight); block->attn_out_weight = NULL;
        free(block->ffn_w1); block->ffn_w1 = NULL;
        free(block->ffn_w2); block->ffn_w2 = NULL;
        free(block->ffn_w3); block->ffn_w3 = NULL;
    }
#else
    (void)use_gpu; (void)dim; (void)ffn_dim; (void)mmap_f32_weights;
#endif
    return 1;
}

static void zi_free_block(zi_block_t *block, int free_f32_weights) {
    if (free_f32_weights) {
        free(block->attn_q_weight);
        free(block->attn_k_weight);
        free(block->attn_v_weight);
        free(block->attn_out_weight);
        free(block->attn_norm_q);
        free(block->attn_norm_k);
        free(block->attn_norm1);
        free(block->attn_norm2);
        free(block->ffn_w1);
        free(block->ffn_w2);
        free(block->ffn_w3);
        free(block->ffn_norm1);
        free(block->ffn_norm2);
        free(block->adaln_weight);
        free(block->adaln_bias);
    }
#ifdef USE_METAL
    free(block->attn_q_weight_bf16);
    free(block->attn_k_weight_bf16);
    free(block->attn_v_weight_bf16);
    free(block->attn_qkv_weight_bf16);
    free(block->attn_out_weight_bf16);
    free(block->ffn_w1_bf16);
    free(block->ffn_w2_bf16);
    free(block->ffn_w3_bf16);
    free(block->ffn_w13_weight_bf16);
#endif
}

/* Loads Z-Image transformer weights from sharded safetensors files.
 * Auto-discovers shards from index JSON, probes weights to determine FFN dim
 * and timestep MLP size. In CPU mode: uses mmap zero-copy pointers for f32
 * weights. In GPU mode: converts all large weight matrices to bf16 and builds
 * fused QKV/W13 concatenations for faster matmuls. Pre-warms Metal buffer
 * cache after loading. */
zi_transformer_t *zi_transformer_load_safetensors(const char *model_dir,
                                                     int dim, int n_heads,
                                                     int n_layers, int n_refiner,
                                                     int cap_feat_dim, int in_channels,
                                                     int patch_size, float rope_theta,
                                                     const int *axes_dims) {
    zi_transformer_t *tf = calloc(1, sizeof(zi_transformer_t));
    if (!tf) return NULL;

    char name[256];

    /* Set config */
    tf->dim = dim;
    tf->n_heads = n_heads;
    tf->head_dim = dim / n_heads;
    tf->n_layers = n_layers;
    tf->n_refiner = n_refiner;
    tf->ffn_dim = (8 * dim / 3 + 255) / 256 * 256;  /* Round up to 256 */
    tf->in_channels = in_channels;
    tf->patch_size = patch_size;
    tf->adaln_dim = dim < 256 ? dim : 256;
    tf->rope_theta = rope_theta;
    tf->cap_feat_dim = cap_feat_dim;

    for (int i = 0; i < 3; i++) {
        tf->axes_dims[i] = axes_dims[i];
        tf->axes_lens[i] = 1024;  /* Default max positions */
    }

    /* Open safetensors files */
    char path[1024];

    /* Try index file first for sharded models */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json", model_dir);
    FILE *idx_f = fopen(path, "r");

    safetensors_file_t *files[ZI_MAX_SHARDS] = {0};
    int n_files = 0;

    if (idx_f) {
        /* Sharded: parse index to find shard files */
        fseek(idx_f, 0, SEEK_END);
        long fsize = ftell(idx_f);
        fseek(idx_f, 0, SEEK_SET);
        char *json = (char *)malloc(fsize + 1);
        if (!json) {
            fclose(idx_f);
            goto error;
        }
        fread(json, 1, fsize, idx_f);
        json[fsize] = 0;
        fclose(idx_f);

        /* Find unique shard filenames */
        char seen[32][128];
        int n_seen = 0;
        char *p = json;
        while ((p = strstr(p, ".safetensors")) != NULL) {
            /* Find start of filename */
            char *end = p + strlen(".safetensors");
            char *start = p;
            while (start > json && *(start - 1) != '"') start--;

            int len = (int)(end - start);
            if (len < 128) {
                char fname[128];
                memcpy(fname, start, len);
                fname[len] = 0;

                /* Check if already seen */
                int found = 0;
                for (int i = 0; i < n_seen; i++) {
                    if (strcmp(seen[i], fname) == 0) { found = 1; break; }
                }
                if (!found && n_seen < ZI_MAX_SHARDS) {
                    strcpy(seen[n_seen], fname);
                    n_seen++;
                }
            }
            p = end;
        }
        free(json);

        /* Open each shard */
        for (int i = 0; i < n_seen && n_files < ZI_MAX_SHARDS; i++) {
            snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, seen[i]);
            files[n_files] = safetensors_open(path);
            if (files[n_files]) n_files++;
        }
    } else {
        /* Single file */
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        files[0] = safetensors_open(path);
        if (files[0]) n_files = 1;
    }

    if (n_files == 0) {
        fprintf(stderr, "Z-Image: failed to open transformer safetensors\n");
        goto error;
    }

    tf->num_sf_files = n_files;
    for (int i = 0; i < n_files; i++) tf->sf_files[i] = files[i];

    if (iris_verbose)
        fprintf(stderr, "  Loading Z-Image transformer (%d shards)...\n", n_files);

    /* Determine FFN dimension from weights */
    const safetensor_t *w1_probe = NULL;
    for (int f = 0; f < n_files && !w1_probe; f++)
        w1_probe = safetensors_find(files[f], "layers.0.feed_forward.w1.weight");
    if (w1_probe) {
        tf->ffn_dim = (int)w1_probe->shape[0];
    }

    /* Determine t_embedder mid_size from weights */
    tf->t_emb_mid_size = 1024;  /* Default */
    const safetensor_t *t_probe = NULL;
    for (int f = 0; f < n_files && !t_probe; f++)
        t_probe = safetensors_find(files[f], "t_embedder.mlp.0.weight");
    if (t_probe) {
        tf->t_emb_mid_size = (int)t_probe->shape[0];
    }

    /* Check if GPU acceleration is available. The native Metal path is used
     * only without the device backend, which uploads its own bf16 copy of
     * the mmapped f32 weights. */
    int use_gpu = 0;
#ifdef USE_METAL
    if (iris_metal_available() && iris_metal_shaders_available() && !iris_dev_available()) {
        use_gpu = 1;
        tf->use_gpu = 1;
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: GPU acceleration enabled (bf16 weights)\n");
    }
#endif
    /* BLAS/CPU fast-load mode: keep mmap files open and use direct f32 pointers. */
    int mmap_f32_weights = (!use_gpu && zi_all_tensors_f32(files, n_files));
    tf->mmap_f32_weights = mmap_f32_weights;
    if (mmap_f32_weights) {
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: CPU mmap mode enabled (zero-copy f32 weights)\n");
    }

    /* Load timestep embedder */
    tf->t_emb_mlp0_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.0.weight", mmap_f32_weights);
    tf->t_emb_mlp0_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.0.bias", mmap_f32_weights);
    tf->t_emb_mlp2_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.2.weight", mmap_f32_weights);
    tf->t_emb_mlp2_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.2.bias", mmap_f32_weights);
    if (!tf->t_emb_mlp0_weight || !tf->t_emb_mlp0_bias ||
        !tf->t_emb_mlp2_weight || !tf->t_emb_mlp2_bias) {
        goto error;
    }

    /* Load caption embedder: RMSNorm + Linear */
    tf->cap_emb_norm = zi_get_tensor(files, n_files, "cap_embedder.0.weight", mmap_f32_weights);
    tf->cap_emb_linear_w = zi_get_tensor(files, n_files, "cap_embedder.1.weight", mmap_f32_weights);
    tf->cap_emb_linear_b = zi_get_tensor(files, n_files, "cap_embedder.1.bias", mmap_f32_weights);
    if (!tf->cap_emb_norm || !tf->cap_emb_linear_w || !tf->cap_emb_linear_b) {
        goto error;
    }

    /* Load image embedder */
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.weight", patch_size);
    tf->x_emb_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.bias", patch_size);
    tf->x_emb_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->x_emb_weight || !tf->x_emb_bias) {
        goto error;
    }

    /* Pad tokens */
    tf->x_pad_token = zi_get_tensor(files, n_files, "x_pad_token", mmap_f32_weights);
    tf->cap_pad_token = zi_get_tensor(files, n_files, "cap_pad_token", mmap_f32_weights);
    if (!tf->x_pad_token || !tf->cap_pad_token) {
        goto error;
    }

    /* Load noise refiner blocks */
    tf->noise_refiner = calloc(n_refiner, sizeof(zi_block_t));
    if (!tf->noise_refiner) goto error;
    for (int i = 0; i < n_refiner; i++) {
        snprintf(name, sizeof(name), "noise_refiner.%d", i);
        if (!zi_load_block(&tf->noise_refiner[i], files, n_files, name, 1,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights)) {
            goto error;
        }
    }

    /* Load context refiner blocks (no modulation) */
    tf->context_refiner = calloc(n_refiner, sizeof(zi_block_t));
    if (!tf->context_refiner) goto error;
    for (int i = 0; i < n_refiner; i++) {
        snprintf(name, sizeof(name), "context_refiner.%d", i);
        if (!zi_load_block(&tf->context_refiner[i], files, n_files, name, 0,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights)) {
            goto error;
        }
    }

    /* Load main transformer blocks */
    tf->layers = calloc(n_layers, sizeof(zi_block_t));
    if (!tf->layers) goto error;
    for (int i = 0; i < n_layers; i++) {
        snprintf(name, sizeof(name), "layers.%d", i);
        if (!zi_load_block(&tf->layers[i], files, n_files, name, 1,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights)) {
            goto error;
        }
    }

    /* Load final layer */
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.weight", patch_size);
    tf->final_layer.adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.bias", patch_size);
    tf->final_layer.adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.norm_final.weight", patch_size);
    tf->final_layer.norm_weight = zi_get_tensor_optional(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.weight", patch_size);
    tf->final_layer.linear_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.bias", patch_size);
    tf->final_layer.linear_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->final_layer.adaln_weight || !tf->final_layer.adaln_bias ||
        !tf->final_layer.linear_weight || !tf->final_layer.linear_bias) {
        goto error;
    }

    /* Precompute RoPE tables */
    zi_precompute_rope(tf);

    /* Allocate initial working memory (will be resized as needed) */
    tf->work_alloc = 0;
    tf->work_x = NULL;
    tf->work_tmp = NULL;
    tf->work_qkv = NULL;
    tf->work_attn = NULL;
    tf->work_ffn = NULL;
    tf->max_seq = 0;

#ifdef ZI_DEV
    if (iris_dev_available()) zi_dev_load(tf);
#endif

    /* Close safetensors files unless CPU mmap mode is active. */
    if (!mmap_f32_weights) {
        for (int f = 0; f < n_files; f++) {
            if (tf->sf_files[f]) {
                safetensors_close(tf->sf_files[f]);
                tf->sf_files[f] = NULL;
            }
        }
        tf->num_sf_files = 0;
    }

    if (iris_verbose) {
        fprintf(stderr, "  Z-Image transformer loaded: dim=%d, heads=%d, layers=%d+%d+%d, ffn=%d\n",
                dim, n_heads, n_refiner, n_refiner, n_layers, tf->ffn_dim);
    }

#ifdef USE_METAL
    /* Pre-warm bf16->Metal buffer cache so first denoising step avoids misses. */
    iris_warmup_bf16_zimage(tf);
#endif

    return tf;

error:
    iris_transformer_free_zimage(tf);
    return NULL;
}

void iris_transformer_free_zimage(zi_transformer_t *tf) {
    if (!tf) return;
#ifdef ZI_DEV
    zi_dev_free(tf);
#endif

    int free_f32_weights = !tf->mmap_f32_weights;

    if (free_f32_weights) {
        free(tf->t_emb_mlp0_weight);
        free(tf->t_emb_mlp0_bias);
        free(tf->t_emb_mlp2_weight);
        free(tf->t_emb_mlp2_bias);
        free(tf->cap_emb_norm);
        free(tf->cap_emb_linear_w);
        free(tf->cap_emb_linear_b);
        free(tf->x_emb_weight);
        free(tf->x_emb_bias);
        free(tf->x_pad_token);
        free(tf->cap_pad_token);
    }

    if (tf->noise_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->noise_refiner[i], free_f32_weights);
        free(tf->noise_refiner);
    }
    if (tf->context_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->context_refiner[i], free_f32_weights);
        free(tf->context_refiner);
    }
    if (tf->layers) {
        for (int i = 0; i < tf->n_layers; i++)
            zi_free_block(&tf->layers[i], free_f32_weights);
        free(tf->layers);
    }

    if (free_f32_weights) {
        free(tf->final_layer.adaln_weight);
        free(tf->final_layer.adaln_bias);
        free(tf->final_layer.norm_weight);
        free(tf->final_layer.linear_weight);
        free(tf->final_layer.linear_bias);
    }

    for (int i = 0; i < tf->num_sf_files; i++) {
        if (tf->sf_files[i]) {
            safetensors_close(tf->sf_files[i]);
            tf->sf_files[i] = NULL;
        }
    }
    tf->num_sf_files = 0;

    for (int i = 0; i < 3; i++) {
        free(tf->rope_cos[i]);
        free(tf->rope_sin[i]);
    }

    free(tf->work_x);
    free(tf->work_tmp);
    free(tf->work_qkv);
    free(tf->work_attn);
    free(tf->work_ffn);

#ifdef USE_METAL
    zi_gpu_rope_cache_clear(tf);
#endif

    free(tf);
}
