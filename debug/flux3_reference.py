#!/usr/bin/env python3
"""
FLUX.3 Image Klein reference pipeline, for parity checks of the C code.

Reimplements the official diffusers modular pipeline (text encoding,
latent/ids preparation, Euler sampling, VAE decode) on top of the official
Flux3Transformer2DModel source file, and dumps every intermediate tensor as
raw little-endian float32 so C test harnesses can compare against them.

The Flux3 transformer is not in a diffusers release yet: its source is
fetched from a pinned diffusers commit and loaded against the installed
diffusers (0.36 in flux_env is enough with two small shims).

Usage:
    ./flux_env/bin/python debug/flux3_reference.py -d flux-3-klein \\
        -p "a fish" -W 256 -H 256 --steps 4 --seed 42 -o /tmp/flux3_ref

Outputs in the -o directory:
    tokens.txt            token ids (padded window), one per line
    emb.bin               text embeddings [txt_seq, 20480]
    noise.bin             initial noise [128, H/16, W/16]
    sigmas.txt            the sigma schedule (num_steps + 1 values)
    step<i>_v.bin         velocity predicted at step i [img_seq, 128]
    step<i>_x.bin         latent tokens after step i [img_seq, 128]
    latent.bin            final latent [128, H/16, W/16] (before BN denorm)
    image.png             decoded image

Pass --noise FILE to start from a noise file (e.g. one dumped by C).
--ref IMAGE (repeatable) conditions on reference images, prepared like the
official Flux3ImageProcessor; ref<i>.bin holds their normalized latents.
"""

import argparse
import json
import math
import os
import sys
import types
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

DIFFUSERS_COMMIT = "0c7d56d6f4ea28c684d563cfae96fa375e0a6d4c"
FLUX3_SRC_URL = ("https://raw.githubusercontent.com/huggingface/diffusers/"
                 f"{DIFFUSERS_COMMIT}/src/diffusers/models/transformers/transformer_flux3.py")


def load_flux3_class(cache_path):
    """Return Flux3Transformer2DModel, from diffusers if available, else
    from the pinned upstream source adapted to the installed diffusers."""
    try:
        from diffusers.models import Flux3Transformer2DModel
        return Flux3Transformer2DModel
    except ImportError:
        pass
    if not os.path.exists(cache_path):
        print(f"Fetching {FLUX3_SRC_URL}")
        urllib.request.urlretrieve(FLUX3_SRC_URL, cache_path)
    src = open(cache_path).read()
    src = src.replace("from ...utils import apply_lora_scale, logging",
                      "from diffusers.utils import logging")
    src = src.replace("from ...utils.torch_utils import maybe_adjust_dtype_for_device", "")
    src = src.replace("from ...", "from diffusers.").replace("from ..", "from diffusers.models.")
    shims = (
        "def apply_lora_scale(_name):\n"
        "    return lambda fn: fn\n"
        "def maybe_adjust_dtype_for_device(dtype, device):\n"
        "    return torch.float32 if device.type == 'mps' and dtype == torch.float64 else dtype\n"
    )
    src = src.replace("logger = logging.get_logger", shims + "logger = logging.get_logger", 1)
    mod = types.ModuleType("transformer_flux3")
    exec(compile(src, cache_path, "exec"), mod.__dict__)
    return mod.Flux3Transformer2DModel


def dump(path, t):
    t.detach().to(torch.float32).cpu().contiguous().numpy().tofile(path)


def encode_prompt(te_dir, prompt, dtype, pad_multiple=80, max_len=8192,
                  layers=(4, 8, 12, 16, 20, 24, 28, 32)):
    """Same as Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds."""
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(te_dir)
    template = json.load(open(os.path.join(te_dir, "chat_template.json")))["chat_template"]
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], chat_template=template,
                                   tokenize=False, add_generation_prompt=True)
    real_len = tok(text, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].shape[1]
    padded_len = min(math.ceil(real_len / pad_multiple) * pad_multiple, max_len)
    inputs = tok(text, return_tensors="pt", padding="max_length", padding_side="right",
                 truncation=True, max_length=padded_len)
    input_ids, attn_mask = inputs["input_ids"], inputs["attention_mask"]
    position_ids = torch.arange(padded_len)[None].masked_fill(attn_mask == 0, 1)

    model = Qwen3VLForConditionalGeneration.from_pretrained(te_dir, dtype=dtype)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[k] for k in layers], dim=1)  # [1, L, seq, hidden]
    b, nl, seq, hid = hs.shape
    emb = hs.permute(0, 2, 1, 3).reshape(b, seq, nl * hid)
    del model
    return input_ids[0].tolist(), real_len, emb


def image_ids(h, w, rope_scale=0.5, l_coord=0):
    ids = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w),
                               torch.tensor([l_coord])).to(torch.float32)
    ids[:, 1:3] *= rope_scale
    return ids


def text_ids(seq):
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                torch.arange(seq)).to(torch.float32)


def prepare_references(paths):
    """Official preprocessing: one reference is downscaled so that its canvas
    (size rounded up to 16) fits under 4 MP and reflect-padded to it, several
    are downscaled under 1 MP and center-cropped to multiples of 16. Returns
    tensors in [-1, 1] and the crop box of a single padded reference."""
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    canvas = lambda side: min(4096, max(64, -(-side // 16) * 16))
    out, crop = [], None
    for img in images:
        w, h = img.size
        if len(images) == 1:
            scale = min(1.0, math.sqrt(4 * 1024 * 1024 / (w * h)), 4096 / max(w, h))
            while True:
                nh, nw = (max(1, round(side * scale)) for side in (h, w))
                if canvas(nh) * canvas(nw) <= 4 * 1024 * 1024:
                    break
                scale *= 0.995
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), Image.LANCZOS)
            ch, cw = canvas(nh), canvas(nw)
            left, top = (cw - nw) // 2, (ch - nh) // 2
            mode = "reflect" if min(nw, nh) > 16 else "symmetric"
            arr = np.pad(np.asarray(img), ((top, ch - nh - top), (left, cw - nw - left), (0, 0)), mode=mode)
            crop = (left, top, left + nw, top + nh)
        else:
            if w * h > 1024 * 1024:
                scale = math.sqrt(1024 * 1024 / (w * h))
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            nw, nh = img.size
            cw, ch = nw - nw % 16, nh - nh % 16
            left, top = (nw - cw) // 2, (nh - ch) // 2
            arr = np.asarray(img)[top:top + ch, left:left + cw]
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
        out.append(t)
    return out, crop


def sigma_schedule(num_steps, shift=6.93):
    s = np.linspace(1.0, 1.0 / num_steps, num_steps)
    s = shift * s / (1 + (shift - 1) * s)
    return np.append(s, 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--model-dir", required=True)
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-W", "--width", type=int, default=None, help="default: 256, or the reference size")
    ap.add_argument("-H", "--height", type=int, default=None)
    ap.add_argument("--ref", action="append", default=[], help="reference image (repeatable)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-g", "--guidance", type=float, default=3.5, help="CFG scale (base model only)")
    ap.add_argument("--noise", default=None, help="raw float32 noise file [128, H/16, W/16]")
    ap.add_argument("--bf16", action="store_true", help="run in bfloat16 (default float32)")
    ap.add_argument("--text-only", action="store_true", help="stop after the text embeddings")
    ap.add_argument("-o", "--out-dir", default="/tmp/flux3_ref")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    torch.set_grad_enabled(False)

    # 1. Text
    ids, real_len, emb = encode_prompt(os.path.join(args.model_dir, "text_encoder"), args.prompt, dtype)
    with open(out("tokens.txt"), "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")
    dump(out("emb.bin"), emb[0])
    print(f"text: {real_len} real tokens, window {len(ids)}, emb {tuple(emb.shape)}")
    uncond = None
    tf_cfg = json.load(open(os.path.join(args.model_dir, "transformer", "config.json")))
    flow_map = tf_cfg.get("flow_map", False)
    if not flow_map:
        _, _, uncond = encode_prompt(os.path.join(args.model_dir, "text_encoder"), "", dtype)
        dump(out("emb_uncond.bin"), uncond[0])
    if args.text_only:
        return

    # 2. Reference images: VAE encode, patchify, batch norm, ids
    from diffusers.models import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(args.model_dir, "vae"), torch_dtype=torch.float32)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    cond_tokens, cond_ids, crop = [], [], None
    size_given = args.width is not None
    if args.ref:
        ref_images, crop = prepare_references(args.ref)
        for i, img in enumerate(ref_images):
            lat = vae.encode(img).latent_dist.mode()
            b, c, lh2, lw2 = lat.shape
            lat = lat.view(b, c, lh2 // 2, 2, lw2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, lh2 // 2, lw2 // 2)
            lat = (lat - bn_mean) / bn_std
            dump(out(f"ref{i}.bin"), lat[0])
            cond_tokens.append(lat.flatten(2).transpose(1, 2))
            cond_ids.append(image_ids(lat.shape[2], lat.shape[3], l_coord=-(1 + i)))
            if i == 0 and not size_given:
                args.height, args.width = lat.shape[2] * 16, lat.shape[3] * 16
        print(f"references: {[tuple(t.shape) for t in cond_tokens]}")
    if size_given:
        crop = None
    args.width = args.width or 256
    args.height = args.height or 256

    # 3. Latents and ids
    h, w = args.height // 16, args.width // 16
    if args.noise:
        noise = torch.from_numpy(np.fromfile(args.noise, dtype=np.float32)).view(1, 128, h, w)
    else:
        noise = torch.randn((1, 128, h, w), generator=torch.Generator().manual_seed(args.seed),
                            dtype=torch.float32)
    dump(out("noise.bin"), noise[0])
    x = noise.flatten(2).transpose(1, 2).to(dtype)  # [1, img_seq, 128]
    img_ids = image_ids(h, w)

    steps = args.steps or (16 if flow_map else 50)
    sigmas = sigma_schedule(steps)
    np.savetxt(out("sigmas.txt"), sigmas, fmt="%.8f")

    # 4. Denoise
    cls = load_flux3_class(os.path.join(args.out_dir, "transformer_flux3.py"))
    model = cls.from_pretrained(os.path.join(args.model_dir, "transformer"), torch_dtype=dtype).eval()
    for i in range(steps):
        t = torch.tensor([sigmas[i]], dtype=dtype)
        tn = torch.tensor([sigmas[i + 1]], dtype=dtype)
        kw = dict(hidden_states=x, timestep=t, img_ids=img_ids, return_dict=False)
        if cond_tokens:
            kw["image_cond_hidden_states"] = torch.cat(cond_tokens, dim=1).to(dtype)
            kw["image_cond_ids"] = torch.cat(cond_ids, dim=0)
        if flow_map:
            v = model(encoder_hidden_states=emb.to(dtype), txt_ids=text_ids(emb.shape[1]),
                      target_timestep=tn, **kw)[0]
        else:
            vc = model(encoder_hidden_states=emb.to(dtype), txt_ids=text_ids(emb.shape[1]), **kw)[0]
            vu = model(encoder_hidden_states=uncond.to(dtype), txt_ids=text_ids(uncond.shape[1]), **kw)[0]
            v = vu + args.guidance * (vc - vu)
        x = (x.float() + (sigmas[i + 1] - sigmas[i]) * v.float()).to(dtype)
        dump(out(f"step{i}_v.bin"), v[0])
        dump(out(f"step{i}_x.bin"), x[0])
        print(f"step {i + 1}/{steps} sigma {sigmas[i]:.5f} -> {sigmas[i + 1]:.5f}")
    del model

    # 5. Decode (a single padded reference: crop back to the image)
    lat = x.float().transpose(1, 2).reshape(1, 128, h, w)
    dump(out("latent.bin"), lat[0])
    lat = lat * bn_std + bn_mean
    lat = lat.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, h * 2, w * 2)
    img = vae.decode(lat, return_dict=False)[0]
    img = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if crop is not None:
        left, top, right, bottom = crop
        img = img[top:bottom, left:right]
    from PIL import Image
    Image.fromarray(img).save(out("image.png"))
    print(f"wrote {out('image.png')}")


if __name__ == "__main__":
    main()
