# High-resolution reference conditioning fix

## Problem and evidence

The 1920x1280 img2img run reported that its reference was reduced to 16x16.
`fit_refs_for_attention()` assumes every backend materializes a float32
`[heads, tokens, tokens]` score matrix and repeatedly shrinks references to
fit a 4 GiB allocation limit. The output alone can exceed that estimate, so
shrinking references cannot solve the problem. The loop nevertheless runs
to its minimum reference size and continues generation.

The Metal backend already prefers native MPSGraph scaled dot-product
attention for long sequences. Its memory behavior differs from the manual
score-matrix fallback. Raising the allocation budget unconditionally would
make that fallback unsafe.

## Implementation plan

1. Expose whether the active Metal backend can use native memory-efficient
   attention, respecting `IRIS_USE_MANUAL_SDPA`.
2. Keep original reference dimensions (aligned to 16) in single-reference
   generation instead of resizing the reference to the output dimensions.
3. Skip the quadratic reference-shrinking estimate only for native Metal
   attention. Preserve the conservative policy for other paths, but return
   a clear error when output plus minimum references cannot fit.
4. Guard the manual Metal graph itself against oversized score matrices so
   native-attention failure cannot silently select an unsafe fallback. Defer
   the host score-buffer reservation until fallback is actually needed, reject
   oversized manual fallbacks, and propagate failures through all four
   reference samplers without dereferencing a null prediction.
5. Make the VAE dimension ceiling a documented build option, defaulting to
   the existing 1792. Build with 1920 for the requested 1920x1280 test.
6. Add regression coverage for impossible budgets and retained reference
   dimensions. Run `make test`, compile the generic and BLAS backends, and
   generate the user's reference at 1920x1280 using Metal. Inspect the
   result; successful execution alone does not establish faithful upscaling.
7. Record validation and limitations here and commit only related changes.

## Acceptance criteria

- Impossible conservative budgets fail before reference encoding instead
  of continuing with a 16x16 reference.
- Native Metal img2img retains the 1536x1024 reference for 1920x1280 output.
- Manual attention never requests a score tensor at or above 4 GiB.
- Existing small-image inference tests remain within their tolerances.
- No new inference dependencies, unrelated artifacts, or model files enter
  the commit.

## Implementation and validation (2026-10-20)

- Implemented native SDPA capability detection, conservative-budget rejection,
  independent reference resolution, manual graph/fallback guards, sampler error
  propagation, and the configurable VAE ceiling. The default remains 1792.
- `make mps MAX_IMAGE_DIM=1920` succeeds. Both `iris` and the local compatibility
  copy `iris-1920` contain the fix.
- `make test`: **7 passed, 0 failed**, using Metal on an M1 Max, 24 GPU cores,
  64 GB unified memory. Existing image-reference mean differences: 4.91
  (64x64), 3.83 (512x512), 7.84 (256x256 img2img), all below 20. The added
  forced-manual numerical check has mean difference 4.97.
- Added runner coverage for retaining a 256x256 reference with 64x64 output,
  impossible single/multiple-reference budgets, and manual-attention parity.
  The older full-suite shrinking test now explicitly forces manual attention.
- Generic and BLAS backends compile in isolated temporary build directories;
  both pass single- and multiple-reference budget-rejection checks. They emit
  the existing unused `zi_time_ms` warning in unchanged Z-Image code.
- A throwaway AddressSanitizer harness injects failed transformer predictions
  into each of the four reference samplers, including either CFG prediction;
  failure propagation and successful sampling checks pass.
- Four-step native img2img at 1920x1280 retained the user's original 1536x1024
  reference with no attention-driven reduction. End-to-end wall time: 250.77 s;
  denoising: 217.96 s; VAE decoding: 23.8 s. Visual inspection confirms the
  original man/dog/living-room scene and illustrated style are retained, with
  some fine-detail changes. The local image and log are in
  `workdir/scene4_5-upscaled-fixed-1920x1280.{png,log}` and are not committed.
- The final build also passed a one-step 1920x1280 smoke check after fallback
  hardening and the host score-buffer change, retaining the 1536x1024
  reference. Total reported generation time: 88.0 s. The seven-test inference
  suite and both CPU backend rejection checks passed again after those changes.

The test runner's existing NumPy/Pillow dependencies were installed in an
isolated `/tmp/iris-fix-test-env`; no inference dependencies were added. GPU
tests ran outside the sandbox because it cannot initialize Metal. Optional
Z-Image tests were skipped because its weights are not installed.

## Reproduction

```bash
make mps MAX_IMAGE_DIM=1920
./iris -d flux-klein-4b -i workdir/scene4_5.png \
  -W 1920 -H 1280 -s 4 -S 42 \
  -p "Create an exact copy of the input image." \
  -o workdir/scene4_5-upscaled-fixed-1920x1280.png
```

Native API availability is not a guarantee of enough total system memory.
The original artwork may still change because this is generative reference
conditioning, not a dedicated super-resolution model. Older/manual attention
paths retain conservative shrinking for requests that can fit, and reject
requests that cannot. There is no fidelity guarantee for arbitrary inputs,
and larger VAE ceilings than 1920 have not been tested here.
