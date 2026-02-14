# Local iris.c Patches

This project uses a [fork of iris.c](https://github.com/tednaleid/iris.c) that carries patches on top of [antirez/iris.c](https://github.com/antirez/iris.c). If upstream adds equivalent functionality, these patches can be dropped.

## Fork workflow

The justfile clones from the fork (`tednaleid/iris.c`). Users just run `just iris-update` and get the patches automatically.

### For users

```bash
just iris-update    # pull latest from fork, check API, rebuild, link dylib
```

### For the fork maintainer

The fork's `main` branch is kept rebased on top of upstream. To pick up new upstream changes:

```bash
cd iris.c
git fetch prime                     # prime = antirez/iris.c
git rebase prime/main               # replay patches on top of upstream
# resolve any conflicts
git push --force-with-lease origin main
```

The `prime` remote needs to be set up once:

```bash
cd iris.c
git remote add prime git@github.com:antirez/iris.c.git
```

## Patches

### `iris_img2img_precomputed()` — single-ref precomputed generation

**Files:** `iris.c`, `iris.h`

Accepts pre-computed text embeddings and a single pre-encoded image latent, skipping text encoding and VAE encoding. Only runs sampling and VAE decode.

```c
iris_image *iris_img2img_precomputed(iris_ctx *ctx,
                                      const float *text_emb, int text_seq,
                                      const float *text_emb_uncond, int text_seq_uncond,
                                      const float *img_latent, int latent_h, int latent_w,
                                      const iris_params *params);
```

**Why:** When generating multiple images with the same prompt and input but different seeds (`--count N`), text encoding (~1-5s) and image VAE encoding (~0.2-2s) are redundant after the first image. This function accepts the pre-encoded results so they can be computed once and reused.

**Implementation:** Extracted from the tail of `iris_img2img()` — loads transformer, initializes noise, calls the appropriate sampler (distilled or CFG), decodes with VAE. Does not free the input pointers (caller owns them for reuse).

### `iris_multiref_precomputed()` — multi-ref precomputed generation

**Files:** `iris.c`, `iris.h`

Same as above but for multiple reference images. Takes parallel arrays of pre-encoded latents.

```c
iris_image *iris_multiref_precomputed(iris_ctx *ctx,
                                       const float *text_emb, int text_seq,
                                       const float *text_emb_uncond, int text_seq_uncond,
                                       const float **ref_latents, const int *ref_hs,
                                       const int *ref_ws, int num_refs,
                                       const iris_params *params);
```

**Why:** Extends the precomputed optimization to multi-reference generation (when multiple `-i` flags are used). All reference images can be VAE-encoded once and reused across seeds.

**Implementation:** Constructs internal `iris_ref_t` array from the parallel arrays with `t_offset = 10*(i+1)`, then calls the multiref sampler. Dispatches to `iris_img2img_precomputed` for single-ref case.

## Existing public API used

These functions already exist upstream and are used by the precomputed path:

- `iris_encode_text()` — pre-encode text to embeddings
- `iris_release_text_encoder()` — free Qwen3 after encoding
- `iris_encode_image()` — pre-encode image to latent space
- `iris_is_distilled()` — check if empty prompt encoding is needed for CFG

## Rebuilding after upstream pull

If a rebase against upstream conflicts (upstream modified the same area), resolve manually in `iris.c/iris.c` and `iris.c/iris.h`, then:

```bash
cd iris.c && git rebase --continue
just iris-check     # verify API snapshot still matches
just iris-build     # rebuild
just iris-dylib     # relink shared library
```
