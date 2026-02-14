# ABOUTME: Ctypes wrapper for the iris C library (libiris.dylib)
# ABOUTME: Provides load-once/generate-many semantics for efficient multi-seed generation

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

# Locate the shared library relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_DYLIB_PATH = _SCRIPT_DIR / "iris.c" / "libiris.dylib"

# Schedule constants matching iris.h
IRIS_SCHEDULE_DEFAULT = 0
IRIS_SCHEDULE_LINEAR = 1
IRIS_SCHEDULE_POWER = 2

# Substep types matching iris_substep_type_t enum
_SUBSTEP_DOUBLE_BLOCK = 0
_SUBSTEP_SINGLE_BLOCK = 1
_SUBSTEP_FINAL_LAYER = 2

# Global callback ctypes function types matching iris_kernels.h typedefs
# void (*)(int step, int total)
_STEP_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
# void (*)(int type, int index, int total)
_SUBSTEP_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int)
# void (*)(const char *phase, int done)
_PHASE_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int)

# Terminal graphics protocol constants matching terminals.h
TERM_PROTO_NONE = 0
TERM_PROTO_KITTY = 1
TERM_PROTO_ITERM2 = 2


class IrisImage(ctypes.Structure):
    """Mirrors struct iris_image { int width, height, channels; uint8_t *data; }"""
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


# Step image callback type (needs IrisImage defined first)
# void (*)(int step, int total, const iris_image *img)
_STEP_IMAGE_CB_TYPE = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(IrisImage),
)


class IrisParams(ctypes.Structure):
    """Mirrors iris_params from iris.h with IRIS_PARAMS_DEFAULT values."""
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("seed", ctypes.c_int64),
        ("guidance", ctypes.c_float),
        ("schedule", ctypes.c_int),
        ("power_alpha", ctypes.c_float),
    ]

    @classmethod
    def default(cls):
        """Return params matching IRIS_PARAMS_DEFAULT."""
        return cls(
            width=256, height=256, num_steps=0, seed=-1,
            guidance=0.0, schedule=IRIS_SCHEDULE_DEFAULT, power_alpha=2.0,
        )


def _run_interruptible(func, *args):
    """Run func(*args) in a thread so the main thread can handle SIGINT."""
    result = [None]
    error = [None]

    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted, shutting down...", file=sys.stderr, flush=True)
        os._exit(130)

    if error[0]:
        raise error[0]
    return result[0]


_libc = ctypes.CDLL(None)  # libc for free()
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def _load_lib():
    """Load libiris.dylib and set up function signatures."""
    lib = ctypes.CDLL(str(_DYLIB_PATH))

    # iris_ctx *iris_load_dir(const char *model_dir)
    lib.iris_load_dir.argtypes = [ctypes.c_char_p]
    lib.iris_load_dir.restype = ctypes.c_void_p

    # void iris_free(iris_ctx *ctx)
    lib.iris_free.argtypes = [ctypes.c_void_p]
    lib.iris_free.restype = None

    # iris_image *iris_image_load(const char *path)
    lib.iris_image_load.argtypes = [ctypes.c_char_p]
    lib.iris_image_load.restype = ctypes.POINTER(IrisImage)

    # int iris_image_save(const iris_image *img, const char *path)
    lib.iris_image_save.argtypes = [ctypes.POINTER(IrisImage), ctypes.c_char_p]
    lib.iris_image_save.restype = ctypes.c_int

    # int iris_image_save_with_seed(const iris_image *img, const char *path, int64_t seed)
    lib.iris_image_save_with_seed.argtypes = [
        ctypes.POINTER(IrisImage), ctypes.c_char_p, ctypes.c_int64,
    ]
    lib.iris_image_save_with_seed.restype = ctypes.c_int

    # void iris_image_free(iris_image *img)
    lib.iris_image_free.argtypes = [ctypes.POINTER(IrisImage)]
    lib.iris_image_free.restype = None

    # iris_image *iris_generate(iris_ctx *, const char *, const iris_params *)
    lib.iris_generate.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(IrisParams),
    ]
    lib.iris_generate.restype = ctypes.POINTER(IrisImage)

    # iris_image *iris_img2img(iris_ctx *, const char *, const iris_image *, const iris_params *)
    lib.iris_img2img.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(IrisImage), ctypes.POINTER(IrisParams),
    ]
    lib.iris_img2img.restype = ctypes.POINTER(IrisImage)

    # iris_image *iris_multiref(iris_ctx *, const char *, const iris_image **, int, const iris_params *)
    lib.iris_multiref.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(IrisImage)), ctypes.c_int,
        ctypes.POINTER(IrisParams),
    ]
    lib.iris_multiref.restype = ctypes.POINTER(IrisImage)

    # const char *iris_get_error(void)
    lib.iris_get_error.argtypes = []
    lib.iris_get_error.restype = ctypes.c_char_p

    # int iris_metal_init(void) — from iris_metal.h, not public API but exported
    lib.iris_metal_init.argtypes = []
    lib.iris_metal_init.restype = ctypes.c_int

    # void iris_set_mmap(iris_ctx *ctx, int enable)
    lib.iris_set_mmap.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.iris_set_mmap.restype = None

    # void iris_set_step_image_callback(iris_ctx *ctx, iris_step_image_cb_t callback)
    lib.iris_set_step_image_callback.argtypes = [ctypes.c_void_p, _STEP_IMAGE_CB_TYPE]
    lib.iris_set_step_image_callback.restype = None

    # term_graphics_proto detect_terminal_graphics(void)
    lib.detect_terminal_graphics.argtypes = []
    lib.detect_terminal_graphics.restype = ctypes.c_int

    # int terminal_display_image(const iris_image *img, term_graphics_proto proto)
    lib.terminal_display_image.argtypes = [ctypes.POINTER(IrisImage), ctypes.c_int]
    lib.terminal_display_image.restype = ctypes.c_int

    # int terminal_display_png(const char *path, term_graphics_proto proto)
    lib.terminal_display_png.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.terminal_display_png.restype = ctypes.c_int

    # float *iris_encode_text(iris_ctx *ctx, const char *prompt, int *out_seq_len)
    lib.iris_encode_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.iris_encode_text.restype = ctypes.c_void_p

    # void iris_release_text_encoder(iris_ctx *ctx)
    lib.iris_release_text_encoder.argtypes = [ctypes.c_void_p]
    lib.iris_release_text_encoder.restype = None

    # float *iris_encode_image(iris_ctx *ctx, const iris_image *img, int *out_h, int *out_w)
    lib.iris_encode_image.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(IrisImage),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.iris_encode_image.restype = ctypes.c_void_p

    # int iris_is_distilled(iris_ctx *ctx)
    lib.iris_is_distilled.argtypes = [ctypes.c_void_p]
    lib.iris_is_distilled.restype = ctypes.c_int

    # iris_image *iris_img2img_precomputed(iris_ctx *, text_emb, text_seq,
    #     text_emb_uncond, text_seq_uncond, img_latent, latent_h, latent_w, params)
    lib.iris_img2img_precomputed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,                     # text_emb, text_seq
        ctypes.c_void_p, ctypes.c_int,                     # text_emb_uncond, text_seq_uncond
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,       # img_latent, latent_h, latent_w
        ctypes.POINTER(IrisParams),                         # params
    ]
    lib.iris_img2img_precomputed.restype = ctypes.POINTER(IrisImage)

    # iris_image *iris_multiref_precomputed(iris_ctx *, text_emb, text_seq,
    #     text_emb_uncond, text_seq_uncond, ref_latents, ref_hs, ref_ws, num_refs, params)
    lib.iris_multiref_precomputed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,                     # text_emb, text_seq
        ctypes.c_void_p, ctypes.c_int,                     # text_emb_uncond, text_seq_uncond
        ctypes.POINTER(ctypes.c_void_p),                    # ref_latents (float**)
        ctypes.POINTER(ctypes.c_int),                       # ref_hs
        ctypes.POINTER(ctypes.c_int),                       # ref_ws
        ctypes.c_int,                                       # num_refs
        ctypes.POINTER(IrisParams),                         # params
    ]
    lib.iris_multiref_precomputed.restype = ctypes.POINTER(IrisImage)

    return lib


def _make_progress_callbacks():
    """Create CLI-matching progress callbacks and return (step_cb, substep_cb, phase_cb).

    Output format matches iris CLI: phase messages, step/substep progress on stderr.
    The returned objects must be kept alive to prevent garbage collection.
    """
    current_step = [0]
    legend_printed = [False]
    phase_start = [0.0]

    @_STEP_CB_TYPE
    def step_cb(step, total):
        if not legend_printed[0]:
            print("Denoising (d=double block, s=single blocks, F=final):",
                  file=sys.stderr, flush=True)
            legend_printed[0] = True
        if current_step[0] > 0:
            print(file=sys.stderr)
        current_step[0] = step
        print(f"  Step {step}/{total} ", end="", file=sys.stderr, flush=True)

    @_SUBSTEP_CB_TYPE
    def substep_cb(stype, index, total):
        if stype == _SUBSTEP_DOUBLE_BLOCK:
            print("d", end="", file=sys.stderr, flush=True)
        elif stype == _SUBSTEP_SINGLE_BLOCK:
            if (index + 1) % 5 == 0:
                print("s", end="", file=sys.stderr, flush=True)
        elif stype == _SUBSTEP_FINAL_LAYER:
            print("F", end="", file=sys.stderr, flush=True)

    @_PHASE_CB_TYPE
    def phase_cb(phase_bytes, done):
        phase = phase_bytes.decode() if phase_bytes else ""
        if not done:
            if current_step[0] > 0:
                print(file=sys.stderr)
                current_step[0] = 0
            display = phase[0].upper() + phase[1:] if phase else ""
            print(f"{display}...", end="", file=sys.stderr, flush=True)
            phase_start[0] = time.monotonic()
        else:
            elapsed = time.monotonic() - phase_start[0]
            print(f" done ({elapsed:.1f}s)", file=sys.stderr, flush=True)

    return step_cb, substep_cb, phase_cb, current_step


class IrisContext:
    """Wrapper around iris_ctx providing load-once/generate-many semantics."""

    def __init__(self, model_dir):
        self._lib = _load_lib()
        self._callbacks = None  # prevent GC of active callbacks
        self._current_step = None
        self._lib.iris_metal_init()
        self._ctx = self._lib.iris_load_dir(str(model_dir).encode())
        if not self._ctx:
            self._raise_error(f"Failed to load model from {model_dir}")
        # Match CLI default: mmap text encoder weights (lower memory, reads from page cache)
        self._lib.iris_set_mmap(self._ctx, 1)

    def close(self):
        if self._ctx:
            self._clear_callbacks()
            self._lib.iris_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _raise_error(self, context):
        """Raise RuntimeError with the last iris error message."""
        err = self._lib.iris_get_error()
        msg = err.decode() if err else "unknown error"
        raise RuntimeError(f"{context}: {msg}")

    def _save_result(self, result, output_path, seed):
        """Save a generated image with seed metadata and free it."""
        save_ret = self._lib.iris_image_save_with_seed(
            result, str(output_path).encode(), seed,
        )
        if save_ret != 0:
            self._lib.iris_image_free(result)
            raise RuntimeError(f"Failed to save image to {output_path}")
        self._lib.iris_image_free(result)

    def _detect_graphics(self):
        """Detect terminal graphics protocol, caching the result."""
        if not hasattr(self, "_graphics_proto"):
            self._graphics_proto = self._lib.detect_terminal_graphics()
        return self._graphics_proto

    def enable_show_steps(self):
        """Enable display of intermediate images after each denoising step.

        Returns True if the terminal supports graphics, False otherwise (with warning).
        """
        proto = self._detect_graphics()
        if proto == TERM_PROTO_NONE:
            print("Warning: --show-steps requires a supported terminal "
                  "(Kitty, Ghostty, iTerm2, WezTerm, or Konsole)",
                  file=sys.stderr)
            return False

        @_STEP_IMAGE_CB_TYPE
        def _cb(step, total, img):
            print(f"\n[Step {step}]", file=sys.stderr, flush=True)
            self._lib.terminal_display_image(img, proto)

        self._step_image_cb = _cb  # prevent GC
        self._lib.iris_set_step_image_callback(self._ctx, self._step_image_cb)
        return True

    def display_png(self, path):
        """Display a PNG file in the terminal. Returns True on success."""
        proto = self._detect_graphics()
        if proto == TERM_PROTO_NONE:
            print("Warning: --show requires a supported terminal "
                  "(Kitty, Ghostty, iTerm2, WezTerm, or Konsole)",
                  file=sys.stderr)
            return False
        self._lib.terminal_display_png(str(path).encode(), proto)
        return True

    def _set_callbacks(self):
        """Install CLI-matching progress callbacks into the global function pointers."""
        step_cb, substep_cb, phase_cb, current_step = _make_progress_callbacks()
        # Keep references alive
        self._callbacks = (step_cb, substep_cb, phase_cb)
        self._current_step = current_step
        # Set global function pointer variables in the dylib
        step_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_step_callback")
        substep_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_substep_callback")
        phase_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_phase_callback")
        step_ptr.value = ctypes.cast(step_cb, ctypes.c_void_p).value
        substep_ptr.value = ctypes.cast(substep_cb, ctypes.c_void_p).value
        phase_ptr.value = ctypes.cast(phase_cb, ctypes.c_void_p).value

    def _clear_callbacks(self):
        """Remove progress callbacks (print final newline if mid-step)."""
        if self._current_step and self._current_step[0] > 0:
            print(file=sys.stderr)
        step_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_step_callback")
        substep_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_substep_callback")
        phase_ptr = ctypes.c_void_p.in_dll(self._lib, "iris_phase_callback")
        step_ptr.value = None
        substep_ptr.value = None
        phase_ptr.value = None
        self._callbacks = None
        self._current_step = None

    def multiref(self, prompt, input_paths, output_path, params):
        """Generate an image from multiple reference images.

        Args:
            prompt: Text prompt string.
            input_paths: List of Path objects for reference images.
            output_path: Path for the output image.
            params: IrisParams instance.
        """
        # Load all reference images
        ref_images = []
        for p in input_paths:
            img = self._lib.iris_image_load(str(p).encode())
            if not img:
                self._raise_error(f"Failed to load image {p}")
            ref_images.append(img)

        try:
            self._set_callbacks()

            if len(ref_images) == 0:
                result = _run_interruptible(
                    self._lib.iris_generate,
                    self._ctx, prompt.encode(), ctypes.byref(params),
                )
            elif len(ref_images) == 1:
                result = _run_interruptible(
                    self._lib.iris_img2img,
                    self._ctx, prompt.encode(), ref_images[0], ctypes.byref(params),
                )
            else:
                # Build array of iris_image pointers
                arr_type = ctypes.POINTER(IrisImage) * len(ref_images)
                arr = arr_type(*ref_images)
                result = _run_interruptible(
                    self._lib.iris_multiref,
                    self._ctx, prompt.encode(), arr, len(ref_images),
                    ctypes.byref(params),
                )

            self._clear_callbacks()

            if not result:
                self._raise_error("Generation failed")

            self._save_result(result, output_path, params.seed)
        finally:
            self._clear_callbacks()
            for img in ref_images:
                self._lib.iris_image_free(img)

    def is_distilled(self):
        """Return True if this is a distilled (4-step) model."""
        return bool(self._lib.iris_is_distilled(self._ctx))

    def free_encoded(self, ptr):
        """Free a pointer returned by encode_text() or encode_image()."""
        if ptr:
            _libc.free(ptr)

    def encode_text(self, prompt):
        """Encode text to embeddings. Returns (float_ptr, seq_len).

        The returned pointer must be freed with free_encoded() when done.
        """
        self._set_callbacks()
        try:
            seq_len = ctypes.c_int(0)
            emb = self._lib.iris_encode_text(self._ctx, prompt.encode(), ctypes.byref(seq_len))
        finally:
            self._clear_callbacks()
        if not emb:
            self._raise_error("Failed to encode text")
        return emb, seq_len.value

    def release_text_encoder(self):
        """Release the text encoder to free ~8GB of memory."""
        self._lib.iris_release_text_encoder(self._ctx)

    def encode_image(self, path):
        """Encode an image to latent space. Returns (float_ptr, latent_h, latent_w).

        The returned pointer must be freed with free_encoded() when done.
        """
        img = self._lib.iris_image_load(str(path).encode())
        if not img:
            self._raise_error(f"Failed to load image {path}")
        self._set_callbacks()
        try:
            t0 = time.monotonic()
            lat_h = ctypes.c_int(0)
            lat_w = ctypes.c_int(0)
            latent = self._lib.iris_encode_image(
                self._ctx, img, ctypes.byref(lat_h), ctypes.byref(lat_w),
            )
        finally:
            self._clear_callbacks()
        elapsed = time.monotonic() - t0
        name = Path(path).name
        print(f"Encoding {name}... done ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        self._lib.iris_image_free(img)
        if not latent:
            self._raise_error(f"Failed to encode image {path}")
        return latent, lat_h.value, lat_w.value

    def multiref_precomputed(self, text_emb, text_seq, text_emb_uncond, text_seq_uncond,
                              ref_latents, output_path, params):
        """Generate using pre-computed text embeddings and multiple image latents.

        For batch generation with same prompt/refs, different seeds.
        ref_latents: list of (float_ptr, h, w) tuples from encode_image().
        The caller owns all embedding and latent pointers (not freed here).
        """
        num_refs = len(ref_latents)
        lat_arr = (ctypes.c_void_p * num_refs)(*[r[0] for r in ref_latents])
        h_arr = (ctypes.c_int * num_refs)(*[r[1] for r in ref_latents])
        w_arr = (ctypes.c_int * num_refs)(*[r[2] for r in ref_latents])

        self._set_callbacks()
        try:
            result = _run_interruptible(
                self._lib.iris_multiref_precomputed,
                self._ctx,
                text_emb, text_seq,
                text_emb_uncond, text_seq_uncond,
                lat_arr, h_arr, w_arr, num_refs,
                ctypes.byref(params),
            )
            self._clear_callbacks()

            if not result:
                self._raise_error("Generation failed")

            self._save_result(result, output_path, params.seed)
        finally:
            self._clear_callbacks()

    @classmethod
    def available(cls):
        """Return True if the dylib exists."""
        return _DYLIB_PATH.exists()
