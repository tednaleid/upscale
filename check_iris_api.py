#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
# ABOUTME: Verifies iris.h API surface hasn't changed from what iris_ffi.py expects
# ABOUTME: Extracts struct/function definitions from iris.h and diffs against stored snapshot

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IRIS_DIR = SCRIPT_DIR / "iris.c"
SNAPSHOT = SCRIPT_DIR / "iris_api.snapshot"

# --- iris.h: functions and types iris_ffi.py depends on ---

IRIS_H_FUNCTIONS = [
    "iris_load_dir",
    "iris_free",
    "iris_set_mmap",
    "iris_image_load",
    "iris_image_save",
    "iris_image_save_with_seed",
    "iris_image_free",
    "iris_generate",
    "iris_img2img",
    "iris_multiref",
    "iris_get_error",
    "iris_set_step_image_callback",
    "iris_encode_text",
    "iris_release_text_encoder",
    "iris_encode_image",
    "iris_is_distilled",
    "iris_img2img_precomputed",
    "iris_multiref_precomputed",
]

IRIS_H_BLOCKS = [
    "struct iris_image {",
    "typedef struct {",       # iris_params
    "enum {",                 # schedule constants (first enum in file)
]

# --- iris_kernels.h: progress callback types and globals ---

KERNELS_H_BLOCKS = [
    "} iris_substep_type_t;",       # substep enum (match closing line)
]

# Lines to match exactly (substring match against each line)
KERNELS_H_LINES = [
    "(*iris_substep_callback_t)",   # substep callback typedef
    "(*iris_step_callback_t)",      # step callback typedef
    "(*iris_phase_callback_t)",     # phase callback typedef
    "extern iris_substep_callback_t iris_substep_callback;",
    "extern iris_step_callback_t iris_step_callback;",
    "extern iris_phase_callback_t iris_phase_callback;",
]

# --- terminals.h: terminal graphics display ---

TERMINALS_H_BLOCKS = [
    "} term_graphics_proto;",       # protocol enum
]

TERMINALS_H_FUNCTIONS = [
    "detect_terminal_graphics",
    "terminal_display_image",
    "terminal_display_png",
]


def extract_block(lines, start_idx):
    """Extract a brace-delimited block starting at start_idx, through closing ; or }."""
    result = []
    depth = 0
    for i in range(start_idx, len(lines)):
        line = lines[i]
        result.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and ";" in line:
            break
    return "\n".join(result)


def extract_function(lines, name):
    """Extract a function declaration (possibly multi-line) ending at semicolon."""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Skip comment lines
        if stripped.startswith("*") or stripped.startswith("/*") or stripped.startswith("//"):
            continue
        if name + "(" in line:
            result = []
            for j in range(i, len(lines)):
                result.append(lines[j])
                if ";" in lines[j]:
                    break
            return "\n".join(result)
    return None


def extract_typedef_line(lines, ending):
    """Extract a typedef/enum ending with the given text.

    If the matching line contains '}', walk back to find the opening typedef/enum block.
    Otherwise treat as a single-line declaration.
    """
    for i, line in enumerate(lines):
        if ending in line:
            if "}" not in line:
                # Single-line typedef
                return line
            # Multi-line block: walk back to find opening typedef/enum
            for j in range(i - 1, -1, -1):
                if "typedef" in lines[j] or "enum" in lines[j]:
                    return "\n".join(lines[j:i + 1])
            return line
    return None


def extract_api():
    """Extract the API surface iris_ffi.py depends on from iris.h and iris_kernels.h."""
    iris_h = IRIS_DIR / "iris.h"
    kernels_h = IRIS_DIR / "iris_kernels.h"

    sections = []

    # --- iris.h ---
    if not iris_h.exists():
        return None, f"iris.h not found at {iris_h}"

    h_lines = iris_h.read_text().split("\n")

    # Extract named blocks (structs, enums)
    for marker in IRIS_H_BLOCKS:
        for i, line in enumerate(h_lines):
            if marker in line:
                sections.append(extract_block(h_lines, i))
                break

    # Extract function declarations
    for func in IRIS_H_FUNCTIONS:
        decl = extract_function(h_lines, func)
        if decl:
            sections.append(decl)

    # --- iris_kernels.h (progress callbacks) ---
    if not kernels_h.exists():
        return None, f"iris_kernels.h not found at {kernels_h}"

    k_lines = kernels_h.read_text().split("\n")

    for ending in KERNELS_H_BLOCKS:
        block = extract_typedef_line(k_lines, ending)
        if block:
            sections.append(block)

    for marker in KERNELS_H_LINES:
        for line in k_lines:
            if marker in line:
                sections.append(line)
                break

    # --- terminals.h ---
    terminals_h = IRIS_DIR / "terminals.h"
    if not terminals_h.exists():
        return None, f"terminals.h not found at {terminals_h}"

    t_lines = terminals_h.read_text().split("\n")

    for ending in TERMINALS_H_BLOCKS:
        block = extract_typedef_line(t_lines, ending)
        if block:
            sections.append(block)

    for func in TERMINALS_H_FUNCTIONS:
        decl = extract_function(t_lines, func)
        if decl:
            sections.append(decl)

    return "\n\n".join(sections) + "\n", None


def main():
    current, err = extract_api()
    if err:
        print(err)
        return 1

    if "--update" in sys.argv:
        SNAPSHOT.write_text(current)
        print(f"Updated {SNAPSHOT}")
        return 0

    if not SNAPSHOT.exists():
        print(f"No snapshot found at {SNAPSHOT}")
        print(f"Run: python3 {sys.argv[0]} --update")
        return 1

    expected = SNAPSHOT.read_text()
    if current != expected:
        print("WARNING: iris.h API has changed since last snapshot!")
        print("Definitions used by iris_ffi.py may need updating.")
        print()
        import difflib
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="iris_api.snapshot (expected)",
            tofile="iris.h (current)",
        )
        sys.stdout.writelines(diff)
        print()
        print("After updating iris_ffi.py if needed, refresh the snapshot:")
        print(f"  python3 {sys.argv[0]} --update")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
