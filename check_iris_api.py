#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
# ABOUTME: Verifies iris.h API surface hasn't changed from what iris_ffi.py expects
# ABOUTME: Extracts struct/function definitions from iris.h and diffs against stored snapshot

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IRIS_H = SCRIPT_DIR / "iris.c" / "iris.h"
SNAPSHOT = SCRIPT_DIR / "iris_api.snapshot"

# Functions from iris.h that iris_ffi.py depends on
FUNCTIONS = [
    "iris_load_dir",
    "iris_free",
    "iris_image_load",
    "iris_image_save",
    "iris_image_save_with_seed",
    "iris_image_free",
    "iris_img2img",
    "iris_multiref",
    "iris_get_error",
]

# Named blocks to extract (start marker matched against line content)
BLOCK_MARKERS = [
    "struct iris_image {",
    "typedef struct {",       # iris_params
    "enum {",                 # schedule constants (first enum in file)
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


def extract_api(header_text):
    """Extract the API surface iris_ffi.py depends on from iris.h."""
    lines = header_text.split("\n")
    sections = []

    # Extract named blocks (structs, enums)
    for marker in BLOCK_MARKERS:
        for i, line in enumerate(lines):
            if marker in line:
                sections.append(extract_block(lines, i))
                break

    # Extract function declarations
    for func in FUNCTIONS:
        decl = extract_function(lines, func)
        if decl:
            sections.append(decl)

    return "\n\n".join(sections) + "\n"


def main():
    if not IRIS_H.exists():
        print(f"iris.h not found at {IRIS_H}")
        return 1

    current = extract_api(IRIS_H.read_text())

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
