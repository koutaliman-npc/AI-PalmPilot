"""
Export the AI PalmPilot embedding model to ONNX int8.

The PalmPilot embedder is deliberately network-inert: it never downloads
anything at runtime. This module is the supported way to produce the two
files it expects inside $MEMPALM_PATH/models/:

    all-MiniLM-L6-v2_int8.onnx
    tokenizer.json

Usage
-----
    pip install "agent-palm[export]"
    python -m agent_palm.export_model

    # or write somewhere else
    python -m agent_palm.export_model --output ./models

Requires the optional `export` extra (sentence-transformers + optimum).
Everything here runs locally - no data leaves the machine.
"""

import argparse
import os
import shutil
import sys

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILENAME = "all-MiniLM-L6-v2_int8.onnx"
TOKENIZER_FILENAME = "tokenizer.json"


def default_output_dir() -> str:
    """$MEMPALM_PATH/models, else ~/.mempalm/models."""
    mempalm_path = os.environ.get("MEMPALM_PATH")
    if mempalm_path:
        return os.path.join(mempalm_path, "models")
    return os.path.join(os.path.expanduser("~/.mempalm"), "models")


def export_model(output_dir: str, model_name: str = MODEL_NAME) -> str:
    """
    Export the sentence-transformer to ONNX int8 and copy its tokenizer.

    Returns the output directory containing the exported artefacts.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "The export extra is required. Install it with:\n"
            '    pip install "agent-palm[export]"\n'
            "or: pip install sentence-transformers optimum onnx"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"[export] loading {model_name} (this downloads once, then caches)...")
    model = SentenceTransformer(model_name)

    print(f"[export] exporting ONNX int8 to {output_dir}...")
    # sentence-transformers>=2.7 can export directly to ONNX with int8
    # quantisation via the `backend="onnx"` argument.
    model.save_pretrained(output_dir, backend="onnx")

    # The exported ONNX file may be named after the model folder rather than
    # the canonical name the embedder expects - normalise it.
    candidates = [
        os.path.join(output_dir, ONNX_FILENAME),
        os.path.join(output_dir, "model.onnx"),
        os.path.join(output_dir, "onnx", "model.onnx"),
    ]
    onnx_path = next((c for c in candidates if os.path.exists(c)), None)
    if onnx_path is None:
        # Fall back to searching the output tree.
        for root, _dirs, files in os.walk(output_dir):
            for fn in files:
                if fn.endswith(".onnx"):
                    onnx_path = os.path.join(root, fn)
                    break
            if onnx_path:
                break
    if onnx_path is None:
        raise SystemExit(
            f"[export] no .onnx file produced in {output_dir}. "
            "The installed sentence-transformers/optimum version may not support "
            "ONNX export - try: pip install -U sentence-transformers optimum onnx"
        )

    canonical_onnx = os.path.join(output_dir, ONNX_FILENAME)
    if os.path.abspath(onnx_path) != os.path.abspath(canonical_onnx):
        shutil.copy2(onnx_path, canonical_onnx)
        print(f"[export] copied {os.path.basename(onnx_path)} -> {ONNX_FILENAME}")

    # Tokenizer: the exported directory usually contains tokenizer.json already.
    tokenizer_src = os.path.join(output_dir, TOKENIZER_FILENAME)
    if not os.path.exists(tokenizer_src):
        # Try the hub cache via the model's own tokenizer files.
        try:
            model.save_pretrained(output_dir)
        except Exception:
            pass
    if not os.path.exists(tokenizer_src):
        raise SystemExit(
            f"[export] tokenizer.json not found in {output_dir}. "
            "Download it manually from "
            f"https://huggingface.co/{MODEL_NAME}/resolve/main/tokenizer.json"
        )

    size_mb = os.path.getsize(canonical_onnx) / (1024 * 1024)
    print("[export] done:")
    print(f"           {canonical_onnx}  ({size_mb:.1f} MB)")
    print(f"           {tokenizer_src}")
    print(f"\nPoint PalmPilot at it with: export MEMPALM_PATH={os.path.dirname(output_dir)}")
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agent_palm.export_model",
        description="Export the PalmPilot embedding model to ONNX int8 (local, one-off download).",
    )
    parser.add_argument(
        "--output",
        default=default_output_dir(),
        help="Output directory (default: $MEMPALM_PATH/models or ~/.mempalm/models)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"HuggingFace model id to export (default: {MODEL_NAME})",
    )
    args = parser.parse_args(argv)
    export_model(args.output, args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
