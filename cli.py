"""v1 entry point kept for compatibility: ``python cli.py --scan_dir DIR`` == ``autonas mine DIR``."""

import sys

from autonas.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--scan_dir" in args:
        i = args.index("--scan_dir")
        args = [args[i + 1], *args[:i], *args[i + 2 :]]
    if "--output_code" not in args:
        args += ["--output_code", "model_definition.py"]
    raise SystemExit(main(["mine", *args]))
