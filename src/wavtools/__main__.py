"""Entry point for ``python -m wavtools``."""

from __future__ import annotations

import sys


def main() -> None:
    print("wavtools - audio utilities for hardware samplers")
    print()
    print("Available device modules (use python -m wavtools.<device>):")
    print("  wavtools.morphagene   - Make Noise Morphagene reel builder")
    print("  wavtools.octatrack    - Elektron Octatrack chain builder")
    print("  wavtools.play         - 2hp Play SD card prep")
    print("  wavtools.sample_drum  - Sample Drum folder prep")
    sys.exit(0)


main()
