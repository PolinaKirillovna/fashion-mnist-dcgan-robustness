"""Enable ``python -m gan_robustness``."""

from __future__ import annotations

import sys

from gan_robustness.cli import main

if __name__ == "__main__":
    sys.exit(main())
