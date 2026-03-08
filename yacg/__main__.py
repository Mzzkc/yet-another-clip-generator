"""Allow running as ``python -m yacg``."""
import sys

from yacg.cli import main

sys.exit(main())
