"""Path setup shared by the whole test suite.

The entry points live in ``scripts/``, and the probes import them by module name -- they check
the functions those files define, not just their command-line behaviour. ``scripts/`` is a
directory of standalone programs rather than a package, so there is no import path to it; it has
to be on ``sys.path`` before any test module is imported, which is what a conftest guarantees and
a per-module insert does not.

Each test module still puts the repository root on the path itself. That covers the TransEHR2,
reporting and hp_tuning packages, which are importable the ordinary way.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))
