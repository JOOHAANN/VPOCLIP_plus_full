"""Active multi-view reinforcement-learning extension for VPOCLIP.

This package is intentionally isolated from the existing recognition training
code.  VPOCLIP remains the frozen 55-class recognizer; the RL policy only
chooses the next camera destination from a discrete set of views.
"""

__all__ = ["__version__"]

__version__ = "0.1.0-scaffold"


# Implementation guide
# 1. Keep this module free of heavy imports so CLI startup and tests stay cheap.
# 2. Export only stable public contracts after the individual modules work.
# 3. Do not import or mutate the existing VPOCLIP training pipeline here.
