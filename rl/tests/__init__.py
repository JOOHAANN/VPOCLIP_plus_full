"""Acceptance-test scaffolds for the active-view RL package."""


# Implementation guide
# 1. Keep tests deterministic and independent of V100 availability when possible.
# 2. Put expensive checkpoint/cache integration tests behind an explicit marker.
# 3. Remove skip decorators one contract at a time as modules are implemented.
