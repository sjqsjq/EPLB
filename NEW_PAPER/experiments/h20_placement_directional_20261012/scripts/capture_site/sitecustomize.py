"""Install the opt-in H20 placement capture patch in every SGLang worker."""

import os

if os.environ.get("H20_PLACEMENT_CAPTURE_TRIGGER"):
    from placement_capture_patch import install

    install()
