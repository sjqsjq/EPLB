"""Install prediction-study capture hooks in every SGLang worker."""

import os

if os.environ.get("H20_PREDICTION_CAPTURE_TRIGGER"):
    from placement_prediction_capture_patch import install

    install()
