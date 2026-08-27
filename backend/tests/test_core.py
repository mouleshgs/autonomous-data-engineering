import pandas as pd
from app.core import plan_for, profile_frame, quality

def test_profile_and_adaptive_plan():
    frame = pd.DataFrame({"price": [10, None, 10], "region": [" india ", "US", "US"]})
    profile = profile_frame(frame)
    assert profile["missing_values"] == 1
    assert profile["duplicates"] == 0
    assert {step["operation"] for step in plan_for(profile)} >= {"fill_missing", "normalize_strings", "validate"}

def test_quality_improves_after_cleaning():
    before = quality(pd.DataFrame({"value": [1, None, 1]}))
    after = quality(pd.DataFrame({"value": [1, 2]}))
    assert after["overall"] > before["overall"]
