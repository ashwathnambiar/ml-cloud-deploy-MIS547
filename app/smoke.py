# app/smoke.py
import os
import json
import pandas as pd
from edcpvsw3 import create_features

def run_smoke():
    # use tiny sample file for CI-only smoke test
    sample_path = os.path.join("tests","data","sample_small.json")

    with open(sample_path, "r") as f:
        sample = json.load(f)

    df = pd.DataFrame(sample)
    df_feat = create_features(df)
    assert df_feat is not None
    return df_feat

if __name__ == "__main__":
    df = run_smoke()
    print("smoke ok", df.shape)
