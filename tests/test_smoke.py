from app.smoke import run_smoke

def test_smoke_run():
    df = run_smoke()
    assert df is not None
    assert df.shape[0] >= 1
    assert 'community_participation' in df.columns
