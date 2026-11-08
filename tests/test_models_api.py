import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from exofinder import service  # noqa: E402
from exofinder.api.main import app  # noqa: E402
from exofinder.data import labels  # noqa: E402
from exofinder.models import cnn, train  # noqa: E402


def test_cnn_forward_shape():
    m = cnn.ExoNet()
    out = m(torch.randn(4, 2001), torch.randn(4, 201), torch.randn(4, 9))
    assert out.shape == (4,)


def test_group_split_keeps_stars_together():
    ids = np.array([f"K{i // 3}" for i in range(90)])
    y = np.array([(i // 3) % 2 for i in range(90)])
    tr, va, te = train.group_split(ids, y)
    groups = [set(ids[i]) for i in (tr, va, te)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(tr) + len(va) + len(te) == 90


def test_parse_table_variants():
    t, f, e = service.parse_table(b"time,flux\n1,1.0\n2,0.9\n3,1.1\n")
    assert list(t) == [1, 2, 3] and e is None
    t, f, e = service.parse_table(b"1 1.0 0.01\n2 0.9 0.01\n3 1.1 0.01\n")
    assert list(f) == [1.0, 0.9, 1.1] and e is not None
    with pytest.raises(service.InputError):
        service.parse_table(b"a,b\nx,y\n")


def test_standardize_koi():
    raw = pd.DataFrame({
        "kepid": [1, 2, 3], "kepoi_name": ["a", "b", "c"],
        "koi_disposition": ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"],
        "koi_period": [3.0, 5.0, 7.0], "koi_time0bk": [1.0, 2.0, 3.0],
        "koi_duration": [2.4, 4.8, 1.2], "koi_depth": [100, 200, 300],
    })
    out = labels.standardize_koi(raw)
    assert list(out["label"]) == [1, 0, -1]
    assert out["duration_days"].iloc[0] == pytest.approx(0.1)


def test_api_without_models_is_graceful(tmp_path, monkeypatch):
    from exofinder.predictor import Predictor

    with TestClient(app) as client:
        app.state.predictor = Predictor(tmp_path / "x.pt", tmp_path / "y.joblib", tmp_path / "m.json")
        assert client.get("/api/health").json()["models"] == []
        assert client.get("/api/demo?kind=planet&seed=1").status_code == 503
        r = client.post("/api/analyze/upload", files={"file": ("bad.csv", b"a,b\nx,y\n")})
        assert r.status_code in (422, 503)
        assert client.get("/").status_code == 200
