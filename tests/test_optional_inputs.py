import numpy as np
import pandas as pd
import torch

from geoindex_daily.ap3h import CME_FEATURES, CME_SLOTS, optional_aux
from geoindex_daily.models.image_fusion import ImageFusion
from geoindex_daily.models.optional_inputs import OptionalInputFusion

KW = dict(widths=(8, 8, 16, 16), ts_width=8, hidden=16, lasco_channels=1, out_dim=4, ts_pool="avg_last")


def _inputs(n=3):
    torch.manual_seed(1)
    return (torch.randint(0, 256, (n, 2, 4, 64, 64), dtype=torch.uint8), torch.randn(n, 8, 3),
            torch.randint(0, 256, (n, 1, 6, 64, 64), dtype=torch.uint8))


def test_base_weights_equal_model0_from_the_same_seed():
    torch.manual_seed(0); m0 = ImageFusion(2, 3, **KW)
    torch.manual_seed(0); mb = OptionalInputFusion(2, 3, fp_features=6, **KW)
    sb = mb.state_dict()
    for k, v in m0.state_dict().items():
        assert torch.equal(v, sb[k]), k


def test_empty_cme_slots_give_exactly_the_base_head():
    torch.manual_seed(0); m = OptionalInputFusion(2, 3, **KW).eval()
    for p in m.cme_res.parameters():                     # non-zero residual, as after training
        torch.nn.init.normal_(p)
    img, ts, las = _inputs()
    aux = torch.randn(3, CME_SLOTS * CME_FEATURES + CME_SLOTS)
    aux[:, CME_SLOTS * CME_FEATURES:] = 0                # every mask 0; slot values left non-zero on purpose
    with torch.no_grad():
        z = torch.cat([m.image(img), m.lasco(las), m.ts.features(ts)], 1)
        assert torch.equal(m(img, ts, las, aux), m.head(z))
        aux[0, CME_SLOTS * CME_FEATURES] = 1             # one slot filled on sample 0 only
        out = m(img, ts, las, aux)
        assert not torch.equal(out[0], m.head(z)[0]) and torch.equal(out[1:], m.head(z)[1:])


def test_model_a_and_b_start_as_model0():
    torch.manual_seed(0); m0 = ImageFusion(2, 3, **KW).eval()
    torch.manual_seed(0); mb = OptionalInputFusion(2, 3, fp_features=6, **KW).eval()
    img, ts, las = _inputs()
    aux = torch.randn(3, CME_SLOTS * CME_FEATURES + CME_SLOTS + 6); aux[:, CME_SLOTS * CME_FEATURES:][:, :CME_SLOTS] = 1
    with torch.no_grad():
        assert torch.allclose(m0(img, ts, las), mb(img, ts, las, aux))


def test_optional_aux_layout_and_training_standardisation():
    days = pd.date_range("2020-01-01", periods=4)
    t = pd.DataFrame({"issue_day": days, "cme_status": ["available", "no_cme", "available", "available"]})
    for k in (1, 2, 3):
        on = np.array([1, 0, 1, 1]) if k == 1 else np.zeros(4, int)
        t[f"cme{k}_mask"] = on
        for c, v in (("speed", [400.0, np.nan, 800.0, 1200.0]), ("lat", [0.0, np.nan, 10.0, -10.0]),
                     ("lon", [90.0, np.nan, 0.0, -90.0]), ("elapsed_h", [10.0, np.nan, 20.0, 30.0])):
            t[f"cme{k}_{c}"] = np.where(on == 1, v, np.nan)
    for v in (300, 400, 600):
        t[f"fp{v}_lon"] = [0.0, 90.0, 180.0, 270.0]
    a = optional_aux(t, days, np.array([True, True, True, False]), footpoint=True, fp_shift_deg=90.0)
    assert a.shape == (4, CME_SLOTS * CME_FEATURES + CME_SLOTS + 6)
    x = a[:, :CME_SLOTS * CME_FEATURES].reshape(4, CME_SLOTS, CME_FEATURES)
    assert np.allclose(x[[0, 2], 0, 0], [-1, 1])           # speed standardised on the two filled training slots
    assert np.allclose(x[1], 0) and np.allclose(x[:, 1:], 0)
    assert np.allclose(x[0, 0, 2:4], [1, 0], atol=1e-6)     # sin, cos of 90°
    assert a[:, CME_SLOTS * CME_FEATURES].tolist() == [1, 0, 1, 1]
    assert np.allclose(a[0, -6:-3], 1, atol=1e-6)           # footpoint 0° shifted by 90°: sin = 1
