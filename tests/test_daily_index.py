import numpy as np
import pandas as pd

from geoindex_daily.daily_index import DAILY_COLUMNS, mask_fills, to_daily


def _hourly(days=3):
    """Synthetic OMNI-like hourly frame: 3-h ap repeated per block, daily F10.7/SN repeated."""
    idx = pd.date_range("2015-01-01", periods=24 * days, freq="h")
    ap3h = np.array([2, 5, 7, 12, 22, 9, 4, 3])  # eight 3-hourly values per day
    ap = np.repeat(np.tile(ap3h, days), 3)
    return pd.DataFrame({
        "datetime": idx,
        "ap_index_nt": ap.astype(float),
        "kp_index": np.repeat(np.tile([7, 13, 17, 27, 37, 20, 10, 7], days), 3).astype(float),
        "f10_7_index_sfu": np.repeat([120.5, 130.0, 999.9][:days], 24).astype(float),
        "sunspot_number_r": np.repeat([50, 60, 70][:days], 24).astype(float),
        "dst_index_nt": np.tile(np.linspace(-5, -60, 24), days),
    })


def test_daily_ap_equals_mean_of_eight_3h_values():
    d = to_daily(mask_fills(_hourly()))
    assert list(d.columns) == DAILY_COLUMNS
    assert np.isclose(d.ap.iloc[0], np.mean([2, 5, 7, 12, 22, 9, 4, 3]))
    assert (d.n_ap_hours == 24).all()
    assert np.isclose(d.kp.iloc[0], 3.7)
    assert np.isclose(d.dst.iloc[0], -60)


def test_fill_values_masked_and_missing_day_kept_as_nan():
    h = _hourly(3)
    h = h[h.datetime.dt.day != 2]  # drop the whole second day
    d = to_daily(mask_fills(h))
    assert len(d) == 3                      # contiguous index
    assert d.n_ap_hours.iloc[1] == 0 and np.isnan(d.ap.iloc[1])
    assert np.isnan(d.f107.iloc[2])         # 999.9 masked
    assert np.isclose(d.f107.iloc[0], 120.5)


def test_gfz_3h_unrolls_eight_bins_per_day(tmp_path):
    from geoindex_daily.features import gfz_3h
    line = ("2015 01 01 30316.5 30316.5 2475 1  0.333 1.000 1.333 2.000 3.000 1.667 0.667 0.333"
            "   2   4   5   7  15   6   3   2   6  50  120.5  118.0 1\n")
    f = tmp_path / "gfz.txt"
    f.write_text("# header\n" + line)
    d = gfz_3h(f)
    assert len(d) == 8 and d.index[1] == pd.Timestamp("2015-01-01 03:00")
    assert list(d.ap) == [2, 4, 5, 7, 15, 6, 3, 2] and np.isclose(d.kp.iloc[4], 3.0)
    assert np.round(d.ap.mean()) == 6
