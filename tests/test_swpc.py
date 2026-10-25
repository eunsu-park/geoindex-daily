import json

import pandas as pd

from geoindex_daily.swpc import parse_45day_json, parse_45day_txt, parse_rsga

TXT_SNIPPET = """:Product: 45 Day AP and F10.7cm Flux Forecast 45-day-forecast.txt
:Issued: 2026 Sep 02 0000 UTC
#
45-DAY AP FORECAST
02Sep26 005 03Sep26 005 04Sep26 005 05Sep26 005 06Sep26 010
07Sep26 012 08Sep26 008
45-DAY F10.7 CM FLUX FORECAST
02Sep26 100 03Sep26 102 04Sep26 104 05Sep26 106 06Sep26 108
07Sep26 110 08Sep26 112
"""


def test_45day_txt():
    df = parse_45day_txt(TXT_SNIPPET)
    assert len(df) == 7 and df.issue_date.iloc[0] == pd.Timestamp("2026-09-02")
    assert df.ap.tolist() == [5, 5, 5, 5, 10, 12, 8] and df.f107.iloc[-1] == 112
    assert df.lead.tolist() == list(range(7))


def test_45day_json():
    j = {"issued": "2026-03-02T00:00:00Z", "data": [
        {"time": "2026-03-02T00:00:00Z", "ap": 5, "f107": 150},
        {"time": "2026-03-03T00:00:00Z", "ap": 8, "f107": 152}]}
    df = parse_45day_json(json.dumps(j))
    assert df.lead.tolist() == [0, 1] and df.ap.tolist() == [5.0, 8.0]


RSGA_DEC = """:Product: Report of Solar-Geophysical Activity
:Issued: 2022 Dec 31 2200 UTC
Predicted   01 Jan-03 Jan 165/165/165
Observed Afr/Ap 30 Dec  022/028
Estimated Afr/Ap 31 Dec  013/014
Predicted Afr/Ap 01 Jan-03 Jan  009/012-009/008-005/005
"""


def test_rsga_year_wrap():
    df = parse_rsga(RSGA_DEC).set_index("kind")
    assert df.loc["observed", "target_date"] == pd.Timestamp("2022-12-30") and df.loc["observed", "ap"] == 28
    assert df.loc["estimated", "lead"] == 0 and df.loc["estimated", "ap"] == 14
    pred = df.loc["predicted"]
    assert pred.target_date.tolist() == list(pd.date_range("2023-01-01", periods=3))
    assert pred.ap.tolist() == [12, 8, 5] and pred.lead.tolist() == [1, 2, 3]
    assert (pred.issue_time == "2200").all()


def test_rsga_na():
    text = RSGA_DEC.replace("009/012-009/008-005/005", "007/ NA-014/ NA-007/ NA")
    pred = parse_rsga(text).query("kind == 'predicted'")
    assert pred.ap.isna().all() and pred.afr.tolist() == [7, 14, 7]
