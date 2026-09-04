import json

import pandas as pd

from geoindex_daily.swpc import parse_45day_json, parse_45day_txt

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
