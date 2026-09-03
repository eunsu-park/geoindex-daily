import json

import pandas as pd

from geoindex_daily.swpc import parse_45day_json, parse_45day_txt, parse_prf_outlook

PRF_SNIPPET = """
                       SWPC PRF 2314 06 January 2020                    3
                           Twenty-seven Day Outlook

         Radio Flux Planetary Largest                         Radio Flux Planetary Largest
Date      10.7cm    A Index Kp Index              Date         10.7cm    A Index Kp Index

06 Jan      72        12        4                 20 Jan         70          5       2
07          72         5        2                 21             70          5       2
08          72         8        3                 22             70          5       2
09          72         8        3                 23             70          5       2
10          72         8        3                 24             70          5       2
11          72         5        2                 25             71          5       2
12          71         5        2                 26             72          5       2
13          70         5        2                 27             72          5       2
14          70        12        4                 28             72          5       2
15          70        12        4                 29             72          5       2
16          70         5        2                 30             72          5       2
17          70         5        2                 31             72          5       2
18          70         5        2                 01 Feb         72          8       3
19          70         5        2



4                             SWPC PRF 2314 06 January 2020
                                        Energetic Events
"""

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


def test_prf_outlook_two_columns_and_month_roll():
    df = parse_prf_outlook(PRF_SNIPPET)
    assert len(df) == 27
    assert df.issue_date.iloc[0] == pd.Timestamp("2020-01-06")
    assert df.target_date.iloc[0] == pd.Timestamp("2020-01-06") and df.lead.iloc[0] == 0
    assert df.target_date.iloc[-1] == pd.Timestamp("2020-02-01") and df.lead.iloc[-1] == 26
    assert df.target_date.is_monotonic_increasing and df.target_date.diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert df.ap.iloc[8] == 12 and df.f107.iloc[26] == 72 and df.kp.iloc[26] == 3


def test_prf_year_wrap():
    snippet = PRF_SNIPPET.replace("06 January 2020", "23 December 2019").replace("06 Jan", "23 Dec").replace("20 Jan", "06 Jan").replace("01 Feb", "18 Jan")
    df = parse_prf_outlook(snippet)
    assert df.issue_date.iloc[0] == pd.Timestamp("2019-12-23")
    assert df.target_date.iloc[-1].year == 2020


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
