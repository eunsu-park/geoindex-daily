"""geoindex-daily — daily-cadence geomagnetic/solar index forecasting, weeks ahead.

Sibling of geoindex-model (30-min ap30/hp30, hours ahead). Reads the shared PostgreSQL
tables (OMNI hourly → daily Ap/F10.7/Kp/SN) and the SuryaBench catalog; owns the
daily windows, the recurrence baselines, the evaluation, and the foundation-model
encoders (Surya for images, MOMENT for the index history).
"""
__version__ = "0.1.0"
