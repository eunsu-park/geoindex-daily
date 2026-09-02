"""PostgreSQL access to the shared `space_weather` / `solar_images` databases.

Credentials come from the SOLARIS_DB_* env vars (the same ones solaris-data uses),
with the geoindex-data DB_* names accepted as a fallback. Nothing here writes.
"""
import os

import psycopg2


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def connect(dbname: str = "space_weather", timeout: int = 10):
    """Read-only-by-convention connection to one of the shared databases."""
    return psycopg2.connect(
        host=_env("SOLARIS_DB_HOST", "DB_HOST", default="localhost"),
        port=int(_env("SOLARIS_DB_PORT", "DB_PORT", default="5432")),
        user=_env("SOLARIS_DB_USER", "DB_USER"),
        password=_env("SOLARIS_DB_PASSWORD", "DB_PASSWORD"),
        dbname=dbname,
        connect_timeout=timeout,
    )
