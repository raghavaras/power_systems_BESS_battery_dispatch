"""QuestDB ingestion helper for micro_ems telemetry and dispatch logs."""

from datetime import datetime, timezone
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from questdb.ingress import IngressError, Sender, TimestampNanos


QUESTDB_CONF = "http::addr=localhost:9000;auto_flush_rows=10;"
LOCAL_TZ = ZoneInfo("Europe/Paris")
TABLE_NAME = "bess_log"


def sanitize_for_questdb(data_dict):
    """Convert NumPy/pandas scalar and datetime types to native Python types."""
    sanitized = {}
    for key, value in data_dict.items():
        if isinstance(value, np.floating):
            sanitized[key] = float(value)
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, (np.datetime64, pd.Timestamp)):
            sanitized[key] = pd.to_datetime(value).to_pydatetime()
        else:
            sanitized[key] = value
    return sanitized


def db_write(
    last_avg_timestamp,
    last_soc_percent,
    last_solar_W,
    last_load_W,
    last_aux_W,
    last_grid_W,
    last_ess_W,
    final_setpoint_W,
    solver_setpoint_W,
    flg_feasibility,
):
    """Write a single dispatch and meter telemetry row to QuestDB."""
    try:
        local_dt = (
            last_avg_timestamp.replace(tzinfo=LOCAL_TZ)
            if last_avg_timestamp.tzinfo is None
            else last_avg_timestamp.astimezone(LOCAL_TZ)
        )
        utc_dt = local_dt.astimezone(timezone.utc)

        columns = sanitize_for_questdb(
            {
                "soc_percent": last_soc_percent,
                "solar_W": last_solar_W,
                "load_W": last_load_W,
                "aux_W": last_aux_W,
                "grid_W": last_grid_W,
                "ess_W": last_ess_W,
                "final_setpoint_W": final_setpoint_W,
                "solver_setpoint_W": solver_setpoint_W,
                "feasibility": flg_feasibility,
            }
        )

        with Sender.from_conf(QUESTDB_CONF) as sender:
            sender.row(
                TABLE_NAME,
                symbols={},
                columns=columns,
                at=TimestampNanos.from_datetime(utc_dt),
            )
            sender.flush()

    except IngressError as exc:
        sys.stderr.write(f"Got error: {exc}\n")