"""Main controller loop for micro_ems battery energy storage dispatching."""

from collections import deque
import csv
from datetime import datetime, timedelta
import os
import signal
import threading
import time

import numpy as np

from bess import BESS
from gitbased_code.ghv_micro_ems.profiles import (
    create_load_vector,
    create_price_vector,
    create_solar_vector,
)
from gitbased_code.ghv_micro_ems.programmes import (
    linear_program_optimal_setpoint_10min,
    mixed_integer_linear_program_optimal_setpoint_10min,
)
from meters import Meters
from questdb_api import db_write


SCRIPT_LOOP_PERIOD_MIN = 1
AUX_METER_ADDR = 7
ESS_METER_ADDR = 8
LOAD_METER_ADDR = 9
PV_METER_ADDR = 10
GRID_METER_ADDR = 12
GRID_MAX_DISCHARGING_POWER_W = 80_000
POLL_INTERVAL_SEC = 5
SCHEDULER_MARGIN_SEC = 5
BUFFER_WINDOW_SIZE = 5

# Installation and operating parameters
BATT_ENERGY_CAPACITY_KWH = 372.0
BATT_MAX_CHARGING_POWER_KW = 70.0
BATT_MAX_DISCHARGING_POWER_KW = 200.0
GRID_MAX_CHARGING_POWER_KW = 300.0
GRID_MAX_DISCHARGING_POWER_KW = 80.0
SOC_PERCENT_LOWER_LIMIT = 15.0
SOC_PERCENT_UPPER_LIMIT = 100.0

LOG_HEADER = [
    "current_timestamp",
    "last_avg_timestamp",
    "soc_percent",
    "solar_W",
    "load_W",
    "aux_W",
    "grid_W",
    "ess_W",
    "final_setpoint_W",
    "solver_setpoint_W",
    "feasibility",
]

# Backward-compatible module names
script_loop_period_min = SCRIPT_LOOP_PERIOD_MIN
aux_meter_addr = AUX_METER_ADDR
ess_meter_addr = ESS_METER_ADDR
load_meter_addr = LOAD_METER_ADDR
pv_meter_addr = PV_METER_ADDR
grid_meter_addr = GRID_METER_ADDR
grid_max_discharging_power_W = GRID_MAX_DISCHARGING_POWER_W
log_header = LOG_HEADER

background_thread_stop_event = threading.Event()
main_thread_stop_event = threading.Event()
final_setpoint_W = 0


def _clamp_battery_power_w(
    power_w: float,
    max_charge_kw: float,
    max_discharge_kw: float,
) -> float:
    """Clamp active power (W) to battery charge/discharge limits (positive = discharge)."""
    return float(
        np.clip(
            power_w,
            -max_charge_kw * 1000.0,
            max_discharge_kw * 1000.0,
        )
    )


def _month_bounds(now: datetime) -> tuple[datetime, datetime]:
    """Return start and end datetimes for the calendar month of `now`."""
    start = datetime(now.year, now.month, 1)
    next_month = (
        datetime(now.year + 1, 1, 1)
        if now.month == 12
        else datetime(now.year, now.month + 1, 1)
    )
    end = next_month - timedelta(days=1)
    return start, end


def _read_meter_snapshot(meters: Meters, bess: BESS) -> dict:
    """Read averaged meter powers, current timestamp, and battery SoC."""
    return {
        "last_avg_timestamp": meters.read_date_of_last_avg(PV_METER_ADDR),
        "current_timestamp": datetime.now(),
        "solar_W": meters.read_avg_active_power(PV_METER_ADDR),
        "load_W": meters.read_avg_active_power(LOAD_METER_ADDR),
        "aux_W": meters.read_avg_active_power(AUX_METER_ADDR),
        "grid_W": meters.read_avg_active_power(GRID_METER_ADDR),
        "ess_W": meters.read_avg_active_power(ESS_METER_ADDR),
        "soc_percent": bess.read_soc(),
    }


def _print_cycle_summary(
    snapshot: dict,
    final_setpoint: int,
    solver_setpoint: int,
    feasibility: int,
    execution_time: float,
    meters: Meters,
    bess: BESS,
) -> None:
    """Print telemetry and dispatch summary for one optimization cycle."""
    print(
        f"\nExecution time for optimize_bess_active_power_sp: {execution_time:.4f} seconds"
    )
    print(f"Current timestamp = {snapshot['current_timestamp']}")
    print(f"Last averaging timestamp = {snapshot['last_avg_timestamp']}")
    print(f"Last soc (%) = {snapshot['soc_percent']}%")
    print(f"Last solar power (W) = {snapshot['solar_W']}W")
    print(f"Last load (W) = {snapshot['load_W']}W")
    print(f"Solver setpoint (W) = {solver_setpoint}W")
    print(f"Final setpoint (W) = {final_setpoint}W")
    print(
        f"Feasibility of solution (1 = feasible, 0 = infeasible) = {feasibility}"
    )
    print(f"BESS TCP reconnection counter = {bess.get_connection_lost_counter()}")
    print(
        f"Meters TCP reconnection counter = {meters.get_connection_lost_counter()}"
    )
    print("--------------------------------------------------")


def post_solver_setpoint_adjustment(
    solver_setpoint_W,
    effective_load_W,
    batt_max_charging_power_kW,
    batt_max_discharging_power_kW,
    batt_energy_upper_limit_kWh,
    batt_energy_init_kWh,
):
    """Adjust solver setpoint locally for surplus solar or net load."""
    adjusted_w = float(solver_setpoint_W)

    if effective_load_W < 0:
        slot_length_minutes = 1
        factor = 60.0 / slot_length_minutes

        capped_load_w = max(
            float(effective_load_W), -batt_max_charging_power_kW * 1000.0
        )
        available_capacity_kwh = max(
            0.0, batt_energy_upper_limit_kWh - batt_energy_init_kWh
        )
        surplus_energy_kwh = min(
            abs(capped_load_w) / (1000.0 * factor),
            available_capacity_kwh,
        )
        adjusted_w = -surplus_energy_kwh * 1000.0 * factor

    elif effective_load_W > 0:
        adjusted_w = min(adjusted_w, float(effective_load_W))

    return _clamp_battery_power_w(
        adjusted_w,
        batt_max_charging_power_kW,
        batt_max_discharging_power_kW,
    )


def generate_optimal_setpoint(
    current_soc_percent,
    current_timestamp,
    price_vector_EUR_kWh,
    load_vector_kW,
    solar_vector_kW,
    load_buffer_kW,
    solar_buffer_kW,
    program_type,
):
    """Compute battery setpoint (W) from SoC, forecasts, and LP/MILP model."""
    batt_energy_lower_limit_kWh = round(
        (SOC_PERCENT_LOWER_LIMIT / 100.0) * BATT_ENERGY_CAPACITY_KWH, 2
    )
    batt_energy_upper_limit_kWh = round(
        (SOC_PERCENT_UPPER_LIMIT / 100.0) * BATT_ENERGY_CAPACITY_KWH, 2
    )
    batt_energy_init_kWh = round(
        (current_soc_percent / 100.0) * BATT_ENERGY_CAPACITY_KWH, 2
    )

    parameters = np.array(
        [
            batt_energy_lower_limit_kWh,
            batt_energy_upper_limit_kWh,
            BATT_MAX_CHARGING_POWER_KW,
            BATT_MAX_DISCHARGING_POWER_KW,
            GRID_MAX_CHARGING_POWER_KW,
            GRID_MAX_DISCHARGING_POWER_KW,
        ],
        dtype=float,
    )

    if current_soc_percent < SOC_PERCENT_LOWER_LIMIT:
        return -BATT_MAX_CHARGING_POWER_KW * 1000.0, 0, 0, []

    if program_type == "milp":
        batt_discharging_kW, batt_charging_kW, solution = (
            mixed_integer_linear_program_optimal_setpoint_10min(
                batt_energy_init_kWh,
                current_timestamp,
                price_vector_EUR_kWh,
                load_vector_kW,
                solar_vector_kW,
                parameters,
                load_buffer_kW,
                solar_buffer_kW,
            )
        )
        solver_setpoint_kW = batt_discharging_kW - batt_charging_kW
    else:
        solver_setpoint_kW, solution = linear_program_optimal_setpoint_10min(
            batt_energy_init_kWh,
            current_timestamp,
            price_vector_EUR_kWh,
            load_vector_kW,
            solar_vector_kW,
            parameters,
            load_buffer_kW,
            solar_buffer_kW,
        )

    solver_setpoint_W = int(solver_setpoint_kW * 1000)
    flg_feasibility = 1 if solution.x is not None else 0
    solver_variable_vector = solution.x if solution.x is not None else []

    effective_load_W = (load_buffer_kW[0] - solar_buffer_kW[0]) * 1000.0
    final_setpoint = post_solver_setpoint_adjustment(
        solver_setpoint_W,
        effective_load_W,
        BATT_MAX_CHARGING_POWER_KW,
        BATT_MAX_DISCHARGING_POWER_KW,
        batt_energy_upper_limit_kWh,
        batt_energy_init_kWh,
    )

    return final_setpoint, solver_setpoint_W, flg_feasibility, solver_variable_vector


def log_write(
    current_timestamp,
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
    """Append a telemetry and dispatch record to the monthly CSV log."""
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)

    start, end = _month_bounds(datetime.now())
    log_path = os.path.join(log_dir, f"bess_log_{start:%Y%m%d}_{end:%Y%m%d}.csv")
    write_header = not os.path.isfile(log_path)

    row = [
        current_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        last_avg_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        last_soc_percent,
        last_solar_W,
        last_load_W,
        last_aux_W,
        last_grid_W,
        last_ess_W,
        final_setpoint_W,
        solver_setpoint_W,
        flg_feasibility,
    ]

    with open(log_path, mode="a", newline="") as log_file:
        writer = csv.writer(log_file)
        if write_header:
            writer.writerow(LOG_HEADER)
        writer.writerow(row)


def background_loop(myMeters, myBESS):
    """Periodically poll meters, run the dispatch optimizer, and persist results."""
    global final_setpoint_W
    print(
        f"Script started with a loop period of {SCRIPT_LOOP_PERIOD_MIN} minutes."
    )

    price_vector_EUR_kWh = create_price_vector()
    load_vector_kW = create_load_vector()
    solar_vector_kW = create_solar_vector()

    load_history = deque([0.0] * BUFFER_WINDOW_SIZE, maxlen=BUFFER_WINDOW_SIZE)
    solar_history = deque([0.0] * BUFFER_WINDOW_SIZE, maxlen=BUFFER_WINDOW_SIZE)

    # Linear program is faster and sufficient without efficiency constraints.
    program_type = "lp"  # options: "milp" or "lp"
    period_sec = SCRIPT_LOOP_PERIOD_MIN * 60

    while not background_thread_stop_event.is_set():
        now_sec = time.time()
        next_mark = ((now_sec // period_sec) + 1) * period_sec
        sleep_time = next_mark - now_sec + SCHEDULER_MARGIN_SEC

        if background_thread_stop_event.wait(sleep_time):
            break

        snap = _read_meter_snapshot(myMeters, myBESS)
        total_load_kw = (abs(snap["load_W"]) + abs(snap["aux_W"])) / 1000.0
        solar_kw = snap["solar_W"] / 1000.0

        load_history.appendleft(total_load_kw)
        solar_history.appendleft(solar_kw)

        start_time = time.time()
        setpoint, solver_setpoint_w, feasibility, _ = generate_optimal_setpoint(
            snap["soc_percent"],
            snap["last_avg_timestamp"],
            price_vector_EUR_kWh,
            load_vector_kW,
            solar_vector_kW,
            list(load_history),
            list(solar_history),
            program_type,
        )
        execution_time = time.time() - start_time
        final_setpoint_W = setpoint

        log_write(
            snap["current_timestamp"],
            snap["last_avg_timestamp"],
            snap["soc_percent"],
            snap["solar_W"],
            snap["load_W"],
            snap["aux_W"],
            snap["grid_W"],
            snap["ess_W"],
            final_setpoint_W,
            solver_setpoint_w,
            feasibility,
        )

        db_write(
            snap["last_avg_timestamp"],
            snap["soc_percent"],
            snap["solar_W"],
            snap["load_W"],
            snap["aux_W"],
            snap["grid_W"],
            snap["ess_W"],
            final_setpoint_W,
            solver_setpoint_w,
            feasibility,
        )

        _print_cycle_summary(
            snap,
            final_setpoint_W,
            solver_setpoint_w,
            feasibility,
            execution_time,
            myMeters,
            myBESS,
        )

    print("background_loop stopped.\n")


def signal_handler(sig, frame):
    """Handle SIGINT by signaling both worker and main loops to stop."""
    print("Arrêt demandé...\n")
    background_thread_stop_event.set()
    main_thread_stop_event.set()


def main():
    """Connect hardware clients, run real-time setpoint loop, and clean up."""
    myBESS = BESS(ip="10.67.43.3")
    myMeters = Meters(ip="10.67.43.5")
    background_thread = None

    try:
        myBESS.connect()
        myMeters.connect()
        signal.signal(signal.SIGINT, signal_handler)

        background_thread = threading.Thread(
            target=background_loop, args=(myMeters, myBESS)
        )
        background_thread.start()

        while not main_thread_stop_event.is_set():
            myBESS.reconnect()
            myMeters.reconnect()

            grid_active_power = myMeters.read_inst_active_power(GRID_METER_ADDR)
            grid_excess = min(0.0, GRID_MAX_DISCHARGING_POWER_W - grid_active_power)
            if grid_excess != 0:
                print(
                    f"Grid power exceeds 80kW, the BESS will be used to compensate the difference ({grid_excess}W)"
                )

            myBESS.write_active_power_sp(int(final_setpoint_W - grid_excess))
            myBESS.write_heartbeat()
            print(".", end="", flush=True)

            main_thread_stop_event.wait(POLL_INTERVAL_SEC)

        print("main_loop stopped.\n")
        background_thread.join()
        myMeters.disconnect()
        myBESS.disconnect()
        raise SystemExit(0)

    except SystemExit:
        raise
    except Exception as exc:
        background_thread_stop_event.set()
        if background_thread is not None and background_thread.is_alive():
            background_thread.join()
        try:
            myMeters.disconnect()
        finally:
            myBESS.disconnect()
        print(f"An error occured: {exc}")
        print("The connection with the meters and the BESS has been closed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()