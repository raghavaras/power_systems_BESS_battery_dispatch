"""Battery dispatch optimization models for the micro_ems controller.

The public function signatures and return values are kept compatible with the
original implementation. Power is expressed in kW, energy in kWh and prices
in EUR/kWh. The optimization horizon is 24 hours with 10-minute intervals.
"""

from datetime import timedelta
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


SLOT_MINUTES = 10
SLOTS_PER_HOUR = 60 // SLOT_MINUTES
HORIZON_HOURS = 24
HORIZON = HORIZON_HOURS * SLOTS_PER_HOUR
PENALTY = 10.0


def _read_parameters(parameters: Sequence[float]) -> tuple[float, ...]:
    """Unpack battery and grid limits in the legacy parameter order."""
    if len(parameters) != 6:
        raise ValueError("parameters must contain exactly six values")
    return tuple(float(value) for value in parameters)


def _forecast_at(
    slot: int,
    starting_timestamp,
    load_vector_kW,
    solar_vector_kW,
    load_buffer_kW,
    solar_buffer_kW,
) -> float:
    """Return net load for one slot, retaining the original first-slot buffer."""
    if slot == 0:
        return float(load_buffer_kW[0]) - float(solar_buffer_kW[0])

    timestamp = starting_timestamp + timedelta(minutes=SLOT_MINUTES * slot)
    day = timestamp.weekday()
    hour = timestamp.hour
    slot_in_hour = timestamp.minute // SLOT_MINUTES

    load = load_vector_kW[day][hour][slot_in_hour]
    solar = solar_vector_kW[hour][slot_in_hour]
    return float(load) - float(solar)


def _price_at(slot: int, starting_timestamp, price_vector_EUR_kWh) -> float:
    timestamp = starting_timestamp + timedelta(minutes=SLOT_MINUTES * slot)
    return float(price_vector_EUR_kWh[timestamp.hour]) / SLOTS_PER_HOUR


def _make_constraint(rows, lower, upper, coefficients, lb, ub):
    """Append one sparse linear constraint row."""
    rows.append(coefficients)
    lower.append(lb)
    upper.append(ub)


def _solve_milp(objective, integrality, lower_bounds, upper_bounds,
                rows, constraint_lower, constraint_upper):
    """Build and solve a sparse MILP/LP using SciPy's HiGHS interface."""
    row_indices, col_indices, values = [], [], []
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            if value:
                row_indices.append(row_index)
                col_indices.append(column)
                values.append(value)

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(rows), len(objective)),
    ).tocsr()

    constraints = LinearConstraint(matrix, constraint_lower, constraint_upper)
    bounds = Bounds(lower_bounds, upper_bounds)

    return milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )


def mixed_integer_linear_program_optimal_setpoint_10min(
    batt_initial_energy_kWh,
    starting_timestamp,
    price_vector_EUR_kWh,
    load_vector_kW,
    solar_vector_kW,
    parametres,
    load_buffer_kW,
    solar_buffer_kW,
):
    """Optimize battery/grid dispatch with mutually exclusive charge/discharge.

    Variables per slot: battery discharge, battery charge, grid discharge,
    grid charge, battery energy, battery mode binary and grid mode binary.
    """
    (
        batt_energy_lower,
        batt_energy_upper,
        batt_max_charge,
        batt_max_discharge,
        grid_max_charge,
        grid_max_discharge,
    ) = _read_parameters(parametres)

    # Per-slot variable offsets: XD, XC, ZD, ZC, Y, LB, LG.
    block = 7
    n_variables = HORIZON * block

    objective = np.zeros(n_variables)
    integrality = np.zeros(n_variables, dtype=np.uint8)
    lower_bounds = np.zeros(n_variables)
    upper_bounds = np.zeros(n_variables)

    rows, constraint_lower, constraint_upper = [], [], []

    for slot in range(HORIZON):
        base = slot * block

        # The objective prices grid discharge, as in the original model.
        objective[base + 2] = _price_at(
            slot, starting_timestamp, price_vector_EUR_kWh
        )

        upper_bounds[base:base + 4] = (
            batt_max_discharge,
            batt_max_charge,
            grid_max_discharge,
            grid_max_charge,
        )
        lower_bounds[base + 4] = batt_energy_lower
        upper_bounds[base + 4] = batt_energy_upper

        upper_bounds[base + 5:base + 7] = 1
        integrality[base + 5:base + 7] = 1

        net_load = _forecast_at(
            slot, starting_timestamp, load_vector_kW, solar_vector_kW,
            load_buffer_kW, solar_buffer_kW,
        )

        # XD - XC + ZD - ZC = load - solar.
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base: 1, base + 1: -1, base + 2: 1, base + 3: -1},
            net_load, net_load,
        )

        # Y[0] is fixed to initial energy; subsequent energy follows
        # the previous slot's charge/discharge setpoints.
        if slot == 0:
            energy_row = {base + 4: 1}
            energy_value = batt_initial_energy_kWh
        else:
            previous = base - block
            energy_row = {
                base + 4: 1,
                previous + 4: -1,
                previous + 1: -1 / SLOTS_PER_HOUR,
                previous: 1 / SLOTS_PER_HOUR,
            }
            energy_value = 0

        _make_constraint(
            rows, constraint_lower, constraint_upper,
            energy_row, energy_value, energy_value,
        )

        # Battery charge and discharge cannot occur simultaneously.
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base: 1, base + 5: -batt_max_discharge},
            -batt_max_discharge, 0,
        )
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base + 1: 1, base + 5: batt_max_charge},
            0, batt_max_charge,
        )

        # Grid import and export cannot occur simultaneously.
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base + 2: 1, base + 6: -grid_max_discharge},
            -grid_max_discharge, 0,
        )
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base + 3: 1, base + 6: grid_max_charge},
            0, grid_max_charge,
        )

    solution = _solve_milp(
        objective, integrality, lower_bounds, upper_bounds,
        rows, constraint_lower, constraint_upper,
    )

    battery_discharge = 0.0
    battery_charge = 0.0
    if solution.x is not None:
        battery_discharge = round(float(solution.x[0]), 2)
        battery_charge = round(float(solution.x[1]), 2)

    return battery_discharge, battery_charge, solution


def linear_program_optimal_setpoint_10min(
    batt_initial_energy_kWh,
    starting_timestamp,
    price_vector_EUR_kWh,
    load_vector_kW,
    solar_vector_kW,
    parametres,
    load_buffer_kW,
    solar_buffer_kW,
):
    """Optimize continuous battery dispatch with a penalized balance slack.

    Variables per slot: signed battery setpoint, grid discharge, grid charge,
    battery energy and nonnegative balance penalty.
    A positive battery setpoint means discharge; a negative one means charge.
    """
    (
        batt_energy_lower,
        batt_energy_upper,
        batt_max_charge,
        batt_max_discharge,
        grid_max_charge,
        grid_max_discharge,
    ) = _read_parameters(parametres)

    # Per-slot variable offsets: X, ZD, ZC, Y, E.
    block = 5
    n_variables = HORIZON * block

    objective = np.zeros(n_variables)
    integrality = np.zeros(n_variables, dtype=np.uint8)
    lower_bounds = np.zeros(n_variables)
    upper_bounds = np.zeros(n_variables)

    rows, constraint_lower, constraint_upper = [], [], []

    for slot in range(HORIZON):
        base = slot * block

        objective[base + 1] = _price_at(
            slot, starting_timestamp, price_vector_EUR_kWh
        )
        objective[base + 4] = PENALTY

        lower_bounds[base] = -batt_max_charge
        upper_bounds[base] = batt_max_discharge
        upper_bounds[base + 1] = grid_max_discharge
        upper_bounds[base + 2] = grid_max_charge
        lower_bounds[base + 3] = batt_energy_lower
        upper_bounds[base + 3] = batt_energy_upper
        upper_bounds[base + 4] = 0
        upper_bounds[base + 4] = np.inf

        net_load = _forecast_at(
            slot, starting_timestamp, load_vector_kW, solar_vector_kW,
            load_buffer_kW, solar_buffer_kW,
        )

        # X + ZD - ZC + E = load - solar.
        _make_constraint(
            rows, constraint_lower, constraint_upper,
            {base: 1, base + 1: 1, base + 2: -1, base + 4: 1},
            net_load, net_load,
        )

        # Y[t] = Y[t-1] - X[t-1] * slot_duration_hours.
        if slot == 0:
            energy_row = {base + 3: 1}
            energy_value = batt_initial_energy_kWh
        else:
            previous = base - block
            energy_row = {
                base + 3: 1,
                previous + 3: -1,
                previous: 1 / SLOTS_PER_HOUR,
            }
            energy_value = 0

        _make_constraint(
            rows, constraint_lower, constraint_upper,
            energy_row, energy_value, energy_value,
        )

    solution = _solve_milp(
        objective, integrality, lower_bounds, upper_bounds,
        rows, constraint_lower, constraint_upper,
    )

    battery_setpoint = 0.0
    if solution.x is not None:
        battery_setpoint = round(float(solution.x[0]), 2)

    return battery_setpoint, solution
