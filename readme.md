# Micro EMS — Battery Energy Storage Dispatch Controller

**Micro EMS** is a real-time energy management controller for a Battery Energy Storage System (BESS). It periodically computes an optimal 24-hour active power dispatch policy over 10-minute intervals and applies real-time setpoints to minimize grid electricity costs while respecting battery and grid constraints.

---

## How It Works

### 1. Horizon & Sign Convention
* **Time Horizon:** 24 hours divided into **144 intervals of 10 minutes** ($t = 1 \dots 144$, from `00:00` to `24:00`). Because each slot is $\frac{1}{6}\text{ h}$, power in $\text{kW}$ converts to energy in $\text{kWh}$ by dividing by $6$.
* **Setpoint Convention ($\pi(t)$):**
  * **Positive ($\pi(t) > 0$):** Battery **discharges** (supplies power to the site).
  * **Negative ($\pi(t) < 0$):** Battery **charges** (absorbs power).

### 2. Forecasting & Profiles
At each optimization step starting at timestamp $t_0$, the controller builds 144-slot vectors for the upcoming 24 hours:
* **Load Profile ($L(t)$):** Expected site demand ($\text{kW}$) indexed by day of the week and 10-minute slot, with slot $0$ anchored to recent live meter measurements.
* **PV Generation Profile ($S(t)$):** Expected solar production ($\text{kW}$) indexed by 10-minute slot, with slot $0$ anchored to live PV meter data.
* **Tariff Profile ($p(t)$):** Hourly grid import price ($\text{EUR/kWh}$) mapped across the 144 slots.

### 3. Optimization & Real-Time Control Loop
* **Optimization Models (`LP` / `MILP`):** Solves a receding-horizon dispatch problem over the 144 intervals subject to power balance, battery State of Charge (SoC) dynamics and bounds ($[Z_1, Z_2]$), battery charge/discharge power limits ($[-X_1, X_2]$), and grid import/export limits ($[0, Y_1]$, $[0, Y_2]$).
* **Post-Solver Adjustment:** Refines the first-interval setpoint against real-time net load (`load - solar`) and available battery energy headroom.
* **Two-Tier Control Execution:**
  * **Background Optimizer Loop (every 1 min):** Polls averaged meter values and battery SoC, runs the optimizer, logs results to CSV and QuestDB, and updates the target setpoint.
  * **Fast Protection & Heartbeat Loop (every 5 s):** Maintains Modbus TCP connections, checks instantaneous grid import against the $80\text{ kW}$ threshold (compensating any excess with the BESS), writes the active power setpoint, and increments the PMS heartbeat.

---

## Project Structure

```text
.
├── controller.py       # Main controller: fast setpoint/heartbeat loop, 1-min optimizer loop, logging
├── programmes.py      # LP and MILP 24-hour (144-slot) dispatch optimization formulations (SciPy/HiGHS)
├── profiles.py        # Static 10-minute load, solar PV, and electricity price profiles
├── bess.py            # Modbus TCP client for the BESS (SoC, regulation mode, setpoint, heartbeat)
├── meters.py          # Modbus TCP client for site energy meters (PV, load, aux, grid, ESS)
├── questdb_api.py     # Time-series telemetry ingestion helper for QuestDB
└── log/               # Auto-generated monthly CSV dispatch logs (bess_log_YYYYMMDD_YYYYMMDD.csv)
```

---

## Hardware & Telemetry Interfaces

* **BESS (`bess.py`):** Communicates via Modbus TCP using SunSpec (models 704 & 802) registers to manage ESS/GRID regulation mode, read SoC, write active power setpoints ($\text{W}$), and maintain the controller heartbeat.
* **Meters (`meters.py`):** Reads 32-bit instantaneous power, averaged power, and averaging timestamps across five meter addresses (`aux=7`, `ess=8`, `load=9`, `pv=10`, `grid=12`).
* **Storage (`controller.py`, `questdb_api.py`):** Persists every optimization cycle to monthly CSV files under `log/` and streams rows into the `bess_log` table in QuestDB.