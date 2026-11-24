"""Modbus interface for the Socomec battery energy storage system."""

from enum import Enum

import pymodbus.client as ModbusClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder


# SunSpec / Socomec register addresses
SUNSPEC_802_MODEL_BASE = 0x9FB6  # Battery base model
SUNSPEC_704_MODEL_BASE = 0x9D68  # DER AC controls

SUNSPEC_WSET_ENA_ADDR = SUNSPEC_704_MODEL_BASE + 22
SUNSPEC_WSET_MOD_ADDR = SUNSPEC_704_MODEL_BASE + 23
SUNSPEC_WSET_ADDR = SUNSPEC_704_MODEL_BASE + 24

SOCOMEC_REGUL_MODE_REG_ADDR = 0x15B0
SOCOMEC_HEARTBEAT_REG_ADDR = 0x9FB2

# Backward-compatible names used by the original module.
sunspec_802_model_base = SUNSPEC_802_MODEL_BASE
sunspec_703_model_base = 0x9D55
sunspec_704_model_base = SUNSPEC_704_MODEL_BASE
sunspec_WSetEna_addr = SUNSPEC_WSET_ENA_ADDR
sunspec_WSetMod_addr = SUNSPEC_WSET_MOD_ADDR
sunspec_WSet_addr = SUNSPEC_WSET_ADDR
socomec_regul_mode_reg_addr = SOCOMEC_REGUL_MODE_REG_ADDR
socomec_heartbeat_reg_addr = SOCOMEC_HEARTBEAT_REG_ADDR


class RegulationMode(Enum):
    ESS = 0
    GRID = 1


class BESS:
    def __init__(self, ip):
        self.ip = ip
        self.hComPms = None
        self.connection_lost_counter = 0
        self.heartbeat = 0

    @staticmethod
    def _check_response(response, operation):
        if response.isError():
            raise RuntimeError(f"Modbus error: {operation}")

    @staticmethod
    def _int32_registers(value):
        builder = BinaryPayloadBuilder(
            byteorder=Endian.BIG,
            wordorder=Endian.BIG,
        )
        builder.add_32bit_int(int(value))
        return builder.to_registers()

    def is_connected(self):
        return self.hComPms is not None and self.hComPms.connected

    def connect(self):
        if self.is_connected():
            return

        # Discard a stale client before creating a new connection.
        self._close_client()

        client = ModbusClient.ModbusTcpClient(self.ip)
        self.hComPms = client

        try:
            if not client.connect():
                raise ConnectionError(f"Could not connect to BESS at {self.ip}")

            self.init_registers()
        except Exception:
            # Initialization may have enabled control before a later write failed.
            try:
                if self.is_connected():
                    self.deinit_registers()
            except Exception:
                pass  # Preserve the original initialization error.
            finally:
                self._close_client()
            raise

    def reconnect(self):
        if not self.is_connected():
            self.connection_lost_counter += 1
            # Do not attempt register writes over a lost connection.
            self._close_client()
            self.connect()

    def get_connection_lost_counter(self):
        return self.connection_lost_counter

    def read_soc(self):
        if not self.is_connected():
            raise ConnectionError("BESS is not connected")

        address = SUNSPEC_802_MODEL_BASE + 11
        response = self.hComPms.read_holding_registers(address, 1)
        self._check_response(response, "reading SOC")
        return response.registers[0] / 10

    def write_heartbeat(self):
        if not self.is_connected():
            raise ConnectionError("BESS is not connected")

        next_heartbeat = self.heartbeat + 1
        response = self.hComPms.write_registers(
            SOCOMEC_HEARTBEAT_REG_ADDR,
            self._int32_registers(next_heartbeat),
        )
        self._check_response(response, "writing PMS heartbeat")
        self.heartbeat = next_heartbeat

    def write_active_power_sp(self, Pess):
        if not self.is_connected():
            raise ConnectionError("BESS is not connected")

        response = self.hComPms.write_registers(
            SUNSPEC_WSET_ADDR,
            self._int32_registers(Pess),
        )
        self._check_response(response, "writing active power setpoint")

    def init_registers(self):
        self._write_register(SUNSPEC_WSET_ENA_ADDR, 1, "enabling active power control")
        self._write_register(SUNSPEC_WSET_MOD_ADDR, 1, "setting active power mode")
        self._write_register(
            SOCOMEC_REGUL_MODE_REG_ADDR,
            RegulationMode.ESS.value,
            "setting PMS regulation mode to ESS",
        )

    def deinit_registers(self):
        self._write_register(SUNSPEC_WSET_ENA_ADDR, 0, "disabling active power control")
        self._write_register(
            SOCOMEC_REGUL_MODE_REG_ADDR,
            RegulationMode.GRID.value,
            "setting PMS regulation mode to GRID",
        )

    def _write_register(self, address, value, operation):
        if not self.is_connected():
            raise ConnectionError("BESS is not connected")

        response = self.hComPms.write_register(address, value)
        self._check_response(response, operation)

    def _close_client(self):
        client, self.hComPms = self.hComPms, None
        if client is not None:
            client.close()

    def disconnect(self):
        if self.hComPms is None:
            return

        try:
            # Restore grid mode only while communication is still available.
            if self.is_connected():
                self.deinit_registers()
        finally:
            self._close_client()
