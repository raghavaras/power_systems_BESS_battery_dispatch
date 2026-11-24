"""Modbus TCP client for reading energy meter registers."""

from datetime import datetime, timedelta

import pymodbus.client as ModbusClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder


INST_ACTIVE_POWER_ADDR = 0x482C
AVG_ACTIVE_POWER_ADDR = 0x48AC
DATE_OF_LAST_AVG_ADDR = 0x4881
EPOCH_2000 = datetime(2000, 1, 1)


class Meters:
    """Client for polling instantaneous, averaged, and timestamp data from meters."""

    def __init__(self, ip):
        self.ip = ip
        self.hComGtw = None
        self.connection_lost_counter = 0

    def is_connected(self):
        return self.hComGtw is not None and self.hComGtw.connected

    def connect(self):
        if not self.is_connected():
            self.disconnect()
            client = ModbusClient.ModbusTcpClient(self.ip, timeout=2)
            client.connect()
            self.hComGtw = client

    def reconnect(self):
        if not self.is_connected():
            self.connection_lost_counter += 1
            self.disconnect()
            self.connect()

    def get_connection_lost_counter(self):
        return self.connection_lost_counter

    def _read_32bit(self, slave_addr, addr, *, signed):
        if not self.is_connected():
            raise ConnectionError(f"Meters client at {self.ip} is not connected")

        result = self.hComGtw.read_holding_registers(addr, count=2, slave=slave_addr)
        if result.isError():
            raise RuntimeError(
                f"Error reading {'int32' if signed else 'uint32'} from slave {slave_addr} at 0x{addr:X}"
            )

        decoder = BinaryPayloadDecoder.fromRegisters(
            result.registers, byteorder=Endian.BIG
        )
        return decoder.decode_32bit_int() if signed else decoder.decode_32bit_uint()

    def read_int(self, slave_addr, addr):
        return self._read_32bit(slave_addr, addr, signed=True)

    def read_uint(self, slave_addr, addr):
        return self._read_32bit(slave_addr, addr, signed=False)

    def read_inst_active_power(self, slave_addr):
        return self.read_int(slave_addr, INST_ACTIVE_POWER_ADDR)

    def read_avg_active_power(self, slave_addr):
        return self.read_int(slave_addr, AVG_ACTIVE_POWER_ADDR)

    def read_date_of_last_avg(self, slave_addr):
        elapsed_sec = self.read_uint(slave_addr, DATE_OF_LAST_AVG_ADDR)
        return EPOCH_2000 + timedelta(seconds=elapsed_sec)

    def disconnect(self):
        client, self.hComGtw = self.hComGtw, None
        if client is not None:
            client.close()