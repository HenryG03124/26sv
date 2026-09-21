import ctypes
import json
import math
import struct
import unittest
from unittest.mock import Mock , patch

from MyAutoPilot.scs_telemetry import SCSTelemetry , snapshot_size


def sdk_snapshot():
    data = bytearray(snapshot_size)
    data[0] = 1
    struct.pack_into("<QQ" , data , 8 , 1234567 , 1200000)
    struct.pack_into("<I" , data , 40 , 12)
    struct.pack_into("<I" , data , 80 , 6)
    struct.pack_into("<i" , data , 504 , 7)
    struct.pack_into("<10f" , data , 948 , 12.5 , 1500 , -0.08 , 0.4 , 0.2 , 0 , -0.06 , 0.3 , 0.1 , 0)
    struct.pack_into("<16f" , data , 1200 , 0.01 , 0.01 , *([0] * 14))
    struct.pack_into("<6f" , data , 1868 , 1 , 0 , -12.5 , 0 , 0.02 , 0)
    struct.pack_into("<6d" , data , 2200 , 100 , 20 , -300 , 0.75 , 0 , 0)
    return data


class TelemetryTests(unittest.TestCase):
    def test_revision_12_units_and_signs(self):
        value = SCSTelemetry.decode(sdk_snapshot())
        self.assertEqual(value["status"] , "ok")
        self.assertEqual(value["speed_kmh"] , 45)
        self.assertEqual(value["gear"] , 7)
        self.assertAlmostEqual(value["user_steer_right"] , 0.08)
        self.assertAlmostEqual(value["game_steer_right"] , 0.06)
        self.assertAlmostEqual(value["yaw_rate_ccw_rad_s"] , 0.02 * 2 * math.pi)
        self.assertAlmostEqual(value["wheel_steering_deg"][0] , 3.6 , places = 6)
        self.assertAlmostEqual(value["game_brake"] , 0.1)
        self.assertEqual(value["position_m"] , [100 , 20 , -300])
        self.assertEqual(value["rotation_turns"][0] , 0.75)
        json.dumps(value , allow_nan = False)

    def test_reverse_speed_is_signed(self):
        data = sdk_snapshot()
        struct.pack_into("<f" , data , 948 , -2)
        self.assertAlmostEqual(SCSTelemetry.decode(data)["speed_kmh"] , -7.2)

    def test_paused_and_inactive_data_are_not_used_as_live_speed(self):
        for offset , status in ((4 , "paused") , (0 , "inactive")):
            data = sdk_snapshot()
            data[offset] = 1 if offset == 4 else 0
            value = SCSTelemetry.decode(data)
            self.assertEqual(value["status"] , status)
            self.assertIsNone(value["speed_kmh"])

    def test_unknown_revision_is_not_decoded(self):
        data = sdk_snapshot()
        struct.pack_into("<I" , data , 40 , 99)
        self.assertEqual(SCSTelemetry.decode(data) , {"status": "unsupported" , "revision": 99 , "speed_kmh": None})

    def test_missing_mapping_does_not_create_memory(self):
        reader = SCSTelemetry()
        reader.kernel = Mock()
        reader.kernel.OpenFileMappingW.return_value = None
        value = reader.read()
        self.assertEqual(value["status"] , "missing")
        reader.kernel.OpenFileMappingW.assert_called_once_with(4 , False , "Local\\SCSTelemetry")
        reader.kernel.MapViewOfFile.assert_not_called()
        reader.close()

    def test_stale_snapshot_is_rejected_and_reconnects(self):
        buffer = ctypes.create_string_buffer(bytes(sdk_snapshot()))
        reader = SCSTelemetry()
        reader.kernel = Mock()
        reader.address = ctypes.addressof(buffer)
        reader.handle = 42
        with patch("MyAutoPilot.scs_telemetry.time.monotonic" , side_effect = [10 , 10.6]):
            self.assertEqual(reader.read()["status"] , "ok")
            value = reader.read()
        self.assertEqual(value["status"] , "stale")
        self.assertIsNone(value["speed_kmh"])
        self.assertIsNone(reader.address)
        reader.kernel.CloseHandle.assert_called_once_with(42)

    def test_advancing_sdk_clock_remains_fresh(self):
        buffer = ctypes.create_string_buffer(bytes(sdk_snapshot()))
        reader = SCSTelemetry()
        reader.address = ctypes.addressof(buffer)
        with patch("MyAutoPilot.scs_telemetry.time.monotonic" , side_effect = [10 , 11]):
            reader.read()
            struct.pack_into("<Q" , buffer , 8 , 2234567)
            value = reader.read()
        self.assertEqual(value["status"] , "ok")
        self.assertEqual(value["age_s"] , 0)


if __name__ == "__main__":
    unittest.main()
