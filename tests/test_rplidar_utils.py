"""
Unit tests for utility functions added in test_rplidar.py in this PR.

Functions tested:
  - find_serial(lidar)        — finds serial attribute on lidar object
  - flush(lidar, serial_obj)  — flushes serial buffers
  - try_raw_serial(lidar, serial_obj) — returns False when serial_obj is None
  - try_iter_measures_normal / try_iter_measures_express — returns False when
    neither iter_measurements nor iter_measurments is present

The module-level code in test_rplidar.py (which opens a real RPLidar) is
bypassed by mocking rplidar.RPLidar to raise an exception before import.
"""
import sys
import unittest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Prevent the module-level RPLidar connection attempt
# ---------------------------------------------------------------------------

_rplidar_mod_mock = MagicMock(name="rplidar_module")
_rplidar_mod_mock.RPLidar.side_effect = Exception("mocked: no hardware")

if "rplidar" not in sys.modules:
    sys.modules["rplidar"] = _rplidar_mod_mock

# The module-level try/except in test_rplidar.py will print "FATAL ERROR …"
# but the import still succeeds and the functions become accessible.
import test_rplidar as rplidar_diag  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lidar_mock(**attrs):
    """Return a MagicMock that behaves like RPLidar with configurable attrs."""
    m = MagicMock(name="RPLidar")
    for key, val in attrs.items():
        setattr(m, key, val)
    return m


def _make_serial_mock():
    """Return a mock with reset_input_buffer and reset_output_buffer methods."""
    s = MagicMock(name="serial")
    s.reset_input_buffer = MagicMock()
    s.reset_output_buffer = MagicMock()
    return s


# ===========================================================================
# Tests for find_serial
# ===========================================================================

class TestFindSerial(unittest.TestCase):
    def test_finds_underscore_serial(self):
        serial = _make_serial_mock()
        lidar = _make_lidar_mock(_serial=serial)
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial)

    def test_finds_underscore_serial_port(self):
        serial = _make_serial_mock()
        # _serial is None, _serial_port is the real one
        lidar = MagicMock()
        lidar._serial = None
        lidar._serial_port = serial
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial)

    def test_finds_serial_attr(self):
        serial = _make_serial_mock()
        lidar = MagicMock()
        del lidar._serial          # not present
        del lidar._serial_port     # not present
        lidar.serial = serial
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial)

    def test_finds_ser_attr(self):
        serial = _make_serial_mock()
        lidar = MagicMock(spec=[])  # no attributes by default
        lidar.ser = serial
        # ser has reset_input_buffer
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial)

    def test_finds_port_attr(self):
        serial = _make_serial_mock()
        lidar = MagicMock(spec=[])
        lidar._port = serial
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial)

    def test_returns_none_when_no_serial_attr(self):
        lidar = MagicMock(spec=[])  # empty spec — no matching attributes
        result = rplidar_diag.find_serial(lidar)
        self.assertIsNone(result)

    def test_skips_attr_without_reset_input_buffer(self):
        """An attribute without reset_input_buffer should not be returned."""
        fake = MagicMock(spec=[])   # no reset_input_buffer
        lidar = MagicMock(spec=[])
        lidar._serial = fake
        result = rplidar_diag.find_serial(lidar)
        self.assertIsNone(result)

    def test_priority_order_underscore_serial_first(self):
        """_serial should be found before other attributes."""
        serial1 = _make_serial_mock()
        serial2 = _make_serial_mock()
        lidar = _make_lidar_mock(_serial=serial1, _serial_port=serial2)
        result = rplidar_diag.find_serial(lidar)
        self.assertIs(result, serial1)

    def test_none_attr_skipped(self):
        """getattr returning None should not trigger an error."""
        lidar = MagicMock()
        lidar._serial = None
        lidar._serial_port = None
        lidar.serial = None
        lidar.ser = None
        lidar._port = None
        result = rplidar_diag.find_serial(lidar)
        self.assertIsNone(result)


# ===========================================================================
# Tests for flush
# ===========================================================================

class TestFlush(unittest.TestCase):
    def test_calls_clean_input_when_present(self):
        lidar = MagicMock()
        lidar.clean_input = MagicMock()
        serial_obj = _make_serial_mock()
        rplidar_diag.flush(lidar, serial_obj)
        lidar.clean_input.assert_called_once()

    def test_no_clean_input_attr_is_ok(self):
        """Lidar without clean_input should not raise."""
        lidar = MagicMock(spec=[])  # no clean_input
        serial_obj = _make_serial_mock()
        # Should not raise
        rplidar_diag.flush(lidar, serial_obj)

    def test_calls_reset_input_buffer(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        rplidar_diag.flush(lidar, serial_obj)
        serial_obj.reset_input_buffer.assert_called_once()

    def test_calls_reset_output_buffer(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        rplidar_diag.flush(lidar, serial_obj)
        serial_obj.reset_output_buffer.assert_called_once()

    def test_serial_obj_none_does_not_raise(self):
        lidar = MagicMock(spec=[])
        rplidar_diag.flush(lidar, None)  # must not raise

    def test_clean_input_and_reset_buffers_both_called(self):
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        rplidar_diag.flush(lidar, serial_obj)
        lidar.clean_input.assert_called_once()
        serial_obj.reset_input_buffer.assert_called_once()
        serial_obj.reset_output_buffer.assert_called_once()


# ===========================================================================
# Tests for try_raw_serial
# ===========================================================================

class TestTryRawSerial(unittest.TestCase):
    def test_returns_false_when_serial_none(self):
        lidar = _make_lidar_mock()
        result = rplidar_diag.try_raw_serial(lidar, None)
        self.assertFalse(result)

    def test_returns_true_when_serial_provided_and_valid(self):
        """A valid 7-byte descriptor should allow the function to return True."""
        lidar = MagicMock()
        serial_obj = _make_serial_mock()

        # Simulate a valid 7-byte RPLIDAR response descriptor
        # start1=0xA5, start2=0x5A, data_size bytes (5 bytes), data_type
        descriptor = bytes([0xA5, 0x5A, 0x05, 0x00, 0x00, 0x40, 0x81])
        # Simulate 5 packets of 5 bytes each
        packet = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        serial_obj.read.side_effect = [descriptor] + [packet] * 5

        result = rplidar_diag.try_raw_serial(lidar, serial_obj)
        self.assertTrue(result)

    def test_returns_false_when_descriptor_incomplete(self):
        """Short descriptor (< 7 bytes) should cause early return False."""
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        serial_obj.read.side_effect = [bytes([0xA5, 0x5A, 0x05])]  # only 3 bytes

        result = rplidar_diag.try_raw_serial(lidar, serial_obj)
        self.assertFalse(result)

    def test_sends_scan_command(self):
        """Should write 0xA5 0x20 to start scan."""
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        # Provide a valid descriptor so execution proceeds
        descriptor = bytes([0xA5, 0x5A, 0x01, 0x00, 0x00, 0x40, 0x81])
        packet = bytes([0xFF])
        serial_obj.read.side_effect = [descriptor] + [packet] * 10
        rplidar_diag.try_raw_serial(lidar, serial_obj)
        # First write call should be the SCAN command
        first_write_call = serial_obj.write.call_args_list[0]
        self.assertEqual(first_write_call[0][0], b"\xa5\x20")

    def test_sends_stop_command_after_packets(self):
        """Should write 0xA5 0x25 to stop scan."""
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        descriptor = bytes([0xA5, 0x5A, 0x01, 0x00, 0x00, 0x40, 0x81])
        packet = bytes([0xFF])
        serial_obj.read.side_effect = [descriptor] + [packet] * 10
        rplidar_diag.try_raw_serial(lidar, serial_obj)
        write_calls = serial_obj.write.call_args_list
        stop_written = any(c[0][0] == b"\xa5\x25" for c in write_calls)
        self.assertTrue(stop_written)


# ===========================================================================
# Tests for try_iter_measures_normal
# ===========================================================================

class TestTryIterMeasuresNormal(unittest.TestCase):
    def test_returns_false_when_no_iter_method(self):
        """If neither iter_measurements nor iter_measurments exists, return False."""
        lidar = MagicMock(spec=[])  # no iter methods
        serial_obj = _make_serial_mock()
        result = rplidar_diag.try_iter_measures_normal(lidar, serial_obj)
        self.assertFalse(result)

    def test_uses_iter_measurements_if_available(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        # Provide iter_measurements that yields a few measurements then stops
        measurements = [(True, 15, 90.0, 1000.0)] * 25  # > 20 items
        lidar.iter_measurements = MagicMock(return_value=iter(measurements))
        result = rplidar_diag.try_iter_measures_normal(lidar, serial_obj)
        self.assertTrue(result)

    def test_uses_iter_measurments_typo_fallback(self):
        """Should fall back to iter_measurments (typo) if iter_measurements absent."""
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        measurements = [(False, 10, 45.0, 500.0)] * 25
        lidar.iter_measurments = MagicMock(return_value=iter(measurements))
        result = rplidar_diag.try_iter_measures_normal(lidar, serial_obj)
        self.assertTrue(result)

    def test_calls_with_scan_type_normal(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        measurements = [(True, 15, 90.0, 1000.0)] * 25
        lidar.iter_measurements = MagicMock(return_value=iter(measurements))
        rplidar_diag.try_iter_measures_normal(lidar, serial_obj)
        lidar.iter_measurements.assert_called_once_with(max_buf_meas=500, scan_type="normal")


# ===========================================================================
# Tests for try_iter_measures_express
# ===========================================================================

class TestTryIterMeasuresExpress(unittest.TestCase):
    def test_returns_false_when_no_iter_method(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        result = rplidar_diag.try_iter_measures_express(lidar, serial_obj)
        self.assertFalse(result)

    def test_calls_with_scan_type_express(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        measurements = [(True, 15, 90.0, 1000.0)] * 25
        lidar.iter_measurements = MagicMock(return_value=iter(measurements))
        rplidar_diag.try_iter_measures_express(lidar, serial_obj)
        lidar.iter_measurements.assert_called_once_with(max_buf_meas=500, scan_type="express")

    def test_returns_true_on_success(self):
        lidar = MagicMock(spec=[])
        serial_obj = _make_serial_mock()
        measurements = [(True, 15, 90.0, 1000.0)] * 25
        lidar.iter_measurements = MagicMock(return_value=iter(measurements))
        result = rplidar_diag.try_iter_measures_express(lidar, serial_obj)
        self.assertTrue(result)


# ===========================================================================
# Tests for try_iter_scans
# ===========================================================================

class TestTryIterScans(unittest.TestCase):
    def test_returns_true_after_iterating(self):
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        # Provide at least 3 scans so the loop runs to i >= 2
        scan_data = [[(1, 90.0, 1000.0), (1, 180.0, 2000.0)]] * 5
        lidar.iter_scans = MagicMock(return_value=iter(scan_data))
        result = rplidar_diag.try_iter_scans(lidar, serial_obj)
        self.assertTrue(result)

    def test_calls_iter_scans_with_max_buf(self):
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        scan_data = [[(1, 90.0, 1000.0)]] * 5
        lidar.iter_scans = MagicMock(return_value=iter(scan_data))
        rplidar_diag.try_iter_scans(lidar, serial_obj)
        lidar.iter_scans.assert_called_once_with(max_buf_meas=500)


# ===========================================================================
# Tests for try_iter_scans_min_len_0
# ===========================================================================

class TestTryIterScansMinLen0(unittest.TestCase):
    def test_returns_true(self):
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        scan_data = [[(1, 90.0, 1000.0)]] * 5
        lidar.iter_scans = MagicMock(return_value=iter(scan_data))
        result = rplidar_diag.try_iter_scans_min_len_0(lidar, serial_obj)
        self.assertTrue(result)

    def test_calls_iter_scans_with_min_len_0(self):
        lidar = MagicMock()
        serial_obj = _make_serial_mock()
        scan_data = [[(1, 90.0, 1000.0)]] * 5
        lidar.iter_scans = MagicMock(return_value=iter(scan_data))
        rplidar_diag.try_iter_scans_min_len_0(lidar, serial_obj)
        lidar.iter_scans.assert_called_once_with(max_buf_meas=500, min_len=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
