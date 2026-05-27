"""
Diagnostic + fix script for RPLidar "Wrong body size" on Raspberry Pi 5.
Tries multiple strategies to get scan data working.
"""
import time
import sys
from rplidar import RPLidar

PORT = "/dev/ttyUSB0"
BAUD = 115200


def find_serial(lidar):
    """Find the underlying pyserial object regardless of library version."""
    for attr in ("_serial", "_serial_port", "serial", "ser", "_port"):
        obj = getattr(lidar, attr, None)
        if obj is not None and hasattr(obj, "reset_input_buffer"):
            return obj
    return None


def flush(lidar, serial_obj):
    """Flush any stale bytes from the serial buffer."""
    if hasattr(lidar, "clean_input"):
        lidar.clean_input()
    if serial_obj:
        serial_obj.reset_input_buffer()
        serial_obj.reset_output_buffer()


def try_iter_scans(lidar, serial_obj):
    """Strategy 1: iter_scans with flush."""
    print("\n=== Strategy 1: iter_scans (flush + delay) ===")
    flush(lidar, serial_obj)
    for i, scan in enumerate(lidar.iter_scans(max_buf_meas=500)):
        print(f"  scan #{i} len={len(scan)}, first={scan[0] if scan else None}")
        if i >= 2:
            break
    return True


def try_iter_scans_min_len_0(lidar, serial_obj):
    """Strategy 2: iter_scans with min_len=0 (skip empty-scan errors)."""
    print("\n=== Strategy 2: iter_scans(min_len=0) ===")
    flush(lidar, serial_obj)
    for i, scan in enumerate(lidar.iter_scans(max_buf_meas=500, min_len=0)):
        print(f"  scan #{i} len={len(scan)}, first={scan[0] if scan else None}")
        if i >= 2:
            break
    return True


def try_iter_measures_normal(lidar, serial_obj):
    """Strategy 3: iter_measurments / iter_measurements with scan_type='normal'."""
    print("\n=== Strategy 3: iter_measur(e)ments(scan_type='normal') ===")
    flush(lidar, serial_obj)
    # handle typo in original lib vs fixed spelling in fork
    fn = getattr(lidar, "iter_measurements", None) or getattr(lidar, "iter_measurments", None)
    if fn is None:
        print("  SKIP: no iter_measur(e)ments method found")
        return False
    for i, meas in enumerate(fn(max_buf_meas=500, scan_type="normal")):
        if len(meas) == 4:
            new_scan, quality, angle, distance = meas
        else:
            new_scan, quality, angle, distance = meas[0], meas[1], meas[2], meas[3]
        print(f"  #{i} new_scan={new_scan} quality={quality} angle={angle:.1f} dist={distance:.1f}")
        if i >= 20:
            break
    return True


def try_iter_measures_express(lidar, serial_obj):
    """Strategy 4: express scan mode (some models default to this)."""
    print("\n=== Strategy 4: iter_measur(e)ments(scan_type='express') ===")
    flush(lidar, serial_obj)
    fn = getattr(lidar, "iter_measurements", None) or getattr(lidar, "iter_measurments", None)
    if fn is None:
        print("  SKIP: no iter_measur(e)ments method found")
        return False
    for i, meas in enumerate(fn(max_buf_meas=500, scan_type="express")):
        if len(meas) == 4:
            new_scan, quality, angle, distance = meas
        else:
            new_scan, quality, angle, distance = meas[0], meas[1], meas[2], meas[3]
        print(f"  #{i} new_scan={new_scan} quality={quality} angle={angle:.1f} dist={distance:.1f}")
        if i >= 20:
            break
    return True


def try_raw_serial(lidar, serial_obj):
    """Strategy 5: manually send scan command and read raw bytes (debug)."""
    print("\n=== Strategy 5: raw serial debug ===")
    if serial_obj is None:
        print("  SKIP: could not find serial object")
        return False

    flush(lidar, serial_obj)

    # Send SCAN command: 0xA5 0x20
    serial_obj.write(b"\xa5\x20")
    time.sleep(0.5)

    # Read response descriptor (7 bytes)
    descriptor = serial_obj.read(7)
    print(f"  Descriptor ({len(descriptor)} bytes): {descriptor.hex()}")
    if len(descriptor) < 7:
        print("  ERROR: incomplete descriptor")
        return False

    # Parse: start1=0xA5, start2=0x5A, size(30-bit), send_mode(2-bit), data_type(1 byte)
    start1, start2 = descriptor[0], descriptor[1]
    print(f"  start1=0x{start1:02x}, start2=0x{start2:02x}")
    size = descriptor[2] | (descriptor[3] << 8) | (descriptor[4] << 16) | ((descriptor[5] & 0x3F) << 24)
    send_mode = descriptor[5] >> 6
    data_type = descriptor[6]
    print(f"  data_size={size}, send_mode={send_mode}, data_type={data_type}")

    # Read a few data packets
    for i in range(5):
        data = serial_obj.read(size)
        print(f"  packet #{i} ({len(data)} bytes): {data.hex()}")

    # Send STOP command: 0xA5 0x25
    serial_obj.write(b"\xa5\x25")
    time.sleep(0.1)
    flush(lidar, serial_obj)
    return True


# ─── Main ────────────────────────────────────────────────────────────────
lidar = None
try:
    lidar = RPLidar(PORT, baudrate=BAUD, timeout=3)

    serial_obj = find_serial(lidar)
    print(f"Serial object found: {serial_obj is not None}")

    # Print all public + private attrs for debugging
    print(f"RPLidar attrs: {[a for a in dir(lidar) if not a.startswith('__')]}")

    info = lidar.get_info()
    health = lidar.get_health()
    print(f"Info: {info}")
    print(f"Health: {health}")

    if hasattr(lidar, "start_motor"):
        lidar.start_motor()
        print("Motor started")
    time.sleep(3.0)

    strategies = [
        try_iter_measures_normal,     # most likely to work
        try_iter_scans,
        try_iter_scans_min_len_0,
        try_iter_measures_express,
        try_raw_serial,               # debug fallback
    ]

    for strat in strategies:
        try:
            ok = strat(lidar, serial_obj)
            if ok:
                print(f"\n  >>> SUCCESS with {strat.__name__}")
                break
        except Exception as exc:
            print(f"  >>> FAILED: {exc}")
            # Reset lidar state between attempts
            try:
                lidar.stop()
            except Exception:
                pass
            time.sleep(1.0)
            flush(lidar, serial_obj)
            time.sleep(1.0)
    else:
        print("\n!!! All strategies failed. See raw serial debug above for clues.")

except Exception as exc:
    print(f"FATAL ERROR: {exc}")
    import traceback
    traceback.print_exc()

finally:
    if lidar is not None:
        try:
            lidar.stop()
        except Exception:
            pass
        if hasattr(lidar, "stop_motor"):
            try:
                lidar.stop_motor()
            except Exception:
                pass
        try:
            lidar.disconnect()
        except Exception:
            pass
