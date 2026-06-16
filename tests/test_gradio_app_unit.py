"""
Unit tests for gradio_app.py — covers functions added/changed in this PR.

All heavy dependencies (gradio, torch, PIL, cv2, onnxruntime, openvino,
matplotlib, picamera2, rplidar, src.networks) are replaced with mocks so
the suite runs with only stdlib + numpy.
"""
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np

# ---------------------------------------------------------------------------
# Module-level mock setup — must happen before `import gradio_app`
# ---------------------------------------------------------------------------

def _make_torch_mock():
    """Return a mock that mimics the torch API subset used by gradio_app."""
    torch_mod = MagicMock(name="torch")
    torch_mod.float32 = "float32"  # used in hole_tensor cast

    # torch.from_numpy
    def _from_numpy(arr):
        m = MagicMock(name="from_numpy_result")
        m._arr = arr
        m.numpy.return_value = arr
        m.squeeze.return_value = m
        m.cpu.return_value = m
        m.unsqueeze.return_value = m
        return m

    torch_mod.from_numpy.side_effect = _from_numpy

    # torch.no_grad() — used as context manager
    no_grad_ctx = MagicMock()
    no_grad_ctx.__enter__ = MagicMock(return_value=None)
    no_grad_ctx.__exit__ = MagicMock(return_value=False)
    torch_mod.no_grad.return_value = no_grad_ctx

    # torch.nn.functional.pad
    def _pad(tensor, padding, mode="constant", value=0):
        return tensor  # return unchanged for testing
    torch_mod.nn = MagicMock()
    torch_mod.nn.functional = MagicMock()
    torch_mod.nn.functional.pad.side_effect = _pad

    return torch_mod


def _make_pil_mock():
    """Return a PIL mock that wraps real numpy operations via PIL.Image."""
    try:
        from PIL import Image as _RealImage, ImageDraw as _RealImageDraw
        pil_mod = MagicMock(name="PIL")
        pil_mod.Image = _RealImage
        pil_mod.ImageDraw = _RealImageDraw
        return pil_mod, _RealImage, _RealImageDraw
    except ImportError:
        pil_mod = MagicMock(name="PIL")
        return pil_mod, pil_mod.Image, pil_mod.ImageDraw


# Try to use the real PIL if available; fall back to mock.
try:
    from PIL import Image as _RealImage, ImageDraw as _RealImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _setup_sys_modules():
    mocks = {}

    # torch
    torch_mock = _make_torch_mock()
    mocks["torch"] = torch_mock
    mocks["torch.nn"] = torch_mock.nn
    mocks["torch.nn.functional"] = torch_mock.nn.functional

    # gradio
    gr_mock = MagicMock(name="gradio")
    mocks["gradio"] = gr_mock

    if not _HAS_PIL:
        pil_mock = MagicMock(name="PIL")
        mocks["PIL"] = pil_mock
        mocks["PIL.Image"] = pil_mock.Image
        mocks["PIL.ImageDraw"] = pil_mock.ImageDraw
    # If PIL is real we leave it alone.

    # cv2 — optional in gradio_app, but imported via try/except
    cv2_mock = MagicMock(name="cv2")
    mocks["cv2"] = cv2_mock

    # onnxruntime
    ort_mock = MagicMock(name="onnxruntime")
    mocks["onnxruntime"] = ort_mock

    # openvino
    ov_mock = MagicMock(name="openvino")
    mocks["openvino"] = ov_mock

    # matplotlib
    mpl_mock = MagicMock(name="matplotlib")
    mocks["matplotlib"] = mpl_mock
    mocks["matplotlib.cm"] = mpl_mock.cm

    # picamera2, rplidar
    mocks["picamera2"] = MagicMock(name="picamera2")
    mocks["rplidar"] = MagicMock(name="rplidar")

    # src.networks
    src_mock = types.ModuleType("src")
    src_networks_mock = MagicMock(name="src.networks")
    mocks["src"] = src_mock
    mocks["src.networks"] = src_networks_mock

    for name, mock in mocks.items():
        if name not in sys.modules:
            sys.modules[name] = mock

    return mocks


_MOCKS = _setup_sys_modules()

# Now safe to import
import gradio_app  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_rgb_array(h=64, w=64, c=3):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, c), dtype=np.uint8)


def _make_depth_uint16(h=64, w=64, fill=32000):
    return np.full((h, w), fill, dtype=np.uint16)


# ===========================================================================
# Tests for scan_models
# ===========================================================================

class TestScanModels(unittest.TestCase):
    def test_nonexistent_dir_returns_empty_list(self):
        result = gradio_app.scan_models(Path("/nonexistent/path/xyz"))
        self.assertEqual(result, [])

    def test_returns_only_supported_extensions(self, tmp_path=None):
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create files with various extensions
            (tmp / "model.pth").touch()
            (tmp / "model.onnx").touch()
            (tmp / "model.xml").touch()
            (tmp / "model.bin").touch()    # should be excluded
            (tmp / "readme.txt").touch()   # should be excluded

            result = gradio_app.scan_models(tmp)
            names = [Path(p).name for p in result]
            self.assertIn("model.pth", names)
            self.assertIn("model.onnx", names)
            self.assertIn("model.xml", names)
            self.assertNotIn("model.bin", names)
            self.assertNotIn("readme.txt", names)

    def test_returns_sorted_order(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "z_model.pth").touch()
            (tmp / "a_model.onnx").touch()
            (tmp / "m_model.xml").touch()

            result = gradio_app.scan_models(tmp)
            self.assertEqual(result, sorted(result))

    def test_case_insensitive_extension(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "model.PTH").touch()
            (tmp / "model.ONNX").touch()

            result = gradio_app.scan_models(tmp)
            self.assertEqual(len(result), 2)


# ===========================================================================
# Tests for scan_test_ids
# ===========================================================================

class TestScanTestIds(unittest.TestCase):
    def test_nonexistent_dir_returns_empty_list(self):
        result = gradio_app.scan_test_ids(Path("/nonexistent/path/xyz"))
        self.assertEqual(result, [])

    def test_returns_correct_ids(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rgb_dir = tmp / "Rgb"
            rgb_dir.mkdir()
            (rgb_dir / "001_rgb.png").touch()
            (rgb_dir / "002_rgb.png").touch()
            (rgb_dir / "other_file.png").touch()  # no _rgb suffix → excluded

            result = gradio_app.scan_test_ids(tmp)
            self.assertIn("001", result)
            self.assertIn("002", result)
            self.assertNotIn("other_file", result)

    def test_ids_are_sorted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rgb_dir = tmp / "Rgb"
            rgb_dir.mkdir()
            (rgb_dir / "z10_rgb.png").touch()
            (rgb_dir / "a01_rgb.png").touch()

            result = gradio_app.scan_test_ids(tmp)
            self.assertEqual(result, sorted(result))


# ===========================================================================
# Tests for is_live_capture_supported
# ===========================================================================

class TestIsLiveCaptureSupported(unittest.TestCase):
    def test_returns_false_when_picamera2_none(self):
        orig = gradio_app.Picamera2
        try:
            gradio_app.Picamera2 = None
            self.assertFalse(gradio_app.is_live_capture_supported())
        finally:
            gradio_app.Picamera2 = orig

    def test_returns_false_when_rplidar_none(self):
        orig_p = gradio_app.Picamera2
        orig_r = gradio_app.RPLidar
        try:
            gradio_app.Picamera2 = MagicMock()
            gradio_app.RPLidar = None
            self.assertFalse(gradio_app.is_live_capture_supported())
        finally:
            gradio_app.Picamera2 = orig_p
            gradio_app.RPLidar = orig_r

    def test_returns_false_when_cv2_none(self):
        orig_p = gradio_app.Picamera2
        orig_r = gradio_app.RPLidar
        orig_c = gradio_app.cv2
        try:
            gradio_app.Picamera2 = MagicMock()
            gradio_app.RPLidar = MagicMock()
            gradio_app.cv2 = None
            self.assertFalse(gradio_app.is_live_capture_supported())
        finally:
            gradio_app.Picamera2 = orig_p
            gradio_app.RPLidar = orig_r
            gradio_app.cv2 = orig_c

    def test_returns_true_when_all_present(self):
        orig_p = gradio_app.Picamera2
        orig_r = gradio_app.RPLidar
        orig_c = gradio_app.cv2
        try:
            gradio_app.Picamera2 = MagicMock()
            gradio_app.RPLidar = MagicMock()
            gradio_app.cv2 = MagicMock()
            self.assertTrue(gradio_app.is_live_capture_supported())
        finally:
            gradio_app.Picamera2 = orig_p
            gradio_app.RPLidar = orig_r
            gradio_app.cv2 = orig_c


# ===========================================================================
# Tests for angle_distance_to_lidar_xz  (pure math)
# ===========================================================================

class TestAngleDistanceToLidarXZ(unittest.TestCase):
    def test_zero_angle_points_along_z(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(0.0, 1000.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(z, 1000.0, places=5)

    def test_90_degree_points_along_x(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(90.0, 1000.0)
        self.assertAlmostEqual(x, 1000.0, places=5)
        self.assertAlmostEqual(z, 0.0, places=4)

    def test_180_degree_points_negative_z(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(180.0, 500.0)
        self.assertAlmostEqual(x, 0.0, places=4)
        self.assertAlmostEqual(z, -500.0, places=4)

    def test_45_degree_equal_x_and_z(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(45.0, 1.0)
        self.assertAlmostEqual(x, z, places=10)

    def test_zero_distance_returns_zero(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(45.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=10)
        self.assertAlmostEqual(z, 0.0, places=10)

    def test_negative_distance(self):
        x, z = gradio_app.angle_distance_to_lidar_xz(0.0, -100.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(z, -100.0, places=5)

    def test_pythagorean_relationship(self):
        """Distance should equal sqrt(x^2 + z^2)."""
        dist = 750.0
        for angle in [0.0, 30.0, 60.0, 90.0, 135.0, 270.0]:
            x, z = gradio_app.angle_distance_to_lidar_xz(angle, dist)
            computed = np.sqrt(x ** 2 + z ** 2)
            self.assertAlmostEqual(computed, abs(dist), places=5,
                                   msg=f"Pythagorean check failed at angle={angle}")


# ===========================================================================
# Tests for build_sparse_depth_from_lidar_points
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestBuildSparseDepthFromLidarPoints(unittest.TestCase):
    def test_empty_points_returns_zero_image(self):
        image_shape = (32, 64, 3)
        result = gradio_app.build_sparse_depth_from_lidar_points(image_shape, [])
        arr = np.array(result)
        self.assertTrue((arr == 0).all())
        self.assertEqual(arr.shape, (32, 64))

    def test_valid_point_is_stored(self):
        image_shape = (100, 200, 3)
        visible_points = [(50, 40, 10000.0)]  # (x, y, distance)
        result = gradio_app.build_sparse_depth_from_lidar_points(image_shape, visible_points)
        arr = np.array(result, dtype=np.float32)
        # pixel at (y=40, x=50) should be non-zero
        self.assertGreater(arr[40, 50], 0)

    def test_out_of_bounds_point_ignored(self):
        image_shape = (10, 10, 3)
        visible_points = [(20, 20, 5000.0)]  # outside 10x10
        result = gradio_app.build_sparse_depth_from_lidar_points(image_shape, visible_points)
        arr = np.array(result)
        self.assertTrue((arr == 0).all())

    def test_multiple_points(self):
        image_shape = (50, 50, 3)
        pts = [(0, 0, 1000.0), (25, 25, 2000.0), (49, 49, 3000.0)]
        result = gradio_app.build_sparse_depth_from_lidar_points(image_shape, pts)
        arr = np.array(result, dtype=np.float32)
        self.assertGreater(arr[0, 0], 0)
        self.assertGreater(arr[25, 25], 0)
        self.assertGreater(arr[49, 49], 0)

    def test_distance_normalization(self):
        """Distance stored as distance/65535 scaled back to uint16."""
        image_shape = (10, 10, 3)
        distance = 32767.5
        visible_points = [(5, 5, distance)]
        result = gradio_app.build_sparse_depth_from_lidar_points(image_shape, visible_points)
        arr = np.array(result, dtype=np.float32)
        # stored_uint16 = round(distance / 65535.0 * 65535.0) = round(distance)
        expected = round(distance)
        self.assertAlmostEqual(arr[5, 5], expected, delta=1.0)


# ===========================================================================
# Tests for build_sparse_depth_from_lidar (reads JSON)
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestBuildSparseDepthFromLidar(unittest.TestCase):
    def _write_json(self, tmp_dir, points):
        path = Path(tmp_dir) / "lidar.json"
        data = {"labels": {"image_pixel_points": points}}
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_points_copied_from_gt_depth(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            h, w = 20, 20
            gt = np.zeros((h, w), dtype=np.float32)
            gt[5, 10] = 0.5
            gt[10, 15] = 0.8

            json_path = self._write_json(tmp, [[10, 5], [15, 10]])
            sparse = gradio_app.build_sparse_depth_from_lidar(gt, json_path)

            self.assertAlmostEqual(sparse[5, 10], 0.5, places=5)
            self.assertAlmostEqual(sparse[10, 15], 0.8, places=5)

    def test_out_of_bounds_points_skipped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            h, w = 10, 10
            gt = np.ones((h, w), dtype=np.float32) * 0.5
            json_path = self._write_json(tmp, [[100, 100]])  # out of bounds
            sparse = gradio_app.build_sparse_depth_from_lidar(gt, json_path)
            self.assertTrue((sparse == 0).all())

    def test_pil_image_input(self):
        """gt_depth can be a PIL Image."""
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            h, w = 16, 16
            arr = np.full((h, w), 32768, dtype=np.uint16)
            pil_img = Image.fromarray(arr)
            json_path = self._write_json(tmp, [[8, 8]])
            sparse = gradio_app.build_sparse_depth_from_lidar(pil_img, json_path)
            # pixel [8, 8] should be non-zero
            self.assertGreater(sparse[8, 8], 0.0)

    def test_empty_pixel_points(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            gt = np.ones((8, 8), dtype=np.float32) * 0.3
            path = Path(tmp) / "lidar.json"
            with open(path, "w") as f:
                json.dump({"labels": {}}, f)  # missing "image_pixel_points"
            sparse = gradio_app.build_sparse_depth_from_lidar(gt, path)
            self.assertTrue((sparse == 0).all())

    def test_rgb_channel_extraction(self):
        """3-channel gt_depth uses first channel only."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            h, w = 8, 8
            gt = np.zeros((h, w, 3), dtype=np.float32)
            gt[4, 4, 0] = 0.9
            json_path = self._write_json(tmp, [[4, 4]])
            sparse = gradio_app.build_sparse_depth_from_lidar(gt, json_path)
            self.assertAlmostEqual(sparse[4, 4], 0.9, places=5)


# ===========================================================================
# Tests for read_rgb_image
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestReadRgbImage(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(gradio_app.read_rgb_image(None))

    def test_pil_rgb_returns_numpy_array(self):
        from PIL import Image
        img = Image.fromarray(_make_rgb_array(32, 32))
        result = gradio_app.read_rgb_image(img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[2], 3)

    def test_numpy_array_input(self):
        arr = _make_rgb_array(16, 16)
        result = gradio_app.read_rgb_image(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (16, 16, 3))

    def test_pil_rgba_converted_to_rgb(self):
        from PIL import Image
        rgba = Image.fromarray(np.zeros((10, 10, 4), dtype=np.uint8), mode="RGBA")
        result = gradio_app.read_rgb_image(rgba)
        self.assertEqual(result.shape[2], 3)


# ===========================================================================
# Tests for read_depth_image
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestReadDepthImage(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(gradio_app.read_depth_image(None))

    def test_uint16_normalized_to_0_1(self):
        from PIL import Image
        arr = np.array([[65535, 0]], dtype=np.uint16)
        img = Image.fromarray(arr)
        result = gradio_app.read_depth_image(img)
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(result[0, 1]), 0.0, places=5)

    def test_uint8_normalized_to_0_1(self):
        arr = np.array([[255, 0]], dtype=np.uint8)
        result = gradio_app.read_depth_image(arr)
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(result[0, 1]), 0.0, places=5)

    def test_3d_array_uses_first_channel(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[2, 2, 0] = 128
        result = gradio_app.read_depth_image(arr)
        self.assertEqual(result.ndim, 2)

    def test_large_float_divided_by_65535(self):
        arr = np.array([[32767.5]], dtype=np.float32)
        result = gradio_app.read_depth_image(arr)
        self.assertAlmostEqual(float(result[0, 0]), 32767.5 / 65535.0, places=5)

    def test_already_normalized_float_unchanged(self):
        arr = np.array([[0.5]], dtype=np.float32)
        result = gradio_app.read_depth_image(arr)
        self.assertAlmostEqual(float(result[0, 0]), 0.5, places=5)


# ===========================================================================
# Tests for resize_to_match
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestResizeToMatch(unittest.TestCase):
    def test_same_size_returns_unchanged(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        depth = np.ones((64, 64), dtype=np.float32) * 0.5
        result = gradio_app.resize_to_match(rgb, depth)
        np.testing.assert_array_equal(result, depth)

    def test_different_size_resizes_depth(self):
        rgb = np.zeros((128, 256, 3), dtype=np.uint8)
        depth = np.ones((64, 64), dtype=np.float32) * 0.5
        result = gradio_app.resize_to_match(rgb, depth)
        self.assertEqual(result.shape, (128, 256))

    def test_output_in_float_range(self):
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        depth = np.ones((16, 24), dtype=np.float32) * 0.7
        result = gradio_app.resize_to_match(rgb, depth)
        self.assertTrue(result.max() <= 1.0)
        self.assertTrue(result.min() >= 0.0)


# ===========================================================================
# Tests for resize_for_model
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestResizeForModel(unittest.TestCase):
    def test_output_shape_matches_target(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(100, 150))
        depth = Image.fromarray(_make_depth_uint16(100, 150))
        rgb_out, sparse_out, original_shape = gradio_app.resize_for_model(
            rgb, depth, target_h=64, target_w=128
        )
        self.assertEqual(rgb_out.shape, (64, 128, 3))
        self.assertEqual(sparse_out.shape, (64, 128))

    def test_original_shape_preserved(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(100, 150))
        depth = Image.fromarray(_make_depth_uint16(100, 150))
        _, _, original_shape = gradio_app.resize_for_model(rgb, depth, target_h=64, target_w=128)
        self.assertEqual(original_shape, (100, 150))

    def test_sparse_values_in_range(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(48, 48))
        depth_arr = np.full((48, 48), 32000, dtype=np.uint16)
        depth = Image.fromarray(depth_arr)
        _, sparse_out, _ = gradio_app.resize_for_model(rgb, depth, target_h=48, target_w=48)
        self.assertTrue(sparse_out.max() <= 1.0)
        self.assertTrue(sparse_out.min() >= 0.0)

    def test_numpy_array_inputs_accepted(self):
        rgb = _make_rgb_array(32, 32)
        depth = _make_depth_uint16(32, 32).astype(np.float32) / 65535.0
        rgb_out, sparse_out, original_shape = gradio_app.resize_for_model(
            rgb, depth, target_h=16, target_w=16
        )
        self.assertEqual(rgb_out.shape, (16, 16, 3))
        self.assertEqual(original_shape, (32, 32))


# ===========================================================================
# Tests for pad_to_multiple (uses real torch.nn.functional if available)
# ===========================================================================

class TestPadToMultiple(unittest.TestCase):
    """Tests pad_to_multiple using a real tensor-like mock."""

    def _make_tensor(self, h, w):
        """Return a mock tensor that tracks its shape."""
        t = MagicMock(name=f"tensor_{h}x{w}")
        t.shape = (1, 1, h, w)
        # pad returns a new tensor with modified shape
        def _pad_side_effect(padding, mode="constant", value=0):
            pad_w, pad_h = padding[1], padding[3]
            new_t = MagicMock(name=f"padded_{h+pad_h}x{w+pad_w}")
            new_t.shape = (1, 1, h + pad_h, w + pad_w)
            return new_t
        # We rely on the real F.pad from gradio_app (which is mocked to return tensor)
        return t

    def test_already_multiple_returns_none_pad_info(self):
        """When h and w are already multiples of 64, pad_info should be None."""
        # Use real numpy arrays wrapped as mock tensors
        t = self._make_tensor(128, 192)  # already multiples of 64
        # Patch F so pad is NOT called
        orig_F = gradio_app.F
        try:
            F_mock = MagicMock()
            gradio_app.F = F_mock
            padded, pad_info = gradio_app.pad_to_multiple(t, multiple=64)
            F_mock.pad.assert_not_called()
            self.assertIsNone(pad_info)
        finally:
            gradio_app.F = orig_F

    def test_non_multiple_calls_pad(self):
        t = self._make_tensor(100, 100)  # 100 % 64 != 0
        orig_F = gradio_app.F
        try:
            F_mock = MagicMock()
            padded_result = MagicMock()
            F_mock.pad.return_value = padded_result
            gradio_app.F = F_mock
            padded, pad_info = gradio_app.pad_to_multiple(t, multiple=64)
            F_mock.pad.assert_called_once()
            self.assertIsNotNone(pad_info)
            pad_h, pad_w = pad_info
            self.assertEqual(pad_h, (-100) % 64)
            self.assertEqual(pad_w, (-100) % 64)
        finally:
            gradio_app.F = orig_F

    def test_pad_values_correct(self):
        """Verify computed padding amounts for various sizes."""
        cases = [
            (65, 65, 63, 63),   # pad 63 each
            (64, 63, 0, 1),
            (127, 128, 1, 0),
        ]
        for h, w, expected_ph, expected_pw in cases:
            t = self._make_tensor(h, w)
            orig_F = gradio_app.F
            try:
                F_mock = MagicMock()
                F_mock.pad.return_value = MagicMock()
                gradio_app.F = F_mock
                _, pad_info = gradio_app.pad_to_multiple(t, multiple=64)
                if expected_ph == 0 and expected_pw == 0:
                    self.assertIsNone(pad_info)
                else:
                    self.assertIsNotNone(pad_info)
                    self.assertEqual(pad_info[0], expected_ph)
                    self.assertEqual(pad_info[1], expected_pw)
            finally:
                gradio_app.F = orig_F


# ===========================================================================
# Tests for adjust_domain
# ===========================================================================

class TestAdjustDomain(unittest.TestCase):
    """adjust_domain takes a mock/tensor and bool, returns np.uint16 array."""

    def _make_pred_tensor(self, arr):
        """Wrap a numpy array in a squeeze().cpu().numpy() chain."""
        t = MagicMock()
        t.squeeze.return_value = t
        t.cpu.return_value = t
        t.numpy.return_value = arr.copy()
        return t

    def test_relative_false_clips_and_scales(self):
        arr = np.array([[0.5, 1.0, 0.0]], dtype=np.float32)
        t = self._make_pred_tensor(arr)
        result = gradio_app.adjust_domain(t, relative=False)
        self.assertEqual(result.dtype, np.uint16)
        self.assertAlmostEqual(int(result[0, 0]), int(0.5 * 65535), delta=1)
        self.assertAlmostEqual(int(result[0, 1]), 65535, delta=1)
        self.assertEqual(int(result[0, 2]), 0)

    def test_relative_true_normalizes_to_0_1(self):
        arr = np.array([[1.0, 3.0, 5.0]], dtype=np.float32)
        t = self._make_pred_tensor(arr)
        result = gradio_app.adjust_domain(t, relative=True)
        self.assertEqual(result.dtype, np.uint16)
        self.assertEqual(int(result[0, 0]), 0)        # min → 0
        self.assertEqual(int(result[0, 2]), 65535)    # max → 65535

    def test_relative_true_constant_image_returns_zeros(self):
        arr = np.full((4, 4), 0.5, dtype=np.float32)
        t = self._make_pred_tensor(arr)
        result = gradio_app.adjust_domain(t, relative=True)
        self.assertTrue((result == 0).all())

    def test_values_clamped_above_1(self):
        arr = np.array([[2.0, -1.0]], dtype=np.float32)
        t = self._make_pred_tensor(arr)
        result = gradio_app.adjust_domain(t, relative=False)
        self.assertEqual(int(result[0, 0]), 65535)
        self.assertEqual(int(result[0, 1]), 0)


# ===========================================================================
# Tests for depth_to_colormap
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestDepthToColormap(unittest.TestCase):
    def test_returns_pil_rgb_image(self):
        from PIL import Image
        depth = _make_depth_uint16(16, 16, fill=32000)
        orig_cm = gradio_app.cm
        orig_mpl = gradio_app.matplotlib
        try:
            # Force fallback path (cm is None)
            gradio_app.cm = None
            result = gradio_app.depth_to_colormap(depth)
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.size, (16, 16))
        finally:
            gradio_app.cm = orig_cm
            gradio_app.matplotlib = orig_mpl

    def test_output_size_matches_input(self):
        from PIL import Image
        depth = _make_depth_uint16(32, 48, fill=16000)
        orig_cm = gradio_app.cm
        try:
            gradio_app.cm = None
            result = gradio_app.depth_to_colormap(depth)
            self.assertEqual(result.size, (48, 32))  # PIL size is (w, h)
        finally:
            gradio_app.cm = orig_cm

    def test_zero_depth_returns_black_without_matplotlib(self):
        from PIL import Image
        depth = np.zeros((8, 8), dtype=np.uint16)
        orig_cm = gradio_app.cm
        try:
            gradio_app.cm = None
            result = gradio_app.depth_to_colormap(depth)
            arr = np.array(result)
            self.assertTrue((arr == 0).all())
        finally:
            gradio_app.cm = orig_cm


# ===========================================================================
# Tests for prepare_model_inputs
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestPrepareModelInputs(unittest.TestCase):
    """Uses real numpy/PIL but mock torch."""

    def _assert_tensor_shape(self, mock_tensor, expected_calls_description):
        """Just verify the tensor was constructed (mock chain)."""
        pass  # We check indirectly via return values

    def test_returns_four_values(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(32, 32))
        depth = Image.fromarray(_make_depth_uint16(32, 32, fill=1000))
        result = gradio_app.prepare_model_inputs(rgb, depth)
        self.assertEqual(len(result), 4)

    def test_relative_false_when_depth_nonzero(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(32, 32))
        # depth has non-zero values
        depth = Image.fromarray(_make_depth_uint16(32, 32, fill=10000))

        # We need torch.from_numpy to return something sum-able
        # Patch torch so raw_tensor.sum() returns non-zero float
        import gradio_app as ga
        orig_torch = ga.torch

        class FakeTensor:
            def __init__(self, arr):
                self._arr = arr.copy()
            def unsqueeze(self, dim): return self
            def __gt__(self, val): return FakeTensor(self._arr > val)
            def to(self, dtype): return self
            def sum(self): return 100.0   # non-zero → relative=False
            def permute(self, *args): return self
            def numpy(self): return self._arr
            def squeeze(self): return self
            def cpu(self): return self

        def fake_from_numpy(arr):
            return FakeTensor(arr)

        try:
            ga.torch = MagicMock()
            ga.torch.from_numpy.side_effect = fake_from_numpy
            _, _, _, relative = ga.prepare_model_inputs(rgb, depth)
            self.assertFalse(relative)
        finally:
            ga.torch = orig_torch

    def test_relative_true_when_depth_all_zero(self):
        from PIL import Image
        rgb = Image.fromarray(_make_rgb_array(32, 32))
        depth = Image.fromarray(np.zeros((32, 32), dtype=np.uint16))

        import gradio_app as ga
        orig_torch = ga.torch

        class FakeTensorZero:
            def __init__(self, arr):
                self._arr = arr.copy()
            def unsqueeze(self, dim): return self
            def __gt__(self, val): return FakeTensorZero(self._arr > val)
            def to(self, dtype): return self
            def sum(self): return 0.0   # zero → relative=True
            def permute(self, *args): return self
            def numpy(self): return self._arr
            def squeeze(self): return self
            def cpu(self): return self

        try:
            ga.torch = MagicMock()
            ga.torch.from_numpy.side_effect = lambda a: FakeTensorZero(a)
            _, _, _, relative = ga.prepare_model_inputs(rgb, depth)
            self.assertTrue(relative)
        finally:
            ga.torch = orig_torch


# ===========================================================================
# Tests for infer_pipeline (high-level, no model)
# ===========================================================================

class TestInferPipeline(unittest.TestCase):
    def test_returns_error_when_model_path_empty(self):
        result = gradio_app.infer_pipeline("", None, None)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIn("model", result[2].lower())

    def test_returns_error_when_model_path_none(self):
        result = gradio_app.infer_pipeline(None, None, None)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_returns_error_when_rgb_missing(self):
        result = gradio_app.infer_pipeline("/some/model.onnx", None, MagicMock())
        self.assertIsNone(result[0])

    def test_returns_error_when_depth_missing(self):
        result = gradio_app.infer_pipeline("/some/model.onnx", MagicMock(), None)
        self.assertIsNone(result[0])

    def test_returns_four_values(self):
        result = gradio_app.infer_pipeline("", None, None)
        self.assertEqual(len(result), 4)


# ===========================================================================
# Tests for run_model_inference (no model path cases)
# ===========================================================================

class TestRunModelInference(unittest.TestCase):
    def test_empty_model_path_returns_status(self):
        result = gradio_app.run_model_inference(None, None, "")
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIsInstance(result[2], str)

    def test_none_model_path_returns_status(self):
        result = gradio_app.run_model_inference(None, None, None)
        self.assertIsNone(result[0])

    def test_load_failure_returns_error_message(self):
        orig_load = gradio_app.model_manager.load
        orig_path = gradio_app.model_manager.model_path
        try:
            gradio_app.model_manager.model_path = None  # force reload
            gradio_app.model_manager.load = MagicMock(side_effect=RuntimeError("fail"))
            result = gradio_app.run_model_inference(MagicMock(), MagicMock(), "/fake/model.pth")
            self.assertIsNone(result[0])
            self.assertIn("model", result[2].lower())
        finally:
            gradio_app.model_manager.load = orig_load
            gradio_app.model_manager.model_path = orig_path


# ===========================================================================
# Tests for ModelManager (mocked backends)
# ===========================================================================

class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.mgr = gradio_app.ModelManager()

    def test_load_unsupported_extension_raises(self):
        with self.assertRaises(RuntimeError):
            self.mgr.load("/fake/model.bin")

    def test_load_onnx_raises_when_ort_none(self):
        orig_ort = gradio_app.ort
        try:
            gradio_app.ort = None
            with self.assertRaises(RuntimeError):
                self.mgr.load("/fake/model.onnx")
        finally:
            gradio_app.ort = orig_ort

    def test_load_xml_raises_when_ov_none(self):
        orig_ov = gradio_app.ov
        try:
            gradio_app.ov = None
            with self.assertRaises(RuntimeError):
                self.mgr.load("/fake/model.xml")
        finally:
            gradio_app.ov = orig_ov

    def test_infer_onnx_raises_when_session_none(self):
        self.mgr.model_type = ".onnx"
        self.mgr.session = None
        with self.assertRaises(RuntimeError):
            self.mgr.infer(MagicMock(), MagicMock(), MagicMock())

    def test_infer_xml_raises_when_compiled_none(self):
        self.mgr.model_type = ".xml"
        self.mgr.compiled = None
        with self.assertRaises(RuntimeError):
            self.mgr.infer(MagicMock(), MagicMock(), MagicMock())

    def test_infer_unsupported_type_raises(self):
        self.mgr.model_type = ".bin"
        with self.assertRaises(RuntimeError):
            self.mgr.infer(MagicMock(), MagicMock(), MagicMock())

    def test_load_onnx_success(self):
        orig_ort = gradio_app.ort
        try:
            mock_ort = MagicMock()
            mock_session = MagicMock()
            mock_ort.InferenceSession.return_value = mock_session
            gradio_app.ort = mock_ort
            msg = self.mgr.load("/fake/model.onnx")
            self.assertIn("ONNX", msg)
            self.assertIs(self.mgr.session, mock_session)
        finally:
            gradio_app.ort = orig_ort

    def test_load_xml_success(self):
        orig_ov = gradio_app.ov
        try:
            mock_ov = MagicMock()
            mock_compiled = MagicMock()
            mock_ov.Core.return_value.compile_model.return_value = mock_compiled
            gradio_app.ov = mock_ov
            msg = self.mgr.load("/fake/model.xml")
            self.assertIn("OpenVINO", msg)
            self.assertIs(self.mgr.compiled, mock_compiled)
        finally:
            gradio_app.ov = orig_ov

    def test_load_resets_previous_state(self):
        """Loading a new model resets all backend attributes."""
        self.mgr.model = MagicMock()
        self.mgr.session = MagicMock()
        self.mgr.compiled = MagicMock()

        orig_ort = gradio_app.ort
        try:
            mock_ort = MagicMock()
            mock_ort.InferenceSession.return_value = MagicMock()
            gradio_app.ort = mock_ort
            self.mgr.load("/fake/new_model.onnx")
            self.assertIsNone(self.mgr.model)
            self.assertIsNone(self.mgr.compiled)
        finally:
            gradio_app.ort = orig_ort


# ===========================================================================
# Tests for live_infer_pipeline (boundary/validation)
# ===========================================================================

class TestLiveInferPipeline(unittest.TestCase):
    def test_returns_error_when_model_path_empty(self):
        result = gradio_app.live_infer_pipeline("")
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])
        self.assertIn("model", result[4].lower())

    def test_returns_error_when_live_capture_unsupported(self):
        orig_p = gradio_app.Picamera2
        orig_r = gradio_app.RPLidar
        orig_c = gradio_app.cv2
        try:
            gradio_app.Picamera2 = None
            gradio_app.RPLidar = None
            gradio_app.cv2 = None
            result = gradio_app.live_infer_pipeline("/fake/model.pth")
            self.assertIsNone(result[0])
            self.assertIn("Picamera2", result[4])
        finally:
            gradio_app.Picamera2 = orig_p
            gradio_app.RPLidar = orig_r
            gradio_app.cv2 = orig_c

    def test_returns_six_values(self):
        result = gradio_app.live_infer_pipeline("")
        self.assertEqual(len(result), 6)


# ===========================================================================
# Draw lidar overlay tests
# ===========================================================================

@unittest.skipUnless(_HAS_PIL, "PIL not installed")
class TestDrawLidarOverlay(unittest.TestCase):
    def test_empty_points_returns_copy_of_frame(self):
        from PIL import Image
        frame = _make_rgb_array(32, 32)
        result = gradio_app.draw_lidar_overlay(frame, [])
        self.assertIsInstance(result, Image.Image)
        # Should match original
        np.testing.assert_array_equal(np.array(result), frame)

    def test_single_point_does_not_raise(self):
        frame = _make_rgb_array(100, 100)
        result = gradio_app.draw_lidar_overlay(frame, [(50, 50, 1000.0)])
        self.assertIsNotNone(result)

    def test_multiple_points_all_same_distance_no_crash(self):
        """When all distances are equal, d_range=1.0 — no divide-by-zero."""
        frame = _make_rgb_array(100, 100)
        pts = [(10, 10, 5000.0), (20, 20, 5000.0), (30, 30, 5000.0)]
        result = gradio_app.draw_lidar_overlay(frame, pts)
        self.assertIsNotNone(result)

    def test_color_gradient_close_red_far_blue(self):
        """Closest point (d_min) should tend towards red, farthest towards blue."""
        from PIL import Image
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Two points at very different distances
        close_pt = (20, 20, 100.0)
        far_pt = (80, 80, 10000.0)
        result = gradio_app.draw_lidar_overlay(frame, [close_pt, far_pt])
        arr = np.array(result)
        # Close point area should be more red than blue
        self.assertGreater(int(arr[20, 20, 0]), int(arr[20, 20, 2]))
        # Far point area should be more blue than red
        self.assertGreater(int(arr[80, 80, 2]), int(arr[80, 80, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)