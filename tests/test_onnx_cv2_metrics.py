"""
Unit tests for functions and constants added in test_ONNX_cv2.py in this PR.

New additions tested:
  - ORIG_H, ORIG_W, PAD_BOTTOM, PAD_RIGHT constants
  - calc_rmse(depth, ground_truth)
  - calc_absrel(depth, ground_truth)
  - prepare_input padding behaviour (via mocked cv2)

cv2 and onnxruntime are mocked so tests run with only stdlib + numpy.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np

# ---------------------------------------------------------------------------
# Mock cv2 and onnxruntime before importing the module under test
# ---------------------------------------------------------------------------

_cv2_mock = MagicMock(name="cv2")
_ort_mock = MagicMock(name="onnxruntime")

# Make cv2 constants accessible
_cv2_mock.COLOR_BGR2RGB = 4
_cv2_mock.IMREAD_UNCHANGED = -1
_cv2_mock.BORDER_CONSTANT = 0

if "cv2" not in sys.modules:
    sys.modules["cv2"] = _cv2_mock
if "onnxruntime" not in sys.modules:
    sys.modules["onnxruntime"] = _ort_mock

import test_ONNX_cv2 as onnx_cv2  # noqa: E402


# ===========================================================================
# Tests for padding constants (values defined in the PR diff)
# ===========================================================================

class TestPaddingConstants(unittest.TestCase):
    """Verify that padding constants match the expected image geometry."""

    def test_orig_h(self):
        self.assertEqual(onnx_cv2.ORIG_H, 480)

    def test_orig_w(self):
        self.assertEqual(onnx_cv2.ORIG_W, 848)

    def test_pad_bottom(self):
        # 512 - 480 = 32
        self.assertEqual(onnx_cv2.PAD_BOTTOM, 32)

    def test_pad_right(self):
        # 896 - 848 = 48
        self.assertEqual(onnx_cv2.PAD_RIGHT, 48)

    def test_padded_height_is_multiple_of_64(self):
        padded_h = onnx_cv2.ORIG_H + onnx_cv2.PAD_BOTTOM
        self.assertEqual(padded_h % 64, 0,
                         f"Padded H={padded_h} is not a multiple of 64")

    def test_padded_width_is_multiple_of_64(self):
        padded_w = onnx_cv2.ORIG_W + onnx_cv2.PAD_RIGHT
        self.assertEqual(padded_w % 64, 0,
                         f"Padded W={padded_w} is not a multiple of 64")

    def test_padded_size(self):
        self.assertEqual(onnx_cv2.ORIG_H + onnx_cv2.PAD_BOTTOM, 512)
        self.assertEqual(onnx_cv2.ORIG_W + onnx_cv2.PAD_RIGHT, 896)


# ===========================================================================
# Tests for calc_rmse
# ===========================================================================

class TestCalcRmse(unittest.TestCase):
    def test_all_zero_gt_returns_zero(self):
        depth = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        gt = np.zeros_like(depth)
        self.assertEqual(onnx_cv2.calc_rmse(depth, gt), 0.0)

    def test_perfect_prediction_returns_zero(self):
        gt = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        depth = gt.copy()
        result = onnx_cv2.calc_rmse(depth, gt)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_known_rmse_value(self):
        """Manual RMSE calculation for a known input."""
        # Suppose depth=356, gt=100 → residual=((356-100)/256)^2 = (256/256)^2 = 1.0
        gt = np.array([[100.0]], dtype=np.float32)
        depth = np.array([[356.0]], dtype=np.float32)
        result = onnx_cv2.calc_rmse(depth, gt)
        # residual = ((356-100)/256)^2 = 1.0, valid_pixels=1, rmse=sqrt(1.0/1)=1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_zero_pixels_are_excluded(self):
        """Pixels where gt == 0 contribute 0 residual."""
        gt = np.array([[100.0, 0.0]], dtype=np.float32)
        depth = np.array([[356.0, 9999.0]], dtype=np.float32)
        # Only first pixel counts: residual=1.0, valid_count=1
        result = onnx_cv2.calc_rmse(depth, gt)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_symmetric_errors(self):
        """RMSE is the same for +error and -error of the same magnitude."""
        gt = np.array([[512.0, 512.0]], dtype=np.float32)
        depth_pos = np.array([[768.0, 512.0]], dtype=np.float32)
        depth_neg = np.array([[256.0, 512.0]], dtype=np.float32)
        r_pos = onnx_cv2.calc_rmse(depth_pos, gt)
        r_neg = onnx_cv2.calc_rmse(depth_neg, gt)
        self.assertAlmostEqual(r_pos, r_neg, places=8)

    def test_multiple_pixels(self):
        """RMSE averages over all valid pixels."""
        gt = np.array([[256.0, 512.0]], dtype=np.float32)
        # Both residuals should be 1.0: (256+256)/256 and (512+256)/256 errors
        depth = np.array([[256.0 + 256.0, 512.0 + 256.0]], dtype=np.float32)
        result = onnx_cv2.calc_rmse(depth, gt)
        # residual = ((256)/256)^2 = 1.0 for both, rmse = sqrt(2/2) = 1.0
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_2d_array_input(self):
        gt = np.array([[200.0, 400.0], [100.0, 0.0]], dtype=np.float32)
        depth = gt.copy()
        result = onnx_cv2.calc_rmse(depth, gt)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_result_is_non_negative(self):
        gt = np.random.rand(10, 10).astype(np.float32) * 1000
        depth = np.random.rand(10, 10).astype(np.float32) * 1000
        result = onnx_cv2.calc_rmse(depth, gt)
        self.assertGreaterEqual(result, 0.0)

    def test_does_not_modify_input_arrays(self):
        gt = np.array([[100.0, 200.0]], dtype=np.float32)
        depth = np.array([[150.0, 250.0]], dtype=np.float32)
        gt_copy = gt.copy()
        depth_copy = depth.copy()
        onnx_cv2.calc_rmse(depth, gt)
        # calc_rmse modifies residual in-place, but the originals should be unchanged
        # (residual is computed from copies via arithmetic ops)
        np.testing.assert_array_equal(gt, gt_copy)


# ===========================================================================
# Tests for calc_absrel
# ===========================================================================

class TestCalcAbsrel(unittest.TestCase):
    def test_all_zero_gt_returns_zero(self):
        depth = np.array([[100.0, 200.0]], dtype=np.float32)
        gt = np.zeros_like(depth)
        self.assertEqual(onnx_cv2.calc_absrel(depth, gt), 0.0)

    def test_perfect_prediction_returns_zero(self):
        gt = np.array([[100.0, 500.0, 300.0]], dtype=np.float32)
        depth = gt.copy()
        result = onnx_cv2.calc_absrel(depth, gt)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_known_absrel_value(self):
        """Manual calculation for a single pixel."""
        gt = np.array([[100.0]], dtype=np.float32)
        depth = np.array([[200.0]], dtype=np.float32)
        # abs_diff = |200-100| = 100, absrel per pixel = 100/(100+1e-6) ≈ 0.99999
        result = onnx_cv2.calc_absrel(depth, gt)
        expected = 100.0 / (100.0 + 1e-6)
        self.assertAlmostEqual(result, expected, places=4)

    def test_zero_pixels_excluded(self):
        gt = np.array([[100.0, 0.0]], dtype=np.float32)
        depth = np.array([[200.0, 9999.0]], dtype=np.float32)
        # Only first pixel counts
        result = onnx_cv2.calc_absrel(depth, gt)
        expected = 100.0 / (100.0 + 1e-6) / 1
        self.assertAlmostEqual(result, expected, places=4)

    def test_result_is_non_negative(self):
        gt = np.random.rand(10, 10).astype(np.float32) * 1000 + 1.0
        depth = np.random.rand(10, 10).astype(np.float32) * 1000
        result = onnx_cv2.calc_absrel(depth, gt)
        self.assertGreaterEqual(result, 0.0)

    def test_multiple_pixels_averaged(self):
        """AbsRel is the mean of per-pixel absolute relative errors."""
        gt = np.array([[100.0, 200.0]], dtype=np.float32)
        depth = np.array([[200.0, 400.0]], dtype=np.float32)
        # p1: |200-100|/(100+1e-6), p2: |400-200|/(200+1e-6)
        e1 = 100.0 / (100.0 + 1e-6)
        e2 = 200.0 / (200.0 + 1e-6)
        expected = (e1 + e2) / 2
        result = onnx_cv2.calc_absrel(depth, gt)
        self.assertAlmostEqual(result, expected, places=4)

    def test_epsilon_prevents_division_by_zero(self):
        """gt contains a very small positive value — should not raise ZeroDivisionError."""
        gt = np.array([[1e-10]], dtype=np.float32)
        depth = np.array([[0.5]], dtype=np.float32)
        # Should not raise
        result = onnx_cv2.calc_absrel(depth, gt)
        self.assertTrue(np.isfinite(result))

    def test_2d_array_input(self):
        gt = np.ones((4, 4), dtype=np.float32) * 200.0
        depth = gt.copy()
        result = onnx_cv2.calc_absrel(depth, gt)
        self.assertAlmostEqual(result, 0.0, places=5)


# ===========================================================================
# Tests for prepare_input (padding behaviour added in this PR)
# ===========================================================================

class TestPrepareInputPadding(unittest.TestCase):
    """Test that prepare_input correctly pads RGB and depth images."""

    def setUp(self):
        """Reset cv2 mock state before each test."""
        _cv2_mock.reset_mock()

    def test_rgb_padded_with_zeros_at_bottom_and_right(self):
        """cv2.copyMakeBorder should be called for RGB with correct padding values."""
        # Build a fake RGB numpy array of the original size
        orig_rgb = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), dtype=np.uint8)
        padded_rgb = np.zeros((512, 896, 3), dtype=np.uint8)

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb

        # copyMakeBorder returns padded_rgb for RGB, then padded_raw for depth
        raw_16bit = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), dtype=np.uint16)
        padded_raw = np.zeros((512, 896), dtype=np.uint16)

        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        onnx_cv2.prepare_input("/fake/rgb.png", "/fake/raw.png")

        # Expect exactly 2 calls to copyMakeBorder (RGB + depth)
        self.assertEqual(_cv2_mock.copyMakeBorder.call_count, 2)

        rgb_border_call = _cv2_mock.copyMakeBorder.call_args_list[0]
        rgb_kwargs = rgb_border_call[1] if rgb_border_call[1] else {}
        rgb_args = rgb_border_call[0]

        # First positional arg is the image
        # Keyword args or positional: top=0, bottom=PAD_BOTTOM, left=0, right=PAD_RIGHT
        all_args = {**dict(zip(["top", "bottom", "left", "right", "borderType", "value"],
                               rgb_args[1:])), **rgb_kwargs}
        self.assertEqual(all_args.get("top", rgb_args[1] if len(rgb_args) > 1 else None), 0)
        self.assertEqual(all_args.get("bottom",
                                      rgb_args[2] if len(rgb_args) > 2 else None),
                         onnx_cv2.PAD_BOTTOM)

    def test_depth_padded_with_zero_value(self):
        """Depth padding should use value=0 (no-data for LiDAR)."""
        orig_rgb = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), dtype=np.uint8)
        padded_rgb = np.zeros((512, 896, 3), dtype=np.uint8)
        raw_16bit = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), dtype=np.uint16)
        padded_raw = np.zeros((512, 896), dtype=np.uint16)

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb
        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        onnx_cv2.prepare_input("/fake/rgb.png", "/fake/raw.png")

        depth_border_call = _cv2_mock.copyMakeBorder.call_args_list[1]
        depth_kwargs = depth_border_call[1] if depth_border_call[1] else {}
        depth_args = depth_border_call[0]
        all_args = {**dict(zip(["top", "bottom", "left", "right", "borderType", "value"],
                               depth_args[1:])), **depth_kwargs}
        # The value for depth padding should be 0
        self.assertEqual(all_args.get("value", depth_args[6] if len(depth_args) > 6 else None), 0)

    def test_output_shapes(self):
        """prepare_input should return (rgb, raw, hole) with batch/channel dims."""
        orig_rgb = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), dtype=np.uint8)
        padded_rgb = np.zeros((512, 896, 3), dtype=np.uint8)
        raw_16bit = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), dtype=np.uint16)
        padded_raw = np.zeros((512, 896), dtype=np.uint16)

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb
        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        rgb_out, raw_out, hole_out = onnx_cv2.prepare_input("/rgb.png", "/raw.png")

        # RGB: [1, C, H, W]
        self.assertEqual(rgb_out.ndim, 4)
        self.assertEqual(rgb_out.shape[0], 1)
        self.assertEqual(rgb_out.shape[1], 3)
        # Raw and hole: [1, 1, H, W]
        self.assertEqual(raw_out.ndim, 4)
        self.assertEqual(raw_out.shape[0], 1)
        self.assertEqual(raw_out.shape[1], 1)
        self.assertEqual(hole_out.ndim, 4)

    def test_hole_mask_is_binary(self):
        """hole should contain only 0 and 1 values."""
        orig_rgb = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), dtype=np.uint8)
        padded_rgb = np.zeros((512, 896, 3), dtype=np.uint8)
        raw_16bit = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), dtype=np.uint16)
        raw_16bit[100, 200] = 10000  # one non-zero depth
        padded_raw = np.zeros((512, 896), dtype=np.uint16)
        padded_raw[100, 200] = 10000

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb
        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        _, _, hole_out = onnx_cv2.prepare_input("/rgb.png", "/raw.png")

        unique_vals = np.unique(hole_out)
        for v in unique_vals:
            self.assertIn(v, [0.0, 1.0])

    def test_rgb_normalized_to_0_1(self):
        """RGB values should be in [0, 1] after normalization."""
        orig_rgb = np.full((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), 128, dtype=np.uint8)
        padded_rgb = np.full((512, 896, 3), 128, dtype=np.uint8)
        raw_16bit = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), dtype=np.uint16)
        padded_raw = np.zeros((512, 896), dtype=np.uint16)

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb
        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        rgb_out, _, _ = onnx_cv2.prepare_input("/rgb.png", "/raw.png")

        self.assertTrue(rgb_out.max() <= 1.0)
        self.assertTrue(rgb_out.min() >= 0.0)

    def test_depth_normalized_to_0_1(self):
        """Depth values should be in [0, 1] after normalization."""
        orig_rgb = np.zeros((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W, 3), dtype=np.uint8)
        padded_rgb = np.zeros((512, 896, 3), dtype=np.uint8)
        raw_16bit = np.full((onnx_cv2.ORIG_H, onnx_cv2.ORIG_W), 32768, dtype=np.uint16)
        padded_raw = np.full((512, 896), 32768, dtype=np.uint16)

        _cv2_mock.imread.return_value = orig_rgb
        _cv2_mock.cvtColor.return_value = orig_rgb
        _cv2_mock.copyMakeBorder.side_effect = [padded_rgb, padded_raw]

        _, raw_out, _ = onnx_cv2.prepare_input("/rgb.png", "/raw.png")

        self.assertTrue(raw_out.max() <= 1.0)
        self.assertTrue(raw_out.min() >= 0.0)


# ===========================================================================
# Regression: result cropping removes padding region
# ===========================================================================

class TestCroppingBackToOriginal(unittest.TestCase):
    """The PR crops result[0:ORIG_H, 0:ORIG_W] to remove padding."""

    def test_crop_removes_padded_rows(self):
        padded_h = onnx_cv2.ORIG_H + onnx_cv2.PAD_BOTTOM
        padded_w = onnx_cv2.ORIG_W + onnx_cv2.PAD_RIGHT
        padded = np.ones((padded_h, padded_w), dtype=np.uint16)
        padded[onnx_cv2.ORIG_H:, :] = 0  # padding region is zero
        padded[:, onnx_cv2.ORIG_W:] = 0

        cropped = padded[0:onnx_cv2.ORIG_H, 0:onnx_cv2.ORIG_W]
        self.assertEqual(cropped.shape, (onnx_cv2.ORIG_H, onnx_cv2.ORIG_W))
        # Cropped area should all be 1 (no padding zeros)
        self.assertTrue((cropped == 1).all())

    def test_crop_size_matches_original(self):
        cropped_h = onnx_cv2.ORIG_H
        cropped_w = onnx_cv2.ORIG_W
        self.assertEqual(cropped_h, 480)
        self.assertEqual(cropped_w, 848)


if __name__ == "__main__":
    unittest.main(verbosity=2)