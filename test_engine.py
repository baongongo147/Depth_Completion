from multiprocessing import context
import os
import cv2
from httpcore import stream
import numpy as np
import argparse
import time
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    raise ImportError(
        "TensorRT inference requires both 'tensorrt' and 'pycuda' packages. "
        f"Install them in your NVIDIA GPU environment. Original error: {e}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference using a TensorRT .engine model for G2-MonoDepth"
    )
    parser.add_argument(
        "--rgbd_dir",
        type=Path,
        default=Path("Test_Datasets/Private_Test"),
        help="Path to RGBD dataset folder",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default="checkpoints/models/g2_monodepth_epoch_30.engine",
        help="Path to TensorRT engine file",
    )
    return parser.parse_args()


# Ảnh Target camera gốc: 640x480 (WxH) -> Padding lên 640x512
ORIG_H, ORIG_W = 480, 640
PAD_BOTTOM = 32   # 512 - 480 = 32
PAD_RIGHT = 0    # 640 - 640 = 0


def prepare_input(rgb_path, raw_path):
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (ORIG_W, ORIG_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.copyMakeBorder(
        rgb,
        top=0,
        bottom=PAD_BOTTOM,
        left=0,
        right=PAD_RIGHT,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    rgb = rgb.astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    raw = cv2.resize(raw, (ORIG_W, ORIG_H), interpolation=cv2.INTER_NEAREST)
    raw = cv2.copyMakeBorder(
        raw,
        top=0,
        bottom=PAD_BOTTOM,
        left=0,
        right=PAD_RIGHT,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    raw = (raw.astype(np.float32) / 65535.0)
    raw = raw[np.newaxis, np.newaxis, ...]

    hole = np.where(raw == 0, 0, 1).astype(np.float32)
    return rgb, raw, hole


def adjust_domain_numpy(pred):
    pred = np.clip(pred, 0, 1)
    pred = (pred * 65535.0).astype(np.uint16)
    return pred[0, 0, :, :]


def calc_rmse(depth, ground_truth):
    residual = ((depth - ground_truth) / 256.0) ** 2
    residual[ground_truth == 0.0] = 0.0
    valid_pixels = np.count_nonzero(ground_truth)
    if valid_pixels == 0:
        return 0.0
    return np.sqrt(np.sum(residual) / valid_pixels)


def calc_absrel(depth, ground_truth):
    diff = depth - ground_truth
    diff[ground_truth == 0.0] = 0.0
    valid_pixels = np.count_nonzero(ground_truth)
    if valid_pixels == 0:
        return 0.0
    return np.sum(np.abs(diff) / (ground_truth + 1e-6)) / valid_pixels


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Không thể load TensorRT engine từ {engine_path}")
    return engine


# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()

#     for binding_idx in range(engine.num_bindings):
#         name = engine.get_binding_name(binding_idx)
#         dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
#         shape = engine.get_binding_shape(binding_idx)
#         if any(dim < 0 for dim in shape):
#             shape = tuple(1 if dim < 0 else dim for dim in shape)
#         size = trt.volume(shape)

#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)

#         bindings.append(int(device_mem))
#         tensor = {
#             "name": name,
#             "dtype": dtype,
#             "shape": shape,
#             "host": host_mem,
#             "device": device_mem,
#         }

#         if engine.binding_is_input(binding_idx):
#             inputs.append(tensor)
#         else:
#             outputs.append(tensor)

#     return inputs, outputs, bindings, stream

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(
            engine.get_tensor_dtype(name)
        )

        shape = engine.get_tensor_shape(name)

        size = trt.volume(shape)

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        tensor = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "host": host_mem,
            "device": device_mem,
        }

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(tensor)
        else:
            outputs.append(tensor)

    return inputs, outputs, bindings, stream

def infer_with_engine(context, inputs, outputs, bindings, stream, input_arrays):
    for tensor, array in zip(inputs, input_arrays):
        array = np.ascontiguousarray(array.astype(np.float32))
        expected_size = int(np.prod(tensor["shape"]))
        flat_host = np.frombuffer(tensor["host"], dtype=tensor["dtype"])
        if flat_host.size != array.size:
            raise ValueError(
                f"Input shape mismatch for '{tensor['name']}': "
                f"expected {flat_host.size}, got {array.size}"
            )
        flat_host[:] = array.ravel()
        cuda.memcpy_htod_async(tensor["device"], tensor["host"], stream)

    # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    for tensor in inputs:
        context.set_tensor_address(
            tensor["name"],
            int(tensor["device"])
        )

    for tensor in outputs:
        context.set_tensor_address(
            tensor["name"],
            int(tensor["device"])
        )

    context.execute_async_v3(
        stream_handle=stream.handle
    )

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()

    results = []
    for out in outputs:
        result = np.frombuffer(out["host"], dtype=out["dtype"]).reshape(out["shape"])
        results.append(result.copy())
    return results


def run_inference():
    args = parse_arguments()
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"Không tìm thấy engine file: {engine_path}")

    engine = load_engine(str(engine_path))
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    rgb_dir = args.rgbd_dir / "rgb"
    raw_dir = args.rgbd_dir / "raw_1%"
    gt_dir = args.rgbd_dir / "gt"
    save_dir = args.rgbd_dir / f"result_ENGINE_{engine_path.stem}"
    os.makedirs(save_dir, exist_ok=True)

    total_rmse = 0.0
    total_absrel = 0.0
    valid_count = 0

    files = list(rgb_dir.rglob("*.png"))
    print(f"Bắt đầu xử lý {len(files)} ảnh bằng TensorRT engine: {engine_path.name}...")

    start_total = time.time()
    for i, file_path in enumerate(files):
        rel_path = file_path.relative_to(rgb_dir)
        raw_path = raw_dir / rel_path
        save_path = save_dir / rel_path
        os.makedirs(save_path.parent, exist_ok=True)

        input_rgb, input_raw, input_hole = prepare_input(file_path, raw_path)
        output_arrays = infer_with_engine(
            context,
            inputs,
            outputs,
            bindings,
            stream,
            [input_rgb, input_raw, input_hole],
        )

        result = adjust_domain_numpy(output_arrays[0])
        result = result[0:ORIG_H, 0:ORIG_W]
        cv2.imwrite(str(save_path), result)

        gt_path = gt_dir / rel_path
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            if gt is not None:
                gt = cv2.resize(gt, (ORIG_W, ORIG_H), interpolation=cv2.INTER_NEAREST)
                gt = gt.astype(np.float32)
                pred_eval = result.astype(np.float32)
                total_rmse += calc_rmse(pred_eval, gt)
                total_absrel += calc_absrel(pred_eval, gt)
                valid_count += 1

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(files)}] - Đã infer {i+1} ảnh")

    elapsed = time.time() - start_total
    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng thời gian: {elapsed:.2f}s")
    print(f"Trung bình: {elapsed / max(1, len(files)):.4f}s/ảnh")
    if valid_count > 0:
        print(f"\n--- KẾT QUẢ ĐÁNH GIÁ (trên {valid_count} ảnh) ---")
        print(f"RMSE   = {total_rmse / valid_count:.4f}")
        print(f"AbsRel = {total_absrel / valid_count:.4f}")


if __name__ == "__main__":
    run_inference()
