#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch


def _import_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is required to build or smoke-test an engine.") from exc
    return trt


def _dtype_to_torch(trt, dtype) -> torch.dtype:
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if hasattr(trt, "bfloat16") and dtype == trt.bfloat16:
        return torch.bfloat16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def build_engine(onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_gb: float) -> None:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.INFO)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))
    using_fp16 = bool(fp16 and builder.platform_has_fast_fp16)
    if using_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        print("FP16 requested but platform_has_fast_fp16 is false; building default precision.", flush=True)

    print(
        json.dumps(
            {
                "onnx": str(onnx_path),
                "engine": str(engine_path),
                "num_inputs": network.num_inputs,
                "num_outputs": network.num_outputs,
                "fp16": using_fp16,
                "workspace_gb": workspace_gb,
            },
            indent=2,
        ),
        flush=True,
    )
    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Saved TensorRT engine to {engine_path} ({engine_path.stat().st_size / (1024**3):.2f} GiB)")
    print(f"Build seconds: {time.time() - start:.1f}", flush=True)


def smoke_engine(engine_path: Path) -> None:
    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    torch.cuda.init()
    stream = torch.cuda.current_stream()
    tensors: dict[str, torch.Tensor] = {}
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        torch_dtype = _dtype_to_torch(trt, dtype)

        if mode == trt.TensorIOMode.INPUT and any(dim < 0 for dim in shape):
            raise RuntimeError(f"Dynamic input shape is not set by this smoke script: {name} {shape}")
        if any(dim < 0 for dim in shape):
            shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved tensor shape for {name}: {shape}")

        if mode == trt.TensorIOMode.INPUT and torch_dtype == torch.bool:
            tensor = torch.ones(shape, device="cuda", dtype=torch_dtype)
        else:
            tensor = torch.zeros(shape, device="cuda", dtype=torch_dtype)
        if mode == trt.TensorIOMode.OUTPUT:
            tensor = torch.empty(shape, device="cuda", dtype=torch_dtype)
        tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))
        io.append({"name": name, "mode": str(mode), "shape": list(shape), "dtype": str(dtype)})

    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 returned false")
    stream.synchronize()

    outputs = {}
    for name, tensor in tensors.items():
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        out = tensor.detach().float()
        outputs[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "mean": float(out.mean().item()),
            "std": float(out.std().item()),
            "min": float(out.min().item()),
            "max": float(out.max().item()),
            "has_nan": bool(torch.isnan(out).any().item()),
        }
    print(json.dumps({"io": io, "outputs": outputs}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and smoke-test a TensorRT engine from an exported ONNX file.")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace-gb", type=float, default=20.0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    if not args.skip_build and args.onnx is None:
        parser.error("--onnx is required unless --skip-build is set")

    if not args.skip_build:
        build_engine(args.onnx, args.engine, fp16=not args.no_fp16, workspace_gb=args.workspace_gb)
    if not args.skip_smoke:
        smoke_engine(args.engine)


if __name__ == "__main__":
    main()
