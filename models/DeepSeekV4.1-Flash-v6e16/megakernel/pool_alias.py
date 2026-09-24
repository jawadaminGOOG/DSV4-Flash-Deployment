# Copyright 2026 DeepSeek-V4.1-Flash TPU Deployment Authors.
# Adapted from Inferact/tpu-megakernels/pool_alias.py (https://inferact.ai/blog/tpu-megakernels)
# Apache-2.0 License.
"""Zero-copy VMEM buffer pool aliasing for TPU Pallas megakernels (`Reinterpret(ReshapeTransform)`).

Hooks `jax._src.pallas.mosaic.lowering._reshape_memref` to emit `tpu.reinterpret_cast`
with exact `#tpu.tiled<({rows},128)...>` layouts on a 2D `uint8[pool_bytes // 128, 128]`
VMEM scratch pool (`9.40 MiB` on TPU v6e).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from jax import tree_util
from jax._src.lib import tpu as libtpu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering
from jax._src.state import types
import jax.numpy as jnp
from jax.experimental import pallas as pl


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    pass


_original_reshape_memref = getattr(lowering, "_orig_reshape_memref", lowering._reshape_memref)
lowering._orig_reshape_memref = _original_reshape_memref


def _lower_reinterpret(ref, transform, aval, block_shape):
    if not isinstance(transform, Reinterpret):
        return _original_reshape_memref(ref, transform, aval, block_shape)
    transform._validate_shape(tuple(block_shape))
    original_type = ir.MemRefType(ref.type)
    bits = aval.dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    columns = transform.shape[-1] // 128
    strides = [columns, 1]
    minor_dimension = transform.shape[-2]
    for dimension in reversed(transform.shape[:-2]):
        minor_tiles = (minor_dimension + rows - 1) // rows
        strides.insert(0, strides[0] * minor_tiles)
        minor_dimension = dimension
    inner = f"({packing},1)" if packing > 1 else ""
    stride_str = ",".join(map(str, strides))
    layout = ir.Attribute.parse(f"#tpu.tiled<({rows},128){inner},[{stride_str}]>")
    result_type = ir.MemRefType.get(
        transform.shape,
        original_type.element_type,
        layout=layout,
        memory_space=original_type.memory_space,
    )
    alias = libtpu.reinterpret_cast(result_type, ref)
    erased = ir.MemRefType.get(
        transform.shape,
        original_type.element_type,
        memory_space=original_type.memory_space,
    )
    return libtpu.erase_memref_layout(erased, alias), transform.shape


lowering._reshape_memref = _lower_reinterpret


def storage_size(shape: Sequence[int], dtype: Any) -> int:
    """Computes the exact byte size of `dtype[shape]` in TPU VMEM `(rows, 128)` tiles."""
    dt = jnp.dtype(dtype)
    bits = dt.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return int(math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096)


def view(ref, shape: Sequence[int], dtype: Any):
    """Reinterprets a 2D `uint8[N, 128]` VMEM `Ref` slice into `dtype[shape]` via `tpu.reinterpret_cast`."""
    dt = jnp.dtype(dtype)
    bitcast_ref = ref.bitcast(dt)
    return types.TransformedRef(
        bitcast_ref.ref,
        (*bitcast_ref.transforms, Reinterpret(tuple(shape))),
    )


class VmemPoolAllocator:
    """Sub-allocates aligned typed VMEM `TransformedRef` views from `uint8[pool_bytes // 128, 128]`."""

    def __init__(self, pool_ref):
        self.pool_ref = pool_ref
        # Support either 2D `(rows, 128)` uint8 or 1D `uint32` dummy ref
        if len(pool_ref.shape) == 2:
            self.pool_bytes = pool_ref.shape[0] * pool_ref.shape[1]
        else:
            self.pool_bytes = pool_ref.shape[0] * 4
        self.cursor_bytes = 0
        self.peak_bytes = 0

    def reset(self) -> None:
        self.cursor_bytes = 0

    def alloc(self, shape: Sequence[int], dtype: Any):
        dt = jnp.dtype(dtype)
        n_bytes = storage_size(shape, dt)
        start_bytes = ((self.cursor_bytes + 4095) // 4096) * 4096
        end_bytes = start_bytes + n_bytes
        if end_bytes > self.pool_bytes:
            raise ValueError(
                f"VMEM Pool overflow: requested {n_bytes / 1024:.1f} KiB "
                f"at offset {start_bytes / 1024:.1f} KiB, exceeding pool size "
                f"{self.pool_bytes / (1024 * 1024):.2f} MiB."
            )
        self.cursor_bytes = end_bytes
        self.peak_bytes = max(self.peak_bytes, end_bytes)
        if len(self.pool_ref.shape) == 2:
            sub_ref = self.pool_ref.at[pl.ds(start_bytes // 128, n_bytes // 128)]
            return view(sub_ref, shape, dt)
        return None
