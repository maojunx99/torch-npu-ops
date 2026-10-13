# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compile and check the qualified entries through the existing AOT pytest flow."""
import pytest
import torch
import torch_npu

from joyo_v2 import (
    joyo_attention_finish_b64,
    joyo_ec_sum_reduce_f32_e48_b256_c48,
    joyo_expert_swiglu_d4096,
    joyo_moe_post_rms_d6144,
    joyo_qk_half_prepare_b32_0_1,
    joyo_rms_gate_residual_d6144,
    joyo_rms_modulate_d6144,
    joyo_rms_sandwich_d6144,
)

OPTIONS = {"enable_fp_fusion": False, "multibuffer": False}


@pytest.fixture(autouse=True)
def _require_npu() -> None:
    if not torch.npu.is_available():
        pytest.skip("NPU is required for the existing Triton AOT build flow")
    torch.manual_seed(12345)


def _value(*shape: int) -> torch.Tensor:
    return torch.randn(shape, device="npu", dtype=torch.bfloat16)


def _norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_rms_norm(value, weight, epsilon=1e-6)[0]


@pytest.mark.parametrize("rows", [1, 49, 12915])
def test_rms_entries(rows: int) -> None:
    x, residual = _value(rows, 6144), _value(rows, 6144)
    weight, next_weight = _value(6144), _value(6144)
    tables = _value(2, 6, 6144)
    gate, scale, shift = tables[:, 0], tables[:, 2], tables[:, 4]
    ids = (torch.arange(rows, device="npu", dtype=torch.int64) % 2).contiguous()
    h, y = torch.empty_like(x), torch.empty_like(x)
    cores = min(rows, 48)
    common = [
        x,
        residual,
        weight,
        next_weight,
        gate,
        scale,
        shift,
        ids,
        h,
        y,
        rows,
        gate.stride(0),
        scale.stride(0),
        shift.stride(0),
        cores,
    ]

    affine = common.copy()
    affine[4:7] = [scale, shift, scale]
    joyo_rms_modulate_d6144[(cores,)](*affine, **OPTIONS)
    expected = _norm(x, weight) * (1 + scale[ids]) + shift[ids]
    torch.testing.assert_close(y, expected, atol=0.03, rtol=0.03)

    joyo_rms_gate_residual_d6144[(cores,)](*common, **OPTIONS)
    expected_h = residual + _norm(x, weight) * gate[ids]
    torch.testing.assert_close(y, expected_h, atol=0.03, rtol=0.03)

    joyo_rms_sandwich_d6144[(cores,)](*common, **OPTIONS)
    expected_y = _norm(expected_h, next_weight) * (1 + scale[ids]) + shift[ids]
    torch.testing.assert_close(h, expected_h, atol=0.03, rtol=0.03)
    torch.testing.assert_close(y, expected_y, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("rows", [1, 49, 12915])
def test_ec_sum(rows: int) -> None:
    capacity = min(rows, max(1, (rows + 7) // 8))
    ids = torch.stack(
        [torch.randperm(rows, device="npu")[:capacity] for _ in range(48)]
    )
    weights = torch.rand((48, capacity), device="npu", dtype=torch.float32)
    dense = torch.full((48, rows), -1.0, device="npu", dtype=torch.float32)
    dense.scatter_(1, ids, weights)
    sums = torch.empty(rows, device="npu", dtype=torch.float32)
    counts = torch.empty_like(sums)
    joyo_ec_sum_reduce_f32_e48_b256_c48[(48,)](dense, sums, counts, rows, **OPTIONS)
    expected_sums = torch.zeros_like(sums)
    expected_counts = torch.zeros_like(counts)
    for expert in range(48):
        assigned = dense[expert] >= 0
        expected_sums += torch.where(assigned, dense[expert], 0.0)
        expected_counts += assigned.float()
    assert torch.equal(sums, expected_sums)
    assert torch.equal(counts, expected_counts)


@pytest.mark.parametrize("rows", [1, 49, 4845])
def test_expert_swiglu(rows: int) -> None:
    value = _value(rows, 8192)
    output = torch.empty((rows, 4096), device="npu", dtype=value.dtype)
    joyo_expert_swiglu_d4096[(48,)](value, output, rows, **OPTIONS)
    gate, up = value.chunk(2, -1)
    assert torch.equal(output, torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize("rows", [1, 49, 809])
def test_moe_post(rows: int) -> None:
    routed, shared, residual = [_value(rows, 6144) for _ in range(3)]
    shared_gate = _value(rows, 1)
    weight = _value(6144)
    table = _value(2, 6, 6144)[:, 2]
    ids = torch.arange(rows, device="npu", dtype=torch.int64) % 2
    output = torch.empty_like(routed)
    joyo_moe_post_rms_d6144[(48,)](
        routed,
        shared,
        shared_gate,
        weight,
        table,
        residual,
        ids,
        output,
        rows,
        table.stride(0),
        **OPTIONS
    )
    combined = routed * 10.0 + shared * shared_gate
    expected = residual + _norm(combined, weight) * table[ids]
    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("rows", [1, 49, 12915])
def test_attention_entries(rows: int) -> None:
    first_length = rows if rows == 1 else rows // 2 + 1
    qk, v = _value(rows, 512), _value(rows, 128)
    qw, kw = _value(128), _value(128)
    angles = torch.randn((rows, 64), device="npu")
    freqs = torch.cat([angles.cos(), angles.sin()], -1)
    prepared = [torch.empty(rows * 384, device="npu", dtype=qk.dtype) for _ in range(3)]
    joyo_qk_half_prepare_b32_0_1[(48,)](
        qk, v, qw, kw, freqs, *prepared, rows, first_length, **OPTIONS
    )
    expected_q = _norm(qk[:, :384].reshape(-1, 128), qw).reshape(rows, 3, 128)
    expected_k = _norm(qk[:, 384:].contiguous(), kw).reshape(rows, 1, 128)
    cos, sin = freqs.to(qk.dtype).chunk(2, -1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, -1)
        return torch.cat(
            [
                first * cos[:, None] - second * sin[:, None],
                second * cos[:, None] + first * sin[:, None],
            ],
            -1,
        )

    expected = [
        rotate(expected_q),
        rotate(expected_k).expand(-1, 3, -1),
        v[:, None].expand(-1, 3, -1),
    ]
    offset = 0
    for length in [first_length, rows - first_length]:
        if length == 0:
            continue
        for value, reference in zip(prepared, expected):
            actual = value[offset * 384 : (offset + length) * 384].view(3, length, 128)
            target = reference[offset : offset + length].permute(1, 0, 2)
            torch.testing.assert_close(actual, target, atol=0.05, rtol=0.03)
        offset += length

    first = _value(1, 3, first_length, 128)
    second = first if first_length == rows else _value(1, 3, rows - first_length, 128)
    gate = torch.sigmoid(_value(rows, 3))
    output = torch.empty((rows, 3, 128), device="npu", dtype=qk.dtype)
    joyo_attention_finish_b64[(48,)](
        first, second, gate, output, rows, first_length, **OPTIONS
    )
    samples = [first] if first_length == rows else [first, second]
    expected = torch.cat([value.squeeze(0).permute(1, 0, 2) for value in samples], 0)
    assert torch.equal(output, expected * gate[:, :, None])
