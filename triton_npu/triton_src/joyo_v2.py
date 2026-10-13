# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0
"""The eight qualified Joyo V2 AOT entries consumed by triton_adapter.

Keep staged BF16 rounding, FP32 reduction order and runtime scalar ABI.
"""
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value: tl.tensor) -> tl.tensor:
    bits = value.to(tl.uint32, bitcast=True)
    bits = bits + 32767 + (bits >> 16 & 1) & 4294901760
    return bits.to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["sample_start", "sample_length"])
def _prepare_sample(
    QK: tl.tensor,
    V: tl.tensor,
    QW: tl.tensor,
    KW: tl.tensor,
    FREQS: tl.tensor,
    QO: tl.tensor,
    KO: tl.tensor,
    VO: tl.tensor,
    sample_start: tl.tensor,
    sample_length: tl.tensor,
    BLOCK: tl.constexpr,
    SUM_MODE: tl.constexpr,
    INVERSE: tl.constexpr,
) -> None:
    local_row = tl.arange(0, BLOCK)
    column = tl.arange(0, 64)
    tiles = tl.cdiv(sample_length, BLOCK) * 3
    for iteration in range(tl.cdiv(tiles, 48)):
        tile = tl.program_id(0) + iteration * 48
        row = tile // 3 * BLOCK + local_row
        head = tile % 3
        active = row[:, None] < sample_length
        global_row = sample_start + row
        base = global_row[:, None] * 512 + column[None, :]
        destination = (
            sample_start * 384
            + (head * sample_length + row[:, None]) * 128
            + column[None, :]
        )
        qw1 = tl.load(QW + column).to(tl.float32)
        qw2 = tl.load(QW + column + 64).to(tl.float32)
        kw1 = tl.load(KW + column).to(tl.float32)
        kw2 = tl.load(KW + column + 64).to(tl.float32)
        q1 = tl.load(QK + base + head * 128, mask=active, other=0).to(tl.float32)
        q2 = tl.load(QK + base + head * 128 + 64, mask=active, other=0).to(tl.float32)
        k1 = tl.load(QK + base + 384, mask=active, other=0).to(tl.float32)
        k2 = tl.load(QK + base + 448, mask=active, other=0).to(tl.float32)
        if SUM_MODE == 0:
            qsum = tl.sum(q1 * q1 + q2 * q2, axis=1)
            ksum = tl.sum(k1 * k1 + k2 * k2, axis=1)
        else:
            qsum = tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)
            ksum = tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)
        qmean = qsum * (1.0 / 128) + 1e-06
        kmean = ksum * (1.0 / 128) + 1e-06
        if INVERSE == 0:
            rq, rk = (tl.rsqrt(qmean), tl.rsqrt(kmean))
        else:
            rq, rk = (1.0 / tl.sqrt(qmean), 1.0 / tl.sqrt(kmean))
        q1 = _round_bf16(q1 * rq[:, None] * qw1[None, :])
        q2 = _round_bf16(q2 * rq[:, None] * qw2[None, :])
        k1 = _round_bf16(k1 * rk[:, None] * kw1[None, :])
        k2 = _round_bf16(k2 * rk[:, None] * kw2[None, :])
        frequency = global_row[:, None] * 128 + column[None, :]
        co = _round_bf16(tl.load(FREQS + frequency, mask=active, other=0))
        si = _round_bf16(tl.load(FREQS + frequency + 64, mask=active, other=0))
        tl.store(
            QO + destination, _round_bf16(q1 * co) - _round_bf16(q2 * si), mask=active
        )
        tl.store(
            QO + destination + 64,
            _round_bf16(q2 * co) + _round_bf16(q1 * si),
            mask=active,
        )
        tl.store(
            KO + destination, _round_bf16(k1 * co) - _round_bf16(k2 * si), mask=active
        )
        tl.store(
            KO + destination + 64,
            _round_bf16(k2 * co) + _round_bf16(k1 * si),
            mask=active,
        )
        tl.store(
            VO + destination,
            tl.load(
                V + global_row[:, None] * 128 + column[None, :], mask=active, other=0
            ),
            mask=active,
        )
        tl.store(
            VO + destination + 64,
            tl.load(
                V + global_row[:, None] * 128 + column[None, :] + 64,
                mask=active,
                other=0,
            ),
            mask=active,
        )


@triton.jit(do_not_specialize=["rows", "first_length"])
def _prepare_qk_rstd(
    QK: tl.tensor,
    V: tl.tensor,
    QW: tl.tensor,
    KW: tl.tensor,
    FREQS: tl.tensor,
    QO: tl.tensor,
    KO: tl.tensor,
    VO: tl.tensor,
    rows: tl.tensor,
    first_length: tl.tensor,
    BLOCK: tl.constexpr,
    SUM_MODE: tl.constexpr,
    INVERSE: tl.constexpr,
) -> None:
    _prepare_sample(
        QK, V, QW, KW, FREQS, QO, KO, VO, 0, first_length, BLOCK, SUM_MODE, INVERSE
    )
    _prepare_sample(
        QK,
        V,
        QW,
        KW,
        FREQS,
        QO,
        KO,
        VO,
        first_length,
        rows - first_length,
        BLOCK,
        SUM_MODE,
        INVERSE,
    )


@triton.jit(do_not_specialize=["sample_start", "sample_length"])
def _finish_sample(
    SOURCE: tl.tensor,
    GATE: tl.tensor,
    OUT: tl.tensor,
    sample_start: tl.tensor,
    sample_length: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    local_row = tl.arange(0, BLOCK)
    column = tl.arange(0, 128)
    tiles = tl.cdiv(sample_length, BLOCK) * 3
    for iteration in range(tl.cdiv(tiles, 48)):
        tile = tl.program_id(0) + iteration * 48
        row = tile // 3 * BLOCK + local_row
        head = tile % 3
        active = row[:, None] < sample_length
        global_row = sample_start + row
        source_offset = (head * sample_length + row[:, None]) * 128 + column[None, :]
        value = tl.load(SOURCE + source_offset, mask=active, other=0).to(tl.float32)
        gate = tl.load(
            GATE + global_row * 3 + head, mask=row < sample_length, other=0
        ).to(tl.float32)
        destination = global_row[:, None] * 384 + head * 128 + column[None, :]
        tl.store(OUT + destination, value * gate[:, None], mask=active)


@triton.jit(do_not_specialize=["rows", "first_length"])
def _finish_attention(
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    GATE: tl.tensor,
    OUT: tl.tensor,
    rows: tl.tensor,
    first_length: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    _finish_sample(FIRST, GATE, OUT, 0, first_length, BLOCK)
    _finish_sample(SECOND, GATE, OUT, first_length, rows - first_length, BLOCK)


@triton.jit
def _inverse(a: tl.tensor, b: tl.tensor, c: tl.tensor) -> tl.tensor:
    total = tl.sum(a * a + c * c + b * b, axis=1)
    return tl.rsqrt(total * (1.0 / 6144) + 1e-06)


@triton.jit
def _normalized(value: tl.tensor, weight: tl.tensor, inverse: tl.tensor) -> tl.tensor:
    return (
        (value * inverse[:, None] * weight[None, :].to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )


@triton.jit
def _affine_chunk(
    value: tl.tensor,
    weight: tl.tensor,
    inverse: tl.tensor,
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    sample: tl.tensor,
    column: tl.tensor,
    active: tl.tensor,
    first_stride: tl.tensor,
    second_stride: tl.tensor,
) -> tl.tensor:
    normalized = _normalized(value, weight, inverse)
    scale = tl.load(
        FIRST + sample[:, None] * first_stride + column[None, :], mask=active, other=0
    ).to(tl.float32)
    shift = tl.load(
        SECOND + sample[:, None] * second_stride + column[None, :], mask=active, other=0
    ).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    product = (normalized * factor).to(tl.bfloat16).to(tl.float32)
    return (product + shift).to(tl.bfloat16)


@triton.jit
def _gate_chunk(
    value: tl.tensor,
    weight: tl.tensor,
    inverse: tl.tensor,
    FIRST: tl.tensor,
    RESIDUAL: tl.tensor,
    sample: tl.tensor,
    position: tl.tensor,
    column: tl.tensor,
    active: tl.tensor,
    first_stride: tl.tensor,
) -> tl.tensor:
    normalized = _normalized(value, weight, inverse)
    gate = tl.load(
        FIRST + sample[:, None] * first_stride + column[None, :], mask=active, other=0
    ).to(tl.float32)
    residual = tl.load(RESIDUAL + position, mask=active, other=0).to(tl.float32)
    product = (normalized * gate).to(tl.bfloat16).to(tl.float32)
    return (residual + product).to(tl.bfloat16).to(tl.float32)


@triton.jit(
    do_not_specialize=["rows", "first_stride", "second_stride", "third_stride", "cores"]
)
def _rms_indexed_tiled(
    X: tl.tensor,
    RESIDUAL: tl.tensor,
    W: tl.tensor,
    NEXT_W: tl.tensor,
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    THIRD: tl.tensor,
    IDS: tl.tensor,
    H: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    first_stride: tl.tensor,
    second_stride: tl.tensor,
    third_stride: tl.tensor,
    cores: tl.tensor,
    KIND: tl.constexpr,
    TILE: tl.constexpr,
) -> None:
    column = tl.arange(0, 2048)
    local_rows = tl.arange(0, TILE)
    rows_per_core = tl.cdiv(rows, cores)
    start = tl.program_id(0) * rows_per_core
    w0 = tl.load(W + column)
    w1 = tl.load(W + column + 2048)
    w2 = tl.load(W + column + 4096)
    for iteration in range(tl.cdiv(rows_per_core, TILE)):
        row = start + iteration * TILE + local_rows
        active_row = (row < rows) & (row < start + rows_per_core)
        active = active_row[:, None]
        position = row[:, None] * 6144 + column[None, :]
        sample = tl.load(IDS + row, mask=active_row, other=0)
        a = tl.load(X + position, mask=active, other=0).to(tl.float32)
        b = tl.load(X + position + 2048, mask=active, other=0).to(tl.float32)
        c = tl.load(X + position + 4096, mask=active, other=0).to(tl.float32)
        inverse = _inverse(a, b, c)
        if KIND == 0:
            tl.store(
                Y + position,
                _affine_chunk(
                    a,
                    w0,
                    inverse,
                    FIRST,
                    SECOND,
                    sample,
                    column,
                    active,
                    first_stride,
                    second_stride,
                ),
                mask=active,
            )
            tl.store(
                Y + position + 2048,
                _affine_chunk(
                    b,
                    w1,
                    inverse,
                    FIRST,
                    SECOND,
                    sample,
                    column + 2048,
                    active,
                    first_stride,
                    second_stride,
                ),
                mask=active,
            )
            tl.store(
                Y + position + 4096,
                _affine_chunk(
                    c,
                    w2,
                    inverse,
                    FIRST,
                    SECOND,
                    sample,
                    column + 4096,
                    active,
                    first_stride,
                    second_stride,
                ),
                mask=active,
            )
        else:
            ha = _gate_chunk(
                a,
                w0,
                inverse,
                FIRST,
                RESIDUAL,
                sample,
                position,
                column,
                active,
                first_stride,
            )
            hb = _gate_chunk(
                b,
                w1,
                inverse,
                FIRST,
                RESIDUAL,
                sample,
                position + 2048,
                column + 2048,
                active,
                first_stride,
            )
            hc = _gate_chunk(
                c,
                w2,
                inverse,
                FIRST,
                RESIDUAL,
                sample,
                position + 4096,
                column + 4096,
                active,
                first_stride,
            )
            if KIND == 1:
                tl.store(Y + position, ha, mask=active)
                tl.store(Y + position + 2048, hb, mask=active)
                tl.store(Y + position + 4096, hc, mask=active)
            else:
                tl.store(H + position, ha, mask=active)
                tl.store(H + position + 2048, hb, mask=active)
                tl.store(H + position + 4096, hc, mask=active)
                next_inverse = _inverse(ha, hb, hc)
                n0 = tl.load(NEXT_W + column)
                n1 = tl.load(NEXT_W + column + 2048)
                n2 = tl.load(NEXT_W + column + 4096)
                tl.store(
                    Y + position,
                    _affine_chunk(
                        ha,
                        n0,
                        next_inverse,
                        SECOND,
                        THIRD,
                        sample,
                        column,
                        active,
                        second_stride,
                        third_stride,
                    ),
                    mask=active,
                )
                tl.store(
                    Y + position + 2048,
                    _affine_chunk(
                        hb,
                        n1,
                        next_inverse,
                        SECOND,
                        THIRD,
                        sample,
                        column + 2048,
                        active,
                        second_stride,
                        third_stride,
                    ),
                    mask=active,
                )
                tl.store(
                    Y + position + 4096,
                    _affine_chunk(
                        hc,
                        n2,
                        next_inverse,
                        SECOND,
                        THIRD,
                        sample,
                        column + 4096,
                        active,
                        second_stride,
                        third_stride,
                    ),
                    mask=active,
                )


@triton.jit(do_not_specialize=["rows"])
def _sum(
    DENSE: tl.tensor,
    SUMS: tl.tensor,
    COUNTS: tl.tensor,
    rows: tl.tensor,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    CORES: tl.constexpr,
) -> None:
    for iteration in range(tl.cdiv(rows, BLOCK * CORES)):
        token = (tl.program_id(0) + iteration * CORES) * BLOCK + tl.arange(0, BLOCK)
        active = token < rows
        total = tl.full((BLOCK,), 0.0, tl.float32)
        count = tl.full((BLOCK,), 0.0, tl.float32)
        for expert in range(EXPERTS):
            weight = tl.load(DENSE + expert * rows + token, mask=active, other=-1.0)
            assigned = weight >= 0
            total = total + tl.where(assigned, weight, 0.0)
            count = count + assigned.to(tl.float32)
        tl.store(SUMS + token, total, mask=active)
        tl.store(COUNTS + token, count, mask=active)


@triton.jit(do_not_specialize=["rows"])
def _swiglu_rows(
    X: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    HIDDEN: tl.constexpr,
    TILE: tl.constexpr,
    CORES: tl.constexpr,
) -> None:
    columns = tl.arange(0, HIDDEN)
    local_rows = tl.arange(0, TILE)
    rows_per_core = tl.cdiv(rows, CORES)
    start = tl.program_id(0) * rows_per_core
    for iteration in range(tl.cdiv(rows_per_core, TILE)):
        row = start + iteration * TILE + local_rows
        active = (row < rows) & (row < start + rows_per_core)
        source = row[:, None] * (2 * HIDDEN) + columns[None, :]
        gate = tl.load(X + source, mask=active[:, None], other=0).to(tl.float32)
        up = tl.load(X + source + HIDDEN, mask=active[:, None], other=0).to(tl.float32)
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        output = row[:, None] * HIDDEN + columns[None, :]
        tl.store(Y + output, silu * up, mask=active[:, None])


@triton.jit
def _moe_value(
    ROUTED: tl.tensor,
    SHARED: tl.tensor,
    position: tl.tensor,
    active: tl.tensor,
    gate: tl.tensor,
) -> tl.tensor:
    routed = tl.load(ROUTED + position, mask=active, other=0).to(tl.float32)
    shared = tl.load(SHARED + position, mask=active, other=0).to(tl.float32)
    a = (routed * 10.0).to(tl.bfloat16).to(tl.float32)
    b = (shared * gate[:, None]).to(tl.bfloat16).to(tl.float32)
    return (a + b).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["rows", "gate_stride"])
def _moe_post_rms(
    ROUTED: tl.tensor,
    SHARED: tl.tensor,
    SHARED_GATE: tl.tensor,
    WEIGHT: tl.tensor,
    MLP_GATE: tl.tensor,
    RESIDUAL: tl.tensor,
    IDS: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    gate_stride: tl.tensor,
    CORES: tl.constexpr,
) -> None:
    columns = tl.arange(0, 2048)
    local = tl.arange(0, 1)
    rows_per_core = tl.cdiv(rows, CORES)
    start = tl.program_id(0) * rows_per_core
    w0 = tl.load(WEIGHT + columns)
    w1 = tl.load(WEIGHT + columns + 2048)
    w2 = tl.load(WEIGHT + columns + 4096)
    for iteration in range(rows_per_core):
        row = start + iteration + local
        active_row = row < rows
        active = active_row[:, None]
        position = row[:, None] * 6144 + columns[None, :]
        gate = tl.load(SHARED_GATE + row, mask=active_row, other=0).to(tl.float32)
        sample = tl.load(IDS + row, mask=active_row, other=0)
        a = _moe_value(ROUTED, SHARED, position, active, gate)
        b = _moe_value(ROUTED, SHARED, position + 2048, active, gate)
        c = _moe_value(ROUTED, SHARED, position + 4096, active, gate)
        inverse = _inverse(a, b, c)
        tl.store(
            Y + position,
            _gate_chunk(
                a,
                w0,
                inverse,
                MLP_GATE,
                RESIDUAL,
                sample,
                position,
                columns,
                active,
                gate_stride,
            ),
            mask=active,
        )
        tl.store(
            Y + position + 2048,
            _gate_chunk(
                b,
                w1,
                inverse,
                MLP_GATE,
                RESIDUAL,
                sample,
                position + 2048,
                columns + 2048,
                active,
                gate_stride,
            ),
            mask=active,
        )
        tl.store(
            Y + position + 4096,
            _gate_chunk(
                c,
                w2,
                inverse,
                MLP_GATE,
                RESIDUAL,
                sample,
                position + 4096,
                columns + 4096,
                active,
                gate_stride,
            ),
            mask=active,
        )


@triton.jit(do_not_specialize=["rows", "first_length"])
def joyo_qk_half_prepare_b32_0_1(
    QK: tl.tensor,
    V: tl.tensor,
    QW: tl.tensor,
    KW: tl.tensor,
    FREQS: tl.tensor,
    QO: tl.tensor,
    KO: tl.tensor,
    VO: tl.tensor,
    rows: tl.tensor,
    first_length: tl.tensor,
) -> None:
    _prepare_qk_rstd(QK, V, QW, KW, FREQS, QO, KO, VO, rows, first_length, 32, 0, 1)


@triton.jit(do_not_specialize=["rows", "first_length"])
def joyo_attention_finish_b64(
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    GATE: tl.tensor,
    OUT: tl.tensor,
    rows: tl.tensor,
    first_length: tl.tensor,
) -> None:
    _finish_attention(FIRST, SECOND, GATE, OUT, rows, first_length, 64)


@triton.jit(
    do_not_specialize=["rows", "first_stride", "second_stride", "third_stride", "cores"]
)
def joyo_rms_modulate_d6144(
    X: tl.tensor,
    RESIDUAL: tl.tensor,
    W: tl.tensor,
    NEXT_W: tl.tensor,
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    THIRD: tl.tensor,
    IDS: tl.tensor,
    H: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    first_stride: tl.tensor,
    second_stride: tl.tensor,
    third_stride: tl.tensor,
    cores: tl.tensor,
) -> None:
    _rms_indexed_tiled(
        X,
        RESIDUAL,
        W,
        NEXT_W,
        FIRST,
        SECOND,
        THIRD,
        IDS,
        H,
        Y,
        rows,
        first_stride,
        second_stride,
        third_stride,
        cores,
        0,
        1,
    )


@triton.jit(
    do_not_specialize=["rows", "first_stride", "second_stride", "third_stride", "cores"]
)
def joyo_rms_gate_residual_d6144(
    X: tl.tensor,
    RESIDUAL: tl.tensor,
    W: tl.tensor,
    NEXT_W: tl.tensor,
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    THIRD: tl.tensor,
    IDS: tl.tensor,
    H: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    first_stride: tl.tensor,
    second_stride: tl.tensor,
    third_stride: tl.tensor,
    cores: tl.tensor,
) -> None:
    _rms_indexed_tiled(
        X,
        RESIDUAL,
        W,
        NEXT_W,
        FIRST,
        SECOND,
        THIRD,
        IDS,
        H,
        Y,
        rows,
        first_stride,
        second_stride,
        third_stride,
        cores,
        1,
        1,
    )


@triton.jit(
    do_not_specialize=["rows", "first_stride", "second_stride", "third_stride", "cores"]
)
def joyo_rms_sandwich_d6144(
    X: tl.tensor,
    RESIDUAL: tl.tensor,
    W: tl.tensor,
    NEXT_W: tl.tensor,
    FIRST: tl.tensor,
    SECOND: tl.tensor,
    THIRD: tl.tensor,
    IDS: tl.tensor,
    H: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    first_stride: tl.tensor,
    second_stride: tl.tensor,
    third_stride: tl.tensor,
    cores: tl.tensor,
) -> None:
    _rms_indexed_tiled(
        X,
        RESIDUAL,
        W,
        NEXT_W,
        FIRST,
        SECOND,
        THIRD,
        IDS,
        H,
        Y,
        rows,
        first_stride,
        second_stride,
        third_stride,
        cores,
        2,
        1,
    )


@triton.jit(do_not_specialize=["rows"])
def joyo_ec_sum_reduce_f32_e48_b256_c48(
    DENSE: tl.tensor, SUMS: tl.tensor, COUNTS: tl.tensor, rows: tl.tensor
) -> None:
    _sum(DENSE, SUMS, COUNTS, rows, 48, 256, 48)


@triton.jit(do_not_specialize=["rows"])
def joyo_expert_swiglu_d4096(X: tl.tensor, Y: tl.tensor, rows: tl.tensor) -> None:
    _swiglu_rows(X, Y, rows, 4096, 2, 48)


@triton.jit(do_not_specialize=["rows", "gate_stride"])
def joyo_moe_post_rms_d6144(
    ROUTED: tl.tensor,
    SHARED: tl.tensor,
    SHARED_GATE: tl.tensor,
    WEIGHT: tl.tensor,
    MLP_GATE: tl.tensor,
    RESIDUAL: tl.tensor,
    IDS: tl.tensor,
    Y: tl.tensor,
    rows: tl.tensor,
    gate_stride: tl.tensor,
) -> None:
    _moe_post_rms(
        ROUTED,
        SHARED,
        SHARED_GATE,
        WEIGHT,
        MLP_GATE,
        RESIDUAL,
        IDS,
        Y,
        rows,
        gate_stride,
        48,
    )
