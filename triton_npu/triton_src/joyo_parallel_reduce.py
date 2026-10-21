# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0
"""TP16/SP1 ordered BF16 reductions for qualified Joyo projection geometries.

Build through the existing Triton setup/pytest workflow. The rank/half-band
order is specific to the qualified HCCL configuration; callers enforce it.
The peer table and expert banks remain owned by ECOwnerOutputResource.
"""
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value: tl.tensor) -> tl.tensor:
    # The retained order-identification tests require this round trip after
    # each add to remain observable through the installed NPU compiler.
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit(
    do_not_specialize=["elements", "first_flat", "partial_stride"],
    do_not_specialize_on_alignment=[
        "PARTIALS", "OUTPUT", "elements", "first_flat", "partial_stride"
    ],
)
def joyo_attention_ordered_reduce_bf16_b2048_c48(
    PARTIALS: tl.tensor,
    OUTPUT: tl.tensor,
    elements: tl.tensor,
    first_flat: tl.tensor,
    partial_stride: tl.tensor,
) -> None:
    lane = tl.arange(0, 2048)
    tiles = tl.cdiv(elements, 2048)
    for iteration in range(tl.cdiv(tiles, 48)):
        tile = tl.program_id(0) + iteration * 48
        if tile < tiles:
            position = tile * 2048 + lane
            active = position < elements
            global_start = first_flat + tile * 2048
            # Owner selection is scalar, pure integer arithmetic. Non-power-of-
            # two integer division loses low bits near large boundaries on NPU.
            owner = tl.where(global_start >= 8 * 4959360, 8, 0)
            owner += tl.where(global_start >= (owner + 4) * 4959360, 4, 0)
            owner += tl.where(global_start >= (owner + 2) * 4959360, 2, 0)
            owner += tl.where(global_start >= (owner + 1) * 4959360, 1, 0)
            within_owner = global_start - owner * 4959360
            forward = within_owner < 2479680
            band_end = owner * 4959360 + tl.where(forward, 2479680, 4959360)
            first_source = tl.where(forward, owner + 1, owner + 15) & 15
            value = tl.load(
                PARTIALS + first_source * partial_stride + position,
                mask=active,
                other=0,
            ).to(tl.float32)
            for step in tl.static_range(1, 16):
                source_rank = tl.where(
                    forward, owner + 1 + step, owner + 31 - step
                ) & 15
                partial = tl.load(
                    PARTIALS + source_rank * partial_stride + position,
                    mask=active,
                    other=0,
                ).to(tl.float32)
                value = _round_bf16(value + partial)

            # A 2048-element tile can reach only the immediately following band.
            # The common case performs no secondary loads or additions.
            if global_start + 2048 > band_end:
                owner_next = tl.where(forward, owner, owner + 1)
                forward_next = ~forward
                first_source_next = tl.where(
                    forward_next, owner_next + 1, owner_next + 15
                ) & 15
                secondary_value = tl.load(
                    PARTIALS + first_source_next * partial_stride + position,
                    mask=active,
                    other=0,
                ).to(tl.float32)
                for step in tl.static_range(1, 16):
                    source_rank_next = tl.where(
                        forward_next, owner_next + 1 + step, owner_next + 31 - step
                    ) & 15
                    secondary_partial = tl.load(
                        PARTIALS + source_rank_next * partial_stride + position,
                        mask=active,
                        other=0,
                    ).to(tl.float32)
                    secondary_value = _round_bf16(secondary_value + secondary_partial)
                value = tl.where(
                    first_flat + position < band_end, value, secondary_value
                )
            tl.store(OUTPUT + position, value, mask=active)


@triton.jit(
    do_not_specialize=["elements", "first_flat", "partial_stride", "rank_elements"],
    do_not_specialize_on_alignment=[
        "PARTIALS", "OUTPUT", "elements", "first_flat", "partial_stride",
        "rank_elements",
    ],
)
def joyo_attention_ordered_reduce_dynamic_bf16_b2048_c48(
    PARTIALS: tl.tensor,
    OUTPUT: tl.tensor,
    elements: tl.tensor,
    first_flat: tl.tensor,
    partial_stride: tl.tensor,
    rank_elements: tl.tensor,
) -> None:
    """Bounded chunks; native ring order is independently qualified per geometry."""
    lane = tl.arange(0, 2048)
    tiles = tl.cdiv(elements, 2048)
    for iteration in range(tl.cdiv(tiles, 48)):
        tile = tl.program_id(0) + iteration * 48
        if tile < tiles:
            position = tile * 2048 + lane
            active = position < elements
            global_position = first_flat + position
            # Integer comparisons preserve exact band ownership even above 2**24.
            owner = tl.where(global_position >= 8 * rank_elements, 8, 0)
            owner += tl.where(global_position >= (owner + 4) * rank_elements, 4, 0)
            owner += tl.where(global_position >= (owner + 2) * rank_elements, 2, 0)
            owner += tl.where(global_position >= (owner + 1) * rank_elements, 1, 0)
            # HCCL 200-MiB buffer loops split each owner's slice into two
            # rings. The final loop has a shorter split. Integer comparisons
            # avoid inaccurate quotient lowering at high element addresses.
            within_owner = global_position - owner * rank_elements
            loop_elements: tl.constexpr = 6553600
            loop = tl.where(within_owner >= 16 * loop_elements, 16, 0)
            loop += tl.where(within_owner >= (loop + 8) * loop_elements, 8, 0)
            loop += tl.where(within_owner >= (loop + 4) * loop_elements, 4, 0)
            loop += tl.where(within_owner >= (loop + 2) * loop_elements, 2, 0)
            loop += tl.where(within_owner >= (loop + 1) * loop_elements, 1, 0)
            loop_start = loop * loop_elements
            loop_count = tl.minimum(loop_elements, rank_elements - loop_start)
            forward = within_owner - loop_start < loop_count // 2
            first_source = tl.where(forward, owner + 1, owner + 15) & 15
            value = tl.load(
                PARTIALS + first_source * partial_stride + position,
                mask=active,
                other=0,
            ).to(tl.float32)
            for step in tl.static_range(1, 16):
                source = tl.where(forward, owner + 1 + step, owner + 31 - step) & 15
                partial = tl.load(
                    PARTIALS + source * partial_stride + position,
                    mask=active,
                    other=0,
                ).to(tl.float32)
                value = _round_bf16(value + partial)
            tl.store(OUTPUT + position, value, mask=active)


@triton.jit
def _table_leaf(
    PEER_BASES: tl.tensor, INVERSE: tl.tensor, source: tl.tensor,
    token: tl.tensor, columns: tl.tensor, capacity: tl.tensor,
) -> tl.tensor:
    # Scalar i64 -> BF16 pointer once per source, outside the expert loop.
    # The table is never written; peer-bank ownership is external.
    address = tl.load(PEER_BASES + source)
    peer = address.to(tl.pointer_type(tl.bfloat16))
    total = tl.full((2048,), 0, tl.float32)
    for expert in tl.static_range(0, 3):
        slot = tl.load(INVERSE + (source * 3 + expert) * 12915 + token).to(tl.int32)
        present = (slot >= 0) & (slot < capacity)
        if present:
            position = (expert * capacity + slot) * 6144 + columns
            total = total + tl.load(peer + position).to(tl.float32)
    bits = total.to(tl.uint32, bitcast=True)
    bits = tl.where((bits & 2147483647) == 0, 0, bits)
    return _round_bf16(bits.to(tl.float32, bitcast=True))


@triton.jit
def _table_ordered(
    PEER_BASES: tl.tensor, INVERSE: tl.tensor, owner: tl.tensor,
    forward: tl.tensor, token: tl.tensor, columns: tl.tensor, capacity: tl.tensor,
) -> tl.tensor:
    source = tl.where(forward, owner + 1, owner + 15) & 15
    value = _table_leaf(PEER_BASES, INVERSE, source, token, columns, capacity)
    step = 1
    while step < 16:
        source = tl.where(forward, owner + 1 + step, owner + 31 - step) & 15
        leaf = _table_leaf(PEER_BASES, INVERSE, source, token, columns, capacity)
        # No zero-source elision, reassociation, FP32 global sum or sum tree.
        value = _round_bf16(value + leaf)
        step += 1
    return value


@triton.jit(
    do_not_specialize=['capacity', 'first_row', 'local_rows'],
    do_not_specialize_on_alignment=[
        'PEER_BASES', 'INVERSE', 'OUTPUT', 'capacity', 'first_row', 'local_rows'
    ],
)
def joyo_ec_owner_fused_table_bf16_b2048_c48(
    PEER_BASES: tl.tensor,
    INVERSE: tl.tensor,
    OUTPUT: tl.tensor,
    capacity: tl.tensor,
    first_row: tl.tensor,
    local_rows: tl.tensor,
) -> None:
    lane = tl.arange(0, 2048)
    valid = ((capacity >= 0) & (capacity <= 1615) & (first_row >= 0)
             & (first_row <= 12915) & (local_rows >= 0) & (local_rows <= 809)
             & (first_row + local_rows <= 12915))
    if valid:
        # Small pid0..47 division only. Three column pieces per row and
        # sixteen row stripes cover complete rows, including both halos.
        row_stripe = tl.program_id(0) // 3
        part = tl.program_id(0) % 3
        columns = part * 2048 + lane
        for row in range(row_stripe, local_rows, 16):
            token = first_row + row
            global_start = token * 6144 + part * 2048
            owner = tl.where(global_start >= 8 * 4959360, 8, 0)
            owner += tl.where(global_start >= (owner + 4) * 4959360, 4, 0)
            owner += tl.where(global_start >= (owner + 2) * 4959360, 2, 0)
            owner += tl.where(global_start >= (owner + 1) * 4959360, 1, 0)
            forward = global_start - owner * 4959360 < 2479680
            band_end = owner * 4959360 + tl.where(forward, 2479680, 4959360)
            value = _table_ordered(
                PEER_BASES, INVERSE, owner, forward, token, columns, capacity
            )
            if global_start + 2048 > band_end:
                # Consume the primary result before starting the secondary
                # runtime loop. The failed V2 kept two loop-result tensors
                # alive through one select; lower-level buffer clobber is a
                # hypothesis, not proved by the available TTIR/adapter.
                primary_mask = global_start + lane < band_end
                tl.store(OUTPUT + row * 6144 + columns, value, mask=primary_mask)
                next_owner = tl.where(forward, owner, owner + 1)
                secondary = _table_ordered(
                    PEER_BASES, INVERSE, next_owner, ~forward,
                    token, columns, capacity
                )
                tl.store(
                    OUTPUT + row * 6144 + columns,
                    secondary,
                    mask=global_start + lane >= band_end,
                )
            else:
                tl.store(OUTPUT + row * 6144 + columns, value)


joyo_ec_owner_fused_table_bf16_b2048_c48.tensor_kinds = [0, 0, 1]


@triton.jit
def _dynamic_table_geometry(
    position: tl.tensor, rank_elements: tl.tensor,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    owner = tl.where(position >= 8 * rank_elements, 8, 0)
    owner += tl.where(position >= (owner + 4) * rank_elements, 4, 0)
    owner += tl.where(position >= (owner + 2) * rank_elements, 2, 0)
    owner += tl.where(position >= (owner + 1) * rank_elements, 1, 0)
    within = position - owner * rank_elements
    loop_elements: tl.constexpr = 6553600
    loop = tl.where(within >= 16 * loop_elements, 16, 0)
    loop += tl.where(within >= (loop + 8) * loop_elements, 8, 0)
    loop += tl.where(within >= (loop + 4) * loop_elements, 4, 0)
    loop += tl.where(within >= (loop + 2) * loop_elements, 2, 0)
    loop += tl.where(within >= (loop + 1) * loop_elements, 1, 0)
    loop_start = loop * loop_elements
    count = tl.minimum(loop_elements, rank_elements - loop_start)
    forward = within - loop_start < count // 2
    end = owner * rank_elements + loop_start + tl.where(forward, count // 2, count)
    return owner, forward, end


@triton.jit
def _dynamic_table_leaf(
    PEER_BASES: tl.tensor, INVERSE: tl.tensor, source: tl.tensor,
    token: tl.tensor, columns: tl.tensor, capacity: tl.tensor, rows: tl.tensor,
) -> tl.tensor:
    address = tl.load(PEER_BASES + source)
    peer = address.to(tl.pointer_type(tl.bfloat16))
    total = tl.full((2048,), 0, tl.float32)
    for expert in tl.static_range(0, 3):
        # The dynamic entry uses int32 slots; capacities above 32767 remain
        # representable. Missing experts contribute the native zero leaf.
        slot = tl.load(INVERSE + (source * 3 + expert) * rows + token)
        present = (slot >= 0) & (slot < capacity)
        position = (expert * capacity + tl.maximum(slot, 0)) * 6144 + columns
        total = total + tl.load(peer + position, mask=present, other=0).to(tl.float32)
    bits = total.to(tl.uint32, bitcast=True)
    bits = tl.where((bits & 2147483647) == 0, 0, bits)
    return _round_bf16(bits.to(tl.float32, bitcast=True))


@triton.jit
def _dynamic_table_ordered(
    PEER_BASES: tl.tensor, INVERSE: tl.tensor, owner: tl.tensor,
    forward: tl.tensor, token: tl.tensor, columns: tl.tensor,
    capacity: tl.tensor, rows: tl.tensor,
) -> tl.tensor:
    source = tl.where(forward, owner + 1, owner + 15) & 15
    value = _dynamic_table_leaf(
        PEER_BASES, INVERSE, source, token, columns, capacity, rows
    )
    step = 1
    while step < 16:
        source = tl.where(forward, owner + 1 + step, owner + 31 - step) & 15
        leaf = _dynamic_table_leaf(
            PEER_BASES, INVERSE, source, token, columns, capacity, rows
        )
        value = _round_bf16(value + leaf)
        step += 1
    return value


@triton.jit(
    do_not_specialize=['capacity', 'first_row', 'local_rows', 'rows'],
    do_not_specialize_on_alignment=[
        'PEER_BASES', 'INVERSE', 'OUTPUT', 'capacity', 'first_row',
        'local_rows', 'rows',
    ],
)
def joyo_ec_owner_fused_table_dynamic_bf16_b2048_c48(
    PEER_BASES: tl.tensor, INVERSE: tl.tensor, OUTPUT: tl.tensor,
    capacity: tl.tensor, first_row: tl.tensor, local_rows: tl.tensor,
    rows: tl.tensor,
) -> None:
    lane = tl.arange(0, 2048)
    valid = ((capacity > 0) & (capacity <= 116508) & (rows >= 16)
             & (rows <= 349525) & (first_row >= 0) & (local_rows >= 0)
             & (first_row + local_rows <= rows))
    if valid:
        stripe = tl.program_id(0) // 3
        columns = (tl.program_id(0) % 3) * 2048 + lane
        rank_elements = rows * 384
        for row in range(stripe, local_rows, 16):
            token = first_row + row
            start = token * 6144 + (tl.program_id(0) % 3) * 2048
            owner, forward, end = _dynamic_table_geometry(start, rank_elements)
            if start + 2048 <= end:
                # Scalar peer selection preserves contiguous vector loads.
                value = _dynamic_table_ordered(
                    PEER_BASES, INVERSE, owner, forward, token, columns,
                    capacity, rows,
                )
                tl.store(OUTPUT + row * 6144 + columns, value)
            else:
                # Short final buffer loops can cross several bands in one
                # tile. Exact per-lane ownership handles every such boundary.
                owner, forward, end = _dynamic_table_geometry(start + lane, rank_elements)
                value = _dynamic_table_ordered(
                    PEER_BASES, INVERSE, owner, forward, token, columns,
                    capacity, rows,
                )
                tl.store(OUTPUT + row * 6144 + columns, value)


joyo_ec_owner_fused_table_dynamic_bf16_b2048_c48.tensor_kinds = [0, 0, 1]
