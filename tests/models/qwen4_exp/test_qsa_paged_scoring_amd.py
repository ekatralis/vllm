# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged QSA indexer scoring (qsa_mqa_paged) must match the pre-change full-width
kernel bit for bit on every visible column, and the token selection built on it
must not change. The full-width kernel scored all page-table columns (sized by
max_model_len) for every row; the current one stops at each row's visible length.
"""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops import qsa as qsa_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda_alike()),
    reason="QSA paged scoring needs a GPU and Triton",
)

COMPRESS = 4
HEADS = 4
HEAD_DIM = 128


@triton.jit
def _full_width_kernel(  # verbatim pre-change kernel (reference)
    q_ptr, k_cache_ptr, page_table_ptr, token_to_req_ptr, query_positions_ptr,
    sequence_lengths_ptr, visible_blocks_ptr, logits_ptr, stride_q_row, stride_q_head,
    stride_q_dim, stride_cache_block, stride_cache_token, stride_cache_dim,
    stride_table_req, stride_table_page, stride_logits_row, num_rows, num_columns,
    num_pages, num_requests, score_divisor, PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(sequence_lengths_ptr + safe_request,
                              mask=(request >= 0) & (request < num_requests), other=0)
    visible = tl.minimum((query_position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = ((row < num_rows) & (columns < num_columns) & (columns < visible) & (request >= 0)
             & (request < num_requests) & (logical_page < PAGE_TABLE_WIDTH))
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(page_table_ptr + safe_request * stride_table_req
                            + safe_logical_page * stride_table_page, mask=valid, other=-1)
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
                        mask=dims < HEAD_DIM, other=0.0).to(tl.float32)
        keys = tl.load(k_cache_ptr + safe_physical_page[:, None] * stride_cache_block
                       + page_offset[:, None] * stride_cache_token + dims[None, :] * stride_cache_dim,
                       mask=valid[:, None] & (dims[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    score /= score_divisor
    tl.store(logits_ptr + row * stride_logits_row + columns, tl.where(valid, score, -float("inf")),
             mask=(row < num_rows) & (columns < num_columns))


def _full_width(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
                compress_ratio, num_columns=None, score_scale=None):
    columns = page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    div = (q.shape[2] ** 0.5) if score_scale is None else score_scale
    _full_width_kernel[(q.shape[0], triton.cdiv(columns, 32))](
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible, logits,
        q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
        page_table.stride(0), page_table.stride(1), logits.stride(0), q.shape[0], columns,
        k_cache.shape[0], page_table.shape[0], float(div), PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
        BLOCK_N=32, BLOCK_D=triton.next_power_of_2(q.shape[2]), COMPRESS_RATIO=compress_ratio,
        num_warps=4)
    return logits, visible


def _case(rows, max_model_len, page_size, seed, dtype=torch.bfloat16):
    """Random paged cache and a batch mixing long prefill rows, decode rows, short
    sequences, unallocated pages and padded rows (request -1)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    width = triton.cdiv(max_model_len // COMPRESS, page_size)
    reqs = 5
    max_visible = max_model_len // COMPRESS
    seq = torch.tensor([max_model_len - 7, max_model_len // 3 + 5, 4 * page_size + 3, 9, 1], dtype=torch.int32)
    pages_needed = [triton.cdiv(int(s) // COMPRESS, page_size) for s in seq]
    num_pages = sum(pages_needed) + 3
    perm = torch.randperm(num_pages, generator=g)
    table = torch.full((reqs, width), -1, dtype=torch.int32)
    used = 0
    for r, n in enumerate(pages_needed):
        table[r, :n] = perm[used:used + n].to(torch.int32); used += n
    table[1, 2] = -1  # a hole: unallocated page inside the visible range
    k_cache = (torch.randn((num_pages, page_size, 1, HEAD_DIM), generator=g) * 0.5).to(dtype)
    q = (torch.randn((rows, HEADS, HEAD_DIM), generator=g)).to(dtype)
    token_to_req = torch.randint(0, reqs, (rows,), generator=g, dtype=torch.int32)
    token_to_req[-1] = -1  # padded row
    pos = torch.empty(rows, dtype=torch.int32)
    for i in range(rows):
        s = int(seq[max(int(token_to_req[i]), 0)])
        pos[i] = int(torch.randint(0, s, (1,), generator=g)) if i % 3 else s - 1  # decode rows at the end
    assert max_visible >= int(seq.max()) // COMPRESS
    return [t.to(dev) for t in (q, k_cache, table, token_to_req, pos, seq.to(torch.int32))]


def _check_scores(args):
    new_logits, new_vis = qsa_ops.qsa_mqa_paged(*args, COMPRESS)
    ref_logits, ref_vis = _full_width(*args, COMPRESS)
    torch.testing.assert_close(new_vis, ref_vis, rtol=0, atol=0)
    assert new_logits.shape == ref_logits.shape
    for r in range(new_logits.shape[0]):
        v = int(ref_vis[r]) if int(args[3][r]) >= 0 else 0
        # bitwise: same values, same -inf at unallocated pages
        assert torch.equal(new_logits[r, :v], ref_logits[r, :v]), f"row {r} differs within visible {v}"
        assert torch.all(ref_logits[r, v:] == -float("inf"))
    return ref_logits, ref_vis


def _torch_reference(q, k_cache, table, token_to_req, pos, seq, row):
    r = int(token_to_req[row]); vis = min((int(pos[row]) + 1) // COMPRESS, int(seq[r]) // COMPRESS)
    cols = torch.arange(vis, device=q.device); page = table[r, cols // k_cache.shape[1]]
    keys = k_cache[page.clamp(min=0).long(), cols % k_cache.shape[1], 0].float()
    s = torch.relu(keys @ q[row].float().T).sum(-1) / HEAD_DIM ** 0.5
    return torch.where(page >= 0, s, torch.full_like(s, -float("inf")))


@pytest.mark.parametrize("max_model_len", [262144, 1000000])
@pytest.mark.parametrize("rows", [1, 3, 64, 2048])
@pytest.mark.parametrize("page_size", [64, 400])
def test_paged_scores_match_full_width(max_model_len, rows, page_size):
    args = _case(rows, max_model_len, page_size, seed=rows * 7 + page_size)
    ref_logits, ref_vis = _check_scores(args)
    for row in [0, rows // 2, rows - 2] if rows > 2 else [0]:
        if int(args[3][row]) < 0:
            continue
        v = int(ref_vis[row]); want = _torch_reference(*args, row)
        torch.testing.assert_close(ref_logits[row, :v], want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("max_model_len", [262144, 1000000])
def test_selection_unchanged(max_model_len, monkeypatch):
    rows, topk = 256, 2048
    args = _case(rows, max_model_len, 64, seed=11)
    out_new = qsa_ops.qsa_select_paged_tokens(*args, topk, COMPRESS)
    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", _full_width)
    out_ref = qsa_ops.qsa_select_paged_tokens(*args, topk, COMPRESS)
    assert torch.equal(out_new.sort(dim=1).values, out_ref.sort(dim=1).values)


def test_cuda_graph_replay_follows_new_lengths():
    rows = 8
    args = _case(rows, 1000000, 64, seed=3)
    q, k_cache, table, token_to_req, pos, seq = args
    qsa_ops.qsa_mqa_paged(*args, COMPRESS)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, vis = qsa_ops.qsa_mqa_paged(*args, COMPRESS)
    for step in range(3):  # decode: positions advance, lengths grow, rows move between requests
        pos.copy_(torch.minimum(pos + 4096 * (step + 1), seq[token_to_req.clamp(min=0).long()] - 1))
        token_to_req.copy_(torch.roll(token_to_req, 1)); token_to_req[0] = -1
        graph.replay()
        ref_logits, ref_vis = _full_width(*args, COMPRESS)
        torch.testing.assert_close(vis, ref_vis, rtol=0, atol=0)
        for r in range(rows):
            v = int(ref_vis[r]) if int(token_to_req[r]) >= 0 else 0
            assert torch.equal(logits[r, :v], ref_logits[r, :v])
