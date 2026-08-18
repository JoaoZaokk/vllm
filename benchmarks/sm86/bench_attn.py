# SPDX-License-Identifier: Apache-2.0
"""How much of prefill is attention? The question that killed vllm#10532.

SageAttention was proposed as a vLLM backend in Nov 2024 (PR #10532, 670 lines,
3 files). A maintainer asked one thing before reviewing: show a throughput
benefit for an LLM. Nobody produced it. The PR died of merge conflicts, and the
question is still open 21 months later -- thu-ml/SageAttention#71 as well.

The community's guess, from the SageAttention2 paper, was "mainly prefill, and
only obvious at very long input". That is testable, and it decides whether a
Sage backend is worth writing for THIS deployment, where prompts are whole
repositories.

Sage replaces the softmax attention in the 16 full-attention layers. It cannot
touch the other 48, which are Gated DeltaNet -- a linear recurrence with no
softmax to quantize. So the ceiling on any Sage win is:

    attention_ms / (attention_ms + gdn_ms + everything_else)

and bench_gdn.py already measured the GDN half on this machine: 47.15 ms at
seqlen 2048, 178.84 ms at 8192, across the 48 layers.

Shapes come from the checkpoint (models/awq-w4a16/config.json):
    64 layers, full_attention_interval 4 -> 16 attention layers
    num_attention_heads 24, num_key_value_heads 4, head_dim 256

Usage:
    python bench_attn.py                      # 2048, 8192, 32768
    python bench_attn.py --seqlens 65536
"""

from __future__ import annotations

import argparse
import statistics

import torch

# --- geometry from the checkpoint -----------------------------------------
NUM_ATTN_LAYERS = 16
NUM_GDN_LAYERS = 48
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256

# From bench_gdn.py on this GPU, same dtype, 48 layers total.
GDN_PREFILL_MS = {2048: 47.15, 8192: 178.84}


def time_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def build(seqlen: int, dtype, device):
    q = torch.randn(seqlen, Q_HEADS, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(seqlen, KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(seqlen, KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    cu = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    return q, k, v, cu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seqlens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU")

    # The backend the server actually uses, not a stand-in: measuring a
    # different attention implementation would answer a different question.
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    dtype = getattr(torch, args.dtype)
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"GPU {props.name}  dtype {args.dtype}")
    print(f"{NUM_ATTN_LAYERS} camadas de atencao, {NUM_GDN_LAYERS} de GDN")
    print(f"q_heads {Q_HEADS}  kv_heads {KV_HEADS}  head_dim {HEAD_DIM}\n")

    print(f"{'seqlen':>7} {'ms/camada':>10} {'ms x16':>9} {'GDN x48':>9} {'attn %':>8} {'teto Sage 2x':>13}")
    for s in args.seqlens:
        q, k, v, cu = build(s, dtype, dev)

        def run():
            flash_attn_varlen_func(
                q=q, k=k, v=v,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=s, max_seqlen_k=s,
                causal=True,
            )

        med = time_ms(run, args.iters, max(2, args.iters // 5))
        attn = med * NUM_ATTN_LAYERS
        gdn = GDN_PREFILL_MS.get(s)
        if gdn:
            share = 100 * attn / (attn + gdn)
            # Sage claims ~2x on attention. Even at 2x the saving is bounded by
            # the share: half of it, and only of the attention half.
            teto = 100 * (attn / 2) / (attn + gdn)
            print(f"{s:>7} {med:>10.3f} {attn:>9.2f} {gdn:>9.2f} {share:>7.1f}% {teto:>12.1f}%")
        else:
            print(f"{s:>7} {med:>10.3f} {attn:>9.2f} {'-':>9} {'-':>8} {'-':>13}")

    print(
        "\nLeitura: 'attn %' e a fatia do prefill que Sage pode tocar; a ultima\n"
        "coluna e o que sobraria se ela fosse 2x mais rapida (o numero anunciado).\n"
        "Se essa coluna for pequena, o PR morreu por um bom motivo."
    )


if __name__ == "__main__":
    main()
