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

# Do bench_gdn.py nesta GPU, mesmo dtype, 48 camadas, caminho de chunk (que E'
# o de producao no prefill: forward_native -> fla_chunk_gated_delta_rule).
# Remedido 18/ago/2026 depois de tirar o inspect.signature da janela de tempo.
GDN_PREFILL_MS = {2048: 44.04, 8192: 178.81}


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


# Os GEMM que a versao anterior deste arquivo esquecia. O docstring acima sempre
# disse `attn / (attn + gdn + everything_else)`, mas o codigo dividia so por
# (attn + gdn) -- e num 27B o "everything_else" e' o termo que domina o prefill.
# Sem ele a fatia da atencao inflava, exatamente na direcao que fazia o Sage
# parecer valer a pena.
#
# Formas do checkpoint: hidden 5120, intermediate 17408, head_dim 256.
#   atencao: qkv (24+2*4)*256 = 8192, o_proj 24*256 = 6144
#   GDN:     qkvz = 2*16*128 + 2*48*128 = 16384, ba = 2*48 = 96, out 48*128 = 6144
ATTN_GEMMS = [(5120, 8192), (6144, 5120)]
GDN_GEMMS = [(5120, 16384), (5120, 96), (6144, 5120)]
MLP_GEMMS = [(5120, 17408), (5120, 17408), (17408, 5120)]


def gemm_ms(seqlen: int, dtype, device, iters: int) -> float:
    """Piso do custo de GEMM do prefill inteiro, em ms.

    PISO, nao estimativa: mede matmul denso em bf16, e o deploy roda AWQ W4A16,
    que ainda paga a dequantizacao. O numero real e' maior, entao a fatia da
    atencao calculada contra ele e' um limite SUPERIOR -- que e' o lado seguro
    para decidir se vale escrever um backend.
    """
    total = 0.0
    for n_camadas, gemms in (
        (NUM_ATTN_LAYERS, ATTN_GEMMS + MLP_GEMMS),
        (NUM_GDN_LAYERS, GDN_GEMMS + MLP_GEMMS),
    ):
        for entrada, saida in gemms:
            x = torch.randn(seqlen, entrada, device=device, dtype=dtype)
            w = torch.randn(entrada, saida, device=device, dtype=dtype)
            total += n_camadas * time_ms(lambda: torch.matmul(x, w), iters, 3)
            del x, w
            torch.cuda.empty_cache()
    return total


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

    print(f"{'seqlen':>7} {'ms/camada':>10} {'ms x16':>9} {'GDN x48':>9} {'GEMM':>9} {'attn %':>8} {'teto Sage 2x':>13}")
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
        gemm = gemm_ms(s, dtype, dev, args.iters)
        if gdn:
            denom = attn + gdn + gemm
            share = 100 * attn / denom
            # Sage anuncia ~2x na atencao. Mesmo a 2x a economia e' limitada pela
            # fatia: metade dela, e so' da metade que e' atencao.
            teto = 100 * (attn / 2) / denom
            print(
                f"{s:>7} {med:>10.3f} {attn:>9.2f} {gdn:>9.2f} {gemm:>9.2f}"
                f" {share:>7.1f}% {teto:>12.1f}%"
            )
        else:
            print(
                f"{s:>7} {med:>10.3f} {attn:>9.2f} {'-':>9} {gemm:>9.2f}"
                f" {'-':>8} {'-':>13}"
            )

    print(
        "\nLeitura: 'attn %' e a fatia do prefill que Sage pode tocar, agora contra\n"
        "o denominador inteiro (atencao + GDN + GEMM). A ultima coluna e o que\n"
        "sobraria se a atencao fosse 2x mais rapida. Se ela for pequena, o PR\n"
        "morreu por um bom motivo.\n"
        "\n"
        "A coluna GEMM e' PISO: matmul denso em bf16, enquanto o deploy paga AWQ\n"
        "W4A16 com dequantizacao por cima. Logo 'attn %' aqui e' limite SUPERIOR.\n"
        "A versao anterior deste arquivo omitia essa coluna inteira e inflava a\n"
        "fatia da atencao em quase uma ordem de grandeza."
    )


if __name__ == "__main__":
    main()
