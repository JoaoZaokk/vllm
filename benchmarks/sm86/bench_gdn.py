# SPDX-License-Identifier: Apache-2.0
"""Standalone microbenchmark for the Gated DeltaNet kernels. No model load.

Why this exists: 48 of this model's 64 layers are GDN, so the FLA kernels cover
three quarters of the network. But a server boot costs minutes and every knob
worth sweeping (spec K, KV layout, max_model_len) is fixed at boot, so sweeping
GDN through the server is the slowest possible way to learn anything.

These kernels are importable on their own. Seconds per configuration.

The number this is built to produce is a budget check, not a leaderboard:

    at 44.59 tok/s a token costs 22.4 ms.
    how many of those milliseconds are the 48 GDN layers?

If GDN decode is 2 ms of 22.4, then tuning it by 20% buys 0.4 ms, which is 1.8%
and the whole family is not worth a week. If it is 8 ms, it is worth a lot. Run
this before porting anything.

Shapes come from the real checkpoint (models/awq-w4a16/config.json):
    hidden 5120, 64 layers, full_attention_interval 4 -> 16 attn + 48 GDN
    linear_num_key_heads 16   linear_key_head_dim   128
    linear_num_value_heads 48 linear_value_head_dim 128

Usage:
    python benchmarks/sm86/bench_gdn.py
    python benchmarks/sm86/bench_gdn.py --tokens-per-forward 1 2 3 4 8
"""

from __future__ import annotations

import argparse
import inspect
import statistics

import torch

# Os kernels FLA mudaram de lugar entre as arvores: o 1Cat (base 0.21) os tem em
# model_executor/layers/fla/ops, o 0.27.1 em third_party/flash_linear_attention.
# Importar dos dois mantem o instrumento util em qualquer uma -- e um bench que
# so' roda numa arvore deixa de ser comparavel justamente quando muda a base.
try:
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
except ImportError:  # arvore 1Cat
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

# --- checkpoint geometry ---------------------------------------------------
NUM_GDN_LAYERS = 48
K_HEADS, K_DIM = 16, 128
V_HEADS, V_DIM = 48, 128

# Measured on this machine, 3090, ctx 8192, AWQ W4A16, no draft.
BASELINE_TOK_S = 44.59
BUDGET_MS = 1000.0 / BASELINE_TOK_S


def make_inputs(batch: int, seqlen: int, device: str, dtype: torch.dtype):
    """FLA convention: [B, T, H, D], head_first=False."""
    g = torch.Generator(device=device).manual_seed(0)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=dt)

    return {
        "q": rnd(batch, seqlen, K_HEADS, K_DIM),
        "k": rnd(batch, seqlen, K_HEADS, K_DIM),
        "v": rnd(batch, seqlen, V_HEADS, V_DIM),
        # gate and beta are per value head, and the gate is a log-decay so it
        # must be negative or the recurrence blows up.
        "g": -rnd(batch, seqlen, V_HEADS, dt=torch.float32).abs(),
        "beta": rnd(batch, seqlen, V_HEADS).sigmoid(),
        "initial_state": rnd(batch, V_HEADS, K_DIM, V_DIM, dt=torch.float32),
        "scale": K_DIM**-0.5,
        "output_final_state": True,
        # chunk takes output_final_state; fused_recurrent takes this one instead,
        # and it defaults to True. Left at the default the kernel compiles with
        # INPLACE_FINAL_STATE and then dereferences the ssm_state_indices this
        # bench has no reason to build, so it fails at Triton compile time.
        #
        # False costs representativeness worth naming: real decode does write the
        # state back in place. That write is one 3.1 MiB state per layer, about
        # 4 microseconds of bandwidth against a recurrence measured in hundreds,
        # so it does not move the budget question this bench exists to answer.
        "inplace_final_state": False,
        "use_qk_l2norm_in_kernel": True,
    }


def call(fn, inputs):
    """Pass only the kwargs this function actually declares.

    The chunk and fused_recurrent entry points do not take the same set, and
    they drift between versions. Filtering by signature keeps this bench from
    breaking on a rebase for a reason that has nothing to do with performance.
    """
    accepted = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in inputs.items() if k in accepted})


def time_ms(fn, inputs, iters: int = 50, warmup: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        call(fn, inputs)
    torch.cuda.synchronize()

    samples = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        call(fn, inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    # Median, not mean: a single scheduler hiccup should not move the number,
    # and this machine shares its GPU with other work.
    return statistics.median(samples), samples[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokens-per-forward",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 8],
        help="1 is plain decode; >1 is what MTP/DFlash verify in one pass.",
    )
    p.add_argument("--prefill", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU — este bench nao carrega modelo, mas roda kernel")

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {name}  sm_{cap[0]}{cap[1]}  dtype {args.dtype}")
    print(f"orcamento por token a {BASELINE_TOK_S} tok/s: {BUDGET_MS:.2f} ms")
    print(f"{NUM_GDN_LAYERS} camadas GDN de 64\n")

    print("== decode (fused_recurrent) ==")
    print(f"{'tok/fwd':>8} {'ms/camada':>11} {'ms x48':>9} {'% orcamento':>12}")
    for t in args.tokens_per_forward:
        inputs = make_inputs(args.batch, t, dev, dtype)
        med, _ = time_ms(fused_recurrent_gated_delta_rule, inputs)
        total = med * NUM_GDN_LAYERS
        print(f"{t:>8} {med:>11.4f} {total:>9.2f} {100 * total / BUDGET_MS:>11.1f}%")

    print("\n== prefill (chunk) ==")
    print(f"{'seqlen':>8} {'ms/camada':>11} {'ms x48':>9}")
    for s in args.prefill:
        inputs = make_inputs(args.batch, s, dev, dtype)
        med, _ = time_ms(chunk_gated_delta_rule, inputs, iters=20, warmup=5)
        print(f"{s:>8} {med:>11.4f} {med * NUM_GDN_LAYERS:>9.2f}")

    print(
        "\nLeitura: a coluna de porcentagem e o teto do que QUALQUER tuning de GDN\n"
        "pode comprar no decode. Se ela for pequena, PN365/PN350/PN354/PN299 nao\n"
        "valem o esforco e a fila muda."
    )


if __name__ == "__main__":
    main()
