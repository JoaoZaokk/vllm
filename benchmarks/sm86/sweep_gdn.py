# SPDX-License-Identifier: Apache-2.0
"""Compare GDN decode CTA geometries on SM 8.6, honestly.

NAO roda no 0.27.1 como esta: ele escreve _SM86_FLA_WARPS_OVERRIDE e
_SM86_FLA_BV_OVERRIDE, seletores que existiam so' no ramo SM86 que criei na
arvore do 1Cat. Aquele ramo nao foi portado de proposito -- a varredura nao
achou geometria que batesse o default fora do ruido de 87% entre rodadas.
Fica aqui como instrumento pronto caso a pergunta volte, com o metodo (rodadas
intercaladas, ancora inicial e final) que e' a parte que custou a aprender.

bench_gdn.py answers "how much does GDN cost". This one answers "can the launch
geometry buy any of it back" -- the part that changes without touching a kernel.

The first version of this file measured every configuration once, in order, and
reported deltas against whichever one ran first. It produced a confident -30%
that evaporated on the second run: the GPU had been cold, the baseline ran
first, and everything after it inherited the warmed-up clocks. Run-to-run drift
here is about 10%, and the effects being compared are about 5%, so measurement
order was worth more than the thing being measured.

So: clocks are warmed first, configurations are measured INTERLEAVED within a
round, and the reported number is the median across rounds. Drift that moves
slower than a round cancels, because every configuration eats the same amount
of it.

Usage:
    python benchmarks/sm86/sweep_gdn.py [--rounds 7]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from bench_gdn import BASELINE_TOK_S, BUDGET_MS, NUM_GDN_LAYERS, make_inputs, time_ms  # noqa: E402

import vllm.model_executor.layers.fla.ops.fused_recurrent as fr  # noqa: E402

# (warps, BV). First entry is the stock default and the comparison baseline.
CONFIGS = [(1, 32), (1, 16), (2, 16), (4, 16), (8, 16), (1, 8)]
TOKENS = [1, 4]


def medir(w: int, bv: int, t: int, inputs_cache: dict) -> float:
    fr._SM86_FLA_WARPS_OVERRIDE = w
    fr._SM86_FLA_BV_OVERRIDE = bv
    inputs = inputs_cache.setdefault(t, make_inputs(1, t, "cuda", torch.bfloat16))
    med, _ = time_ms(fr.fused_recurrent_gated_delta_rule, inputs, iters=30, warmup=5)
    return med * NUM_GDN_LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU {props.name}  {props.multi_processor_count} SMs")
    print(f"orcamento {BUDGET_MS:.2f} ms/token, {NUM_GDN_LAYERS} camadas GDN")
    print(f"{args.rounds} rodadas intercaladas, mediana entre rodadas\n")

    cache: dict = {}
    for _ in range(4):          # sobe os clocks antes de qualquer numero valer
        medir(1, 32, 1, cache)

    amostras: dict = {(w, bv, t): [] for w, bv in CONFIGS for t in TOKENS}
    for _ in range(args.rounds):
        for t in TOKENS:
            for w, bv in CONFIGS:
                amostras[(w, bv, t)].append(medir(w, bv, t, cache))

    print(f"{'warps':>5} {'BV':>4} {'CTAs':>5} " + "".join(f"{'T='+str(t):>22}" for t in TOKENS))
    base = {t: statistics.median(amostras[(*CONFIGS[0], t)]) for t in TOKENS}
    for w, bv in CONFIGS:
        linha = f"{w:>5} {bv:>4} {(128 // bv) * 48:>5} "
        for t in TOKENS:
            v = amostras[(w, bv, t)]
            med, lo, hi = statistics.median(v), min(v), max(v)
            delta = 100 * (med - base[t]) / base[t]
            linha += f"{med:>7.2f}ms {delta:>+6.1f}% [{hi - lo:>4.2f}] "
        print(linha)

    espalhamento = max(
        100 * (max(v) - min(v)) / statistics.median(v) for v in amostras.values()
    )
    print(f"\n[ ] = amplitude entre rodadas da MESMA config. Maior espalhamento: {espalhamento:.1f}%.")
    print("Uma diferenca menor que isso nao e um resultado, e ruido com opiniao.")


if __name__ == "__main__":
    main()
