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
import os
import statistics
import subprocess
import sys

import torch

# O tamanho do tile que os kernels FLA realmente usam. Importado, nao assumido:
# se a arvore nao le a variavel de ambiente, o numero impresso tem que ser o que
# o kernel viu, senao a varredura mente sobre o proprio eixo.
try:
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
except ImportError:  # arvore 1Cat, sem o override
    FLA_CHUNK_SIZE = 64

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

# O caminho FlashInfer do prefill GDN. Existe no 0.27.1 e e' inalcancavel nesta
# placa: _resolve_gdn_prefill_backend() so' o libera em SM90 (Hopper) ou na
# familia SM100 (Blackwell) com head_k_dim 128 e CUDA >= 13. sm_86 cai no else e
# recebe "triton" -- por arquitetura, nao por medicao.
#
# Chamar o wrapper direto pula esse portao. E' a unica forma de responder se
# levantar o portao para sm_86 compraria alguma coisa, ou se o kernel nem
# compila aqui. As duas respostas encerram a pergunta; nenhuma delas sai de ler
# codigo.
try:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
    )
except ImportError:
    fi_chunk_gated_delta_rule = None

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


def relaunch_per_chunk_size(sizes: list[int]) -> int:
    """Run this bench once per chunk size, each in its own process.

    FLA_CHUNK_SIZE binds into a dozen default arguments at import time. A loop
    inside one process would move the global for the callers that read it live
    and leave it stale for the ones that captured the default -- two tile sizes
    in one kernel chain, which is a wrong answer, not a slow one. One process
    per value is the only honest sweep.
    """
    argv, skip = [], False
    for a in sys.argv[1:]:
        if a == "--chunk-sizes":
            skip = True
            continue
        if skip:
            if not a.startswith("-"):
                continue
            skip = False
        argv.append(a)

    worst = 0
    for s in sizes:
        print(f"\n{'=' * 60}\nVLLM_FLA_CHUNK_SIZE={s}\n{'=' * 60}", flush=True)
        rc = subprocess.run(
            [sys.executable, __file__, *argv],
            env=dict(os.environ, VLLM_FLA_CHUNK_SIZE=str(s)),
        ).returncode
        if rc != 0:
            # Nao compilar num tamanho e resultado: tile grande demais estoura a
            # shared memory do sm_86. Segue para o proximo em vez de parar.
            print(f"  chunk {s}: saiu com codigo {rc}", flush=True)
            worst = rc
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Varre VLLM_FLA_CHUNK_SIZE, um subprocesso por valor. Ex: 32 64 128.",
    )
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

    if args.chunk_sizes:
        raise SystemExit(relaunch_per_chunk_size(args.chunk_sizes))

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU — este bench nao carrega modelo, mas roda kernel")

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {name}  sm_{cap[0]}{cap[1]}  dtype {args.dtype}")
    print(f"FLA_CHUNK_SIZE {FLA_CHUNK_SIZE}")
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
    print(f"{'seqlen':>8} {'kernel':>10} {'ms/camada':>11} {'ms x48':>9}")
    for s in args.prefill:
        inputs = make_inputs(args.batch, s, dev, dtype)

        # Sem cu_seqlens: e' a chamada que produziu os numeros ja' gravados em
        # RESULTADOS.md. Fica para a serie nao quebrar quando este arquivo muda.
        med, _ = time_ms(chunk_gated_delta_rule, inputs, iters=20, warmup=5)
        print(f"{s:>8} {'fla':>10} {med:>11.4f} {med * NUM_GDN_LAYERS:>9.2f}")

        # A partir daqui, a convencao que o vLLM realmente usa no prefill: uma
        # sequencia empacotada, descrita por cu_seqlens. Comparar fla contra
        # flashinfer exige as duas do mesmo lado dessa linha.
        varlen = dict(inputs, cu_seqlens=torch.tensor([0, s], device=dev, dtype=torch.int32))
        med_v, _ = time_ms(chunk_gated_delta_rule, varlen, iters=20, warmup=5)
        print(f"{s:>8} {'fla+cu':>10} {med_v:>11.4f} {med_v * NUM_GDN_LAYERS:>9.2f}")

        if fi_chunk_gated_delta_rule is None:
            continue
        try:
            # Primeira chamada isolada: o FlashInfer compila por JIT, e essa
            # compilacao ja' contaminou uma medicao minha antes. O custo dela e'
            # um numero de boot, nao de token -- separado, nao escondido.
            jit = torch.cuda.Event(True), torch.cuda.Event(True)
            jit[0].record()
            call(fi_chunk_gated_delta_rule, varlen)
            jit[1].record()
            jit[1].synchronize()
            med_f, _ = time_ms(fi_chunk_gated_delta_rule, varlen, iters=20, warmup=5)
            print(
                f"{s:>8} {'fi':>10} {med_f:>11.4f} {med_f * NUM_GDN_LAYERS:>9.2f}"
                f"   (1a chamada {jit[0].elapsed_time(jit[1]):.0f} ms, JIT)"
            )
            print(f"{'':>8} {'':>10} {'':>11} {'ganho':>9}"
                  f"   {100 * (med_v - med_f) / med_v:+.1f}% vs fla+cu")
        except Exception as e:  # noqa: BLE001
            # Nao compilar aqui e' resultado, nao falha do bench: significa que
            # o portao de arquitetura esta' certo e a alavanca morre.
            print(f"{s:>8} {'fi':>10} {'--':>11} {'--':>9}   {type(e).__name__}: {e}")
            break

    print(
        "\nLeitura: a coluna de porcentagem e o teto do que QUALQUER tuning de GDN\n"
        "pode comprar no decode. Se ela for pequena, PN365/PN350/PN354/PN299 nao\n"
        "valem o esforco e a fila muda.\n"
        "\n"
        "A linha fi so' importa para prefill/TTFT. O decode nao tem alternativa:\n"
        "fused_recurrent e' Triton em qualquer backend."
    )


if __name__ == "__main__":
    main()
