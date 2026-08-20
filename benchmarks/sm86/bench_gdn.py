# SPDX-License-Identifier: Apache-2.0
"""Standalone microbenchmark for the Gated DeltaNet kernels. No model load.

Why this exists: 48 of this model's 64 layers are GDN, so these kernels cover
three quarters of the network. A server boot costs minutes and every knob worth
sweeping is fixed at boot, so sweeping GDN through the server is the slowest
possible way to learn anything. These kernels import on their own; seconds per
configuration.

REESCRITO apos auditoria. A versao anterior tinha dois defeitos que juntos
produziram o numero "GDN e 27,8% do orcamento do token", agora riscado em
RESULTADOS.md:

  1. Cronometrava `fused_recurrent_gated_delta_rule`, que ESTE modelo nao chama.
     O decode do Qwen3.5 chama `fused_sigmoid_gating_delta_rule_update`
     (qwen_gdn_linear_attn.py:1404 para o caminho nao-especulativo). A diferenca
     nao e cosmetica: o kernel de producao funde o gating (calcula g a partir de
     A_log/a/b/dt_bias dentro do kernel) e faz gather num POOL de estados por
     `ssm_state_indices`, em vez de receber `g` pronto e um estado ja isolado.
  2. Punha `inspect.signature()` dentro da janela entre os dois CUDA events. Com
     a stream vazia a GPU fica ociosa esperando o dispatch de Python, e isso
     entrava na medicao.

Aqui os kwargs sao ligados UMA vez, fora do laco, e o caminho de producao e o
que manda. O kernel antigo continua medido, rotulado, so para a serie antiga
continuar interpretavel.

RESSALVA QUE NAO SAI: isto e eager, camada isolada. O servidor roda decode sob
CUDA graph (FULL_AND_PIECEWISE captura decode inteiro). Multiplicar por 48 e
comparar com um token de servidor mistura duas populacoes. A razao resultante e
um limite superior de aperto desconhecido, nao "a fracao do token gasta em GDN".
Por isso este bench imprime ms, e imprime a razao rotulada como limite.

Formas vindas do checkpoint (models/awq-w4a16/config.json):
    hidden 5120, 64 camadas, full_attention_interval 4 -> 16 attn + 48 GDN
    linear_num_key_heads 16   linear_key_head_dim   128
    linear_num_value_heads 48 linear_value_head_dim 128

Uso:
    python benchmarks/sm86/bench_gdn.py
    python benchmarks/sm86/bench_gdn.py --seqs 1 2 4 8
    python benchmarks/sm86/bench_gdn.py --chunk-sizes 32 64 128
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys

import torch

# Sentinela de re-entrada. Sem isto, uma forma de CLI que a limpeza de argv nao
# reconhecesse faria o processo se relancar para sempre -- e esta maquina e a
# mesma que o usuario esta usando.
FILHO = os.getenv("_BENCH_GDN_FILHO") == "1"

# Os kernels FLA mudaram de lugar entre as arvores: o 1Cat (base 0.21) os tem em
# model_executor/layers/fla/ops, o 0.27.1 em third_party/flash_linear_attention.
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

# O caminho de decode DE VERDADE deste modelo.
try:
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
except ImportError:
    fused_sigmoid_gating_delta_rule_update = None

# O caminho FlashInfer do prefill. Existe no 0.27.1 e e' inalcancavel nesta
# placa: _resolve_gdn_prefill_backend() so o libera em SM90 (Hopper) ou na
# familia SM100 (Blackwell) com head_k_dim 128 e CUDA >= 13. sm_86 cai no else e
# recebe "triton" -- por arquitetura, nao por medicao.
#
# Chamar o wrapper direto pula esse portao. E' a unica forma de responder se
# levantar o portao para sm_86 compraria alguma coisa, ou se o kernel nem
# compila aqui. As duas respostas encerram a pergunta.
try:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
    )
except ImportError:
    fi_chunk_gated_delta_rule = None

# O tile que os kernels FLA realmente usam. Importado, nao assumido.
try:
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
except ImportError:  # arvore sem o override
    FLA_CHUNK_SIZE = 64

# --- geometria do checkpoint ----------------------------------------------
NUM_GDN_LAYERS = 48
K_HEADS, K_DIM = 16, 128
V_HEADS, V_DIM = 48, 128

# Medido nesta maquina, 3090, ctx 8192, AWQ W4A16, sem draft.
BASELINE_TOK_S = 44.59
BUDGET_MS = 1000.0 / BASELINE_TOK_S


def rnd(gen, *shape, device, dtype):
    return torch.randn(*shape, generator=gen, device=device, dtype=dtype)


def make_decode_inputs(n_seqs: int, device: str, dtype: torch.dtype) -> dict:
    """Monta a chamada de decode como o modelo a faz.

    Uma sequencia por slot, um token por sequencia -- o ramo `split_non_spec` de
    qwen_gdn_linear_attn.py:1404. `initial_state` e o POOL inteiro e o kernel
    busca a linha por `ssm_state_indices`; passar um estado isolado, como a
    versao anterior deste bench fazia, pula justamente esse gather.

    O layout do estado aqui e [slots, HV, V, K] -- V antes de K --, que e o que
    o wrapper assume ao calcular `stride_init_state_token`. O caminho de chunk
    usa [B, HV, K, V]. Trocar os dois nao levanta erro, so' le lixo.
    """
    g = torch.Generator(device=device).manual_seed(0)
    T = n_seqs  # um token por sequencia

    return {
        "A_log": -rnd(g, V_HEADS, device=device, dtype=torch.float32).abs(),
        "dt_bias": rnd(g, V_HEADS, device=device, dtype=torch.float32),
        # a e b sao [T, HV] achatados: o kernel indexa `bos * HV + i_hv`.
        "a": rnd(g, T, V_HEADS, device=device, dtype=torch.float32),
        "b": rnd(g, T, V_HEADS, device=device, dtype=torch.float32),
        "q": rnd(g, 1, T, K_HEADS, K_DIM, device=device, dtype=dtype),
        "k": rnd(g, 1, T, K_HEADS, K_DIM, device=device, dtype=dtype),
        "v": rnd(g, 1, T, V_HEADS, V_DIM, device=device, dtype=dtype),
        "initial_state": rnd(
            g, n_seqs, V_HEADS, V_DIM, K_DIM, device=device, dtype=torch.float32
        ),
        "inplace_final_state": True,
        "cu_seqlens": torch.arange(n_seqs + 1, device=device, dtype=torch.int32),
        "ssm_state_indices": torch.arange(n_seqs, device=device, dtype=torch.int32),
        "use_qk_l2norm_in_kernel": True,
    }


def make_legacy_inputs(seqlen: int, device: str, dtype: torch.dtype) -> dict:
    """Convencao do caminho de chunk e do kernel legado: [B, T, H, D]."""
    g = torch.Generator(device=device).manual_seed(0)
    return {
        "q": rnd(g, 1, seqlen, K_HEADS, K_DIM, device=device, dtype=dtype),
        "k": rnd(g, 1, seqlen, K_HEADS, K_DIM, device=device, dtype=dtype),
        "v": rnd(g, 1, seqlen, V_HEADS, V_DIM, device=device, dtype=dtype),
        # g e log-decay: tem que ser negativo ou a recorrencia explode.
        "g": -rnd(g, 1, seqlen, V_HEADS, device=device, dtype=torch.float32).abs(),
        "beta": rnd(g, 1, seqlen, V_HEADS, device=device, dtype=dtype).sigmoid(),
        "initial_state": rnd(
            g, 1, V_HEADS, K_DIM, V_DIM, device=device, dtype=torch.float32
        ),
        "scale": K_DIM**-0.5,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }


def bind(fn, inputs: dict) -> dict:
    """Filtra kwargs pela assinatura UMA vez, fora de qualquer janela de tempo.

    As entradas de chunk, do kernel legado e do de producao nao aceitam o mesmo
    conjunto, e isso muda entre versoes. Filtrar aqui mantem o bench vivo num
    rebase sem por introspeccao de Python dentro da medicao.
    """
    import inspect

    aceitos = inspect.signature(fn).parameters
    return {k: v for k, v in inputs.items() if k in aceitos}


def time_ms(fn, kwargs: dict, iters: int = 50, warmup: int = 10) -> tuple[float, float]:
    """Mediana e minimo em ms. `kwargs` ja vem ligado -- nada de Python pesado aqui."""
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    samples = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        fn(**kwargs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    # Mediana, nao media: um soluco do escalonador nao deve mover o numero, e
    # esta maquina divide a GPU com outro trabalho.
    return statistics.median(samples), samples[0]


def relaunch_per_chunk_size(sizes: list[int], args: argparse.Namespace) -> int:
    """Roda o bench uma vez por chunk size, cada um no seu processo.

    FLA_CHUNK_SIZE liga em uma duzia de argumentos default no momento do import.
    Um laco dentro de um processo so' moveria o global para quem o le vivo e
    deixaria obsoleto quem capturou o default -- dois tiles na mesma cadeia de
    kernel, o que e' resposta errada, nao lenta. Um processo por valor e' a
    unica varredura honesta.

    O argv do filho e' RECONSTRUIDO a partir dos argumentos parseados, nunca
    filtrado de sys.argv. Filtrar texto nao enxerga `--chunk-sizes=32` nem
    valores negativos, e a versao anterior deste arquivo se relancava para
    sempre com a primeira forma.
    """
    argv = [
        "--seqs", *[str(x) for x in args.seqs],
        "--prefill", *[str(x) for x in args.prefill],
        "--dtype", args.dtype,
    ]
    if args.pular_legado:
        argv.append("--pular-legado")

    pior = 0
    for s in sizes:
        print(f"\n{'=' * 64}\nVLLM_FLA_CHUNK_SIZE={s}\n{'=' * 64}", flush=True)
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), *argv],
            env=dict(os.environ, VLLM_FLA_CHUNK_SIZE=str(s), _BENCH_GDN_FILHO="1"),
        ).returncode
        if rc != 0:
            # Nao compilar num tamanho e' resultado: tile grande demais estoura a
            # shared memory do sm_86. Segue para o proximo em vez de parar.
            print(f"  chunk {s}: saiu com codigo {rc}", flush=True)
            pior = rc
    return pior


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
        "--seqs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Sequencias simultaneas no decode, um token cada. E' o batch real.",
    )
    p.add_argument("--prefill", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--pular-legado",
        action="store_true",
        help="Nao mede fused_recurrent_gated_delta_rule (fora do caminho de producao).",
    )
    args = p.parse_args()

    if args.chunk_sizes:
        if FILHO:
            raise SystemExit(
                "--chunk-sizes num processo filho: recursao. Isto e' bug do bench."
            )
        raise SystemExit(relaunch_per_chunk_size(args.chunk_sizes, args))

    if any(s < 1 for s in args.seqs) or any(s < 1 for s in args.prefill):
        raise SystemExit("--seqs e --prefill exigem valores positivos")

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU — este bench nao carrega modelo, mas roda kernel")

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  dtype {args.dtype}")
    print(f"FLA_CHUNK_SIZE {FLA_CHUNK_SIZE}")
    print(f"orcamento por token a {BASELINE_TOK_S} tok/s: {BUDGET_MS:.2f} ms")
    print(f"{NUM_GDN_LAYERS} camadas GDN de 64\n")

    # ---- decode: o caminho que o modelo realmente executa ------------------
    if fused_sigmoid_gating_delta_rule_update is None:
        print("== decode == kernel de producao ausente nesta arvore, pulando\n")
    else:
        print("== decode (fused_sigmoid_gating_delta_rule_update) == CAMINHO DE PRODUCAO")
        print(f"{'seqs':>6} {'ms/camada':>11} {'ms x48':>9} {'limite sup.':>12}")
        for n in args.seqs:
            entradas = make_decode_inputs(n, dev, dtype)
            kw = bind(fused_sigmoid_gating_delta_rule_update, entradas)
            med, _ = time_ms(fused_sigmoid_gating_delta_rule_update, kw)
            total = med * NUM_GDN_LAYERS
            print(
                f"{n:>6} {med:>11.4f} {total:>9.2f} {100 * total / BUDGET_MS:>11.1f}%"
            )
        print(
            "  'limite sup.' NAO e' a fracao do token: numerador eager e camada\n"
            "  isolada, denominador e' token de servidor com CUDA graph.\n"
        )

    # ---- decode legado: fora do caminho, mantido so' para a serie antiga ---
    if not args.pular_legado:
        print("== decode (fused_recurrent_gated_delta_rule) == FORA DO CAMINHO")
        print("  este modelo nao chama este kernel; medido so' para comparar com")
        print("  a serie anterior, que foi construida em cima dele por engano")
        print(f"{'seqs':>6} {'ms/camada':>11} {'ms x48':>9}")
        for n in args.seqs:
            entradas = make_legacy_inputs(n, dev, dtype)
            # O legado default a INPLACE_FINAL_STATE, que exige ssm_state_indices
            # que esta chamada nao tem; desligar e' o que o deixa compilar aqui.
            entradas["inplace_final_state"] = False
            kw = bind(fused_recurrent_gated_delta_rule, entradas)
            med, _ = time_ms(fused_recurrent_gated_delta_rule, kw)
            print(f"{n:>6} {med:>11.4f} {med * NUM_GDN_LAYERS:>9.2f}")
        print()

    # ---- prefill -----------------------------------------------------------
    print("== prefill (chunk) ==")
    print(f"{'seqlen':>8} {'kernel':>10} {'ms/camada':>11} {'ms x48':>9}")
    fi_vivo = fi_chunk_gated_delta_rule is not None
    for s in args.prefill:
        entradas = make_legacy_inputs(s, dev, dtype)

        # Sem cu_seqlens: e' a chamada que produziu os numeros da serie antiga.
        kw = bind(chunk_gated_delta_rule, entradas)
        med, _ = time_ms(chunk_gated_delta_rule, kw, iters=20, warmup=5)
        print(f"{s:>8} {'fla':>10} {med:>11.4f} {med * NUM_GDN_LAYERS:>9.2f}")

        # Daqui em diante, a convencao que o vLLM usa no prefill: uma sequencia
        # empacotada descrita por cu_seqlens. Comparar fla contra flashinfer
        # exige as duas do mesmo lado dessa linha.
        varlen = dict(
            entradas,
            cu_seqlens=torch.tensor([0, s], device=dev, dtype=torch.int32),
        )
        kw_v = bind(chunk_gated_delta_rule, varlen)
        med_v, _ = time_ms(chunk_gated_delta_rule, kw_v, iters=20, warmup=5)
        print(f"{s:>8} {'fla+cu':>10} {med_v:>11.4f} {med_v * NUM_GDN_LAYERS:>9.2f}")

        if not fi_vivo:
            continue
        try:
            kw_f = bind(fi_chunk_gated_delta_rule, varlen)
            # Primeira chamada isolada: o FlashInfer compila por JIT, e essa
            # compilacao ja contaminou uma medicao antes. Custo de boot, nao de
            # token -- separado, nao escondido.
            ev = torch.cuda.Event(True), torch.cuda.Event(True)
            ev[0].record()
            fi_chunk_gated_delta_rule(**kw_f)
            ev[1].record()
            ev[1].synchronize()
            med_f, _ = time_ms(fi_chunk_gated_delta_rule, kw_f, iters=20, warmup=5)
            print(
                f"{s:>8} {'fi':>10} {med_f:>11.4f} {med_f * NUM_GDN_LAYERS:>9.2f}"
                f"   (1a chamada {ev[0].elapsed_time(ev[1]):.0f} ms, JIT)"
            )
            print(
                f"{'':>8} {'':>10} {'':>11} {'ganho':>9}"
                f"   {100 * (med_v - med_f) / med_v:+.1f}% vs fla+cu"
            )
        except Exception as e:  # noqa: BLE001
            # Nao compilar aqui e' resultado, nao falha do bench: o portao de
            # arquitetura esta certo e a alavanca morre. Desliga o fi para os
            # proximos seqlens (vai falhar igual) mas NAO aborta a varredura:
            # abortar sumia com a linha de 8192 sem mensagem nenhuma.
            print(f"{s:>8} {'fi':>10} {'--':>11} {'--':>9}   {type(e).__name__}: {e}")
            fi_vivo = False

    print(
        "\nLeitura: a coluna de decode do caminho de PRODUCAO e' a unica que responde\n"
        "quanto qualquer tuning de GDN pode comprar. E responde como limite superior:\n"
        "o servidor captura o decode inteiro em CUDA graph, esta medicao nao.\n"
        "\n"
        "A linha fi so' vale para prefill/TTFT. O decode nao tem alternativa de\n"
        "backend: e' Triton em qualquer configuracao."
    )


if __name__ == "__main__":
    main()
