# SPDX-License-Identifier: Apache-2.0
"""Varredura de M sobre os GEMM reais do Qwen3.5, caminho por caminho.

POR QUE ESTE ARQUIVO EXISTE

A bateria de TTFT mediu 530 ms (ConvRot W4A4) contra 1854 ms (AWQ/Marlin) e eu
dividi 3,5 pela razao de pico de tensor core (4x) para dizer "87% do pico". Isso
foi extrapolacao: TTFT end-to-end carrega atencao, GDN, launch, epilogo, cache e
ate prompts de tamanhos diferentes (2301 contra 2382 tokens). Nao da para tirar
eficiencia de kernel de uma razao dessas.

Aqui a medicao e' do GEMM isolado, nas formas do checkpoint, com M varrido de 1 a
8192. E' isso que responde ONDE as curvas cruzam e POR QUE M=1 doi -- sem
atravessar servidor nenhum.

O QUE SE MEDE

  bf16          F.linear, referencia
  marlin        ops.marlin_gemm com peso AWQ (assimetrico, grupo 128) -- e' o
                caminho que o deploy roda hoje
  convrot_a4    convrot_w4a4_linear(int4): peso e ativacao em int4, MMA nativo
  convrot_a8    convrot_w4a4_linear(int8): ativacao int8 no LAYOUT DE PESO do
                W4A4. E' o que a tabela do compose chama de "W4A8" e nao e'.
  w4a8_real     w4a8_int8_linear: o tier de verdade, layout proprio. Morreu no
                vLLM por quebrar captura de CUDA graph -- mas aqui nao ha model
                runner nem CUDA graph, entao da para medir.

DECOMPOSICAO SEM ABRIR O KERNEL

Nao instrumento por dentro do comfy-kitchen. Em vez disso ajusto uma reta
`tempo = a + b*M` na faixa pequena de M: `a` e' o custo FIXO por chamada
(quantizar ativacao, empacotar, lancar, epilogo) e `b` o custo por linha. Se `a`
dominar em M=1, o misterio acabou -- a GPU nao esta sem potencia, o trabalho
administrativo e' maior que a multiplicacao.

Uso:
    python benchmarks/sm86/sweep_gemm_m.py
    python benchmarks/sm86/sweep_gemm_m.py --ms 1 8 64 512 5856 --formas q_proj down_proj
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

# Formas reais do checkpoint (models/awq-w4a16/config.json):
#   hidden 5120, intermediate 17408, head_dim 256, 24 q heads, 4 kv heads
#   GDN: 16 k heads x 128, 48 v heads x 128 -> qkvz = 2*2048 + 2*6144 = 16384
FORMAS = {
    "q_proj": (5120, 6144),
    "o_proj": (6144, 5120),
    "gate_proj": (5120, 17408),
    "down_proj": (17408, 5120),
    "gdn_qkvz": (5120, 16384),
}

MS_PADRAO = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5856, 8192]


def cronometrar(fn, iters: int, warmup: int) -> float:
    """Mediana em ms. Nada de introspeccao de Python dentro da janela."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    amostras = []
    ini, fim = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        ini.record()
        fn()
        fim.record()
        fim.synchronize()
        amostras.append(ini.elapsed_time(fim))
    return statistics.median(sorted(amostras))


# --- caminhos -------------------------------------------------------------


def montar_bf16(k: int, n: int, dev, dtype):
    w = torch.randn(n, k, device=dev, dtype=dtype) / (k**0.5)

    def fabrica(m):
        x = torch.randn(m, k, device=dev, dtype=dtype)
        return lambda: torch.nn.functional.linear(x, w)

    return fabrica


def montar_marlin(k: int, n: int, dev, dtype, group_size: int = 128):
    """Peso AWQ assimetrico, grupo 128 -- a receita do checkpoint que roda hoje."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        awq_marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    tipo = scalar_types.uint4  # AWQ: 4 bits COM zero point
    b = torch.randn(k, n, device=dev, dtype=dtype) / (k**0.5)
    _, qw, s, zp = awq_marlin_quantize(b, tipo, group_size)
    ws = marlin_make_workspace_new(dev)

    def fabrica(m):
        x = torch.randn(m, k, device=dev, dtype=dtype)
        saida = torch.empty((m, n), dtype=dtype, device=dev)

        def chamar():
            return ops.marlin_gemm(
                x, saida, qw, None, s, None, None, zp, None, None, ws,
                tipo, m, n, k,
                is_k_full=True, use_atomic_add=False,
                use_fp32_reduce=True, is_zp_float=False,
            )

        return chamar

    return fabrica


def montar_convrot(k: int, n: int, dev, dtype, act: str):
    """convrot_w4a4_linear com ativacao int4 ou int8.

    O layout do peso e' o mesmo nos dois -- muda so' o dtype da ativacao. Por
    isso `act=int8` NAO e' o tier W4A8: e' ativacao de 8 bits sobre o layout do
    W4A4.
    """
    from comfy_kitchen import convrot_w4a4_linear
    from comfy_kitchen import quantize_convrot_w4a4_weight as quantizar

    # Os mesmos do quant_config.json do checkpoint que roda hoje.
    CONVROT, QUANT = 256, 64
    b = torch.randn(n, k, device=dev, dtype=dtype) / (k**0.5)
    qw, ws = quantizar(b, convrot_groupsize=CONVROT, quant_group_size=QUANT)

    def fabrica(m):
        x = torch.randn(m, k, device=dev, dtype=dtype)
        return lambda: convrot_w4a4_linear(
            x, qw, ws, None, CONVROT, QUANT, act
        )

    return fabrica


def montar_w4a8(k: int, n: int, dev, dtype):
    """O tier w4a8_int8_linear de verdade: layout qdata + s_rel + s_channel."""
    from comfy_kitchen import quantize_w4a8_int8_weight as quantizar
    from comfy_kitchen import w4a8_int8_linear

    b = torch.randn(n, k, device=dev, dtype=dtype) / (k**0.5)
    qdata, s_rel, s_channel, correcao, codebook = quantizar(b)

    def fabrica(m):
        x = torch.randn(m, k, device=dev, dtype=dtype)
        return lambda: w4a8_int8_linear(
            x, qdata, s_rel, s_channel, codebook=codebook, correction=correcao
        )

    return fabrica


def montar_gemv_awq(k: int, n: int, dev, dtype):
    """gemv_awq_w4a16 do comfy-kitchen: GEMV dedicado para M pequeno.

    Nao e' GEMM: e' o caminho magro. Se ganhar do Marlin em M<=8, e' alavanca de
    DECODE -- justamente onde o W4A4 nao tem o que fazer. Em M grande deve perder
    feio, e isso tambem e' resultado.
    """
    from comfy_kitchen import gemv_awq_w4a16

    GRUPO = 64
    peso = torch.randn(n, k, device=dev, dtype=dtype) / (k**0.5)
    # Empacotamento AWQ simples: 8 pesos de 4 bits por int32, escala e zero por
    # grupo. Nao e' a receita do checkpoint, e' o layout que o kernel espera.
    g = k // GRUPO
    qw = torch.randint(-128, 127, (n, k // 2), device=dev, dtype=torch.int8)
    ws = (peso.view(n, g, GRUPO).abs().amax(-1) / 7.0).to(dtype)
    wz = torch.zeros((n, g), device=dev, dtype=dtype)

    def fabrica(m):
        x = torch.randn(m, k, device=dev, dtype=dtype)
        return lambda: gemv_awq_w4a16(x, qw, ws, wz, None, GRUPO)

    return fabrica


CAMINHOS = {
    "bf16": montar_bf16,
    "gemv_awq": montar_gemv_awq,
    "marlin": montar_marlin,
    "convrot_a4": lambda k, n, d, t: montar_convrot(k, n, d, t, "int4"),
    "convrot_a8": lambda k, n, d, t: montar_convrot(k, n, d, t, "int8"),
    "w4a8_real": montar_w4a8,
}


def ajustar_reta(ms: list[int], tempos: list[float]) -> tuple[float, float]:
    """Minimos quadrados de tempo = a + b*M. `a` e' o custo fixo por chamada."""
    n = len(ms)
    sx, sy = sum(ms), sum(tempos)
    sxx = sum(x * x for x in ms)
    sxy = sum(x * y for x, y in zip(ms, tempos))
    den = n * sxx - sx * sx
    if den == 0:
        return tempos[0], 0.0
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=int, nargs="+", default=MS_PADRAO)
    p.add_argument("--formas", nargs="+", default=list(FORMAS))
    p.add_argument("--caminhos", nargs="+", default=list(CAMINHOS))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json", default=None, help="grava as medicoes em JSONL")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU")

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  {args.dtype}")

    jsonl = open(args.json, "w", encoding="utf-8") if args.json else None

    for nome_forma in args.formas:
        k, n = FORMAS[nome_forma]
        print(f"\n=== {nome_forma}  ({k} -> {n}) ===")

        fabricas, mortos = {}, {}
        for nome in args.caminhos:
            try:
                fabricas[nome] = CAMINHOS[nome](k, n, dev, dtype)
            except Exception as e:  # noqa: BLE001
                # Caminho indisponivel e' resultado, nao falha: e' o que separa
                # "mais lento" de "nao existe nesta placa".
                mortos[nome] = f"{type(e).__name__}: {e}"
        for nome, motivo in mortos.items():
            print(f"  {nome}: INDISPONIVEL — {motivo[:110]}")

        vivos = list(fabricas)
        print("  " + f"{'M':>6}" + "".join(f"{c:>14}" for c in vivos))
        medidas = {c: [] for c in vivos}
        for m in args.ms:
            linha = f"  {m:>6}"
            for c in vivos:
                try:
                    t = cronometrar(fabricas[c](m), args.iters, max(3, args.iters // 5))
                    medidas[c].append((m, t))
                    linha += f"{t:>13.4f}m"
                except Exception as e:  # noqa: BLE001
                    linha += f"{'x':>14}"
                    medidas[c].append((m, None))
                    if m == args.ms[0]:
                        mortos[c] = f"{type(e).__name__}: {e}"
            print(linha, flush=True)
            if jsonl:
                jsonl.write(json.dumps({
                    "forma": nome_forma, "k": k, "n": n, "m": m,
                    "ms": {c: (medidas[c][-1][1]) for c in vivos},
                }) + "\n")

        # Custo fixo por chamada, do ajuste na faixa pequena de M.
        pequenos = [m for m in args.ms if m <= 64]
        if len(pequenos) >= 3:
            print(f"\n  custo fixo por chamada (ajuste em M<=64):")
            for c in vivos:
                pts = [(m, t) for m, t in medidas[c] if m <= 64 and t is not None]
                if len(pts) < 3:
                    continue
                a, b = ajustar_reta([x for x, _ in pts], [y for _, y in pts])
                em1 = next((t for m, t in medidas[c] if m == 1 and t is not None), None)
                frac = f"{100 * a / em1:.0f}% do tempo em M=1" if em1 and em1 > 0 else ""
                print(f"    {c:<12} fixo {a * 1000:>8.1f} us   por linha {b * 1000:>7.3f} us   {frac}")

    if jsonl:
        jsonl.close()
    print(
        "\nLeitura: a coluna do custo fixo diz por que M pequeno doi. Se o fixo for\n"
        "quase todo o tempo em M=1, a placa nao esta sem potencia -- o trabalho\n"
        "administrativo (quantizar ativacao, empacotar, lancar, epilogo) e maior\n"
        "que a multiplicacao."
    )


if __name__ == "__main__":
    main()
