#!/usr/bin/env python3
"""Curva de tempo total contra tokens gerados, num servidor de pe.

Por que existe: eu estimei o cruzamento entre Marlin e ConvRot em ~107 tokens de
saida resolvendo `TTFT + G/taxa` com dois pontos medidos. Dois pontos nao fazem
curva -- a taxa de decode nao precisa ser constante em G, e o TTFT dos dois
caminhos tem ruido diferente (o do ConvRot oscila 12-17%, o do Marlin 0,2%).

Aqui o eixo G e' varrido de verdade, com prompt identico e greedy. O que sai e'
tempo total por requisicao -- que e' o que o usuario espera de fato, nao TTFT
nem tok/s isolados.

O CACHE DE PREFIXO MUDA O QUE ISTO MEDE, e o numero nao se anuncia:

  frio (PREFIX_CACHING=0)  prefill inteiro em toda requisicao. E o que este
                           docstring sempre descreveu.
  quente (=1)              o prompt e' prompt_de(80, "fixa") -- IDENTICO em
                           todo G e toda repeticao dentro do mesmo boot. Da
                           segunda requisicao em diante o prefill e pulado por
                           completo. Isto mede REPETICAO VERBATIM, que e' o teto
                           do beneficio de cache, e NAO o reuso de prefixo que
                           um agente de codigo produz.

Por isso cada linha do jsonl carrega o regime: numero de um arquivo nao pode
depender de lembrar como a fila foi chamada. O regime do agente -- prefixo
grande cacheado mais cauda nova por turno -- e um benchmark que ainda nao
existe (C-003 no registro de evidencia).

Repeticoes por ponto porque o ConvRot e ruidoso: mediana de N, nao uma amostra.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

GS = [1, 16, 32, 64, 96, 128, 192, 256, 512]


def modelo_servido(porta: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{porta}/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def falar(porta: int, modelo: str, prompt: str, g: int) -> tuple[float, dict]:
    corpo = json.dumps({
        "model": modelo, "prompt": prompt, "max_tokens": g,
        "temperature": 0, "seed": 1234,
        # min_tokens obriga o modelo a gerar G de verdade. Sem isto ele pode
        # parar antes e o ponto da curva vira outro G, sem avisar.
        "min_tokens": g,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{porta}/v1/completions", data=corpo,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    return (time.perf_counter() - t0) * 1000.0, d


def prompt_de(n: int, semente: str) -> str:
    return (f"[{semente}] Analise o seguinte trecho de codigo e explique o que ele faz. "
            + " ".join(
                f"linha{i} def funcao_{semente}_{i}(x): return x * {i} + len(str({i}))"
                for i in range(n)))


if __name__ == "__main__":
    porta = int(sys.argv[1])
    rotulo = sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    modelo = modelo_servido(porta)
    p = prompt_de(80, "fixa")

    # Sonda: come o JIT do Triton E a compilacao do caminho de decode. Com
    # max_tokens=1 aquece so' o prefill e a primeira geracao paga a compilacao.
    falar(porta, modelo, prompt_de(80, "sonda"), 4)

    for g in GS:
        amostras = [falar(porta, modelo, p, g) for _ in range(reps)]
        ms = statistics.median(sorted(t for t, _ in amostras))
        d = amostras[-1][1]
        print(json.dumps({
            "config": rotulo, "g": g,
            # O regime vem do ambiente porque quem o escolhe e a bateria. Grava-lo
            # aqui torna cada linha auto-descritiva: numero de um arquivo nao pode
            # depender de lembrar como a fila foi chamada tres dias atras.
            "prefix_caching": os.environ.get("PREFIX_CACHING", "0"),
            "regime": ("quente: prompt identico, prefill pulado da 2a requisicao "
                       "em diante -- mede repeticao verbatim"
                       if os.environ.get("PREFIX_CACHING", "0") not in ("", "0")
                       else "frio: prefill inteiro em toda requisicao"),
            "total_ms": round(ms, 1),
            "gerados": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "espalhamento_pct": round(
                100 * (max(t for t, _ in amostras) - min(t for t, _ in amostras)) / ms, 1),
        }), flush=True)
