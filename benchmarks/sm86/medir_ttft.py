#!/usr/bin/env python3
"""Mede TTFT frio, TTFT quente (acerto de prefix cache) e decode, num servidor de pe.

Ordem das chamadas nao e arbitraria:

  sonda   prompt descartavel. Absorve o JIT do Triton, que ja contaminou uma
          medicao minha antes -- a primeira requisicao de qualquer boot mede
          compilador, nao prefill.
  frio    prompt NOVO, nunca visto. max_tokens=1, entao o relogio e' quase todo
          prefill. Isto e' o TTFT de verdade.
  quente  o MESMO prompt de novo. Se o prefix caching estiver funcionando, o
          prefill nao acontece e o numero desaba. A razao frio/quente e' a
          resposta que ninguem nesta stack tinha.
  decode  max_tokens=128 sobre o prompt ja quente, para tok/s sair sem o
          prefill dentro dele.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def falar(porta: int, prompt: str, max_tokens: int, timeout: int = 600) -> tuple[float, dict]:
    corpo = json.dumps({
        "model": MODELO,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1234,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{porta}/v1/completions",
        data=corpo,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # O corpo diz o que foi recusado; sem ele o 400 nao ensina nada.
        raise SystemExit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}")
    return (time.perf_counter() - t0) * 1000.0, d


def prompt_de(n_palavras: int, semente: str) -> str:
    """Texto deterministico e longo. ~25 tokens por item, entao n=80 fica na casa
    de 2 mil -- folgado sob 4096, que foi onde a versao anterior levou 400.

    A semente muda o prefixo INTEIRO, e isso e' o ponto: um prompt "novo" precisa
    nao casar com nenhum bloco ja cacheado, senao o frio ja nasce quente."""
    base = (
        f"[{semente}] Analise o seguinte trecho de codigo e explique o que ele faz. "
    )
    corpo = " ".join(
        f"linha{i} def funcao_{semente}_{i}(x): return x * {i} + len(str({i}))"
        for i in range(n_palavras)
    )
    return base + corpo


def modelo_servido(porta: int) -> str:
    """Pergunta ao servidor em vez de receber o caminho por argumento.

    Passar "/workspace/models/..." pela linha de comando no Git Bash faz o MSYS
    reescrever para "C:/Program Files/Git/workspace/models/...", e o servidor
    responde 404 sobre um caminho que ninguem digitou. Perguntar elimina a
    classe inteira.
    """
    with urllib.request.urlopen(f"http://127.0.0.1:{porta}/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


if __name__ == "__main__":
    porta = int(sys.argv[1])
    rotulo = sys.argv[2]
    semente = sys.argv[3] if len(sys.argv) > 3 else "a"
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    MODELO = modelo_servido(porta)

    saida = {"config": rotulo}

    # Sonda com max_tokens=4, nao 1. Com 1 token ela exercita so' o PREFILL, e a
    # primeira geracao de verdade paga a compilacao do caminho de decode -- foi
    # assim que a validacao mediu 1,24 tok/s num servidor que faz 44.
    ms, _ = falar(porta, prompt_de(n, f"sonda{semente}"), 4)
    saida["sonda_ms"] = round(ms, 1)

    p = prompt_de(n, semente)

    ms, d = falar(porta, p, 1)
    saida["ttft_frio_ms"] = round(ms, 1)
    saida["prompt_tokens"] = d["usage"]["prompt_tokens"]

    ms, _ = falar(porta, p, 1)
    saida["ttft_quente_ms"] = round(ms, 1)
    saida["ganho_cache_pct"] = round(
        100 * (saida["ttft_frio_ms"] - saida["ttft_quente_ms"]) / saida["ttft_frio_ms"], 1
    )

    ms, d = falar(porta, p, 128)
    gerados = d["usage"]["completion_tokens"]
    # subtrai o TTFT quente: o que sobra e' decode puro
    decode_ms = ms - saida["ttft_quente_ms"]
    saida["gerados"] = gerados
    saida["decode_tok_s"] = round(1000.0 * gerados / decode_ms, 2) if decode_ms > 0 else None
    saida["texto_120"] = d["choices"][0]["text"][:120].replace("\n", " ")

    print(json.dumps(saida, ensure_ascii=False))
