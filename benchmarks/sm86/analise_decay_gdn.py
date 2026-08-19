# SPDX-License-Identifier: Apache-2.0
"""Le do checkpoint quanto tempo o estado do GDN LEMBRA -- e portanto por quanto
tempo um erro de quantizacao sobrevive nele.

Motivo. O argumento padrao contra baixar o estado do SSM de 16 bits e que a
recorrencia acumula erro: le-modifica-escreve todo passo, mil passos, mil
arredondamentos. Mas o gate do GDN nao e um A qualquer, e

    g = -exp(A_log) * softplus(a + dt_bias)

com exp(A_log) > 0 e softplus > 0. Entao g < 0 SEMPRE, e exp(g) esta em (0,1)
estritamente: o estado e uma contracao, por construcao. Erro injetado num passo
decai nos seguintes em vez de somar. A pergunta vira quantitativa -- decai
quao rapido? -- e A_log e dt_bias estao no checkpoint, sem quantizacao.

Ponto de operacao a=0. O termo dependente da entrada desloca isso: a positivo
faz softplus crescer, g ficar mais negativo e a memoria ENCURTAR. a negativo faz
o contrario. Isto e uma referencia, nao um limite; medir o a real exige hook num
forward de verdade.

Duas leituras, e a segunda e a que decide:
  meia-vida     quantos passos ate um erro cair a metade
  ganho 1/(1-d^2)  quanto o erro de regime permanente e amplificado -- e o que
                   diz se int8 sobrevive, porque amplitude cresce com sqrt dele

    python benchmarks/sm86/analise_decay_gdn.py
"""

import os
import pathlib
import sys

import numpy as np
from safetensors import safe_open

# Caminho do checkpoint: argumento de linha de comando, variavel de ambiente,
# ou o palpite padrao relativo a raiz deste repositorio.
P = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "GDN_CHECKPOINT",
    str(pathlib.Path(__file__).resolve().parents[2].parent
        / "models" / "awq-w4a16" / "model.safetensors"),
)
if not os.path.exists(P):
    raise SystemExit(
        f"checkpoint nao encontrado: {P} -- "
        "passe o caminho como argumento ou defina GDN_CHECKPOINT"
    )
softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

rows = []
with safe_open(P, framework="pt") as f:
    keys = [k for k in f.keys() if k.endswith("linear_attn.A_log")]
    keys.sort(key=lambda k: int(k.split(".layers.")[1].split(".")[0]))
    for k in keys:
        L = int(k.split(".layers.")[1].split(".")[0])
        A = f.get_tensor(k).float().numpy().astype(np.float64)
        b = f.get_tensor(k.replace("A_log", "dt_bias")).float().numpy().astype(np.float64)
        # g = -exp(A_log) * softplus(a + dt_bias); ponto de operacao a=0
        g = -np.exp(A) * softplus(b)
        rows.append((L, np.exp(A), softplus(b), np.exp(g)))

d = np.stack([r[3] for r in rows])           # [48 camadas, n_heads]
print(f"camadas {d.shape[0]}  heads {d.shape[1]}")
print(f"\ndecay por passo exp(g), em a=0   (1.0 = nao esquece nada)")
for name, fn in [("min", np.min), ("p1", lambda x: np.percentile(x, 1)),
                 ("mediana", np.median), ("p99", lambda x: np.percentile(x, 99)),
                 ("max", np.max)]:
    v = fn(d)
    hl = np.log(0.5) / np.log(v) if 0 < v < 1 else float("inf")
    print(f"  {name:>8}  {v:.6f}   meia-vida do erro: {hl:10.1f} passos")

print(f"\nfracao de heads com decay > 0.999 (memoria > ~693 passos): "
      f"{100*(d > 0.999).mean():.1f}%")
print(f"fracao com decay > 0.9999 (> ~6931 passos):                 "
      f"{100*(d > 0.9999).mean():.1f}%")

worst = np.max(d, axis=1)
print("\ncamadas com a memoria mais longa (decay maximo do head):")
for L in np.argsort(-worst)[:5]:
    hl = np.log(0.5)/np.log(worst[L])
    print(f"  camada {rows[L][0]:>2}  decay {worst[L]:.6f}  meia-vida {hl:9.1f}")

print("\n=== por camada: quantos heads sao lentos (decay > 0.999) ===")
slow = (d > 0.999).sum(axis=1)
for L in range(d.shape[0]):
    if slow[L]:
        print(f"  camada {rows[L][0]:>2}  {slow[L]:>2}/48 heads lentos  "
              f"max decay {d[L].max():.6f}")
print(f"\ncamadas com ZERO heads lentos: {(slow == 0).sum()} de {d.shape[0]}")
tot = d.size
print(f"heads lentos no total: {slow.sum()}/{tot} ({100*slow.sum()/tot:.1f}%)")
print(f"\nse os heads rapidos fossem int8 e os lentos bf16:")
print(f"  estado por camada = {(tot-slow.sum())/tot*0.5 + slow.sum()/tot*1.0:.3f} x o de hoje")

print("\n=== amplificacao do erro em regime permanente: 1/(1-d^2) ===")
print("int8 injeta ~0,4% por passo; a coluna de amplitude diz no que isso vira")
for lab, v in [("mediana", np.median(d)), ("limiar lento", 0.999),
               ("p99", np.percentile(d, 99)), ("pior head", d.max())]:
    gain = 1.0 / (1.0 - v * v)
    print(f"  {lab:>13}  decay {v:.6f}  variancia {gain:10.1f}x  "
          f"amplitude {np.sqrt(gain):6.1f}x  ->  erro final {0.4*np.sqrt(gain):6.1f}%")
