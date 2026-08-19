#!/usr/bin/env bash
# 2x2: {Marlin, ConvRot} x {CUDA graph, enforce-eager}
#
# Separa de uma vez limitacao de KERNEL de limitacao de INTEGRACAO.
#
# A microvarredura mostrou que em M=1 o Marlin gasta 30,8 us no caminho inteiro
# -- menos que o kernel do ConvRot sozinho -- e que o custo FIXO por chamada do
# ConvRot e' 4,7x o do Marlin, enquanto o custo POR LINHA dele e' 3x melhor.
# Assinatura de pedagio de despacho, nao de matematica ruim.
#
# Se o pedagio for a captura de CUDA graph que o ConvRot nao recebe, entao:
#
#   Marlin  graph -> eager   DESABA        (perdeu a captura)
#   ConvRot graph -> eager   NAO MUDA      (nunca teve)
#
# Se os dois desabarem, o ConvRot tambem e' capturado e o alvo e outro.
# Se nenhum mudar, CUDA graph nao explica nada e sobra wrapper/alocacao.
#
# Cache de prefixo DESLIGADO nas quatro: com ele ligado a segunda chamada nao
# paga prefill e o TTFT deixa de medir o caminho de GEMM.
set -uo pipefail

# --------------------------------------------------------------------------
# ARNES APOSENTADO -- recusa rodar por padrao.
#
# Substituido por:
#
#     ./lab run <perfil> curva-g --set enforce_eager=true
#
# Esta bateria DEIXOU DE EXISTIR como conceito. Graph ligado contra
# enforce-eager e uma variavel de configuracao, entao a matriz 2x2 inteira
# sao quatro execucoes da MESMA tarefa, com quatro config_hash distintos:
#
#     ./lab run marlin_single  curva-g
#     ./lab run marlin_single  curva-g --set enforce_eager=true
#     ./lab run convrot_single curva-g
#     ./lab run convrot_single curva-g --set enforce_eager=true
#
# Cada run e identificavel pelo hash, em vez de por lembrar como o script
# foi chamado.
#
# Os defeitos que ficaram para tras, e que a tarefa nao tem:
#
#   - lista de -e escrita a mao, que diverge sem aparecer em lugar nenhum;
#   - porta fixa, em vez de sorteada pelo Docker;
#   - identidade do servidor por porta, sem conferir o container por ID nem
#     perguntar ao /v1/models qual modelo esta sendo servido;
#   - resultado em arquivo compartilhado na raiz, em vez de runs/<run_id>/;
#   - nenhuma procedencia: sem commit, sem digest de imagem, sem seed.
#
# Para rodar assim mesmo -- comparar comportamentos, reproduzir um defeito:
#
#     LAB_ARNES_LEGADO=1 bash bateria_2x2.sh
#
# Recusa em vez de aviso: aviso e' rolado para cima e o numero sai igual.
if [ "${LAB_ARNES_LEGADO:-0}" != "1" ]; then
  sed -n '/^# ARNES APOSENTADO/,/^# igual\.$/p' "${BASH_SOURCE[0]}" >&2
  echo >&2
  echo "RECUSADO: arnes aposentado. Use:  ./lab run <perfil> curva-g --set enforce_eager=true" >&2
  echo "          ou LAB_ARNES_LEGADO=1 para rodar assim mesmo." >&2
  exit 78
fi
# --------------------------------------------------------------------------
export MSYS_NO_PATHCONV=1

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ_MNT="$(cd "$RAIZ" && (pwd -W 2>/dev/null || pwd))"
source "$RAIZ/gpu_lock.sh"
NOME=cell2x2
PORTA=8012
OUT="$RAIZ/resultados_2x2.jsonl"
OUT_MNT="$RAIZ_MNT/resultados_2x2.jsonl"
mkdir -p "$RAIZ/logs"

gpu_lock_pegar "bateria-2x2" 0 || exit 1
trap gpu_lock_soltar EXIT

# Truncar DEPOIS de pegar o lock. Perder a corrida pelo lock ainda destruia o
# resultado da execucao anterior, e a raiz da stack nao e' repositorio git --
# nao havia de onde recuperar.
: > "$OUT"

. "$(dirname "${BASH_SOURCE[0]}")/banner_bateria.sh"
banner_bateria "2x2" \n    "saida              $OUT" || exit 1

#  nome | entrypoint | modelo | eager
CELULAS=(
  "A_marlin_graph|awq_entry.sh|/workspace/models/awq-w4a16|0"
  "B_marlin_eager|awq_entry.sh|/workspace/models/awq-w4a16|1"
  "C_convrot_graph|dspark_entry.sh|/workspace/models/qwen3.8-27b-heretic-convrot-w4a4|0"
  "D_convrot_eager|dspark_entry.sh|/workspace/models/qwen3.8-27b-heretic-convrot-w4a4|1"
)

for linha in "${CELULAS[@]}"; do
  IFS='|' read -r nome entry modelo eager <<< "$linha"
  echo "=============================================================="
  echo "== $nome  (eager=$eager, cache OFF)"
  docker rm -f $NOME >/dev/null 2>&1

  quant=""
  [ "$entry" = "dspark_entry.sh" ] && quant="convrot_w4a4"

  docker run -d --name $NOME --gpus all --ipc host --shm-size=8g -p ${PORTA}:8000 \
    --entrypoint bash \
    -v "${RAIZ_MNT}/models:/workspace/models:ro" \
    -v "${RAIZ_MNT}/docker:/opt/qwen38/docker:ro" \
    -v "${RAIZ_MNT}/plugin/qwen_w4a4_vllm:/usr/local/lib/python3.12/dist-packages/qwen_w4a4_vllm:ro" \
    -v "${RAIZ_MNT}/docker/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e MODEL_PATH="$modelo" -e QUANTIZATION="$quant" \
    -e NUM_SPECULATIVE_TOKENS=0 \
    -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=4 \
    -e GPU_MEMORY_UTILIZATION=0.93 -e KV_CACHE_MEMORY_BYTES= \
    -e LIMIT_MM_PER_PROMPT='{"image":0,"video":0}' \
    -e ENFORCE_EAGER="$eager" -e PREFIX_CACHING=0 \
    -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    qwen38-w4a4:latest "/opt/qwen38/docker/$entry" >/dev/null

  fim=$((SECONDS + 900)); estado=timeout
  while [ $SECONDS -lt $fim ]; do
    docker ps --format '{{.Names}}' | grep -qx $NOME || { estado=morreu; break; }
    curl -sf "http://127.0.0.1:${PORTA}/health" >/dev/null 2>&1 && { estado=subiu; break; }
    sleep 5
  done
  echo "   boot: $estado em ${SECONDS}s"

  if [ "$estado" = subiu ]; then
    # Semente FIXA: as quatro celulas veem o mesmo prompt, entao as saidas sao
    # comparaveis. Cada servidor sobe limpo, entao continua frio mesmo repetido.
    python "$RAIZ_MNT/medir_ttft.py" $PORTA "$nome" fixa 80 | tee -a "$OUT"
  else
    echo "{\"config\":\"$nome\",\"erro\":\"$estado\"}" | tee -a "$OUT"
    docker logs $NOME 2>&1 | grep -iE "error|Error|Traceback" | tail -4
  fi
  docker logs $NOME > "$RAIZ/logs/2x2_${nome}.log" 2>&1
  docker rm -f $NOME >/dev/null 2>&1
  SECONDS=0
done

echo
echo "=============================================================="
python - "$OUT_MNT" <<'PY'
import json, sys
d = {}
for l in open(sys.argv[1], encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        d[r["config"]] = r

print(f"{'celula':<18}{'ttft frio':>11}{'decode':>10}")
for c in ["A_marlin_graph", "B_marlin_eager", "C_convrot_graph", "D_convrot_eager"]:
    r = d.get(c)
    if not r or "erro" in (r or {}):
        print(f"{c:<18}  {(r or {}).get('erro', 'ausente')}")
        continue
    print(f"{c:<18}{r['ttft_frio_ms']:>10.0f}m{r['decode_tok_s']:>10.2f}")

def razao(a, b, campo):
    ra, rb = d.get(a), d.get(b)
    if not ra or not rb or "erro" in ra or "erro" in rb:
        return None
    return ra[campo] / rb[campo]

print("\n== veredito ==")
m = razao("A_marlin_graph", "B_marlin_eager", "decode_tok_s")
c = razao("C_convrot_graph", "D_convrot_eager", "decode_tok_s")
if m is None or c is None:
    raise SystemExit("  incompleto")
print(f"  Marlin  graph/eager = {m:.2f}x")
print(f"  ConvRot graph/eager = {c:.2f}x")
if m > 1.15 and c < 1.08:
    print("  -> Marlin ganha com a captura, ConvRot NAO. O pedagio do ConvRot e")
    print("     ausencia de CUDA graph, e o alvo do conserto esta identificado.")
elif m > 1.15 and c > 1.15:
    print("  -> Os DOIS sao capturados. CUDA graph nao explica a diferenca;")
    print("     o pedagio esta no wrapper/alocacao/despacho.")
elif m < 1.08 and c < 1.08:
    print("  -> Nenhum muda. CUDA graph nao e o culpado nesta configuracao --")
    print("     conferir se a captura estava mesmo ativa antes de concluir.")
else:
    print("  -> Padrao misto. Ler as quatro linhas antes de concluir.")
PY
