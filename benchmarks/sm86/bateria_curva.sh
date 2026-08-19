#!/usr/bin/env bash
# Curva de tempo total contra G, nos dois caminhos. Dois boots apenas: o eixo G
# varre dentro do mesmo servidor.
#
# Decide o que a estimativa de ~107 tokens so' levantava como hipotese. E' a
# escolha que importa agora, porque manter as duas representacoes de peso na
# VRAM esta descartado -- 17,41 + 17,78 = 35,19 GiB contra 24 da placa.
set -uo pipefail
export MSYS_NO_PATHCONV=1

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ_MNT="$(cd "$RAIZ" && (pwd -W 2>/dev/null || pwd))"
source "$RAIZ/gpu_lock.sh"
NOME=curva
PORTA=8013
CACHE="${PREFIX_CACHING:-0}"
OUT="$RAIZ/resultados_curva${CACHE:+_cache$CACHE}.jsonl"
mkdir -p "$RAIZ/logs"

gpu_lock_pegar "curva-g-cache$CACHE" 0 || exit 1
trap gpu_lock_soltar EXIT

# Truncar DEPOIS de pegar o lock. Perder a corrida pelo lock ainda destruia o
# resultado da execucao anterior, e a raiz da stack nao e' repositorio git --
# nao havia de onde recuperar.
: > "$OUT"

CONFIGS=(
  "marlin|awq_entry.sh|/workspace/models/awq-w4a16|"
  "convrot|dspark_entry.sh|/workspace/models/qwen3.8-27b-heretic-convrot-w4a4|convrot_w4a4"
)

for linha in "${CONFIGS[@]}"; do
  IFS='|' read -r nome entry modelo quant <<< "$linha"
  echo "== $nome =="
  docker rm -f $NOME >/dev/null 2>&1
  docker run -d --name $NOME --gpus all --ipc host --shm-size=8g -p ${PORTA}:8000 \
    --entrypoint bash \
    -v "${RAIZ_MNT}/models:/workspace/models:ro" \
    -v "${RAIZ_MNT}/docker:/opt/qwen38/docker:ro" \
    -v "${RAIZ_MNT}/plugin/qwen_w4a4_vllm:/usr/local/lib/python3.12/dist-packages/qwen_w4a4_vllm:ro" \
    -v "${RAIZ_MNT}/docker/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
    -e CUDA_VISIBLE_DEVICES=0 -e MODEL_PATH="$modelo" -e QUANTIZATION="$quant" \
    -e NUM_SPECULATIVE_TOKENS=0 -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=4 \
    -e GPU_MEMORY_UTILIZATION=0.93 -e KV_CACHE_MEMORY_BYTES= \
    -e LIMIT_MM_PER_PROMPT='{"image":0,"video":0}' \
    -e PREFIX_CACHING="$CACHE" -e ENFORCE_EAGER=0 \
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
    python "$RAIZ_MNT/curva_g.py" $PORTA "$nome" 3 | tee -a "$OUT"
  else
    docker logs $NOME 2>&1 | grep -iE "error|Error" | tail -3
  fi
  docker logs $NOME > "$RAIZ/logs/curva_${nome}.log" 2>&1
  docker rm -f $NOME >/dev/null 2>&1
  SECONDS=0
done

echo
python - "$RAIZ_MNT/$(basename "$OUT")" <<'PY'
import json, sys
d = {}
for l in open(sys.argv[1], encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        d.setdefault(r["config"], {})[r["g"]] = r

gs = sorted(set().union(*[set(v) for v in d.values()])) if d else []

# Sem ponto COMPARAVEL nao ha conclusao a tirar. A versao anterior imprimia
# "ConvRot ganha em TODOS os G medidos" com o arquivo vazio -- afirmacao
# categorica sobre zero medicao, no bloco que decide a escolha de modelo do
# projeto. O bateria_2x2 ja tinha essa guarda; este nao.
pares = [g for g in gs if d.get("marlin", {}).get(g) and d.get("convrot", {}).get(g)]
if not pares:
    print()
    print(f"  SEM VEREDITO: {len(gs)} valores de G no arquivo, nenhum com os dois")
    print("  caminhos medidos. Nada a comparar -- conferir o log das subidas.")
    raise SystemExit(1)

print(f"{'G':>5}{'marlin ms':>12}{'convrot ms':>12}{'razao':>9}{'ruido':>8}")
virou = None
for g in gs:
    m, c = d.get("marlin", {}).get(g), d.get("convrot", {}).get(g)
    if not m or not c:
        continue
    r = m["total_ms"] / c["total_ms"]
    ruido = max(m["espalhamento_pct"], c["espalhamento_pct"])
    marca = ""
    if virou is None and r < 1.0:
        virou, marca = g, "  <- marlin passa a ganhar"
    print(f"{g:>5}{m['total_ms']:>11.0f}m{c['total_ms']:>11.0f}m{r:>8.2f}x{ruido:>7.0f}%{marca}")

print()
if virou is None:
    print("  ConvRot ganha em TODOS os G medidos. O cruzamento, se existe, esta")
    print("  acima do maior G da varredura.")
else:
    print(f"  Cruzamento entre G={gs[gs.index(virou)-1]} e G={virou}.")
print("  'razao' > 1 = ConvRot mais rapido. 'ruido' e' o espalhamento do pior")
print("  dos dois no ponto: diferenca menor que ele nao e sinal.")
PY
