#!/usr/bin/env bash
# Bateria de quatro configuracoes, uma placa (3090), sem draft, mesmo prompt.
#
# Responde quatro perguntas que hoje sao palpite:
#   A vs quente   o prefix caching em modo align acerta? (nunca foi medido)
#   A vs C        int4 MMA nativo ganha do Marlin no PREFILL? (so' mediram decode)
#   A vs B        fp16 recupera qualidade sobre bf16? (mesma vazao no Ampere)
#   C vs D        quanto da diferenca do ConvRot e' a ATIVACAO em int4
#
# Nada aqui requantiza. Os dois checkpoints ja existem no disco.
set -uo pipefail

# --------------------------------------------------------------------------
# ARNES APOSENTADO -- recusa rodar por padrao.
#
# Substituido por:
#
#     ./lab run <perfil> ttft
#
# A tarefa separa a PRIMEIRA requisicao das demais: so ela e fria. Mediana
# das tres, que era o que saia daqui, mistura dois regimes na mesma celula.
#
# E confere no /metrics se o servidor fez o que a config diz -- config
# dizendo uma coisa e servidor fazendo outra ja aconteceu neste projeto.
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
#     LAB_ARNES_LEGADO=1 bash bateria_ttft.sh
#
# Recusa em vez de aviso: aviso e' rolado para cima e o numero sai igual.
if [ "${LAB_ARNES_LEGADO:-0}" != "1" ]; then
  sed -n '/^# ARNES APOSENTADO/,/^# igual\.$/p' "${BASH_SOURCE[0]}" >&2
  echo >&2
  echo "RECUSADO: arnes aposentado. Use:  ./lab run <perfil> ttft" >&2
  echo "          ou LAB_ARNES_LEGADO=1 para rodar assim mesmo." >&2
  exit 78
fi
# --------------------------------------------------------------------------
# Lock unico da GPU. Duas sessoes do usuario dividem a mesma placa; cada uma com
# um monitor esperando "a GPU liberar" dispara no mesmo segundo e estraga as
# medicoes das duas. Este bloco e' obrigatorio em TODO script que sobe container
# com --gpus ou roda bench -- inclusive nos escritos antes do lock existir, que
# foi exatamente como eu furei o proprio protocolo uma vez.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh"           "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh"           /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT

export MSYS_NO_PATHCONV=1

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ_MNT="$(cd "$RAIZ" && (pwd -W 2>/dev/null || pwd))"
NOME=bateria
PORTA=8010
OUT="$RAIZ/resultados_ttft.jsonl"
# `tee -a` acumula. Sem truncar, uma rodada interrompida mais um rerun
# deixam a mesma celula duas vezes no arquivo, sem carimbo que as separe,
# e quem agrega depois nao tem como saber qual e' de quando.
: > "$OUT"

. "$(dirname "${BASH_SOURCE[0]}")/banner_bateria.sh"
banner_bateria "ttft" \n    "saida              $OUT" || exit 1
OUT_MNT="$RAIZ_MNT/resultados_ttft.jsonl"
LOGS="$RAIZ/logs"
mkdir -p "$LOGS"

# nome | imagem | entrypoint | modelo | envs extras
CONFIGS=(
  "A_awq_bf16|qwen38-w4a4:latest|awq_entry.sh|/workspace/models/awq-w4a16|MODEL_DTYPE=bfloat16"
  "B_awq_fp16|qwen38-w4a4:latest|awq_entry.sh|/workspace/models/awq-w4a16|MODEL_DTYPE=float16"
  "C_convrot_int4|qwen38-w4a4:latest|dspark_entry.sh|/workspace/models/qwen3.8-27b-heretic-convrot-w4a4|QWEN_W4A4_ACT_DTYPE=int4"
  "D_convrot_bf16|qwen38-w4a4:latest|dspark_entry.sh|/workspace/models/qwen3.8-27b-heretic-convrot-w4a4|QWEN_W4A4_ACT_DTYPE=bf16"
)

for linha in "${CONFIGS[@]}"; do
  IFS='|' read -r nome img entry modelo extra <<< "$linha"
  echo "=============================================================="
  echo "== $nome  ($entry, $(basename "$modelo"))"
  docker rm -f $NOME >/dev/null 2>&1

  # QUANTIZATION vazio para o AWQ (o vLLM le do config.json e escolhe Marlin);
  # convrot_w4a4 explicito para o outro, que precisa do plugin.
  quant=""
  [ "$entry" = "dspark_entry.sh" ] && quant="convrot_w4a4"

  docker run -d --name $NOME --gpus all --ipc host --shm-size=8g -p ${PORTA}:8000 \
    --entrypoint bash \
    -v "${RAIZ_MNT}/models:/workspace/models:ro" \
    -v "${RAIZ_MNT}/docker:/opt/qwen38/docker:ro" \
    -v "${RAIZ_MNT}/plugin/qwen_w4a4_vllm:/usr/local/lib/python3.12/dist-packages/qwen_w4a4_vllm:ro" \
    -v "${RAIZ_MNT}/docker/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e MODEL_PATH="$modelo" \
    -e QUANTIZATION="$quant" \
    -e NUM_SPECULATIVE_TOKENS=0 \
    -e MAX_MODEL_LEN=4096 \
    -e MAX_NUM_SEQS=4 \
    -e GPU_MEMORY_UTILIZATION=0.93 \
    -e KV_CACHE_MEMORY_BYTES= \
    -e LIMIT_MM_PER_PROMPT='{"image":0,"video":0}' \
    -e "$extra" \
    -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    "$img" "/opt/qwen38/docker/$entry" >/dev/null

  fim=$((SECONDS + 780)); estado=timeout
  while [ $SECONDS -lt $fim ]; do
    docker ps --format '{{.Names}}' | grep -qx $NOME || { estado=morreu; break; }
    curl -sf "http://127.0.0.1:${PORTA}/health" >/dev/null 2>&1 && { estado=subiu; break; }
    sleep 5
  done
  echo "   boot: $estado em ${SECONDS}s"

  if [ "$estado" = subiu ]; then
    python "$RAIZ_MNT/medir_ttft.py" $PORTA "$nome" "sem${RANDOM}" 80 \
      | tee -a "$OUT"
  else
    echo "{\"config\":\"$nome\",\"erro\":\"$estado\"}" | tee -a "$OUT"
    docker logs $NOME 2>&1 | grep -iE "error|Error|Traceback" | tail -4
  fi

  docker logs $NOME > "$LOGS/bateria_${nome}.log" 2>&1
  docker rm -f $NOME >/dev/null 2>&1
  SECONDS=0
done

echo
echo "=============================================================="
python - "$OUT_MNT" <<'PY'
import json, sys
linhas = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
print(f"{'config':<16}{'prompt':>7}{'ttft frio':>11}{'ttft quente':>13}{'cache':>8}{'decode':>9}")
for d in linhas:
    if "erro" in d:
        print(f"{d['config']:<16}  {d['erro']}")
        continue
    print(f"{d['config']:<16}{d['prompt_tokens']:>7}{d['ttft_frio_ms']:>10.0f}ms"
          f"{d['ttft_quente_ms']:>12.0f}ms{d['ganho_cache_pct']:>7.0f}%"
          f"{d['decode_tok_s']:>8.1f}")
PY
