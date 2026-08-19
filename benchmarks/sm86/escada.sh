#!/usr/bin/env bash

# Lock unico da GPU: duas sessoes dividem a placa. Ver gpu-lock-protocolo.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh" "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh" /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT
# Escada empirica: nada de estimativa. Para cada particao, sobe o contexto ate
# quebrar. Quem sobe responde /health e gera tokens; quem nao sobe, morre.
OUT=/c/Users/USER/w4a4/escada.txt
: > $OUT
for P in 56,8 52,12 48,16; do
  for L in 32768 65536 98304 131072; do
    docker rm -f esc >/dev/null 2>&1
    MSYS_NO_PATHCONV=1 docker run -d --name esc --gpus all -p 8000:8000 --entrypoint bash --shm-size=8g \
      -v 'C:\Users\USER\w4a4\models\awq-w4a16:/workspace/models/awq-w4a16:ro' \
      -v 'C:\Users\USER\w4a4\docker:/opt/qwen38/docker:ro' \
      -e PIPELINE_PARALLEL_SIZE=2 -e VLLM_PP_LAYER_PARTITION=$P \
      -e NUM_SPECULATIVE_TOKENS=0 -e MAX_MODEL_LEN=$L -e MAX_NUM_SEQS=2 \
      -e GPU_MEMORY_UTILIZATION=0.88 -e KV_CACHE_DTYPE=auto -e KV_CACHE_MEMORY_BYTES= \
      qwen38-w4a4:latest /opt/qwen38/docker/awq_entry.sh >/dev/null 2>&1
    OK=0
    for i in $(seq 1 75); do
      curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1 && { OK=1; break; }
      docker ps -q -f name=esc | grep -q . || break
      sleep 8
    done
    if [ $OK -eq 1 ]; then
      T0=$(date +%s.%N)
      R=$(curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
          -d '{"model":"/workspace/models/awq-w4a16","prompt":"Explique o que e um cache de KV em uma frase.","max_tokens":128,"temperature":0,"seed":1234}')
      T1=$(date +%s.%N)
      N=$(echo "$R" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+$')
      SEC=$(awk "BEGIN{print $T1-$T0}")
      TPS=$(awk "BEGIN{printf \"%.2f\", $N/$SEC}")
      echo "$P  ctx $L  -> SUBIU  ${N} tok em ${SEC}s = ${TPS} tok/s" >> $OUT
      docker rm -f esc >/dev/null 2>&1
    else
      echo "$P  ctx $L  -> falhou ($(docker logs esc 2>&1 | grep -oE 'No available memory|estimated maximum model length is [0-9]+|OutOfMemory' | tail -1))" >> $OUT
      docker rm -f esc >/dev/null 2>&1
      break
    fi
  done
done
echo FIM >> $OUT
