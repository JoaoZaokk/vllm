#!/usr/bin/env bash

# Lock unico da GPU: duas sessoes dividem a placa. Ver gpu-lock-protocolo.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh" "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh" /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT
# 200000 e' de proposito: abaixo do max_position_embeddings (262144), senao o
# vLLM barra numa validacao anterior e nunca chega a calcular o teto de KV --
# foi o que invalidou a primeira varredura.
OUT=/c/Users/USER/w4a4/sweep_pp.txt
: > $OUT
for P in 46,18 45,19 44,20 43,21 42,22; do
  docker rm -f pps >/dev/null 2>&1
  MSYS_NO_PATHCONV=1 docker run -d --name pps --gpus all -p 8000:8000 --entrypoint bash --shm-size=8g \
    -v 'C:\Users\USER\w4a4\models\awq-w4a16:/workspace/models/awq-w4a16:ro' \
    -v 'C:\Users\USER\w4a4\docker:/opt/qwen38/docker:ro' \
    -e PIPELINE_PARALLEL_SIZE=2 -e VLLM_PP_LAYER_PARTITION=$P \
    -e NUM_SPECULATIVE_TOKENS=0 -e MAX_MODEL_LEN=200000 -e MAX_NUM_SEQS=2 \
    -e GPU_MEMORY_UTILIZATION=0.88 -e KV_CACHE_DTYPE=auto -e KV_CACHE_MEMORY_BYTES= \
    qwen38-w4a4:latest /opt/qwen38/docker/awq_entry.sh >/dev/null 2>&1
  for i in $(seq 1 60); do
    docker ps -q -f name=pps | grep -q . || break
    docker logs pps 2>&1 | grep -qE 'estimated maximum model length|No available memory|OutOfMemory' && break
    sleep 10
  done
  N=$(docker logs pps 2>&1 | grep -oE 'estimated maximum model length is [0-9]+' | tail -1 | grep -oE '[0-9]+$')
  [ -z "$N" ] && N=$(docker logs pps 2>&1 | grep -oiE 'No available memory|OutOfMemory' | tail -1)
  echo "$P -> ${N:-sem resposta}" >> $OUT
  docker rm -f pps >/dev/null 2>&1
done
echo FIM >> $OUT
