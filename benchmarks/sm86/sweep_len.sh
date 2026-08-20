#!/usr/bin/env bash

# Lock unico da GPU: duas sessoes dividem a placa. Ver gpu-lock-protocolo.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh" "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh" /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT
OUT=/c/Users/USER/w4a4/sweep_len.txt
: > $OUT
for L in 120000 100000 80000 65536; do
  docker rm -f qlen >/dev/null 2>&1
  MSYS_NO_PATHCONV=1 docker run -d --name qlen --gpus all -p 8000:8000 --entrypoint bash --shm-size=8g \
    -v 'C:\Users\USER\w4a4\models\awq-w4a16:/workspace/models/awq-w4a16:ro' \
    -v 'C:\Users\USER\w4a4\docker:/opt/qwen38/docker:ro' \
    -e PIPELINE_PARALLEL_SIZE=2 -e VLLM_PP_LAYER_PARTITION=48,16 \
    -e NUM_SPECULATIVE_TOKENS=0 -e MAX_MODEL_LEN=$L -e MAX_NUM_SEQS=2 \
    -e GPU_MEMORY_UTILIZATION=0.88 -e KV_CACHE_DTYPE=auto -e KV_CACHE_MEMORY_BYTES= \
    qwen38-w4a4:latest /opt/qwen38/docker/awq_entry.sh >/dev/null 2>&1
  for i in $(seq 1 60); do
    curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    docker ps -q -f name=qlen | grep -q . || break
    sleep 10
  done
  if curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1; then
    KV=$(docker logs qlen 2>&1 | grep -oE 'Available KV cache memory: [0-9.]+ GiB' | tail -1)
    echo "$L -> SUBIU ($KV)" >> $OUT
    echo FIM >> $OUT
    exit 0
  fi
  echo "$L -> $(docker logs qlen 2>&1 | grep -oE 'estimated maximum model length is [0-9]+|No available memory' | tail -1)" >> $OUT
  docker rm -f qlen >/dev/null 2>&1
done
echo FIM >> $OUT
