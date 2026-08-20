#!/usr/bin/env bash
# Escada de contexto do DSpark quantizado sob PP=2.
#
# Baseline a bater: 4.928 tokens com o drafter bf16 (2,6 GB de arquivo mais um
# lm_head de 2,54 GB que o vLLM constroi porque nao esta no checkpoint). O w4a4
# corta so o arquivo, 2,6 -> 1,3 GB, entao o ganho esperado e ~1,3 GB, nao os
# ~4 que eu tinha estimado antes de olhar o audit.
#
# CUDA_VISIBLE_DEVICES=1,0 nao e enfeite: o drafter mora inteiro no ULTIMO
# estagio, e so a 3090 tem espaco para ele.
#
# Duas armadilhas que a auditoria achou, as duas invisiveis no log:
#
#  1. KV_CACHE_MEMORY_BYTES vazio SO' significa 'deixe o vLLM dimensionar'
#     depois da correcao no dspark_entry.sh. Antes ele usava ${VAR:-default},
#     que substitui tambem no vazio, e travava o KV em 1,25 GiB -- fazendo
#     qualquer teto de contexto virar constante/custo, identico em qualquer
#     particao e insensivel ao tamanho do drafter. Era esse o '4.928'.
#  2. Perfilar a torre de visao aloca o encoder multimodal no estagio 0, que
#     aqui e' a placa de 12 GB. Foi ai que as duas primeiras escadas
#     morreram, em profile_encoder_cache, antes de dimensionar KV nenhum.
#     LIMIT_MM_PER_PROMPT zera imagem e video: esta e uma escada de contexto
#     de TEXTO.
set -u
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

IMG="${IMG:-qwen38-pp-dspark:0.27.1}"
# Sem isto o vLLM divide 64 camadas em 32/32 e a 3080 Ti morre no load, antes
# de qualquer conta de contexto -- os tres degraus falham identicos e o teste
# nao mede nada. A particao e a mesma do baseline de 4.928 para o unico eixo
# que muda ser o drafter.
# Este script morava na raiz da stack e deduzia RAIZ como o proprio
# diretorio. Ao vir para benchmarks/sm86 -- unico lugar versionado dos
# scripts de bench -- a raiz passa a ser tres niveis acima, igual ao que
# validar_dspark_pp.sh ja fazia. Enquanto ele estava fora daqui, o passo
# 2/5 de rodar_tudo.sh apontava para um arquivo inexistente.
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAIZ_MNT="$(cd "$RAIZ" && (pwd -W 2>/dev/null || pwd))"
. "$RAIZ/vllm-fork/benchmarks/sm86/baseline_congelado.env"
# Sem apelidos locais. PARTICAO e DRAFT existiam como nomes proprios da escada e
# venciam qualquer export das variaveis do baseline, entao
# `export VLLM_PP_LAYER_PARTITION=20,44` era lido, ignorado, e a escada rodava
# outra particao sem dizer nada. Um botao, um nome: as variaveis do baseline.
# O default de 28,36 que morava aqui era o unico valor do repo que contradizia o
# congelado -- 28 das 64 camadas no estagio 0 davam KV NEGATIVO na placa de 12 GB.
NOME=escada_dspark
PORTA=8009
export MSYS_NO_PATHCONV=1

for LEN in "$@"; do
  docker rm -f $NOME >/dev/null 2>&1
  export MAX_MODEL_LEN="$LEN"
  baseline_mostrar "$RAIZ/logs/escada_baseline.txt"
  # A lista -e sai de baseline_congelado.env. Enquanto ela morava aqui, a escada
  # divergia do validar em tres eixos sem ninguem notar: desligava multimodal
  # pelos limites (kernel fundido OFF) enquanto o validar usava a flag (ON),
  # MAX_NUM_SEQS=4 contra 2, e cache de compilacao LIGADO contra desligado --
  # compartilhando o mesmo volume vllm-cache. Numeros das duas nao eram
  # comparaveis, que e exatamente o que o baseline congelado existe para impedir.
  baseline_docker_args
  baseline_docker_mounts "$RAIZ_MNT"
  docker run -d --name $NOME --gpus all --ipc host --shm-size=8g -p ${PORTA}:8000 \
    --entrypoint bash \
    -v "$RAIZ_MNT/docker/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
    "${BASELINE_DOCKER_MOUNTS[@]}" "${BASELINE_DOCKER_ARGS[@]}" \
    -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
    $IMG /opt/qwen38/docker/dspark_entry.sh >/dev/null

  fim=$((SECONDS + 480)) ; estado=timeout
  while [ $SECONDS -lt $fim ]; do
    if ! docker ps --format '{{.Names}}' | grep -qx $NOME; then estado=morreu; break; fi
    if curl -sf "http://localhost:${PORTA}/health" >/dev/null 2>&1; then estado=SUBIU; break; fi
    sleep 5
  done

  echo "  -> ${estado} em ${SECONDS}s"
  if [ "$estado" = SUBIU ]; then
    docker logs $NOME 2>&1 | grep -iE "GPU KV cache size|maximum concurrency|Memory profiling|available_kv_cache" | tail -4
  else
    docker logs $NOME 2>&1 | grep -iE "error|Error|ValueError|OutOfMemory|estimated maximum" | tail -6
  fi
  docker logs $NOME > "$RAIZ/logs/escada_dspark_${VLLM_PP_LAYER_PARTITION/,/_}_${LEN}.log" 2>&1
  docker rm -f $NOME >/dev/null 2>&1
  SECONDS=0
  echo
done
