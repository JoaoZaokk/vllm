#!/usr/bin/env bash
# Os tres itens que a fila anterior nao entregou, com as causas corrigidas:
#
#  escada + equivalencia: morreram com "Available KV cache memory: -0.41 GiB" no
#    Worker_PP0. Com CUDA_VISIBLE_DEVICES=1,0 o estagio 0 e' a 3080 Ti de 12 GB,
#    e 28 das 64 camadas nao cabem nela junto com ativacao e nao-torch. 16,48
#    poe 16 la (multiplo de 4, exigencia do full_attention_interval).
#  testes: conftest.py do vLLM importa tblib, ausente na imagem. Instalado agora.
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$RAIZ/logs/resto_$(date +%H%M%S).log"
mkdir -p "$RAIZ/logs"
passo() { local n="$1"; shift; echo | tee -a "$LOG"
  echo "########## $n ($(date +%H:%M:%S)) ##########" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "########## $n rc=${PIPESTATUS[0]} ##########" | tee -a "$LOG"; }

passo "1/3 escada do drafter, particao 16,48" \
  # PART/PARTICAO nao sao lidos por ninguem: o script le
  # VLLM_PP_LAYER_PARTITION, carregado de baseline_congelado.env. O valor
  # coincidia com o padrao, entao a fila parecia funcionar. Para mudar de
  # verdade:  export VLLM_PP_LAYER_PARTITION=20,44
  bash "$RAIZ/escada_dspark_w4a4.sh" 8192 16384
passo "2/3 equivalencia greedy, particao 16,48" \
  # PART/PARTICAO nao sao lidos por ninguem: o script le
  # VLLM_PP_LAYER_PARTITION, carregado de baseline_congelado.env. O valor
  # coincidia com o padrao, entao a fila parecia funcionar. Para mudar de
  # verdade:  export VLLM_PP_LAYER_PARTITION=20,44
  bash "$RAIZ/validar_dspark_pp.sh"
passo "3/3 suite de testes com tblib" \
  bash "$RAIZ/run_tests.sh"
echo "===== RESTO COMPLETO =====" | tee -a "$LOG"
