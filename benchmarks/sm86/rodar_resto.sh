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
# rc coletado E TESTADO. A versao anterior imprimia o rc de cada passo e nunca
# olhava para ele: passo 3 morria, a fila seguia, e a ultima linha dizia
# COMPLETA. Nao havia como distinguir uma fila que rodou de uma que caiu no
# primeiro item -- e a fila roda de madrugada, sem ninguem lendo o meio do log.
FALHAS=()
passo() { local n="$1"; shift; echo | tee -a "$LOG"
  echo "########## $n ($(date +%H:%M:%S)) ##########" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "########## $n rc=$rc ##########" | tee -a "$LOG"
  [ "$rc" -ne 0 ] && FALHAS+=("$n (rc=$rc)")
  return 0
}

# PART/PARTICAO nao sao lidos por ninguem: o script le
# VLLM_PP_LAYER_PARTITION, carregado de baseline_congelado.env. O valor
# coincidia com o padrao, entao a fila parecia funcionar. Para mudar de
# verdade:  export VLLM_PP_LAYER_PARTITION=20,44
passo "1/3 escada do drafter, particao 16,48" \
  bash "$RAIZ/escada_dspark_w4a4.sh" 8192 16384

# PART/PARTICAO nao sao lidos por ninguem: o script le
# VLLM_PP_LAYER_PARTITION, carregado de baseline_congelado.env. O valor
# coincidia com o padrao, entao a fila parecia funcionar. Para mudar de
# verdade:  export VLLM_PP_LAYER_PARTITION=20,44
passo "2/3 equivalencia greedy, particao 16,48" \
  bash "$RAIZ/validar_dspark_pp.sh"

passo "3/3 suite de testes com tblib" \
  bash "$RAIZ/run_tests.sh"
if [ ${#FALHAS[@]} -eq 0 ]; then
  echo "===== RESTO COMPLETO — log em $LOG =====" | tee -a "$LOG"
else
  echo "===== RESTO INCOMPLETO: ${#FALHAS[@]} passo(s) falharam — log em $LOG =====" | tee -a "$LOG"
  printf '  %s
' "${FALHAS[@]}" | tee -a "$LOG"
  exit 1
fi
