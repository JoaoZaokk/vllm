#!/usr/bin/env bash
# Os dois que sobraram, com as causas corrigidas:
#
#  equivalencia: `python3` no HOST aponta para um C:\Python314\python.exe
#    quebrado. As duas funcoes que parseiam a resposta usavam ele, entao a saida
#    vinha vazia e o veredito dizia "incompleto" sem nada errado com o modelo.
#  testes: eu mandei coletar a arvore INTEIRA do vLLM -- 152 erros de coleta em
#    9 minutos, de testes que dependem de Blackwell, ROCm, CPU-only. Agora roda
#    so os cinco que o fork adicionou, que sao os que validam os patches.
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$RAIZ/logs/final_$(date +%H%M%S).log"
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
passo "1/2 equivalencia greedy 16,48" \
  bash "$RAIZ/validar_dspark_pp.sh"

passo "2/2 os cinco testes do fork" \
  bash "$RAIZ/run_tests.sh" \
    v1/worker/test_qwen35_dspark_aux_taps_pp.py \
    v1/worker/test_mamba_hybrid_model_state.py \
    v1/worker/test_eagle3_aux_hidden_states_pp.py \
    v1/worker/test_spec_decode_embed_sharing_pp.py \
    v1/e2e/spec_decode/eagle/test_eagle3_pp.py

if [ ${#FALHAS[@]} -eq 0 ]; then
  echo "===== FINAL COMPLETO — log em $LOG =====" | tee -a "$LOG"
else
  echo "===== FINAL INCOMPLETO: ${#FALHAS[@]} passo(s) falharam — log em $LOG =====" | tee -a "$LOG"
  printf '  %s
' "${FALHAS[@]}" | tee -a "$LOG"
  exit 1
fi
