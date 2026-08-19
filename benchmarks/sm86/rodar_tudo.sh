#!/usr/bin/env bash
# Fila de tudo que precisa de GPU e ainda nao rodou. Serial de proposito.
#
# NAO pega o lock: cada script da fila pega e solta o seu. Rodando em serie nao
# ha sobreposicao, e se a outra sessao pedir a placa no meio ela consegue entrar
# entre dois itens em vez de ficar bloqueada uma hora.
#
# Ordem por valor, nao por custo:
#   1 curva com cache ON   -- e' a que decide a escolha no uso real
#   2 escada do drafter    -- teto de contexto, mensuravel pela primeira vez
#   3 equivalencia greedy  -- criterio de aceite do DSpark sob PP=2
#   4 sweep de GEMM cheio  -- 5 formas, sem boot, minutos
#   5 suite de testes      -- valida os patches do fork com placa
set -uo pipefail
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$RAIZ/logs/fila_$(date +%H%M%S).log"
mkdir -p "$RAIZ/logs"

# rc coletado E TESTADO. A versao anterior imprimia o rc de cada passo e nunca
# olhava para ele: passo 3 morria, a fila seguia, e a ultima linha dizia
# COMPLETA. Nao havia como distinguir uma fila que rodou de uma que caiu no
# primeiro item -- e a fila roda de madrugada, sem ninguem lendo o meio do log.
FALHAS=()
passo() {  # $1=nome  resto=comando
  local nome="$1"; shift
  echo | tee -a "$LOG"
  echo "########## $nome  ($(date +%H:%M:%S)) ##########" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "########## $nome terminou: rc=$rc ##########" | tee -a "$LOG"
  [ "$rc" -ne 0 ] && FALHAS+=("$nome (rc=$rc)")
  return 0
}

passo "1/5 curva com cache LIGADO" \
  env PREFIX_CACHING=1 bash "$RAIZ/bateria_curva.sh"

passo "2/5 escada do drafter w4a4" \
  bash "$RAIZ/escada_dspark_w4a4.sh" 8192 16384

passo "3/5 equivalencia greedy do DSpark sob PP=2" \
  bash "$RAIZ/validar_dspark_pp.sh"

passo "4/5 sweep de GEMM, cinco formas" \
  bash -c 'cd "$1" && . ./gpu_lock.sh && gpu_lock_pegar sweep-cheio 0 || exit 1
    trap gpu_lock_soltar EXIT
    MSYS_NO_PATHCONV=1 docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 \
      -v "$(pwd -W)/vllm-fork/benchmarks:/bench:ro" \
      -v "$(pwd -W)/plugin/qwen_w4a4_vllm:/usr/local/lib/python3.12/dist-packages/qwen_w4a4_vllm:ro" \
      --entrypoint python3 qwen38-w4a4:latest /bench/sm86/sweep_gemm_m.py \
      --ms 1 4 16 64 128 256 1024 5856 \
      | grep -vE "WARNING|INFO|sitecustomize|^W0|torch/utils|Triton"' _ "$RAIZ"

passo "5/5 suite de testes do fork" \
  bash "$RAIZ/run_tests.sh"

echo | tee -a "$LOG"
if [ ${#FALHAS[@]} -eq 0 ]; then
  echo "===== FILA COMPLETA — log em $LOG =====" | tee -a "$LOG"
else
  echo "===== FILA INCOMPLETA: ${#FALHAS[@]} passo(s) falharam — log em $LOG =====" | tee -a "$LOG"
  printf '  %s
' "${FALHAS[@]}" | tee -a "$LOG"
  exit 1
fi
