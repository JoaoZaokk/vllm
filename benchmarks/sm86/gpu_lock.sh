#!/usr/bin/env bash
# Lock unico da GPU. Duas sessoes trabalhando na mesma placa, cada uma com um
# monitor esperando "a GPU liberar", disparam no mesmo segundo e estragam as
# medicoes das duas.
#
#   source gpu_lock.sh
#   gpu_lock_pegar "nome-do-trabalho"   || exit 1
#   trap gpu_lock_soltar EXIT
#   ...trabalho...
#
# O arquivo e' F:/GPU_BENCH.lock. Criado com set -o noclobber, que e' atomico:
# se dois processos tentarem ao mesmo tempo, exatamente um vence.
GPU_LOCK_ARQ="${GPU_LOCK_ARQ:-/f/GPU_BENCH.lock}"
GPU_LOCK_MAX_IDADE="${GPU_LOCK_MAX_IDADE:-7200}"   # 2h: lock mais velho e' orfao

gpu_lock_pegar() {
  local dono="${1:-anonimo}" espera="${2:-0}"
  local fim=$((SECONDS + espera))
  while :; do
    if (set -o noclobber; printf '%s\n' \
          "dono=$dono" "pid=$$" "desde=$(date -Iseconds)" \
          > "$GPU_LOCK_ARQ") 2>/dev/null; then
      echo "[lock] peguei: $dono"
      return 0
    fi

    # Lock existente: orfao ou vivo?
    local idade=$(( $(date +%s) - $(stat -c %Y "$GPU_LOCK_ARQ" 2>/dev/null || date +%s) ))
    if [ "$idade" -gt "$GPU_LOCK_MAX_IDADE" ]; then
      echo "[lock] lock com ${idade}s, acima do teto de ${GPU_LOCK_MAX_IDADE}s. Orfao:"
      sed 's/^/       /' "$GPU_LOCK_ARQ"
      echo "[lock] remova a mao se confirmar: rm $GPU_LOCK_ARQ"
      return 1
    fi

    if [ "$SECONDS" -ge "$fim" ]; then
      echo "[lock] OCUPADO por:"
      sed 's/^/       /' "$GPU_LOCK_ARQ"
      return 1
    fi
    sleep 10
  done
}

gpu_lock_soltar() {
  # So' remove se for meu: nao apagar lock de outra sessao por engano.
  if [ -f "$GPU_LOCK_ARQ" ] && grep -qx "pid=$$" "$GPU_LOCK_ARQ" 2>/dev/null; then
    rm -f "$GPU_LOCK_ARQ"
    echo "[lock] soltei"
  fi
}

gpu_lock_ver() {
  if [ -f "$GPU_LOCK_ARQ" ]; then
    echo "[lock] OCUPADO:"; sed 's/^/       /' "$GPU_LOCK_ARQ"
  else
    echo "[lock] livre"
  fi
}
