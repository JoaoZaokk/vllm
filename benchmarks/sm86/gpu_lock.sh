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

    # Lock existente. Como testar se o dono vive depende de QUEM escreveu:
    #
    #   - Controlador Python (control/runtime.py): escreve hb=<epoch>, atualizado
    #     a cada 15s. O pid dele e' NATIVO do Windows; `kill -0` do MSYS reporta
    #     MORTO um processo vivo (namespaces de pid diferentes). Foi medido: pid
    #     vivo no tasklist, kill -0 diz morto. Entao aqui NAO se usa kill -0 --
    #     usa-se o batimento, exatamente como o _dono_morreu do runtime.py.
    #   - Script bash da geracao antiga: escreve so' pid=, sem hb=. Mesmo namespace
    #     de pid, kill -0 e' confiavel. So' nesse caso se usa kill -0.
    #
    # Sem esta distincao, o bash reivindicava o lock do controlador (kill -0 falso
    # negativo), apagava e subia na placa junto -- exclusao mutua quebrada pelo
    # proprio conserto anterior, que criou o hb= e nunca ensinou o bash a le-lo.
    local GPU_LOCK_MORTO_S="${GPU_LOCK_MORTO_S:-55}"   # = BATIMENTO_S*3+10 no runtime.py
    local dono_hb=$(sed -n 's/^hb=//p'  "$GPU_LOCK_ARQ" 2>/dev/null)
    local dono_pid=$(sed -n 's/^pid=//p' "$GPU_LOCK_ARQ" 2>/dev/null)

    if [ -n "$dono_hb" ]; then
      # Dono com batimento (controlador): vivo se o batimento for recente.
      if [[ "${dono_hb%.*}" =~ ^[0-9]+$ ]]; then
        local idade_hb=$(( $(date +%s) - ${dono_hb%.*} ))
        if [ "$idade_hb" -gt "$GPU_LOCK_MORTO_S" ]; then
          echo "[lock] batimento parado ha ${idade_hb}s (> ${GPU_LOCK_MORTO_S}s). Vazado, retomando."
          rm -f "$GPU_LOCK_ARQ"
          continue
        fi
        # batimento fresco: dono vivo. NAO reivindica; cai para espera/timeout abaixo.
      else
        echo "[lock] batimento ilegivel (hb=$dono_hb). Conservador: nao reivindico."
      fi
    elif [ -n "$dono_pid" ] && ! kill -0 "$dono_pid" 2>/dev/null; then
      # Dono sem batimento (bash antigo), mesmo namespace: kill -0 confiavel.
      echo "[lock] dono pid=$dono_pid nao existe mais. Lock vazado, retomando."
      rm -f "$GPU_LOCK_ARQ"
      continue
    fi

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
