#!/usr/bin/env bash

# Lock unico da GPU: duas sessoes dividem a placa. Ver gpu-lock-protocolo.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh" "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh" /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT
# Monta cada .py alterado sobre o pacote instalado e roda a suite.
#
# Tres defeitos corrigidos apos auditoria:
#
#   1. `... | tail -10` fazia o script sair SEMPRE com 0, porque o status do
#      pipe e' o do ultimo comando. Como portao ("run_tests.sh && medir") ele
#      aprovava suite vermelha, e escondia exit 5 do pytest ("no tests ran") e
#      exit 4 ("file or directory not found").
#   2. Rodava sem --gpus, entao os testes que precisam de placa PULAVAM e a
#      saida verde nao distinguia "passou" de "pulou". O unico patch que mexe no
#      caminho de estado dos 48 layers GDN e' justamente um desses.
#   3. Sem alvo, o pytest coletava a partir do WORKDIR da imagem (/workspace),
#      nao dos testes montados em /tests.
#
# Uso:
#   bash run_tests.sh                          # suite inteira, com GPU
#   bash run_tests.sh v1/worker/test_x.py -k y # alvo relativo a /tests
#   GPUS= bash run_tests.sh                    # sem placa, marca o que pulou
set -uo pipefail

# Raiz do repositorio, deduzida da localizacao deste script -- nada de caminho
# absoluto de uma maquina especifica.
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RAIZ" || exit 1

# O -v do Docker Desktop no Windows quer C:/... e nao /c/...; `pwd -W` faz essa
# conversao no Git Bash e falha em Linux, onde o caminho POSIX ja serve.
RAIZ_MNT="$(pwd -W 2>/dev/null || pwd)"

# Array em vez de string: caminho Windows dentro de string vira escape no
# expand do bash.
ARGS=()
while read -r f; do
  [ -z "$f" ] && continue
  case "$f" in *.py) ;; *) continue ;; esac
  [ -f "$f" ] || continue   # arquivo apagado no diff nao pode ser montado
  ARGS+=(-v "${RAIZ_MNT}/${f}:/usr/local/lib/python3.12/dist-packages/${f}:ro")
done < <(git diff --name-only v0.27.1 -- vllm)

if [ ${#ARGS[@]} -eq 0 ]; then
  echo "ERRO: nenhum arquivo alterado encontrado em git diff v0.27.1 -- vllm" >&2
  echo "      montaria a arvore original e validaria o codigo NAO corrigido." >&2
  exit 1
fi
echo "montando ${#ARGS[@]} arquivos alterados"

ARGS+=(-v "${RAIZ_MNT}/tests:/tests:ro")

GPUS="${GPUS-all}"
if [ -n "${GPUS}" ]; then
  ARGS+=(--gpus "${GPUS}")
else
  echo "AVISO: sem --gpus. Teste que precisa de placa vai PULAR, e pular nao e passar."
fi

# Sem alvo, roda a suite inteira montada em /tests.
ALVO=("$@")
[ ${#ALVO[@]} -eq 0 ] && ALVO=(.)

# -rs lista o motivo de cada skip: e' o que separa "passou" de "nem rodou".
# O status do pytest sai inteiro porque nao ha pipe depois dele.
# NAO usar `exec` aqui. Ele substitui o processo, o trap EXIT nunca dispara, e o
# lock vazaria -- foi por isso que a versao anterior soltava o lock ANTES desta
# linha. So que soltar antes e' pior que vazar: o lock ficava livre enquanto o
# container ainda tinha a placa inteira, e a outra sessao arrancava por cima de
# uma GPU ocupada. Lock livre com placa ocupada e' uma afirmacao falsa.
#
# Sem `exec`, o docker run termina, o script continua, o trap dispara e o lock
# sai DEPOIS que a placa esta mesmo livre. O custo e' um processo bash parado
# durante a suite.
MSYS_NO_PATHCONV=1 docker run --rm "${ARGS[@]}" \
  --entrypoint bash qwen38-w4a4:latest -c \
  "pip install -q pytest tblib >/dev/null 2>&1; cd /tests && python3 -m pytest ${ALVO[*]} -q --no-header -rs -p no:cacheprovider"
rc=$?
# O status do pytest e' o status do script. Sem esta linha o `exit` implicito
# devolveria o do ultimo comando executado -- que seria o trap.
exit $rc
