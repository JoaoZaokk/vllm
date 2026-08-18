#!/usr/bin/env bash
# Monta cada .py alterado sobre o pacote instalado. Array em vez de string:
# caminho Windows dentro de string vira escape no expand do bash.
cd /c/Users/USER/w4a4/vllm-fork || exit 1
ARGS=()
while read -r f; do
  [ -z "$f" ] && continue
  ARGS+=(-v "C:/Users/USER/w4a4/vllm-fork/${f}:/usr/local/lib/python3.12/dist-packages/${f}:ro")
done < <(git diff --name-only v0.27.1 -- 'vllm/*.py')
ARGS+=(-v "C:/Users/USER/w4a4/vllm-fork/tests:/tests:ro")
MSYS_NO_PATHCONV=1 exec docker run --rm "${ARGS[@]}" --entrypoint bash qwen38-w4a4:latest -c \
  "pip install -q pytest 2>&1|tail -1; python3 -m pytest $* -q --no-header -p no:cacheprovider 2>&1 | tail -10"
