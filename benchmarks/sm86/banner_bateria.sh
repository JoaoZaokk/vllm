#!/usr/bin/env bash
# Cabecalho das baterias: diz no log o que a bateria REALMENTE vai usar.
#
# As baterias montam a lista de -e a mao, cada uma com a sua, e isso nao vai
# mudar tao cedo -- elas variam o modelo e a quantizacao de proposito, entao nao
# podem simplesmente carregar o baseline congelado, que descreve UMA config. O
# risco disso nao e' a lista existir: e' ela divergir sem aparecer em lugar
# nenhum. Um script que esquece uma variavel nao quebra, ele roda OUTRA
# configuracao e entrega um numero.
#
# Entao, ate as baterias virarem tarefas do controlador, elas ao menos declaram.
#
#   . "$(dirname "${BASH_SOURCE[0]}")/banner_bateria.sh"
#   banner_bateria "curva" "cache de prefixo=$CACHE"

# A placa e' escolhida por CUDA_VISIBLE_DEVICES=0 cravado nas baterias. Isso e'
# deliberado -- benchmark quer a placa FIXA, nao a que o entrypoint escolher --
# mas so vale se o indice 0 for mesmo a grande. Se um dia a ordem do PCI mudar,
# a bateria mediria a 3080 Ti chamando o resultado de 3090.
bateria_exigir_placa_grande() {
  local minimo="${1:-20000}" nome mem
  nome=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null)
  if [ -z "$mem" ]; then
    echo "  ABORTADO: nvidia-smi nao respondeu. Sem saber qual placa e' o indice 0," >&2
    echo "            o resultado nao sabe de qual placa esta falando." >&2
    return 1
  fi
  if [ "$mem" -lt "$minimo" ]; then
    echo "  ABORTADO: CUDA_VISIBLE_DEVICES=0 aponta para '${nome}' (${mem} MiB)," >&2
    echo "            abaixo do minimo de ${minimo} MiB. A bateria mediria a placa" >&2
    echo "            pequena e rotularia o resultado como se fosse a grande." >&2
    return 1
  fi
  printf '  placa (indice 0)   %s, %s MiB\n' "$nome" "$mem"
  return 0
}

banner_bateria() {  # $1=nome  resto=linhas extras
  local nome="$1"; shift
  echo "======================================================================"
  echo "BATERIA ${nome}"
  echo "======================================================================"
  bateria_exigir_placa_grande || return 1
  # Nenhuma bateria monta vllm-cache nem triton-cache. Isso e' bom -- o que for
  # compilado morre com o container, entao nao ha grafo de outra configuracao
  # sendo servido a esta. Mas e' DIFERENTE do validar_dspark_pp.sh, que monta os
  # dois volumes nomeados. Numero de bateria e numero de validacao nao vieram do
  # mesmo regime de cache, e isso precisa estar escrito.
  echo "  cache de compilacao  frio (nenhum volume montado; morre com o container)"
  echo "  imagem               qwen38-w4a4:latest"
  local l
  for l in "$@"; do echo "  $l"; done
  echo "======================================================================"
  return 0
}
