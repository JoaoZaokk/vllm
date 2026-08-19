#!/usr/bin/env bash
# Criterio de aceite do DSpark sob PP=2. NAO e "subiu = funciona".
#
# O upstream avisa que essa feature nao falha alto: um drafter alimentado com
# taps faltando, duplicados ou fora de ordem continua propondo tokens validos,
# so' que rejeitados com mais frequencia. O sintoma unico e' aceitacao baixa.
#
# Entao o teste e' equivalencia: sob temperatura 0, spec decode e' obrigado a
# produzir EXATAMENTE a mesma saida do decode normal. Se divergir, a verificacao
# esta aceitando o que nao devia. Se bater, a aceitacao diz se vale a pena.
set -uo pipefail

IMG=${IMG:-qwen38-pp-dspark:0.27.1}
# O drafter mora INTEIRO no ultimo estagio, e so a 3090 tem espaco. Sem esta
# variavel o dspark_entry.sh escolhe sozinho e poe a 3090 como rank 0 ("3090
# primeiro"), o que joga o drafter na placa de 12 GB e faz a rodada B nunca
# subir -- o veredito caia em "incompleto" e a equivalencia greedy, unica coisa
# capaz de detectar tap faltando ou fora de ordem, nunca era avaliada.
#
# Com 1,0 o estagio 0 e' a 3080 Ti, entao a particao poe POUCAS camadas nele.
# 56,8 so' faria sentido com a 3090 no estagio 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,0}"
PART=${PART:-28,36}
UTIL=${UTIL:-0.88}
LEN=${LEN:-32768}
K=${K:-7}
# Vazio = comportamento de fabrica. '{"image":0,"video":0}' pula o perfilamento
# da torre de visao, que aloca o encoder no estagio 0 -- a placa de 12 GB.
LIMIT_MM=${LIMIT_MM_PER_PROMPT-}
PROMPT='Escreva uma funcao Python que inverte uma lista ligada. Explique cada passo.'
# Raiz da stack (a que tem models/ e docker/), deduzida da localizacao deste
# script: benchmarks/sm86 -> raiz do repo -> pai. Sobrescrevivel por ambiente
# para quem nao guarda o fork ao lado dos modelos.
RAIZ_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STACK="${W4A4_ROOT:-$(cd "$RAIZ_REPO/.." && pwd)}"
# O -v do Docker Desktop no Windows quer C:/..., nao /c/...
STACK_MNT="$(cd "$STACK" && (pwd -W 2>/dev/null || pwd))"

OUT="$STACK/validacao_dspark.txt"
SAIDA_REF="$STACK/saida_sem_draft.json"
SAIDA_SPEC="$STACK/saida_dspark.json"
: > "$OUT"

subir() {  # $1=nome  $2=entry  $3=spec_k
  docker rm -f val >/dev/null 2>&1
  MSYS_NO_PATHCONV=1 docker run -d --name val --gpus all -p 8000:8000 --entrypoint bash --shm-size=8g \
    -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton \
    -v "${STACK_MNT}/models:/workspace/models:ro" \
    -v "${STACK_MNT}/docker:/opt/qwen38/docker:ro" \
    -e PIPELINE_PARALLEL_SIZE=2 -e VLLM_PP_LAYER_PARTITION="$PART" \
    -e MODEL_PATH=/workspace/models/awq-w4a16 -e QUANTIZATION= \
    -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    -e NUM_SPECULATIVE_TOKENS="$3" -e MAX_MODEL_LEN="$LEN" -e MAX_NUM_SEQS=2 \
    -e GPU_MEMORY_UTILIZATION="$UTIL" -e KV_CACHE_MEMORY_BYTES= \
    -e LIMIT_MM_PER_PROMPT="$LIMIT_MM" \
    "$IMG" "/opt/qwen38/docker/$2" >/dev/null 2>&1
  for i in $(seq 1 75); do
    curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    docker ps -q -f name=val | grep -q . || return 1
    sleep 8
  done
  return 1
}

gerar() {  # imprime os ids gerados, temperatura 0
  curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"/workspace/models/awq-w4a16\",\"prompt\":\"$PROMPT\",\"max_tokens\":160,\"temperature\":0,\"seed\":1234,\"logprobs\":0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]; print(json.dumps(c.get('logprobs',{}).get('tokens') or c['text']))"
}

aceitacao() {
  curl -s http://127.0.0.1:8000/metrics | python3 -c "
import sys,re
t=sys.stdin.read()
def g(n):
    m=re.search(r'vllm:%s[^ ]* ([0-9.e+]+)'%n, t)
    return float(m.group(1)) if m else 0.0
a,d=g('spec_decode_num_accepted_tokens_total'),g('spec_decode_num_draft_tokens_total')
print(f'aceitos={a:.0f} rascunhos={d:.0f} taxa={a/d:.4f}' if d else 'sem contadores de spec')
"
}

# As DUAS rodadas usam dspark_entry.sh, com k=0 desligando o spec na primeira.
# Antes A usava awq_entry.sh e B usava dspark_entry.sh: os dois tem defaults
# diferentes de dtype de cache mamba e montam flags diferentes, entao a
# comparacao tinha mais de uma variavel e a equivalencia nao provava nada.
echo "== A: PP=2 sem draft (referencia) ==" | tee -a $OUT
if subir A dspark_entry.sh 0; then
  gerar > "$SAIDA_REF"
  echo "  subiu, $(wc -c < "$SAIDA_REF") bytes de saida" | tee -a $OUT
else
  echo "  FALHOU: $(docker logs val 2>&1 | grep -oE 'No available memory|estimated maximum model length is [0-9]+|NotImplementedError[^\"]*' | tail -1)" | tee -a $OUT
fi
docker rm -f val >/dev/null 2>&1

echo "== B: PP=2 + DSpark k=$K ==" | tee -a $OUT
if subir B dspark_entry.sh "$K"; then
  gerar > "$SAIDA_SPEC"
  echo "  subiu | $(aceitacao)" | tee -a $OUT
else
  echo "  FALHOU: $(docker logs val 2>&1 | grep -oE 'No available memory|SupportsPP|NotImplementedError[^\"]*|Unsupported context manager' | tail -1)" | tee -a $OUT
fi
docker rm -f val >/dev/null 2>&1

echo "== veredito ==" | tee -a $OUT
if [ -s "$SAIDA_REF" ] && [ -s "$SAIDA_SPEC" ]; then
  if diff -q "$SAIDA_REF" "$SAIDA_SPEC" >/dev/null; then
    echo "  IDENTICO — verificacao correta sob greedy" | tee -a $OUT
  else
    echo "  DIVERGIU — spec decode aceitou token que o alvo nao produziria" | tee -a $OUT
  fi
else
  echo "  incompleto: um dos dois nao subiu" | tee -a $OUT
fi
