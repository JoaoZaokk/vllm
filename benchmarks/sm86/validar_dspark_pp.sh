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
# Lock unico da GPU. Duas sessoes do usuario dividem a mesma placa; cada uma com
# um monitor esperando "a GPU liberar" dispara no mesmo segundo e estraga as
# medicoes das duas. Este bloco e' obrigatorio em TODO script que sobe container
# com --gpus ou roda bench -- inclusive nos escritos antes do lock existir, que
# foi exatamente como eu furei o proprio protocolo uma vez.
for _l in "$(dirname "${BASH_SOURCE[0]}")/gpu_lock.sh"           "$(dirname "${BASH_SOURCE[0]}")/../../../gpu_lock.sh"           /c/Users/USER/w4a4/gpu_lock.sh; do
  [ -f "$_l" ] && { . "$_l"; break; }
done
gpu_lock_pegar "$(basename "${BASH_SOURCE[0]}" .sh)" 0 || exit 1
trap gpu_lock_soltar EXIT


# Baseline congelado: a unica config que ja subiu com draft. Sourced em vez de
# repetido aqui, porque repetir foi exatamente como as duas metades divergiram e
# custaram quatro rodadas. Exportar a variavel antes continua sobrescrevendo.
. "$(dirname "${BASH_SOURCE[0]}")/baseline_congelado.env"

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
PART=${PART:-$VLLM_PP_LAYER_PARTITION}
UTIL=${UTIL:-$GPU_MEMORY_UTILIZATION}
LEN=${LEN:-$MAX_MODEL_LEN}
K=${K:-$NUM_SPECULATIVE_TOKENS}
# Estes dois defaults sao copiados da UNICA configuracao que ja subiu com draft:
# escada_dspark_w4a4.sh em 16,48 / 8192, que deu 15.454 tokens de KV. Antes eles
# divergiam e a rodada B morria com "No available memory" -- e o diagnostico
# apontou para a particao, que era a variavel IGUAL entre as duas.
#
#   drafter    dspark-qwen38 (bf16, 2,6 G) contra dspark-qwen38-w4a4 (1,3 G)
#              PP1 carregava 17,57 GiB de peso em vez de 15,48. Diferenca 2,09.
#   LIMIT_MM   vazio perfila a torre de visao e aloca o encoder no estagio 0,
#              a placa de 12 GB: 6,39 GiB de peso em vez de 5,52. Diferenca 0,87.
#
# Somadas, as duas explicam a morte inteira sem tocar em uma camada sequer.
DRAFT=$DRAFT_PATH
LIMIT_MM=$LIMIT_MM_PER_PROMPT
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

# VRAM base: medida com a placa ociosa, ANTES do primeiro boot. Sem isto a
# rodada B arranca enquanto o container de A ainda esta devolvendo memoria,
# e o estagio 1 perfila com menos VRAM do que realmente tem -- foi o que fez
# B morrer com "No available memory" num boot que sozinho sobe.
VRAM_BASE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
echo "VRAM base (ociosa): ${VRAM_BASE} MiB" | tee -a $OUT

esperar_vram() {
  local i usados folga=700
  for i in $(seq 1 60); do
    usados=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
    local ok=1 n=1
    while IFS= read -r u; do
      local b=$(echo "$VRAM_BASE" | cut -d, -f$n)
      [ "$u" -gt $((b + folga)) ] && ok=0
      n=$((n+1))
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$ok" = 1 ]; then
      echo "  VRAM devolvida apos ${i}s: ${usados} MiB (base ${VRAM_BASE})" | tee -a $OUT
      return 0
    fi
    sleep 1
  done
  echo "  AVISO: VRAM nao voltou ao baseline em 60s: ${usados} MiB (base ${VRAM_BASE})" | tee -a $OUT
  return 0
}

subir() {  # $1=nome  $2=entry  $3=spec_k
  docker logs val > "$STACK/logs/validar_${rodada:-x}.log" 2>&1 || true
docker rm -f val >/dev/null 2>&1
  esperar_vram
  MSYS_NO_PATHCONV=1 docker run -d --name val --gpus all -p 8000:8000 --entrypoint bash --shm-size=8g \
    -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton \
    -v "${STACK_MNT}/models:/workspace/models:ro" \
    -v "${STACK_MNT}/docker:/opt/qwen38/docker:ro" \
    -v "${STACK_MNT}/plugin/qwen_w4a4_vllm:/usr/local/lib/python3.12/dist-packages/qwen_w4a4_vllm:ro" \
    -e PIPELINE_PARALLEL_SIZE=2 -e VLLM_PP_LAYER_PARTITION="$PART" \
    -e MODEL_PATH=/workspace/models/awq-w4a16 -e QUANTIZATION= \
    -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    -e DRAFT_PATH="$DRAFT" \
    -e NUM_SPECULATIVE_TOKENS="$3" -e MAX_MODEL_LEN="$LEN" -e MAX_NUM_SEQS=2 \
    -e GPU_MEMORY_UTILIZATION="$UTIL" -e KV_CACHE_MEMORY_BYTES= \
    -e LIMIT_MM_PER_PROMPT="$LIMIT_MM" \
    -e VLLM_DISABLE_COMPILE_CACHE=1 \
    "$IMG" "/opt/qwen38/docker/$2" >/dev/null 2>&1
  for i in $(seq 1 75); do
    curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    docker ps -q -f "name=^val$" | grep -q . || return 1
    sleep 8
  done
  return 1
}

gerar() {  # imprime os ids gerados, temperatura 0
  curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"/workspace/models/awq-w4a16\",\"prompt\":\"$PROMPT\",\"max_tokens\":160,\"temperature\":0,\"seed\":1234,\"logprobs\":0}" \
    | python -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]; print(json.dumps(c.get('logprobs',{}).get('tokens') or c['text']))"
}

aceitacao() {
  curl -s http://127.0.0.1:8000/metrics | python -c "
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
baseline_mostrar "$OUT"
echo "== A: PP=2 sem draft (referencia) ==" | tee -a $OUT
rodada=A
if subir A dspark_entry.sh 0; then
  gerar > "$SAIDA_REF"
  echo "  subiu, $(wc -c < "$SAIDA_REF") bytes de saida" | tee -a $OUT
else
  echo "  FALHOU: $(docker logs val 2>&1 | grep -oE 'No available memory|estimated maximum model length is [0-9]+|NotImplementedError[^\"]*' | tail -1)" | tee -a $OUT
fi
docker logs val > "$STACK/logs/validar_${rodada:-x}.log" 2>&1 || true
docker rm -f val >/dev/null 2>&1

echo "== B: PP=2 + DSpark k=$K ==" | tee -a $OUT
rodada=B
if subir B dspark_entry.sh "$K"; then
  gerar > "$SAIDA_SPEC"
  echo "  subiu | $(aceitacao)" | tee -a $OUT
else
  echo "  FALHOU: $(docker logs val 2>&1 | grep -oE 'No available memory|SupportsPP|NotImplementedError[^\"]*|Unsupported context manager' | tail -1)" | tee -a $OUT
fi
docker logs val > "$STACK/logs/validar_${rodada:-x}.log" 2>&1 || true
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
