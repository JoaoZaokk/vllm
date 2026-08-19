#!/usr/bin/env bash
# bf16 contra fp16, MESMO prompt, greedy: a saida bate?
#
# A bateria de TTFT nao responde isto e nao tinha como: cada configuracao usava
# uma semente de prompt diferente, entao as saidas nunca foram comparaveis. Aqui
# a semente e' FIXA -- cada servidor sobe limpo, entao o prompt continua frio
# mesmo repetido entre as duas rodadas.
#
# Por que importa: fp16 tem 3 bits a mais de mantissa que bf16, mas paga em
# alcance -- estoura em 65504. Um contraexemplo medido em outro modelo hoje
# (Z-Image) tinha ativacao de 344064, cinco vezes o teto, virando `inf`.
#
# E o detalhe que quebra guarda ingenua: overflow em fp16 produz `inf`, NAO
# `NaN`. `isnan` devolve False e a degradacao passa em silencio. A verificacao
# certa e `isfinite` -- e, aqui, a saida greedy divergir do baseline.
set -uo pipefail
export MSYS_NO_PATHCONV=1

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ_MNT="$(cd "$RAIZ" && (pwd -W 2>/dev/null || pwd))"
NOME=equiv
PORTA=8011
mkdir -p "$RAIZ/logs"

gerar() {  # $1 = arquivo de saida
  curl -s "http://127.0.0.1:${PORTA}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$1"'","prompt":"Escreva uma funcao Python que inverte uma lista ligada, sem usar recursao. Depois explique por que a versao iterativa gasta memoria constante.","max_tokens":200,"temperature":0,"seed":1234,"logprobs":1}' \
    | python -c "
import json,sys,math
d=json.load(sys.stdin)
c=d['choices'][0]
lp=(c.get('logprobs') or {}).get('token_logprobs') or []
# isfinite, nao isnan: overflow em fp16 vira inf e passaria por isnan.
ruins=[x for x in lp if x is not None and not math.isfinite(x)]
print(json.dumps({'texto':c['text'],'n_logprob':len(lp),'nao_finitos':len(ruins)},ensure_ascii=False))
"
}

for dt in bfloat16 float16; do
  echo "== $dt =="
  docker rm -f $NOME >/dev/null 2>&1
  docker run -d --name $NOME --gpus all --ipc host --shm-size=8g -p ${PORTA}:8000 \
    --entrypoint bash \
    -v "${RAIZ_MNT}/models:/workspace/models:ro" \
    -v "${RAIZ_MNT}/docker:/opt/qwen38/docker:ro" \
    -e CUDA_VISIBLE_DEVICES=0 -e MODEL_PATH=/workspace/models/awq-w4a16 \
    -e QUANTIZATION= -e NUM_SPECULATIVE_TOKENS=0 \
    -e MAX_MODEL_LEN=4096 -e MAX_NUM_SEQS=4 -e GPU_MEMORY_UTILIZATION=0.93 \
    -e KV_CACHE_MEMORY_BYTES= -e LIMIT_MM_PER_PROMPT='{"image":0,"video":0}' \
    -e MODEL_DTYPE=$dt -e VLLM_NO_USAGE_STATS=1 \
    qwen38-w4a4:latest /opt/qwen38/docker/awq_entry.sh >/dev/null

  fim=$((SECONDS + 780)); ok=0
  while [ $SECONDS -lt $fim ]; do
    docker ps --format '{{.Names}}' | grep -qx $NOME || break
    curl -sf "http://127.0.0.1:${PORTA}/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done

  if [ "$ok" = 1 ]; then
    modelo=$(curl -s "http://127.0.0.1:${PORTA}/v1/models" \
      | python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
    gerar "$modelo" > "$RAIZ/saida_${dt}.json"
    echo "   subiu em ${SECONDS}s | $(python -c "
import json;d=json.load(open('$RAIZ/saida_${dt}.json'));print(f\"logprobs {d['n_logprob']} | nao-finitos {d['nao_finitos']} | {len(d['texto'])} chars\")")"
  else
    echo "   FALHOU"
    docker logs $NOME 2>&1 | grep -iE "error|Error" | tail -3
  fi
  docker logs $NOME > "$RAIZ/logs/equiv_${dt}.log" 2>&1
  docker rm -f $NOME >/dev/null 2>&1
  SECONDS=0
done

echo
echo "== veredito =="
python - "$RAIZ_MNT" <<'PY'
import json, sys, pathlib
r = pathlib.Path(sys.argv[1])
try:
    a = json.loads((r / "saida_bfloat16.json").read_text(encoding="utf-8"))
    b = json.loads((r / "saida_float16.json").read_text(encoding="utf-8"))
except Exception as e:
    raise SystemExit(f"  incompleto: {e}")

if a["nao_finitos"] or b["nao_finitos"]:
    print(f"  NAO-FINITOS: bf16={a['nao_finitos']} fp16={b['nao_finitos']}")
    print("  overflow produz inf, e inf passa por isnan. Este e' o sintoma.")

if a["texto"] == b["texto"]:
    print("  IDENTICO sob greedy — fp16 nao mudou nada nesta amostra")
else:
    pre = next((i for i, (x, y) in enumerate(zip(a["texto"], b["texto"])) if x != y),
               min(len(a["texto"]), len(b["texto"])))
    print(f"  DIVERGIU no caractere {pre} de {len(a['texto'])}")
    print(f"    bf16: ...{a['texto'][max(0,pre-60):pre+60]!r}")
    print(f"    fp16: ...{b['texto'][max(0,pre-60):pre+60]!r}")
    print("  Divergir NAO prova que fp16 esta errado: 3 bits a mais de mantissa")
    print("  mudam o desempate entre logits proximos. Prova que nao e' de graca.")
PY
