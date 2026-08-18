# Medições — RTX 3090 + RTX 3080 Ti, 18/ago/2026

Tudo aqui é servidor de pé ou kernel executado. Nada é estimativa do vLLM:
uma varredura anterior coletou o "estimated maximum model length" das mensagens
de erro e produziu uma curva inteira de números que **nenhum boot reproduziu** —
47/17 "mediu" 146.000 e depois não subiu com 135.000. Estimativa e alocação
discordam, e só a segunda conta.

## PP=2 com AWQ W4A16, sem draft (runner V1)

Partição via `VLLM_PP_LAYER_PARTITION`, estágio 0 = `cuda:0`. Exige
`--shm-size=8g` e `KV_CACHE_MEMORY_BYTES=` vazio.

| partição (3090/3080Ti) | 32.768 | 65.536 | 98.304 |
|---|---|---|---|
| **56 / 8** | subiu, 35,28 tok/s | **subiu, 37,88 tok/s** | falhou |
| 52 / 12 | falhou | — | — |
| 48 / 16 | falhou | — | — |

A 3080 Ti não aguenta mais de 8 camadas como último estágio. `util` é alavanca
grande: 0.84 → 0.88 valeu 19 mil tokens de teto. Referência de uma placa só,
sem draft: 44,59 tok/s — PP custa ~15%, não mais.

## GDN: quanto custa

`bench_gdn.py`, kernels FLA isolados, sem carregar modelo.

| tokens/forward | ms por camada | ms × 48 | % do orçamento de 22,43 ms |
|---|---|---|---|
| 1 | 0,1300 | 6,24 | **27,8%** |
| 4 | 0,1541 | 7,40 | 33,0% |

48 × 3,1 MiB de estado = ~300 MiB por token, que a 850 GB/s seriam 0,35 ms.
Medimos 6,24. **Não é limite de banda — é ocupação**: 48 kernels minúsculos em
série, grid de 192 CTAs sobre 82 SMs, um warp cada (~5% de ocupação).

Varredura de geometria (warps × BV, rodadas intercaladas com âncora): nenhuma
configuração bateu o default fora do ruído, que chegou a **87%** entre rodadas
da mesma config. Sem vencedor — e a primeira rodada, sem âncora, dava "−30%"
que era só clock frio.

## Atenção: o custo do contexto longo

`bench_attn.py`, FlashAttention real, 24 q_heads / 4 kv_heads / head_dim 256.

| seqlen | atenção (16 camadas) | GDN (48) | atenção % |
|---|---|---|---|
| 2.048 | 16,69 ms | 47,15 ms | 26,1% |
| 8.192 | 219,72 ms | 178,84 ms | **55,1%** |
| 32.768 | 3.613 ms | ~715 ms (extrap.) | ~83% |

Quadrático visível: 4× o comprimento custou 13,2× e depois 16,4×. Extrapolando,
128k de prefill são da ordem de **1 minuto só de atenção**.

Isso responde a pergunta que travou vllm#10532 (backend SageAttention) desde
nov/2024: o mantenedor pediu benchmark de LLM, ninguém entregou, o PR morreu.
A resposta é sim para contexto longo. **Mas Sage não serve aqui**: o dispatcher
dele recusa `head_dim > 128` nas cinco variantes, e o nosso é 256.

## DSpark + PP=2

Portão de arquitetura **atravessado** — o `NotImplementedError: Pipeline
parallelism is not supported for this model` sumiu, os dois ranks carregam, o
drafter instancia.

Bloqueio agora é orçamento:

```
Worker_PP0 (3080 Ti, 28 camadas + embed)             8,65 GiB de 10,56
Worker_PP1 (3090, 36 camadas + lm_head + drafter)   15,31 GiB de 21,1
drafter sozinho                                      5,4 GiB
```

Teto de contexto: **4.928 tokens**, idêntico em 16/48 e 28/36. Duas partições
muito diferentes, mesmo número — o limite vem do drafter, que mora inteiro no
último estágio, não do corte do alvo.

O drafter (`dspark-qwen38`) é bf16, 5 camadas, hidden 5120, vocab 248.320,
`tie_word_embeddings=False` — embed e lm_head custam 2,37 GiB cada. Quantizar
para W4 libera ~4 GiB, e é o próximo passo.

Ordem das placas importa: o drafter fica no ÚLTIMO estágio, então
`CUDA_VISIBLE_DEVICES=1,0` põe a 3080 Ti como estágio 0 e deixa a 3090 (24 GB)
hospedar o drafter. Com a ordem natural, ele cai na placa de 12 GB e estoura.

## GDN: onde o ganho NAO esta (verificado 18/ago)

Hipotese do `splitting_ops` DERRUBADA pela docstring do proprio campo:
`FULL_AND_PIECEWISE` captura cudagraph FULL para batches de decode, e piecewise
so para prefill e batches mistos. Os ops GDN so ficam fora do grafo no caminho
piecewise, onde cada kernel dura milissegundos e lancamento e ruido. Nao ha
overhead de lancamento a recuperar no decode -- os 6,24 ms sao kernel de verdade.

Patches de fusao do SNDR valem menos que pareciam: no 0.27.1 o gate ja e kernel
fundido (`fused_gdn_gating`) e a projecao QKV+Z ja e um GEMM so (`in_proj_qkvz`).
PN350 nem e codigo, e issue. PN365/PN298/PN345 sao PRs abertos upstream.

## GDN: onde o ganho PODE estar (nao medido)

`ChunkGatedDeltaRule` e um CustomOp com DOIS caminhos:
  forward_cuda   -> flashinfer.gdn_prefill   (instalado: flashinfer 0.6.16.post3, importa)
  forward_native -> fla_chunk_gated_delta_rule (Triton, o que medimos)

Rodamos o nativo porque a config vem com `custom_ops: [none]`, nao por escolha
medida. So o op de chunk (PREFILL) tem alternativa; decode nao tem.

Importa porque prefill e o gargalo real do uso (repo no prompt): GDN custa
178,84 ms contra 219,72 da atencao em 8k, ~45% do prefill. Se o kernel do
FlashInfer for mais rapido em sm_86, e ganho de TTFT sem escrever kernel.

Ressalvas: o kernel pode ser Hopper-only e cair em erro/lentidao no sm_86, e
ligar `custom_ops` muda mais que esse op. Proximo passo: acrescentar o caminho
`fi_` ao bench_gdn ao lado do `fla_` e comparar nas mesmas formas.
