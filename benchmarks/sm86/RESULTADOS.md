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

`ChunkGatedDeltaRule` tem TRES caminhos, nao dois:
  forward_cuda    -> flashinfer.gdn_prefill      (flashinfer 0.6.16.post3, importa)
  forward_cutedsl -> kernel CuteDSL in-tree
  forward_native  -> fla_chunk_gated_delta_rule  (Triton, o que medimos)

Correcao do que estava escrito aqui antes: nao e `custom_ops: [none]` que nos
prende no nativo. O `__init__` atribui `self._forward_method` na mao, passando
por cima do despacho normal de CustomOp. Quem decide e
`_resolve_gdn_prefill_backend` (qwen_gdn_linear_attn.py:84):

    SM90 (Hopper)                                          -> flashinfer
    familia SM100 (Blackwell) + head_k_dim 128 + CUDA >= 13 -> flashinfer/cutedsl
    resto                                                  -> triton

sm_86 cai no `resto`. Nenhuma config muda isso: pedir
`additional_config.gdn_prefill_backend = flashinfer` so emite
"cannot use this kernel on the current platform" e volta para Triton. O portao
e de arquitetura, e a arquitetura aqui nao passa.

So o op de chunk (PREFILL) tem alternativa. Decode nao tem: fused_recurrent e
Triton em qualquer backend.

Importa porque prefill e o gargalo real do uso (repo no prompt): GDN custa
178,84 ms contra 219,72 da atencao em 8k, ~45% do prefill. Se o kernel do
FlashInfer rodar e ganhar em sm_86, levantar o portao e uma linha no resolver.

O bench agora chama `fi_chunk_gated_delta_rule` direto, pulando o portao, com
cu_seqlens dos dois lados para a comparacao ser justa (a linha `fla` sem
cu_seqlens fica para nao quebrar a serie ja gravada acima). Tres desfechos, e
os tres fecham a pergunta:

    nao compila     -> portao esta certo, alavanca morre
    compila e perde -> portao esta certo, alavanca morre
    compila e ganha -> patch de uma linha, ganho de TTFT sem escrever kernel

Falta rodar. Precisa de GPU, ~1 min, sem subir servidor.

## Triagem da lista externa (Qwen 3.8 Max, sem internet) — 18/ago/2026

Sete itens, ordenados por ele. Reordenados aqui pelo que o codigo e os entrypoints
dizem. Cinco morrem na leitura; dois entram na fila.

### 1. "prefix caching do estado do GDN" — JA ESTA LIGADO

`docker/awq_entry.sh:103-105` e `docker/dspark_entry.sh:111-113` ja passam:

    --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align

Estava ligado em TODAS as medicoes deste arquivo. Nao ha nada para ativar.

E `align` e o teto para este modelo, nao uma escolha conservadora:
  - `qwen3_5.py:313` levanta NotImplementedError em modo `all`
  - `Qwen3_5ForConditionalGeneration` nao declara `supports_mamba_prefix_caching`,
    entao `models/config.py:616` resolveria para `align` sozinho de qualquer jeito
  - `all` tambem obriga `mamba_block_size` a virar multiplo do chunk (interface.py:890),
    o que inflaria o block size da atencao

O que `align` cacheia (cache.py:145): o estado do GDN do ultimo token de cada passo
do scheduler, quando o token cai em `i * block_size`. Com chunked prefill isso
acontece regularmente, entao a restauracao de prefixo e real -- mas so a partir da
SEGUNDA chamada com o mesmo prefixo. Prompt novo paga prefill inteiro. Os 178,84 ms
de GDN e 219,72 de atencao em 8k continuam sendo o custo do primeiro request, e e
esse o numero que manda no TTFT de codigo novo.

NAO MEDIDO e vale medir: a taxa de acerto de verdade. TTFT do 1o contra o 2o request
com prompt identico, na mesma sessao. Se o 2o nao cair muito, `align` esta
checkpointando pouco e ai sim ha trabalho.

### 2. "estado em fp16" — JA E bfloat16

`awq_entry.sh:55-56`: `MAMBA_CACHE_DTYPE` e `MAMBA_SSM_CACHE_DTYPE` ja vem
`bfloat16`. `MambaDType` (cache.py:37) so oferece auto/float32/float16/bfloat16 --
fp16 tem a MESMA largura que bf16. Nao ha trafego para cortar.

E mesmo que houvesse: a premissa dele inverte a nossa medicao. O decode do GDN e
limitado por OCUPACAO (192 CTAs x 1 warp em 82 SMs, ~5%), nao por banda. Os 0,18 ms
de trafego sao 2,9% dos 6,24 ms. Cortar trafego pela metade compraria 0,09 ms de
22,43 -- 0,4% do token. Nao e o item 2 de nada.

### 3. fla vendado vs fla upstream — ENTRA NA FILA

Verdadeiro: o vLLM carrega copia propria em `third_party/flash_linear_attention`.
Comparar com o `flash-linear-attention` do fla-org e A/B de bench, sem servidor,
e o `bench_gdn.py` ja e o lugar. Melhor item da lista dele.

### 4. FlashInfer — RESOLVIDO HOJE, ele estava certo

Ver secao acima: o portao e `_resolve_gdn_prefill_backend`, SM90 ou familia SM100.
sm_86 nunca chega em `forward_cuda`. O bench passa por cima do portao para separar
"o portao esta certo" de "o portao esta conservador".

### 5. SGLang A/B — aberto, mas nao sao 20 min

Exige AWQ de Qwen3.5 hibrido suportado la, mais boot, mais config. Fica na fila
atras do que roda em segundos.

### 6. chunk size 64 — nao e knob

`FLA_CHUNK_SIZE = 64` (third_party/flash_linear_attention/ops/utils.py:31) e
constante de modulo com 34 usos na arvore, nao parametro de funcao. Indexacao de
chunk depende dela. Sweepavel so junto com o item 3, trocando pela versao upstream
onde e argumento.

### 7. TP=2 para TTFT — TROCA CONTRA O QUE MAIS IMPORTA AQUI

Divisibilidade passa: kv_heads 4, linear k/v heads 16/48, todos pares. O problema e
memoria. TP fatia os pesos IGUAIS, entao a 3080 Ti de 12 GB manda nas duas:

    TP=2   pesos ~8 GB/placa  -> ~4 GB livres na 3080 Ti, e a 3090 fica capada
                                 no mesmo valor. KV total ~8 GB.
    PP=2 56/8  3080 Ti segura 8 camadas (~2 GB) -> ~10 GB livres
               3090 segura 56 (~14 GB)          -> ~7 GB livres
               KV total ~17 GB.

TP=2 corta o orcamento de KV pela metade. Isso derruba o teto de contexto de 65k
para a faixa dos 30k. Trocar contexto por TTFT e exatamente o inverso do pedido
("o quanto mais contexto melhor"). Nao entra.

### Fila resultante

    a) fi_ vs fla no bench_gdn                    ~1 min de placa, ja escrito
    b) fla upstream vs vendado no bench_gdn       ~10 min, falta escrever
    c) TTFT 1o vs 2o request, prompt identico     mede se o align serve para algo
    d) drafter dspark-qwen38-w4a4 quantizado      ataca o teto de 4.928 tokens
    e) equivalencia greedy do DSpark              criterio de aceite

## Correcao do item 6: o chunk size E knob, e agora e' variavel de run

Eu disse que `FLA_CHUNK_SIZE` nao era parametro. Errado. Doze dos sitios ja o
escrevem como `chunk_size: int = FLA_CHUNK_SIZE` -- e argumento com default. O que
prende nao e a assinatura, e o MOMENTO: default de funcao liga no import.

Entao o override certo e no ambiente, lido no import, uma vez, por todos:

    VLLM_FLA_CHUNK_SIZE=32   (utils.py, potencia de dois, >= 16)

Isso cobre os 34 sitios de uma vez, inclusive os que leem o global em tempo de
chamada (`chunk.py:38`, `gdn_attn.py:335-387`), porque todos veem o mesmo numero.

O que NAO se pode fazer: mexer no valor com o processo rodando. `chunk.py` le o
global vivo, os sub-ops carregam o default do import. Dois tamanhos de tile na
mesma cadeia de kernel nao e' resultado lento, e resultado errado. Por isso o
`--chunk-sizes` do bench relanca um subprocesso por valor em vez de fazer laco.

    python benchmarks/sm86/bench_gdn.py --chunk-sizes 32 64 128

Nenhum assert na arvore fixa 64, mas tile grande demais estoura a shared memory do
sm_86 e falha no compile do Triton. O bench trata isso como resultado e segue.

Os entrypoints agora imprimem `FLA_CHUNK=` no boot, para nenhuma medicao de
servidor ficar sem o eixo.

## Estado do SSM abaixo de 16 bits: ninguem entrega, e o motivo e estrutural

Pergunta: alguem ja tentou baixar de fp16?

**No vLLM nao ha nem a opcao.** `MambaDType = Literal["auto","float32","float16",
"bfloat16"]` (cache.py:37). Nao e flag que falta validacao, e membro que nao existe
no enum. Grep por fp8/int8 em `layers/mamba/` nao acha nada de estado.

**O motivo nao e preguica, e a recorrencia.** Uma entrada de KV e escrita uma vez e
lida muitas: o erro de quantizacao fica local. O estado do SSM e lido-modificado-
escrito A CADA PASSO. Erro de um passo entra no proximo e acumula. Oito mil tokens
de decode sao oito mil arredondamentos sobre o mesmo tensor.

O proprio vLLM ja sente isso em 16 bits: existe
`--enable-mamba-cache-stochastic-rounding`, com rodadas de Philox configuraveis,
cuja docstring diz "usa bits aleatorios para desviesar o erro de arredondamento,
o que pode melhorar a estabilidade numerica para sequencias longas"
(config/mamba.py:43-49). Se 16 bits precisam de PRNG para nao derivar, e4m3 --
tres bits de mantissa -- e outro esporte.

**A pesquisa que existe quantiza peso, nao estado.** Quamba e Quamba2
(arXiv 2410.13229, 2503.22879) fazem W4A8/W8A8 em Mamba1/2, mas precisaram de
reordenacao de pesos ciente de cluster, agrupando heads e canais de faixa parecida
para dividir escala, mais quantizacao por grupo de estado para B e C. Isto e: 8 bits
da, com um framework inteiro em volta, e e int8 com escalas, nao fp8 cru na memoria.
Ternary Mamba (arXiv 2606.18114) e W1.58**A16** -- o A16 entrega o jogo: ate o
paper ternario mantem o estado em 16 bits.

**O unico caminho credivel que achei contorna o problema em vez de resolver.**
RFC vllm#47572 (ReplaySSM) avalia estado em fp16/fp8 guardando os ENTRADAS do SSM
num ring buffer e reconstruindo, com flush de checkpoint a cada B passos -- assim o
estado de baixa precisao e requantizado muito menos vezes que uma vez por passo.
Esta nesta arvore (`use_replayssm`, `replayssm_buffer_len=16`), e fechado para nos:
`validate_mamba_cached_kernel` exige `supports_replayssm`, que so o Nemotron-H
declara, exige backend Triton e recusa spec decode.

### Se existisse, quanto valeria AQUI

Velocidade: nada. Decode do GDN e limitado por ocupacao; os 0,18 ms de trafego sao
2,9% dos 6,24 ms.

O premio de verdade seria outro, e ninguem na lista externa mencionou. A page do
mamba tem que caber na page da atencao, e isso EMPURRA o block size da atencao.
Do log de boot real:

    interface.py:911  Setting attention block size to 448 tokens
                      to ensure that attention page size is >= mamba page size

Conferindo: estado ssm por camada em bf16 = 48 x 128 x 128 x 2 = 1,5 MiB; page de
atencao por token = 2 x 4 kv_heads x 256 head_dim x 2 = 4 KiB. 1,5 MiB / 4 KiB =
384 tokens, mais o conv state fecha em 448.

Em modo `align`, `mamba_block_size = block_size`. Ou seja: **o estado do GDN so e
checkpointado a cada 448 tokens**, e essa e a granularidade do prefix caching que
ja esta ligado. Estado em 8 bits levaria isso para ~224, dobrando a frequencia de
checkpoint e reduzindo o desperdicio de bloco parcial.

Mas isso exige dtype que nao existe no enum mais load/store nos kernels Triton do
FLA mais gestao de escala. E projeto, nao knob. Fica registrado como o argumento
certo caso alguem volte ao assunto.
