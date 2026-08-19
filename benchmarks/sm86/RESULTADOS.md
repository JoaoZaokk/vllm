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

## O estado do GDN e uma contracao — quanto tempo o erro sobrevive nele

`analise_decay_gdn.py`, so CPU, segundos, le direto do checkpoint.

O argumento "recorrencia acumula erro, entao esqueca sub-16-bit" que escrevi acima
esta forte demais, e o codigo diz por que. O gate do GDN e

    fused_gdn_gating_kernel:  blk_g = -exp(A_log) * softplus_x

com `exp(A_log) > 0` e `softplus > 0`. Logo **g < 0 sempre** e `exp(g)` esta em
(0,1) estritamente. O estado nao e um acumulador: e uma contracao, por construcao.
Erro injetado num passo DECAI nos seguintes.

`A_log` e `dt_bias` estao no checkpoint sem quantizacao (96 tensores, 48 camadas
x 2). No ponto de operacao a=0:

| | decay exp(g) | meia-vida do erro |
|---|---|---|
| min | 0,000117 | 0,1 passo |
| p1 | 0,040 | 0,2 |
| **mediana** | **0,973151** | **25,5 passos** |
| p99 | 0,999731 | 2.574 |
| max | 0,999962 | 18.132 |

Para o head mediano o erro some em 25 tokens, nao em 8.000. A premissa do
argumento padrao nao vale aqui.

Mas meia-vida e a leitura errada. O que decide e a amplificacao em regime
permanente, `1/(1-d^2)`, porque a amplitude cresce com a raiz dela. int8 injeta
~0,4% por passo:

| | decay | amplitude | erro final |
|---|---|---|---|
| mediana | 0,973151 | 4,3x | **1,7%** |
| limiar lento | 0,999 | 22,4x | 8,9% |
| p99 | 0,999731 | 43,1x | 17,2% |
| pior head | 0,999962 | 114,4x | **45,7%** |

Ou seja: **o head mediano aguenta int8 com folga; o pior head e destruido**. Nao e
uma decisao por modelo, e por head. Isso e exatamente a tese do paper de
sensibilidade KL (arXiv 2604.13440) para hibridos SSM+Transformer -- so que aqui a
sensibilidade nao precisa de calibracao nem de forward: sai de `A_log` e `dt_bias`,
que sao pesos.

### Por que a versao barata nao paga

341 de 2.304 heads (14,8%) tem decay > 0,999. Estao espalhados: so **7 das 48
camadas** nao tem nenhum. Decidir por camada obrigaria a manter 41 camadas em
bf16 -- nao compra nada.

Por head, sim: 0,574x o estado de hoje, o que levaria o block size de 448 para
~257 e dobraria quase a frequencia de checkpoint do prefix caching em modo
`align`. Mas exige dtype por head dentro do mesmo tensor, nos kernels Triton do
FLA. E projeto.

### Ressalva que nao pode sumir

Tudo isso e em `a = 0`. O termo dependente da entrada desloca: `a` positivo faz
softplus crescer, g ficar mais negativo, memoria ENCURTAR (melhor para
quantizacao); `a` negativo faz o contrario. E referencia, nao limite. Medir o `a`
real exige hook num forward de verdade, e e o proximo passo se alguem retomar.

### Credito

Achado a partir do argumento de bounded range do
[relu-clip](https://huggingface.co/jiaheguo521/relu-clip): "the unbounded ResNets
reach only 0.045-0.147 -- the two groups do not overlap. In int8 codes that is 256
of 256 against 61-218". O que faz int8 funcionar la e a faixa limitada, nao o
quantizador. A pergunta certa aqui virou "o estado do GDN tem faixa limitada?", e
o gate responde que sim.

# AUDITORIA 18/ago/2026 — O QUE DESTE ARQUIVO CAIU

Cinco auditores adversariais, 45 achados (8 quebra, 17 silencioso). O que segue foi
reconferido a mao antes de ser escrito aqui. Numero riscado aqui NAO deve ser citado
sem refazer a medicao.

## RISCADO: 27,8% do orcamento em GDN (tabela da secao "GDN: quanto custa")

Dois defeitos independentes, cada um sozinho ja bastava.

**1. O bench mede um kernel que este modelo nao chama.** `bench_gdn.py` cronometra
`fused_recurrent_gated_delta_rule`. O caminho de decode do Qwen3.5 chama
`fused_sigmoid_gating_delta_rule_update` (qwen_gdn_linear_attn.py:1377, 1404, 1462) e
`fused_recurrent_gated_delta_rule_packed_decode` (linha 1604). Sao kernels diferentes:
o que roda em producao paga o gating fundido e o gather do pool de estados; o que foi
medido recebe as entradas ja desempacotadas.

**2. O cronometro inclui dispatch de Python.** `call()` roda
`inspect.signature(fn).parameters` a cada chamada, e `call()` esta DENTRO da janela
entre `start.record()` e `end.record()`. Com a stream vazia, a GPU fica parada
esperando o Python. Os 0,1300 ms/camada carregam isso.

Cai junto a conclusao **"nao e limite de banda, e ocupacao"**: ela foi tirada da razao
entre esse numero e um piso de banda calculado com estado em fp32, quando o deploy usa
bf16. Numerador do kernel errado, denominador da largura errada.

Cai junto a razao 27,8% por um terceiro motivo: numerador e 48x uma medicao eager
isolada, denominador e um token de 22,43 ms de servidor com CUDA graph. Populacoes
diferentes.

## RISCADO: "o teto de 4.928 tokens vem do drafter"

`dspark_entry.sh:32` e `KV_BYTES="${KV_CACHE_MEMORY_BYTES:-1342177280}"`. A forma `:-`
substitui tambem quando a variavel esta VAZIA, nao so quando nao existe. Entao passar
`-e KV_CACHE_MEMORY_BYTES=` da 1,25 GiB fixo, e o teto de contexto vira
floor(constante / custo por token) -- identico em 16/48 e em 28/36 porque nao depende
nem da particao nem do drafter.

A inferencia registrada ("dois cortes muito diferentes, mesmo numero, logo o limite e o
drafter") estava lendo uma constante como se fosse medicao.

Consequencia direta: a escada que escrevi hoje para o drafter w4a4 herdou a mesma
armadilha e NAO poderia ter mostrado ganho nenhum, mesmo se tivesse subido.

`awq_entry.sh:32` usa `${KV_CACHE_MEMORY_BYTES-1879048192}` -- um traco so, que respeita
vazio. Os dois entrypoints se comportam DIFERENTE com a mesma variavel vazia, e as
medicoes de PP=2 sem draft (que usaram o awq) nao sao afetadas.

## RISCADO: as duas primeiras escadas do DSpark nao mediram contexto

Os logs mostram a morte real, e nao e KV nem drafter:

    encoder_runner.py:88   profile_encoder_cache
    encoder_runner.py:106  execute_mm_encoder
    qwen3_vl.py:2811       embed_multimodal -> _process_image_input
    torch.OutOfMemoryError: Tried to allocate 144.00 MiB

Morre perfilando a TORRE DE VISAO, antes de qualquer dimensionamento de KV. Some-se a
isso que a primeira rodada nao passou `VLLM_PP_LAYER_PARTITION` e a divisao caiu em
32/32. `--limit-mm-per-prompt` nunca foi setado em lugar nenhum desta stack.

## RISCADO: "atencao e 55,1% do prefill em 8k"

O denominador de `bench_attn.py` e apenas atencao + GDN. Nao inclui nenhum GEMM --
nem as projecoes, nem o MLP, que num 27B dominam o prefill. Chamar aquilo de "fatia do
prefill" infla a fracao em algo entre 8x e 15x, na direcao que faz o SageAttention
parecer valer a pena.

A conclusao ALTERNATIVA sobrevive intacta, porque nao depende dessa razao: Sage recusa
`head_dim > 128` nas cinco variantes e o nosso e 256.

## QUEBRADO: a imagem nao carrega o patch do chunk size

`Dockerfile.pp-dspark` tem 20 linhas COPY; `git diff v0.27.1..HEAD --name-only` lista 21
arquivos .py de codigo. O que falta e
`vllm/third_party/flash_linear_attention/ops/utils.py` -- justamente o do
`VLLM_FLA_CHUNK_SIZE`. Alem disso a imagem foi construida as 18:34 UTC e o commit e das
21:04 UTC: ela e mais velha que o patch de qualquer forma.

Efeito: dentro do container o boot imprime `FLA_CHUNK=32` e o kernel usa 64. O commit
9d40e9871 afirma que o eixo fica registrado no log. Fica registrado errado, que e pior
que nao registrar.

As duas asserts do Dockerfile cobrem 3 dos 20 arquivos. Foi por esse buraco que este
arquivo saiu da lista sem ninguem notar.

## QUEBRADO: caminhos que nao executam como estao no disco

- `docker compose up -d dspark`: aponta para `qwen38-w4a4:latest` (SEM os patches) e
  PP=1. Nenhum spec decode sob pipeline parallel sai desse caminho. E o `.env:38` fixa
  `NUM_SPECULATIVE_TOKENS=2`, que o vLLM rejeita porque o checkpoint declara
  `dspark_block_size=7` (speculative.py:1042-1058). Morre no boot.
- `docker compose up -d awq_dual`: recebe `GPU_MEMORY_UTILIZATION=auto`, mas so o
  `dual_entry.sh` sabe expandir "auto". O `awq_entry.sh` joga a string crua em
  `--gpu-memory-utilization`, que e float. argparse mata antes de carregar peso.
- `.env` sombreia os defaults do compose inteiro: `docker compose up -d` hoje sobe o awq
  com ctx 32768 e KV de 3 GiB, nao com os 8192/1,75 GiB sob os quais os tok/s da tabela
  foram medidos.
- `validar_dspark_pp.sh` -- o criterio de aceite -- nao passa `CUDA_VISIBLE_DEVICES`, e
  o entrypoint entao poe a 3090 como rank 0. O drafter, que mora no ULTIMO estagio, cai
  na placa de 12 GB. A equivalencia greedy nunca chegou a ser avaliada.
- `run_tests.sh` sempre sai 0: o pipe para `tail` engole o status do pytest. Como portao
  ("run_tests.sh && medir") ele aprova suite vermelha, e roda sem `--gpus`, entao nao
  distingue "passou" de "pulou".

## QUEBRADO: tres bugs no bench_gdn, todos meus, todos de hoje

- `--chunk-sizes=32` (com `=`) re-lanca o processo infinitamente. argparse aceita a
  sintaxe sem reclamar; a limpeza de `sys.argv` so remove a forma com espaco.
- `break` dentro do `except` do caminho `fi` aborta a varredura de seqlen inteira em vez
  de pular uma linha. No desfecho mais provavel (fi nao compila em sm_86), o bench
  imprime so 2048 e some com 8192 sem mensagem.
- `--batch > 1` quebra o prefill e, no decode, erra a porcentagem por um fator igual ao
  batch.

## O QUE SOBREVIVEU

- A contracao do GDN e a analise de decay (`analise_decay_gdn.py`): le peso do
  checkpoint, nao depende de kernel nem de servidor.
- O portao de arquitetura do FlashInfer (SM90/SM100): leitura de codigo.
- Sage recusa head_dim 256: leitura de codigo.
- `align` ja ligado nos entrypoints e `all` proibido para Qwen3.5: leitura de codigo.
- PP=2 AWQ sem draft, 56/8, 65.536 tokens a 37,88 tok/s: usou `awq_entry.sh`, cuja
  expansao de KV respeita vazio. Nao caiu na armadilha do `:-`.

# REMEDIDO 18/ago/2026, depois da auditoria

Instrumentos corrigidos, imagem reconstruida, placa livre. Isto substitui o que
ficou riscado acima.

## Piso de ruido, medido de graca

A varredura de chunk size mexeu num knob que o kernel de DECODE nao enxerga --
`fused_sigmoid_gating_delta_rule_update` nao recebe chunk_size. Ainda assim a
coluna de decode andou:

    chunk 16 -> 0,0420 ms/camada
    chunk 32 -> 0,0389
    chunk 64 -> 0,0349

20% de espalhamento num eixo que nao existe. **Diferenca menor que ~20% na coluna
de decode nao e sinal.** Vale para tudo abaixo.

## GDN no decode: 8,5%, nao 27,8%

Caminho de producao (`fused_sigmoid_gating_delta_rule_update`), 3090, bf16:

| seqs | ms/camada | ms x48 | limite superior |
|---|---|---|---|
| 1 | 0,0399 | 1,92 | **8,5%** |
| 2 | 0,0553 | 2,65 | 11,8% |
| 4 | 0,0812 | 3,90 | 17,4% |
| 8 | 0,1101 | 5,28 | 23,6% |

O numero riscado dizia 27,8% com uma sequencia. O real e 8,5% -- **o teto do que
qualquer tuning de GDN pode comprar no decode caiu para menos de um terco**.

Isso encerra a familia SNDR de vez, com aritmetica em vez de opiniao: acelerar o
GDN em 20% compra 1,7% do token. Nao vale uma semana, nao vale um dia.

"Limite superior" continua sendo o rotulo certo: numerador eager e camada
isolada, denominador e token de servidor com CUDA graph.

Comparacao com o kernel fora do caminho, na mesma rodada e mesmas formas:

    producao (fused_sigmoid_gating)  0,0399 ms  <- 40% MAIS RAPIDO
    legado (fused_recurrent)         0,0660 ms

Ou seja, o bench antigo nao so media o kernel errado: media um kernel mais lento,
e por isso o custo do GDN parecia maior do que e.

## Atencao no prefill: 3,6%, nao 55,1%

Com o denominador completo (atencao + GDN + GEMM), 16 camadas de atencao contra
48 de GDN e os GEMM de todas as 64:

| seqlen | atencao x16 | GDN x48 | GEMM | attn % | teto Sage 2x |
|---|---|---|---|---|---|
| 2.048 | 16,62 | 44,04 | 1.450,36 | **1,1%** | 0,6% |
| 8.192 | 215,48 | 178,81 | 5.604,63 | **3,6%** | 1,8% |
| 32.768 | 3.735,91 | — | 21.413,10 | ~14,9% | ~7,4% |

O riscado dizia 55,1% em 8k. O real e 3,6%: **inflacao de 15x**, dentro da faixa
que a auditoria previu, e na direcao que fazia o Sage parecer valer a pena.

Com isso a resposta que este repositorio da para vllm#10532 muda de lado. Eu
tinha escrito "sim para contexto longo". Esta errado em 8k: um backend 2x mais
rapido na atencao compra 1,8% do prefill. Em 32k a fracao sobe para ~15% e um 2x
compraria ~7%, o que ja nao e desprezivel -- mas o PR morreu por um bom motivo
para o regime em que quase todo mundo roda.

A conclusao alternativa continua de pe e nao dependia disso: Sage recusa
`head_dim > 128` nas cinco variantes, e o nosso e 256.

GEMM aqui e PISO -- matmul denso bf16, enquanto o deploy paga AWQ W4A16 com
dequantizacao. Logo a fatia da atencao e limite SUPERIOR.

## FlashInfer no GDN: alavanca MORTA, confirmado na maquina

    2048  fi  --  NotImplementedError: GDN prefill DSL kernel is unavailable

O portao de arquitetura estava certo. Nao e que o kernel seja lento em sm_86: ele
nao existe para esta placa. Pergunta encerrada, sem patch no resolver.

O bench chegou a 8192 e imprimiu as duas linhas de fla. Antes desta correcao o
`break` dentro do `except` teria abortado a varredura inteira e sumido com a
linha de 8192 sem uma palavra. A correcao se provou na primeira execucao.

## Chunk size: 64 ja e o otimo, e 128 e ILEGAL

Prefill 8192, ms x48:

| chunk | fla | fla+cu |
|---|---|---|
| 16 | 212,92 | 212,83 |
| 32 | 174,10 | 170,70 |
| **64 (default)** | **172,30** | 178,24 |
| 128 | AssertionError | — |

16 e claramente pior (+23%). Entre 32 e 64 nao ha vencedor: a ordem se INVERTE
entre as duas colunas da mesma rodada, o que e a assinatura de ruido, nao de
efeito.

E ha um teto rigido que eu tinha afirmado nao existir. Escrevi que "nenhum assert
na arvore fixa 64". Fixa: `solve_tril.py:531` tem

    assert A.shape[-1] in [16, 32, 64]

Entao 128 nem chega a compilar -- morre em assercao Python, nao em shared memory
como eu previa. O espaco util do knob e {16, 32, 64}, e o default ja esta no
melhor dos tres.

A infraestrutura do knob fica: e barata, esta correta, e o eixo agora aparece no
log de boot de verdade (a imagem carrega o patch depois da correcao do
Dockerfile). Mas a alavanca nao tem o que comprar.

## O que sobreviveu da serie antiga

O GDN no PREFILL. O caminho de chunk E' o de producao (`forward_native` ->
`fla_chunk_gated_delta_rule`), entao aquele numero sempre mediu o kernel certo:

    antes  178,84 ms em 8192
    agora  172,30 a 179,18 ms

Dentro do ruido. Era so' o DECODE que estava medindo outra coisa.

## Onde isso deixa a fila

Tres alavancas de GDN mortas por medicao, nao por palpite: SNDR (teto de 8,5%),
FlashInfer (kernel inexistente em sm_86), chunk size (64 ja e o otimo de tres
valores legais). Sage morto por `head_dim`.

O que sobra e o que sempre foi: **contexto e o drafter**. GEMM e' 5,6 s dos 6,0 s
de prefill em 8k -- 94%. Qualquer ganho serio de TTFT esta na quantizacao dos
GEMM, nao em kernel de atencao nem de recorrencia.

# BATERIA DE TTFT — 18/ago/2026, noite

Primeira medicao de TTFT desta stack. Ate hoje so' existia tok/s de decode, o que
e' a metrica errada para o uso declarado (repositorio inteiro no prompt, resposta
curta): quem manda ali e' o tempo ate o primeiro token, nao a vazao depois.

Uma placa (3090), sem draft, `max_model_len` 4096, `util` 0.93, KV dimensionado
pelo vLLM, torre de visao desligada (`--limit-mm-per-prompt {"image":0,"video":0}`).

Cada configuracao mede na ordem sonda -> frio -> quente -> decode. A ordem nao e'
enfeite: a sonda come o JIT do Triton E a compilacao do caminho de decode. Uma
versao anterior sondava com `max_tokens=1`, o que aquece so' o prefill, e a
primeira geracao de verdade mediu 1,24 tok/s num servidor que faz 45.

## Resultado

| config | prompt | TTFT frio | TTFT quente | ganho do cache | decode |
|---|---|---|---|---|---|
| A  AWQ W4A16 bf16 (Marlin) | 2382 | 1854 ms | 346 ms | 81,3% | 45,93 tok/s |
| B  AWQ W4A16 fp16 (Marlin) | 2382 | 1879 ms | 353 ms | 81,2% | 45,59 tok/s |
| C  ConvRot W4A4 int4 MMA | 2301 | **530 ms** | 96 ms | 81,8% | 29,27 tok/s |
| D  ConvRot W4A4, ativacao bf16 | 2382 | 2019 ms | 424 ms | 79,0% | 7,00 tok/s |

## 1. O prefix caching ACERTA. 81%, e nunca tinha sido medido

`--mamba-cache-mode align` estava ligado nos entrypoints desde sempre, e ninguem
sabia se servia para alguma coisa. Serve: o mesmo prefixo repetido cai de 1854 ms
para 346 ms.

Numa validacao separada, com prompt de 2138 tokens e tres repeticoes no MESMO
servidor, o ganho foi **91,3 / 91,4 / 91,5%** -- dentro de 0,2% entre si, com
decode 44,9 a 45,3 (bate com os 44,59 ja registrados, entao o instrumento esta
calibrado).

Na bateria o ganho caiu para ~81%. Duas diferencas: prompt maior (2382 contra
2138) e servidor recem-subido em vez de ja quente. Se o acerto degrada com prompt
maior, seria consistente com a granularidade de checkpoint de 448 tokens que
calculamos -- mas isso e HIPOTESE, nao medicao. Fica na fila.

## 2. fp16 contra bf16: 1,4% em tudo. A hipotese estava certa no mecanismo

A hipotese e do usuario, desta sessao: fp16 tem 10 bits de mantissa contra 7 do
bf16, e num modelo ja quantizado o alcance esta domado pelas escalas, entao os 3
bits a mais poderiam recuperar qualidade.

Velocidade: **nao muda nada**, como previsto -- fp16 e bf16 tem a mesma vazao no
tensor core do Ampere. 1854 vs 1879 ms de TTFT, 45,93 vs 45,59 de decode. Trocar
nao custa.

Confirmacao independente de outra sessao, em difusao na mesma classe de placa:
±5% em quatro backends de atencao, ≤1% com backend forcado no SDPA.

QUALIDADE CONTINUA SEM RESPOSTA, e isto e' um buraco do meu teste, nao um
resultado. A bateria mede TTFT, cache e tok/s -- zero qualidade. Pior: cada
configuracao usou semente de prompt diferente (`sem${RANDOM}`), entao as saidas de
A e B nunca foram comparaveis nem se eu quisesse. Escrito
`equivalencia_dtype.sh` para fechar isso: semente FIXA, greedy, diff da saida.

E o modo de falha que quase passou batido, trazido de outra sessao com
contraexemplo medido: fp16 estoura em 65504, e **overflow em fp16 produz `inf`,
nao `NaN`**. `isnan` devolve False e a degradacao passa em silencio. O guard certo
e' `isfinite`. Contraexemplo real: ativacao de 344064 num Z-Image, cinco vezes o
teto. Modelo diferente, nao prova nada sobre Qwen -- mas mata a premissa de que
ativacao real cabe em 65k.

## 3. O int4 MMA nativo GANHA no prefill. 3,5x

Esta e' a pergunta que a tabela antiga do compose nunca respondeu, porque so'
media decode -- o unico regime onde o int4 nao pode ganhar por construcao.

    A  AWQ W4A16, Marlin desempacota para bf16    1854 ms
    C  ConvRot W4A4, int4 MMA m16n8k64 nativo      530 ms

O mecanismo confere: no prefill o M e' grande (milhares de tokens), a conta e'
limitada por COMPUTE, e o W4A16 faz os mesmos FLOP do bf16 mais o desempacote. O
W4A4 e' o unico caminho que realmente aciona o tensor core int4 da GA102.

E o preco aparece exatamente onde a teoria mandava:

    decode  45,93 (AWQ)  contra  29,27 (ConvRot)   -36%

No decode o M e' 1 a 8, a conta e' limitada por BANDA DE PESO, e o peso e' 4 bits
nos dois casos -- entao o int4 so' paga a quantizacao de ativacao e nao tem onde
cobrar. Era exatamente isso que a tabela antiga de tok/s estava medindo.

### Onde as duas curvas cruzam

Com os dois numeros medidos, para um pedido de 2382 tokens de prompt e G gerados:

    AWQ       1854 + 21,8 G  ms
    ConvRot    530 + 34,2 G  ms
    cruzam em  G ~ 107 tokens

Saida curta, ConvRot ganha. Saida longa, AWQ ganha. E o cruzamento anda para a
DIREITA conforme o prompt cresce, porque a vantagem de prefill escala com P e a
desvantagem de decode nao.

Mas as duas alavancas competem entre si: com prefix caching acertando 81%, a
segunda chamada com o mesmo prefixo quase nao paga prefill, e ai o AWQ volta a
ganhar. Quem decide e' o padrao de uso real, nao a tabela.

### Ressalvas que nao podem sumir

- **Bases diferentes.** O ConvRot foi produzido aqui a partir de `source-heretic`;
  o AWQ veio de outro lugar. Formas e compute sao identicos, entao o TEMPO e
  comparavel -- a QUALIDADE nao e'.
- **Custo de qualidade conhecido e nao medido aqui.** O proprio plugin registra:
  "Quantizacao de ativacao em int4 degrada raciocinio mesmo com a saida
  continuando fluente", com uma pergunta de primos que o W4A4 erra e o W4A16
  acerta. A bateria nao testa nada disso.
- **Prompt 3,4% menor no C** (2301 contra 2382), efeito colateral da semente
  aleatoria por configuracao. Nao explica um fator de 3,5, mas e' imprecisao real.
- **ConvRot nao roda sob PP=2** (morre no Dynamo), entao este ganho nao compoe com
  a particao 56/8 que deu 65k de contexto.

## 4. W4A8: o `int8` que existe NAO e' o tier W4A8

Confirmado no codigo, e vale registrar porque a tabela do compose credita "17,8
tok/s" a uma coisa que nao e' o que o nome sugere:

    QWEN_W4A4_ACT_DTYPE=int8  ->  convrot_w4a4_linear(linear_dtype="int8")
                                  ativacao int8 no LAYOUT DE PESO do W4A4

    tier real                 ->  w4a8_int8_linear
                                  import proprio, layout qdata + s_rel + s_channel

O tier real foi abandonado por quebrar captura de CUDA graph no model runner que o
DSpark exige. Nunca rodou aqui.

Outra sessao esta medindo esse tier em bench de KERNEL PURO -- que nao tem model
runner nem CUDA graph, e portanto consegue medir o que aqui nao da. Numero deles em
difusao, contra W4A4: erro mediano **3,18x menor**, faixa 2,29 a 4,55x, custo de
+3,9% em disco e +2,5% em velocidade. Se confirmar, e' essa a ordem de grandeza que
justifica consertar o CUDA graph aqui.

### D e' o controle, e ele fecha o argumento

C e D sao o MESMO checkpoint, o MESMO peso, o MESMO entrypoint. Muda uma variavel
de ambiente:

    QWEN_W4A4_ACT_DTYPE=int4   TTFT  530 ms   decode 29,3
    QWEN_W4A4_ACT_DTYPE=bf16   TTFT 2019 ms   decode  7,0
                                     3,8x           4,2x

Entao o ganho do C NAO vem de nada do checkpoint ConvRot, da receita de
quantizacao, nem da base heretic. Vem do tensor core int4. Sem o controle, a
tabela A vs C seria ambigua: dois checkpoints, duas bases, dois caminhos. Com ele,
sobra uma variavel so'.

E o D tambem mede o que o comentario do `awq_entry.sh` afirmava sem numero: que o
Marlin e' muito melhor que um dequant ingenuo. Confere, e o tamanho e' brutal no
decode -- 45,9 contra 7,0, seis vezes e meia. No prefill a distancia quase some
(1854 contra 2019), o que faz sentido: la a conta e' de compute e os dois acabam
fazendo os mesmos FLOP em bf16.

Leitura final das quatro linhas:

    Marlin (A)          melhor decode com folga, prefill mediano
    ConvRot int4 (C)    melhor prefill por 3,5x, decode 36% abaixo do Marlin
    ConvRot bf16 (D)    pior dos dois mundos -- e' o dequant ingenuo

## O que ficou de fora, e por que

- **Qualidade**, em todas as quatro. Nenhuma foi submetida a bateria de acertos, e
  o custo conhecido do int4 em raciocinio esta registrado no plugin sem numero.
- **Equivalencia bf16 vs fp16 sob greedy**: ferramenta escrita
  (`equivalencia_dtype.sh`), nao executada.

## Erros deste turno, todos meus, todos custaram uma rodada

1. **CRLF.** Meus edits via Python escreveram em modo texto no Windows, o que
   converte o arquivo inteiro. Dentro do container Linux, `set -euo pipefail\r`
   vira "invalid option name" e o `awq_entry.sh` morreu em 2 segundos. Tres shell
   scripts estavam contaminados; os outros arquivos CRLF (`.env`, compose, `.ps1`,
   Dockerfile, `.py`) ja eram assim e nao quebram -- so' bash nao tolera.
2. **Prompt de 300 itens** dava ~8 mil tokens e estourava `max_model_len`: HTTP
   400. E o medidor engolia o corpo do erro, entao o 400 nao ensinava nada.
3. **Git Bash converteu** `/workspace/models/...` em
   `C:/Program Files/Git/workspace/models/...` ao passar como argumento: HTTP 404
   sobre um caminho que ninguem digitou. Corrigido eliminando a classe -- o medidor
   PERGUNTA o nome ao servidor em `/v1/models`.
4. **Sonda com `max_tokens=1`** aquecia so' o prefill; a primeira geracao pagou a
   compilacao do decode e mediu 1,24 tok/s. Mesma licao do JIT que ja tinha me
   pegado uma vez, agora no outro caminho.
5. **Teto de boot em 540 s** cortou o C por 2 segundos (542) numa rodada em que o D
   subiu em 440.

O backup das configuracoes de antes da bateria esta em
`backup-config/20260818-203718/`.

# VARREDURA DE M E O 2x2 DE CUDA GRAPH — 18/ago/2026, madrugada

Duas medicoes que juntas fecham a pergunta "por que o int4 nativo perde no
decode", e derrubam DUAS explicacoes pelo caminho -- uma minha, uma de fora.

## Antes: uma extrapolacao minha que nao entra em lugar nenhum

Eu disse em conversa que o TTFT de 530 ms contra 1854 ms era "3,5x, ou 87% da
razao de pico de tensor core (4x), logo o kernel e excelente". Isso e' invalido:
TTFT end-to-end carrega atencao, GDN, launch, epilogo, cache e prompts de
tamanhos diferentes (2301 contra 2382 tokens). Nao se tira eficiencia de kernel
de uma razao dessas. Nao foi gravado no registro e nao deve ser citado.

## Varredura de M, formas reais do Qwen, seis caminhos

`sweep_gemm_m.py`, GEMM isolado, sem servidor. Tempo total por chamada em us,
`q_proj` (5120 -> 6144):

| M | bf16 | gemv_awq | marlin | convrot_a4 | convrot_a8 | w4a8_real |
|---|---|---|---|---|---|---|
| 1 | 115,7 | 387,0 | **30,5** | 138,2 | 129,5 | 209,4 |
| 8 | 131,0 | 1267,6 | **31,7** | 146,8 | 245,2 | 212,0 |
| 64 | 128,0 | 1023,9 | **109,4** | 157,2 | 228,8 | 218,0 |
| 128 | 169,5 | 1627,9 | 156,7 | **99,7** | 237,6 | 224,1 |
| 256 | 287,7 | 3193,3 | 266,2 | **126,0** | 287,7 | 244,2 |
| 1024 | 1089,9 | — | 898,5 | **271,2** | 542,1 | 528,9 |
| 5856 | 5130,8 | — | 5198,3 | **1314,2** | 2277,9 | 2203,1 |

`down_proj` (17408 -> 5120) tem a mesma forma de curva, com o cruzamento no mesmo
lugar.

### O ajuste que separa pedagio de trabalho

`tempo = a + b*M` na faixa M<=64. `a` e' o custo FIXO por chamada, `b` o custo
por linha:

| caminho | fixo/chamada | por linha |
|---|---|---|
| marlin | **30,8 us** | 1,189 us |
| bf16 | 117,6 | 0,206 |
| convrot_a4 | 143,5 | **0,388** |
| convrot_a8 | 187,4 | 1,223 |
| w4a8_real | 211,4 | **0,062** |
| gemv_awq | 663,1 | 5,566 |

Assinatura clara: **Marlin tem o menor pedagio e o maior custo marginal;
ConvRot e W4A8 tem o oposto.** Dai o cruzamento.

RESSALVA que precisa andar junto: `b` e' coeficiente de ajuste empirico, nao
throughput intrinseco de tensor core. A curva troca de regime -- o proprio
`convrot_a8` e' NAO MONOTONICO (M=1 da 129 us, M=16 da 351, M=32 volta a 225),
o que so' acontece se o kernel muda de estrategia por tamanho. Ler `b` como
"19x melhor que o Marlin" seria repetir o erro do "87% do pico".

### Cruzamento contra o baseline REAL

    M = 1      marlin 4,5x mais rapido que convrot_a4
    M ~ 100    vira
    M = 5856   convrot_a4 4,0x mais rapido

E' MAIS CEDO que o cruzamento contra bf16 medido em outra sessao (M=256), nao
mais tarde -- porque o Marlin degrada rapido com M enquanto o bf16 fica plano.
Quem compara contra bf16 esta comparando com um baseline fraco nos dois extremos.

### Dois caminhos que ninguem tinha rodado

**`gemv_awq_w4a16` e' o pior da tabela.** Eu tinha apontado como possivel
alavanca de decode por ser GEMV dedicado a M pequeno. Nao e': 663 us de pedagio,
21x o Marlin. Medido, nao suposto.

**`w4a8_int8_linear`, o tier "que nunca rodou", RODA.** As duas falhas iniciais
eram de chamada, minhas:
  - `quantize_w4a8_int8_weight` devolve `(qdata, s_rel, s_channel, correction,
    codebook)` mas o `linear` recebe `codebook` ANTES de `correction`. Posicional
    inverte os dois e o backend recusa com "correction: shape [16] fails:
    exactly 2D" -- estava recebendo o codebook de 16 entradas no lugar da correcao.
  - `gemv_awq_w4a16` quer `qweight` int8 com `k//2` colunas, nao int32.

## O 2x2: CUDA graph nao e o culpado

A hipotese que sobrou da varredura era: o pedagio do ConvRot e' ausencia de
captura de CUDA graph. Teste de quatro celulas, cache de prefixo DESLIGADO nas
quatro, semente de prompt FIXA (2058 tokens identicos em todas):

| celula | TTFT frio | decode |
|---|---|---|
| A marlin + graph | 1590 ms | 45,89 tok/s |
| B marlin + eager | 1725 ms | 15,00 |
| C convrot + graph | **502 ms** | 29,71 |
| D convrot + eager | 719 ms | 8,98 |

    Marlin   graph/eager = 3,06x
    ConvRot  graph/eager = 3,31x

**Os dois sao capturados.** O ConvRot aproveita a captura ate um pouco MAIS que o
Marlin. A hipotese esta morta.

Flags conferidas no log das quatro, nao assumidas:
`enable_prefix_caching=False` nos dois modelos, `enforce_eager` batendo com a
celula em todas.

### O que isso implica

Sob CUDA graph o overhead de host some para os dois. E o ConvRot CONTINUA 1,54x
mais lento no decode (29,71 contra 45,89). Logo a diferenca que sobra nao e'
despacho -- e' o kernel em M=1.

Isso tambem corrige a leitura de fora que dizia "o CUDA do W4A4 (36,3 us) ganha
do BF16 (42,3 us), logo o kernel e melhor". Contra bf16, ganha. Mas o caminho
INTEIRO do Marlin -- host mais GPU -- cabe em 30,8 us, entao o GPU dele em M=1 e'
menor que os 36,3 do ConvRot. A razao que sobra (~1,4x) bate com o 1,54x
observado no servico.

Conclusao que as duas sessoes agora sustentam: **em M=1 o kernel do Marlin e
genuinamente melhor.** Nao ha vantagem escondida por integracao. O Marlin foi
feito para esse regime e o W4A4 para o outro.

### Tres coisas de brinde

**CUDA graph vale 3x no decode**, nos dois caminhos. Nao era o que se procurava, e
e' o maior fator isolado que apareceu nesta rodada.

**O ganho de prefill do ConvRot sobrevive ao controle**: 502 contra 1590 ms, 3,2x,
com cache desligado e prompt identico nas quatro celulas.

**O TTFT do ConvRot e mais ruidoso.** Com cache OFF, o Marlin da ganho
quente/frio de -0,2% e -0,1% (correto, nao ha cache). O ConvRot da +11,6% e
+17,4% -- 58 a 125 ms de oscilacao sobre ~500. Flags conferidas: nao e' cache. E'
variancia, e qualquer comparacao de TTFT do ConvRot abaixo desse patamar nao e
sinal.

## O que sobra na mesa

Some o atalho de "consertar a integracao e o decode melhora". O que fica de pe e
a ideia de **despacho por M**, que fica MAIS forte, nao menos, porque nao ha
conserto de software esperando:

    M pequeno (decode)      -> Marlin W4A16
    M grande (prefill)      -> ConvRot W4A4
    cruzamento medido        ~100

Custo a calcular antes de propor: manter duas representacoes do mesmo peso na
VRAM. Nao esta calculado.

E o W4A8 real continua interessante por outro motivo -- o menor custo marginal da
tabela, com pedagio alto. Se o pedagio cair, muda de figura. Numero de qualidade
de outra sessao, em difusao: erro 2x menor que o `ACT_DTYPE=int8` (0,074 contra
0,157). Nao verificado aqui.
