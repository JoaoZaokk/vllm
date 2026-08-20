<#
Runner de matriz: sobe o servidor uma vez por grupo de configuracao e varre
todo o resto por dentro.

    pwsh -File bench\run_matrix.ps1 -Plan      # so imprime o plano, nao sobe nada
    pwsh -File bench\run_matrix.ps1            # executa

--------------------------------------------------------------------------------
VALIDACAO (por que isso esta correto sem precisar rodar)
--------------------------------------------------------------------------------

(1) A CHAVE DE BOOT esta correta porque cada campo dela e comprovadamente
    congelado na subida:

      num_speculative_tokens  campo de SpeculativeConfig, lido na construcao do
                              engine (vllm/config/speculative.py:81)
      max_model_len,
      kv_cache_memory_bytes,
      gpu_memory_utilization,
      max_num_seqs           argumentos de CLI do vllm serve; nao ha endpoint
                              que os altere em runtime
      kv_cache_dtype          decide o formato dos blocos alocados em
                              vllm/v1/worker/gpu/attn_utils.py:204-224
      VLLM_SM86_PN286         decide o SHAPE do KV. E functools.cache em
                              vllm/sm86_patches.py de proposito: shape e unbind
                              precisam concordar, entao a decisao e congelada na
                              primeira chamada, que acontece na alocacao. Logo
                              nao pode mudar dentro do processo — o cache que
                              garante a correcao tambem prova que isso pertence
                              a chave de boot.

    E cada campo de run varia por requisicao: prompt_tokens, max_tokens,
    temperature e seed sao campos do corpo de /v1/completions.

(2) CONTAGEM DE BOOTS. Sendo B os eixos de boot e R os de requisicao,
    ingenuo = (prod |Bi|) x (prod |Rj|) boots; agrupado = prod |Bi|.
    Fator de economia = prod |Rj|, exato, nao heuristico.
    Matriz atual: |B| = 1x2x4x1x1x1x1x1 = 8 ; |R| = 4x1x2x1 = 8.
      ingenuo 64 boots  ->  agrupado 8 boots. 8x menos.
    A ~3 min por boot: 192 min de subida viram 24 min.
    Repare que acrescentar temperatura DOBROU o fator de economia em vez de
    custar boot: e exatamente o que a separacao compra. Todo eixo que couber no
    corpo da requisicao entra de graca; todo eixo que o servidor congela custa
    o produto inteiro.

(3) ARITMETICA DE KV, para recusar config impossivel ANTES de gastar um boot.
    Bytes de KV por token = camadas_full_attention x kv_heads x head_dim x 2(K,V) x 2(bf16)
                          = 16 x 4 x 256 x 2 x 2 = 65536 = 64 KiB por token, exato.
    Logo, sem compressao: 8k ctx = 512 MiB, 32k = 2 GiB, 64k = 4 GiB, 256k = 16 GiB.
    Confere com o teto de 58.585 tokens que medimos: x64 KiB = 3,58 GiB.

    As 48 camadas GDN NAO entram nessa conta: o estado delas e por sequencia e
    de tamanho fixo. Por isso contexto longo e barato neste modelo e max_num_seqs
    e que e caro:
      estado ssm  = 48 camadas x 48 heads x 128 x 128 x 4 B = 144 MiB por sequencia
      estado conv = 48 x (16x128x2 + 48x128) x 4 x 2 B      = 3,75 MiB por sequencia

    Predicado de viabilidade, puramente aritmetico:
      orcamento = vram x gpu_mem_util - pesos - (estado x max_num_seqs) - workspace
      exigido   = max_model_len x 64 KiB / compressao      (uma sequencia no maximo)
      viavel    <=> kv_cache_memory_bytes esta em [exigido, orcamento]

(4) INVARIANTE DE ANCORA. Celulas do mesmo boot sao comparaveis entre si por
    construcao. Celulas de boots diferentes NAO sao: captura de cudagraph e
    fragmentacao diferem. Por isso toda subida roda uma ancora identica, e toda
    comparacao entre boots e feita sobre valores normalizados pela ancora.
    Foi exatamente isso que invalidou em silencio a escada de contexto anterior,
    que comparava boot de 65536 com numeros historicos de boot 8192.

(5) DERIVA. A ancora roda no inicio e no fim do mesmo boot. Se as duas diferem
    mais que deriva_max_pct, alguma coisa mudou embaixo (termico, outro processo
    na GPU) e o boot inteiro e marcado suspeito. Detecta sem oraculo externo.

(6) ACEITACAO vem de CONTADOR, nao de tok/s. Le-se /metrics antes e depois de
    cada celula e usa-se o delta de vllm:spec_decode_num_accepted_tokens_total
    sobre num_draft_tokens_total. Contador e imune a disputa de GPU; tok/s nao —
    ja apanhamos disso quando o ComfyUI rodava junto e derrubou 80 tok/s para 4.

O QUE ESTA MATEMATICA NAO GARANTE, e por isso o runner instrumenta em vez de
assumir: que o vLLM aceite o kv_cache_memory_bytes pedido (ele pode arredondar
por tamanho de bloco); que a captura de cudagraph caiba no que sobrou; e que
prompt_tokens pedido vire exatamente esse numero de tokens depois do tokenizer.
As tres sao medidas e registradas, nao estimadas.

(7) TTFT, E POR QUE ELE NAO APARECIA. tok_s mede decode. Prefill entra nele
    diluido em 512 tokens de saida, entao qualquer coisa que conserte prefill
    — vllm#44986 e o prefix cache inteiro — era invisivel nesta matriz por
    construcao. Pior: as celulas repetem prompt, entao a segunda do mesmo
    comprimento ja pegava cache quente sem que nada registrasse isso.
    Agora cada celula manda duas sondas de max_tokens=1 no mesmo prompt,
    prefixado por um nonce proprio para a primeira ser mesmo fria. A diferenca
    entre as duas e o que o cache comprou, em ms.
    Custo: 2 requisicoes minusculas por celula, 0 boots.

#>

[CmdletBinding()]
param(
    [switch]$Plan,
    [string]$MatrixPath = "$PSScriptRoot\matrix.json",
    [string]$OutPath    = "$PSScriptRoot\results.jsonl",
    [int]$Port          = 8000,
    [int]$BootTimeoutSec = 900,
    # Raiz da stack (a que tem models/ e docker/): pai da raiz deste repo.
    # $PSScriptRoot e benchmarks/sm86, entao dois niveis acima e a raiz do
    # fork e tres e a stack. Sobrescrevivel na chamada.
    [string]$StackRoot  = (Resolve-Path "$PSScriptRoot\..\..\..").Path,
    [string]$ModelPath  = $null
)

$ErrorActionPreference = 'Stop'
if (-not $ModelPath) { $ModelPath = Join-Path $StackRoot "modelswq-w4a16" }
$KIB = 1024
$GIB = 1073741824

$M = Get-Content $MatrixPath -Raw | ConvertFrom-Json

# ---------------------------------------------------------------- aritmetica
function Get-KvBytesPerToken($mo) {
    # K e V, dtype do modelo (bf16 = 2 bytes). So as camadas de full attention.
    $mo.camadas_full_attention * $mo.kv_heads * $mo.head_dim * 2 * 2
}

function Get-GdnStateBytesPerSeq($mo) {
    $ssm  = $mo.camadas_gdn * $mo.gdn_value_heads * $mo.gdn_key_head_dim *
            $mo.gdn_value_head_dim * $mo.gdn_ssm_dtype_bytes
    $canais = ($mo.kv_heads * 0) + (16 * $mo.gdn_key_head_dim * 2) + ($mo.gdn_value_heads * $mo.gdn_value_head_dim)
    $conv = $mo.camadas_gdn * $canais * $mo.gdn_conv_kernel * 2
    return $ssm + $conv
}

function Test-Viavel($boot, $M) {
    $mo   = $M.modelo
    $bpt  = Get-KvBytesPerToken $mo
    $comp = $M.compressao_kv.($boot.kv_cache_dtype)
    if (-not $comp) { $comp = 1.0 }

    $estado = Get-GdnStateBytesPerSeq $mo
    # O modulo MTP e PESO, nao cache: entra no orcamento assim que o spec liga.
    # Esquecer este termo foi o que fez a primeira matriz pedir contexto demais.
    $pesos = $mo.pesos_bytes
    if ([int]$boot.num_speculative_tokens -gt 0) { $pesos += $mo.pesos_mtp_bytes }

    $orcamento = [long]($M.hardware.vram_bytes * $boot.gpu_memory_utilization) -
                 $pesos - ($estado * $boot.max_num_seqs) - $mo.workspace_bytes
    $exigido   = [long](($boot.max_model_len * $bpt) / $comp)
    $pedido    = [long]$boot.kv_cache_memory_bytes

    $motivos = @()
    if ($pedido -lt $exigido)   { $motivos += "kv pedido $([math]::Round($pedido/$GIB,2)) GiB < exigido $([math]::Round($exigido/$GIB,2)) GiB para uma sequencia de $($boot.max_model_len)" }
    if ($pedido -gt $orcamento) { $motivos += "kv pedido $([math]::Round($pedido/$GIB,2)) GiB > orcamento $([math]::Round($orcamento/$GIB,2)) GiB" }

    [pscustomobject]@{
        viavel        = ($motivos.Count -eq 0)
        motivos       = $motivos
        kv_por_token  = $bpt
        estado_por_seq= $estado
        orcamento     = $orcamento
        exigido       = $exigido
        tokens_kv_max = [long](($pedido * $comp) / $bpt)
    }
}

# ------------------------------------------------------------- produto dos eixos
function Expand-Axes($obj) {
    $nomes = $obj.PSObject.Properties.Name | Where-Object { -not $_.StartsWith('_') }
    $acc = @([ordered]@{})
    foreach ($n in $nomes) {
        $novo = foreach ($base in $acc) {
            foreach ($v in @($obj.$n)) {
                $c = [ordered]@{}; foreach ($k in $base.Keys) { $c[$k] = $base[$k] }
                $c[$n] = $v; $c
            }
        }
        $acc = @($novo)
    }
    $acc | ForEach-Object { [pscustomobject]$_ }
}

$boots = Expand-Axes $M.eixos_boot
$runs  = Expand-Axes $M.eixos_run

function Get-BootKey($b) {
    ($b.PSObject.Properties | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join '|'
}

# ------------------------------------------------------------------- o plano
$plano = foreach ($b in $boots) {
    $v = Test-Viavel $b $M
    [pscustomobject]@{ boot = $b; chave = Get-BootKey $b; viab = $v; celulas = $runs.Count }
}

$ok    = @($plano | Where-Object { $_.viab.viavel })
$ruins = @($plano | Where-Object { -not $_.viab.viavel })

# Celulas cuja requisicao nao cabe no contexto do proprio boot. Aritmetica de
# novo: prompt + saida <= max_model_len. Sem isso o vLLM recusa a requisicao no
# meio do grupo e a celula vira buraco no CSV depois de o boot ja ter custado.
$celulasRuins = foreach ($p in $ok) {
    foreach ($r in $runs) {
        $precisa = [int]$r.prompt_tokens + [int]$r.max_tokens
        if ($precisa -gt [int]$p.boot.max_model_len) {
            [pscustomobject]@{ chave = $p.chave; prompt = $r.prompt_tokens
                               saida = $r.max_tokens; precisa = $precisa
                               limite = $p.boot.max_model_len }
        }
    }
}

Write-Host ""
Write-Host "=========== PLANO ===========" -ForegroundColor Cyan
Write-Host "eixos de boot     : $($boots.Count) combinacoes"
Write-Host "eixos de execucao : $($runs.Count) por boot"
Write-Host "celulas totais    : $($boots.Count * $runs.Count)"
Write-Host "boots ingenuo     : $($boots.Count * $runs.Count)"
Write-Host "boots agrupado    : $($boots.Count)    (economia $($runs.Count)x)"
Write-Host "viaveis           : $($ok.Count)   inviaveis: $($ruins.Count)"
$bpt = Get-KvBytesPerToken $M.modelo
Write-Host ""
Write-Host "KV por token      : $bpt bytes ($([math]::Round($bpt/$KIB,1)) KiB)"
Write-Host "estado GDN por seq: $([math]::Round((Get-GdnStateBytesPerSeq $M.modelo)/1MB,1)) MiB (fixo, nao cresce com contexto)"

if ($ruins) {
    Write-Host ""
    Write-Host "--- RECUSADAS pela aritmetica (nenhum boot gasto) ---" -ForegroundColor Yellow
    foreach ($p in $ruins) { Write-Host "  $($p.chave)"; $p.viab.motivos | ForEach-Object { Write-Host "      $_" } }
}

if ($celulasRuins) {
    Write-Host ""
    Write-Host "--- CELULAS que nao cabem no contexto do boot ---" -ForegroundColor Yellow
    $celulasRuins | Group-Object { "$($_.prompt)+$($_.saida)=$($_.precisa) > $($_.limite)" } |
        ForEach-Object { Write-Host "  $($_.Name)   ($($_.Count) celulas)" }
    throw "matriz invalida: corrija prompt_tokens/max_tokens ou max_model_len antes de gastar boot"
}

Write-Host ""
Write-Host "--- boots a executar ---"
foreach ($p in $ok) {
    Write-Host ("  PN286={0} K={1} ctx_max={2} kv={3} -> cabe {4} tokens de KV" -f `
        $p.boot.VLLM_SM86_PN286, $p.boot.num_speculative_tokens, $p.boot.max_model_len,
        $p.boot.kv_cache_dtype, $p.viab.tokens_kv_max)
}
Write-Host ""

if ($Plan) { Write-Host "modo -Plan: nada foi executado."; return }

# ---------------------------------------------------------------------------
# ARNES LEGADO -- recusa executar por padrao.
#
# Daqui para baixo este script sobe container com `docker run -d --gpus all` e
# NAO passa pelo lock da GPU. Duas sessoes dividem a mesma placa; uma medicao
# concorrente nao da erro, da um numero errado com cara de certo.
#
# Nao ganhou um lock proprio de proposito: uma terceira implementacao de
# exclusao mutua nao e exclusao mutua. Quem coordena a placa e o controlador.
#
# Alem do lock, herda os defeitos ja mapeados do arnes antigo: porta fixa em vez
# de sorteada, lista de -e escrita a mao, e identidade do servidor por porta.
#
# O caminho conferido e:
#
#     ./lab run <perfil> <tarefa>
#
# `-Plan` continua livre: ele so imprime o plano e nao sobe nada.
#
# Para executar assim mesmo:
#
#     $env:LAB_ARNES_LEGADO = "1"; pwsh -File run_matrix.ps1
#
if ($env:LAB_ARNES_LEGADO -ne "1") {
    Write-Error @"
RECUSADO: arnes legado, sem lock da GPU.

Este script sobe container com --gpus all sem passar pelo lock. Com outra
sessao na mesma placa, o resultado e um numero contaminado que nao se anuncia.

Use:  ./lab run <perfil> <tarefa>
Plano sem executar:  pwsh -File run_matrix.ps1 -Plan
Executar assim mesmo:  `$env:LAB_ARNES_LEGADO = "1"
"@
    exit 78
}
# ---------------------------------------------------------------------------

# --------------------------------------------------------------- execucao
function Wait-Ready($timeout) {
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt $timeout) {
        try {
            $r = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 -ErrorAction Stop
            return ((Get-Date) - $t0).TotalSeconds
        } catch { Start-Sleep -Seconds 5 }
    }
    throw "servidor nao ficou pronto em ${timeout}s"
}

function Get-SpecCounters {
    try {
        $m = Invoke-RestMethod "http://127.0.0.1:$Port/metrics" -TimeoutSec 10
        $acc = ([regex]'vllm:spec_decode_num_accepted_tokens_total[^ ]* ([0-9.e+]+)').Match($m)
        $drf = ([regex]'vllm:spec_decode_num_draft_tokens_total[^ ]* ([0-9.e+]+)').Match($m)
        return @{
            aceitos  = if ($acc.Success) { [double]$acc.Groups[1].Value } else { 0 }
            rascunho = if ($drf.Success) { [double]$drf.Groups[1].Value } else { 0 }
        }
    } catch { return @{ aceitos = 0; rascunho = 0 } }
}

function Get-Prompt($promptTokens, $nonce) {
    # O nonce e o que torna a primeira sonda de cada celula genuinamente fria.
    # O prefix cache vive no processo, entao sem ele a segunda celula do mesmo
    # comprimento mediria um acerto e chamaria de miss.
    "n$nonce " + ("hi " * $promptTokens)
}

function Invoke-Probe($prompt) {
    # TTFT por procuracao: uma requisicao de max_tokens=1 e prefill + um passo
    # de decode. Nao e streaming, entao nao ha primeiro chunk para cronometrar,
    # e o passo de decode e ~22 ms contra prefill de centenas — o erro sistematico
    # e conhecido, constante, e nao atrapalha comparar frio contra quente.
    $body = @{
        model = "/workspace/models/awq-w4a16"
        prompt = $prompt; max_tokens = 1; temperature = 0; seed = 1234
    } | ConvertTo-Json -Compress
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Invoke-RestMethod "http://127.0.0.1:$Port/v1/completions" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec 1800 | Out-Null
    $sw.Stop()
    [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
}

function Invoke-Cell($promptTokens, $maxTokens, $temp, $seed, $nonce) {
    # "hi " ~ 1 token. Aproximado de proposito: o numero real vem de
    # usage.prompt_tokens na resposta, que e o que fica registrado.
    $prompt = Get-Prompt $promptTokens $nonce

    # Tres sondas, nao duas. A primeira e descartavel: kernels Triton como
    # _compute_slot_mapping_kernel compilam na primeira requisicao de cada
    # shape novo, e esse JIT chega a segundos — mediria compilador, nao
    # prefill. Ela usa um prompt proprio, entao nao aquece o cache das outras.
    #
    # Depois dela: uma sonda de prefixo FRIO (JIT ja quente) e uma de prefixo
    # QUENTE no mesmo texto. So o par 2/3 responde a pergunta; o 1/2 fica
    # registrado como jit_ms porque um compile escondido dentro de uma medicao
    # e pior que um compile medido.
    $ttftJit    = Invoke-Probe (Get-Prompt $promptTokens "$nonce-jit")
    $ttftFrio   = Invoke-Probe $prompt
    $ttftQuente = Invoke-Probe $prompt

    # Depois das sondas: elas tambem geram draft tokens e contaminariam a
    # aceitacao da celula se entrassem na janela dos contadores.
    $antes = Get-SpecCounters
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $body = @{
        model = "/workspace/models/awq-w4a16"
        prompt = $prompt; max_tokens = $maxTokens
        temperature = $temp; seed = $seed
    } | ConvertTo-Json -Compress
    $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/completions" -Method Post `
            -ContentType 'application/json' -Body $body -TimeoutSec 1800
    $sw.Stop()
    $depois = Get-SpecCounters

    $dRasc = $depois.rascunho - $antes.rascunho
    [pscustomobject]@{
        prompt_tokens_reais = $r.usage.prompt_tokens
        saida_tokens        = $r.usage.completion_tokens
        segundos            = $sw.Elapsed.TotalSeconds
        tok_s               = [math]::Round($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds, 2)
        aceitacao           = if ($dRasc -gt 0) { [math]::Round(($depois.aceitos - $antes.aceitos) / $dRasc, 4) } else { $null }
        draft_tokens        = $dRasc
        ttft_primeira_ms    = $ttftJit
        jit_ms              = [math]::Round($ttftJit - $ttftFrio, 1)
        ttft_frio_ms        = $ttftFrio
        ttft_quente_ms      = $ttftQuente
        ganho_cache_pct     = if ($ttftFrio -gt 0) { [math]::Round(100 * ($ttftFrio - $ttftQuente) / $ttftFrio, 1) } else { $null }
    }
}

$carimbo = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
foreach ($p in $ok) {
    $b = $p.boot
    $nome = "bench_" + ($p.chave -replace '[^a-zA-Z0-9]', '_').Substring(0, [Math]::Min(60, ($p.chave -replace '[^a-zA-Z0-9]','_').Length))
    docker rm -f $nome 2>&1 | Out-Null

    $envs = @(
        "-e","VLLM_SM86_PN286=$($b.VLLM_SM86_PN286)"
        "-e","MAX_MODEL_LEN=$($b.max_model_len)"
        "-e","MAX_NUM_SEQS=$($b.max_num_seqs)"
        "-e","GPU_MEMORY_UTILIZATION=$($b.gpu_memory_utilization)"
        "-e","KV_CACHE_MEMORY_BYTES=$($b.kv_cache_memory_bytes)"
        "-e","KV_CACHE_DTYPE=$($b.kv_cache_dtype)"
        "-e","NUM_SPECULATIVE_TOKENS=$($b.num_speculative_tokens)"
    )
    Write-Host "subindo $nome ..." -ForegroundColor Green
    docker run -d --name $nome --gpus all -p "${Port}:8000" `
        -v "${ModelPath}:/workspace/models/awq-w4a16:ro" `
        -v "$StackRoot\docker:/opt/qwen38/docker:ro" `
        @envs $b.imagem /opt/qwen38/docker/awq_entry.sh | Out-Null

    try {
        $bootSeg = Wait-Ready $BootTimeoutSec
        $a = $M.ancora
        # Pre-aquece a ancora: cobre o JIT e o prefixo de uma vez. Sem isso a
        # ancora inicial mede compilador + prefill frio e a final mede prefill
        # quente, e a "deriva" entre elas seria em boa parte so isso.
        Invoke-Probe (Get-Prompt $a.prompt_tokens "ancora") | Out-Null
        $ancoraIni = Invoke-Cell $a.prompt_tokens $a.max_tokens $a.temperature $a.seed "ancora"

        $iCelula = 0
        foreach ($r in $runs) {
            $iCelula++
            $res = Invoke-Cell $r.prompt_tokens $r.max_tokens $r.temperature $r.seed "c$iCelula"
            $linha = [ordered]@{
                carimbo = $carimbo; chave_boot = $p.chave; boot_segundos = [math]::Round($bootSeg,1)
                ancora_inicial_tok_s = $ancoraIni.tok_s
                ancora_final_tok_s = $null; deriva_pct = $null; suspeito = $null
            }
            foreach ($k in $b.PSObject.Properties.Name) { $linha["boot_$k"] = $b.$k }
            foreach ($k in $r.PSObject.Properties.Name) { $linha["run_$k"]  = $r.$k }
            foreach ($k in $res.PSObject.Properties.Name) { $linha[$k] = $res.$k }
            $linha | ConvertTo-Json -Compress | Add-Content $OutPath
            Write-Host ("  ctx {0,6}  temp {1,-4} -> {2,6} tok/s  aceitacao {3}  ttft {4}->{5} ms ({6}%)" -f $r.prompt_tokens, $r.temperature, $res.tok_s, $res.aceitacao, $res.ttft_frio_ms, $res.ttft_quente_ms, $res.ganho_cache_pct)
        }

        $ancoraFim = Invoke-Cell $a.prompt_tokens $a.max_tokens $a.temperature $a.seed "ancora"
        $deriva = 100 * [math]::Abs($ancoraFim.tok_s - $ancoraIni.tok_s) / $ancoraIni.tok_s
        $susp = $deriva -gt $a.deriva_max_pct
        ([ordered]@{
            carimbo = $carimbo; chave_boot = $p.chave; tipo = "ancora"
            inicial = $ancoraIni.tok_s; final = $ancoraFim.tok_s
            deriva_pct = [math]::Round($deriva,2); suspeito = $susp
        } | ConvertTo-Json -Compress) | Add-Content $OutPath
        if ($susp) { Write-Host "  DERIVA $([math]::Round($deriva,2))% — boot suspeito" -ForegroundColor Yellow }
    }
    finally {
        docker logs $nome --tail 40 2>&1 | Out-File "$PSScriptRoot\log_$nome.txt"
        docker rm -f $nome 2>&1 | Out-Null
    }
}

Write-Host ""
Write-Host "resultados em $OutPath" -ForegroundColor Cyan
