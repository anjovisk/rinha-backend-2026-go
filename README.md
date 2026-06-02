# Fraud Detection API

API de detecção de fraudes em transações de cartão via busca vetorial. Para cada transação recebida, a API normaliza o payload em um vetor de 14 dimensões, busca os 5 vizinhos mais próximos em uma base de referência pré-indexada e retorna uma decisão de aprovação com score de fraude.

## Arquitetura

### Visão geral

```
cliente → load balancer (:9999) → api-1 (:8080)
                                 → api-2 (:8081)
```

O Nginx atua como load balancer em round-robin. Nenhuma lógica de negócio é executada no Nginx.

### Arquitetura hexagonal (Ports & Adapters)

```
┌──────────────────────────────────────────────────────────────┐
│             Adaptador primário  (adapter/http)               │
│                                                              │
│   POST /fraud-score ───► handleFraudScore                   │
│   GET  /ready       ───► handleReady                        │
└─────────────────────────┬────────────────────────────────────┘
                          │  port.FraudScoreUseCase
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              Caso de uso  (usecase.FraudScore)               │
│        Evaluate: vectorize → find neighbors → score          │
└─────────────────┬──────────────────────────┬─────────────────┘
                  │ port.Vectorizer           │ port.NeighborFinder
                  ▼                           ▼
┌──────────────────────┐       ┌──────────────────────────────────────────────┐
│   adapter/vector     │       │  adapter/knn       (VECTOR_SEARCHER=brute)   │
│   Vectorize()        │       │  adapter/hnsw      (VECTOR_SEARCHER=hnsw)    │
│   (14 dimensões)     │       │  adapter/ivf       (VECTOR_SEARCHER=ivf)     │
│   normalization.json │       │  adapter/vamana    (VECTOR_SEARCHER=vamana)  │
│   mcc_risk.json      │       │  adapter/partition (VECTOR_SEARCHER=partition│
│                      │       │  adapter/vptree    (VECTOR_SEARCHER=vptree)  │
│                      │       │                                              │
│                      │       │  FindNearest() via brute O(N),               │
│                      │       │  HNSW aprox. O(log N),                       │
│                      │       │  IVF-SQ8 aprox. O(nprobe·N/nlist),           │
│                      │       │  Vamana aprox. O(L·R),                       │
│                      │       │  partition-brute O(N/bins) ou                │
│                      │       │  VP-Tree exato O(N^α) α < 1 (c/ poda)        │
└──────────────────────┘       └──────────────────────────────────────────────┘
```

As dependências fluem sempre para dentro: adapters conhecem ports, ports conhecem o domínio. O domínio não importa nenhum pacote interno.

### Pacotes

| Pacote | Tipo | Responsabilidade |
|--------|------|-----------------|
| `internal/domain` | Domínio | Entidades: `FraudScoreRequest`, `Vector [14]float64`, `FraudResult`, `FraudThreshold` |
| `internal/port` | Port | Interfaces: `FraudScoreUseCase` (inbound), `Vectorizer` e `NeighborFinder` (outbound) |
| `internal/usecase` | Aplicação | `FraudScore.Evaluate`: orquestra vetorização + KNN + cálculo do score |
| `internal/adapter/http` | Adapter primário | HTTP handlers, parsing de JSON, mapeamento para domínio |
| `internal/adapter/vector` | Adapter secundário | Normaliza o payload em vetor de 14 dimensões |
| `internal/adapter/knn` | Adapter secundário | Busca exata brute-force O(N) via mmap; selecionado por `VECTOR_SEARCHER=brute` (padrão) |
| `internal/adapter/hnsw` | Adapter secundário | Busca aproximada HNSW O(log N) via `github.com/coder/hnsw`; selecionado por `VECTOR_SEARCHER=hnsw` |
| `internal/adapter/ivf` | Adapter secundário | Busca aproximada IVF-SQ8; K-means + quantização uint8; selecionado por `VECTOR_SEARCHER=ivf` |
| `internal/adapter/vamana` | Adapter secundário | Busca aproximada Vamana/DiskANN O(L·R); grafo flat com RobustPrune; mmap compartilhado; selecionado por `VECTOR_SEARCHER=vamana` |
| `internal/adapter/partition` | Adapter secundário | Busca exata particionada; roteia cada query para um dos 6 sub-índices por `(last_tx_null × is_online × card_present)`; ~3,2× mais rápido que brute sem arquivo de índice extra; selecionado por `VECTOR_SEARCHER=partition` |
| `internal/adapter/vptree` | Adapter secundário | VP-Tree exato por bin; mesma partição do `partition` + poda pela desigualdade triangular; construído no startup em O(N log N); selecionado por `VECTOR_SEARCHER=vptree` |

## Stack

- **Go 1.22** — aplicação principal (roteamento nativo via `net/http`)
- **uber-go/zap** — logging estruturado em JSON
- **Docker / Docker Compose** — containerização e orquestração
- **Nginx** — load balancer, configurado via `nginx.conf` montado em volume

## Estrutura do repositório

```
.
├── cmd/
│   ├── preprocess/
│   │   └── main.go                    # converte references.json.gz → .bin + .hnsw + .ivf + .vamana
│   └── server/
│       └── main.go                    # bootstrap: carrega recursos, conecta adapters → usecase → HTTP
├── internal/
│   ├── domain/
│   │   ├── transaction.go             # FraudScoreRequest e entidades de entrada
│   │   ├── vector.go                  # Vector [14]float64
│   │   └── fraud.go                   # FraudResult, FraudThreshold
│   ├── port/
│   │   ├── inbound.go                 # FraudScoreUseCase (driving port)
│   │   └── outbound.go                # Vectorizer, NeighborFinder (driven ports)
│   ├── usecase/
│   │   └── fraud_score.go             # EvaluateFraudScore
│   └── adapter/
│       ├── http/
│       │   ├── server.go              # Server, Routes()
│       │   ├── ready.go               # GET /ready
│       │   ├── ready_test.go
│       │   └── fraud_score.go         # POST /fraud-score
│       ├── vector/
│       │   └── vectorizer.go          # implementa Vectorizer
│       ├── knn/
│       │   └── searcher.go            # implementa NeighborFinder (brute-force exato, mmap)
│       ├── hnsw/
│       │   ├── hnsw.go                # implementa NeighborFinder (HNSW aproximado); Open, Load, Export
│       │   └── hnsw_test.go
│       ├── ivf/
│       │   ├── ivf.go                 # implementa NeighborFinder (IVF-SQ8 aproximado); Open e FindNearest
│       │   ├── build.go               # K-means paralelo, quantização SQ8, Build e New
│       │   └── ivf_test.go
│       ├── vamana/
│       │   ├── vamana.go              # implementa NeighborFinder (Vamana aproximado); Open, New, FindNearest
│       │   ├── build.go               # construção Vamana (RobustPrune, paralelo), Build e helpers SQ8
│       │   └── vamana_test.go
│       ├── partition/
│       │   ├── partition.go           # implementa NeighborFinder (brute-force particionado); Open, New, FindNearest
│       │   └── partition_test.go
│       └── vptree/
│           ├── vptree.go              # implementa NeighborFinder (VP-Tree exato por bin); Open, New, FindNearest
│           └── vptree_test.go
├── resources/
│   ├── references.json.gz             # fonte original: 3M vetores rotulados (não lida em runtime)
│   ├── references.bin                 # gerado por cmd/preprocess; lido via mmap em runtime
│   ├── references.hnsw                # gerado quando BUILD_HNSW=true; grafo HNSW serializado
│   ├── references.ivf                 # gerado quando BUILD_IVF=true; índice IVF-SQ8 (~45 MB)
│   ├── references.vamana              # gerado quando BUILD_VAMANA=true; grafo Vamana (~237 MB R=16 SQ8)
│   ├── mcc_risk.json                  # risco por categoria de merchant (MCC)
│   └── normalization.json             # constantes de normalização
├── docs/                              # documentação em português
├── go.mod
├── .env                               # padrão local: VECTOR_SEARCHER, BUILD_*, parâmetros de índice
├── .env.example                       # template comentado de todas as variáveis
├── Dockerfile                         # build multi-stage; ARGs BUILD_* controlam geração dos índices
├── nginx.conf                         # configuração do load balancer
└── docker-compose.yml                 # nginx (:9999) + api-1 (:8080) + api-2 (:8081); lê .env
```

## Pré-requisitos

### Go (Ubuntu)

```bash
sudo snap install go --classic
go version  # go1.22+
```

### Docker e Docker Compose (Ubuntu)

```bash
# Remova versões antigas, se houver
sudo apt remove -y docker docker-engine docker.io containerd runc

# Instale dependências
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Adicione a chave GPG oficial do Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Adicione o repositório do Docker
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instale o Docker Engine e o plugin Compose
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

Para usar o Docker sem `sudo`, adicione seu usuário ao grupo `docker` e reabra a sessão:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verifique a instalação:

```bash
docker --version
docker compose version
```

## Desenvolvimento

### Pré-processamento da base de referência

O servidor não lê `references.json.gz` diretamente. Antes de rodar a aplicação pela primeira vez, execute o preprocess:

```bash
# Apenas references.bin (padrão — suficiente para VECTOR_SEARCHER=brute)
go run ./cmd/preprocess

# references.bin + references.hnsw (necessário para VECTOR_SEARCHER=hnsw com startup rápido)
BUILD_HNSW=true go run ./cmd/preprocess

# IVF com SQ8 (padrão, ~45 MB) — necessário para VECTOR_SEARCHER=ivf
BUILD_IVF=true go run ./cmd/preprocess

# IVF sem SQ8 (~168 MB, distâncias exatas dentro de cada cluster)
BUILD_IVF=true IVF_SQ8=false go run ./cmd/preprocess

# IVF com mais clusters (melhor recall, build mais lento)
BUILD_IVF=true IVF_NLIST=2048 go run ./cmd/preprocess

# Vamana com SQ8, R=16 (padrão, ~237 MB mmap) — necessário para VECTOR_SEARCHER=vamana
BUILD_VAMANA=true go run ./cmd/preprocess

# Vamana com R=8 (~141 MB mmap, menos recall; melhor para o budget de 350 MB)
BUILD_VAMANA=true VAMANA_R=8 go run ./cmd/preprocess

# Vamana com alpha maior (mais arestas de longo alcance, melhor recall em datasets irregulares)
BUILD_VAMANA=true VAMANA_ALPHA=1.4 go run ./cmd/preprocess
```

O preprocess produz até quatro arquivos:

| Arquivo | Quando gerado | O que é | Usado por |
|---------|--------------|---------|-----------|
| `references.bin` | Sempre | Binário flat SoA: `[uint32 N][N×56 B float32][N×1 B uint8]` | Todos os modos — lido via mmap em runtime |
| `references.hnsw` | `BUILD_HNSW=true` | Grafo HNSW serializado pelo `coder/hnsw` | `VECTOR_SEARCHER=hnsw` com `hnsw.Load` — startup rápido |
| `references.ivf` | `BUILD_IVF=true` | Índice IVF: byte de flags + centroides + vetores (uint8 ou float32) + labels | `VECTOR_SEARCHER=ivf` — obrigatório, Fatal se ausente |
| `references.vamana` | `BUILD_VAMANA=true` | Grafo Vamana: header + adjacência N×R uint32 + vetores (uint8 SQ8 ou float32) + labels | `VECTOR_SEARCHER=vamana` — obrigatório, Fatal se ausente |

**Por que o binário?**

| | float64 + string (JSON) | float32 + uint8 (binário) |
|---|---|---|
| Tamanho por instância | ~408 MB | ~171 MB |
| Tempo de carga | ~30 s (parsing JSON) | < 1 s (mmap) |

Com `mmap`, as duas instâncias da API compartilham as mesmas páginas físicas no page cache do kernel — o arquivo ocupa ~171 MB na RAM, não ~342 MB. Isso mantém o uso total dentro do budget de 350 MB.

Nenhum desses arquivos é versionado no repositório. Em Docker, o preprocess é executado automaticamente durante o build da imagem (ver `Dockerfile`); os `BUILD_*` são passados como build args pelo docker-compose.

### Backend de busca vetorial

O algoritmo usado para encontrar os 5 vizinhos mais próximos é controlado pela variável de ambiente `VECTOR_SEARCHER`:

| `VECTOR_SEARCHER` | Algoritmo | Complexidade de query | Startup | RSS extra por instância | Recall |
|---|---|---|---|---|---|
| `brute` (padrão) | Brute-force exato | O(N) | < 1 s (mmap) | 0 — vetores compartilhados via page cache | 100% |
| `partition` | Brute-force particionado exato por bin | O(N/bins) ≈ O(N×0,31) | < 1 s (mmap + scan único) | ~12 MB heap (índices de ponteiros) | ~100% dentro do bin |
| `vptree` | VP-Tree exato por bin (poda triangular) | O(N^α) α<1 c/ poda efetiva; O(N/bins) pior caso | alguns segundos (build O(N log N)) | ~16–20 MB heap (6 árvores) | ~100% dentro do bin |
| `hnsw` + `references.hnsw` presente | HNSW aprox. (`coder/hnsw`, M=4, EfSearch=50) — **Load** | O(log N) | < 1 s (import) | ~170 MB heap (vetores alocados pelo Import) | ~93–97% |
| `hnsw` + `references.hnsw` ausente | HNSW aprox. — **Open** (build from scratch) | O(log N) | dezenas de segundos (O(N log N)) | ~50–100 MB heap (conexões apenas; vetores no mmap) | ~93–97% |
| `ivf` + `IVF_SQ8=true` (padrão) | IVF aprox. com SQ8 (K-means + uint8) | O(nprobe·N/nlist) | < 1 s (leitura heap) | ~45 MB heap por instância | ~95–99% (nprobe=32) |
| `ivf` + `IVF_SQ8=false` | IVF aprox. sem SQ8 (K-means + float32) | O(nprobe·N/nlist) | < 1 s (leitura heap) | ~168 MB heap por instância | ~96–99% (nprobe=32) |
| `vamana` + R=16 SQ8 (padrão) | Vamana/DiskANN aprox. (grafo flat + RobustPrune) | O(L·R) ≈ O(1024) | < 1 s (mmap) | ~0 — grafo+vetores compartilhados via mmap/page cache | ~95–98% (L=64) |
| `vamana` + R=8 SQ8 | Vamana aprox. — grafo menor | O(L·R) ≈ O(512) | < 1 s (mmap) | ~0 — mmap compartilhado | ~92–96% (L=64) |

Quando `VECTOR_SEARCHER=hnsw`, o servidor detecta automaticamente se `resources/references.hnsw` existe:
- **Arquivo presente** (`cmd/preprocess` foi executado com `BUILD_HNSW=true`): usa `hnsw.Load` — carrega o grafo serializado em O(N), sem custo de build. Vetores ficam em heap (~170 MB por instância).
- **Arquivo ausente**: usa `hnsw.Open` — constrói o grafo em O(N log N), mais lento para iniciar, mas vetores ficam no mmap (~0 MB heap extra). Um `Warn` é emitido no log.

Quando `VECTOR_SEARCHER=ivf`, o arquivo `resources/references.ivf` é obrigatório (Fatal se ausente). A busca:
1. Encontra os `nprobe` clusters mais próximos por L2 entre os `nlist` centroides.
2. Percorre os vetores dentro desses clusters: distância SQ8-aproximada (uint8) ou exata (float32), conforme o flag gravado no arquivo.
3. Retorna os k vizinhos mais próximos encontrados.

`IVF_NPROBE` (padrão 32) e `IVF_NLIST` (padrão 1024, build-time) controlam o tradeoff recall/latência:
- `nprobe=32` / `nlist=1024` → busca ~3% do dataset, ~30× mais rápido que brute-force.
- `nprobe=64` → ~6%, melhor recall com latência 2× maior.

`IVF_SQ8` (padrão `true`, build-time) controla a precisão e o uso de memória:
- `IVF_SQ8=true` → float32 comprimido para uint8 por dimensão: **~45 MB** por instância, pequena perda de precisão nas distâncias.
- `IVF_SQ8=false` → vetores mantidos como float32: **~168 MB** por instância, distâncias exatas dentro de cada cluster (o erro residual vem apenas da partição K-means).

O flag SQ8 é gravado no cabeçalho de `references.ivf` (uint8 flags, bit 0); o servidor o lê automaticamente ao carregar o arquivo — **nenhuma variável de ambiente é necessária em runtime**.

> **Budget de memória (IVF):** com `IVF_SQ8=true`, cada instância usa ~45 MB de heap (vs ~170 MB do `hnsw` + Load). Com `IVF_SQ8=false`, o consumo sobe para ~168 MB por instância — comparável ao HNSW com Load, mas sem o custo de build no startup.

Quando `VECTOR_SEARCHER=vamana`, o arquivo `resources/references.vamana` é obrigatório (Fatal se ausente). A busca executa um **beam search greedy** partindo do medoid do dataset:
1. Mantém uma lista de candidatos ordenada por distância L2, limitada a `L` entradas.
2. A cada passo expande o candidato mais próximo não visitado, adicionando seus vizinhos no grafo.
3. Termina quando todos os candidatos na lista já foram expandidos.
4. Retorna os k mais próximos da lista final.

`VAMANA_L` (padrão 64, runtime) controla o tamanho do beam: maior L = mais recall, latência proporcional a L. `VAMANA_R` e `VAMANA_ALPHA` são parâmetros de build-time e não afetam o runtime.

O flag SQ8 é gravado no cabeçalho de `references.vamana` (uint32 flags, bit 0); o servidor o lê automaticamente — **nenhuma variável de ambiente é necessária em runtime além de `VAMANA_L`**.

> **Budget de memória (Vamana):** o arquivo é mmapeado e as páginas são **compartilhadas entre as duas instâncias** pelo page cache do kernel. Com R=16 SQ8, o arquivo ocupa ~237 MB na RAM (não ~474 MB). Com R=8 SQ8, ~141 MB. Ambas as instâncias mais o Nginx cabem dentro do budget de 350 MB.

#### ⚠️ Problema conhecido: build do índice Vamana levando horas

> **A construção do índice Vamana (`BUILD_VAMANA=true`) está levando horas na prática e precisa ser revisada antes de ser usada em produção ou como opção padrão no contest.**
>
> Causa raiz identificada: o algoritmo faz acessos completamente aleatórios em dois arrays que somam ~360 MB (grafo + vetores), tornando cada acesso um cache miss de ~100 ns. Com 3M nós e `buildL=125`, isso resulta em bilhões de cache misses por passada. O código não usa SIMD nem prefetch, agravando o problema.
>
> **Use `VECTOR_SEARCHER=ivf` como backend ANN enquanto o build do Vamana não é otimizado.**

#### Tempo de build do índice Vamana

O build é dominado por **cache misses aleatórios** em dois arrays grandes que não cabem no cache da CPU:

| Array | Tamanho (R=16) |
|---|---|
| Grafo de adjacência | ~192 MB |
| Vetores float32 | ~168 MB |

A cada nó, o algoritmo faz até `buildL × R` acessos aleatórios a esses arrays (~100 ns por miss). Com `buildL=125`, 2 passadas e N=3 M entradas:

| CPUs disponíveis no build | Tempo estimado |
|:---:|---|
| 1 CPU | 1–2 horas |
| 4 CPUs | 20–40 minutos |
| 8+ CPUs | 10–20 minutos |

> **Se o build ultrapassar 2–3 horas com 1 CPU ou 1 hora com múltiplos CPUs, verifique swap.** O processo de build aloca ~192 MB de grafo em heap + faz mmap de ~168 MB de vetores; se a máquina de build não tiver RAM disponível suficiente, page faults viram seeks em disco e o tempo escala para horas.

Para reduzir o tempo sem rebuild da imagem final:

```bash
# buildL=64 corta o tempo de build ~50%; o arquivo gerado e o recall em runtime são idênticos
BUILD_VAMANA=true VAMANA_BUILD_L=64 go run ./cmd/preprocess

# R=8 reduz o grafo de 192 MB para 96 MB: menos cache miss no build e arquivo final menor
BUILD_VAMANA=true VAMANA_R=8 go run ./cmd/preprocess

# Combinação mais rápida, suficiente para avaliar viabilidade:
BUILD_VAMANA=true VAMANA_R=8 VAMANA_BUILD_L=64 go run ./cmd/preprocess
```

No Docker, passe os ARGs correspondentes:

```bash
BUILD_VAMANA=true VAMANA_BUILD_L=64 docker compose up --build
BUILD_VAMANA=true VAMANA_R=8 VAMANA_BUILD_L=64 docker compose up --build
```

#### Particionamento por features discretas (`VECTOR_SEARCHER=partition`)

O adapter `partition` foi desenvolvido após um estudo da estrutura do dataset. A ideia central é simples: se uma feature discreta do vetor já separa naturalmente os dados em grupos com pouca sobreposição, é possível dividir o dataset em sub-índices e rotear cada query apenas para o sub-índice correspondente — reduzindo o trabalho de busca sem nenhuma perda de precisão dentro da partição.

##### Análise teórica

Três features do vetor de 14 dimensões são binárias ou sentinel:

| Dimensão | Feature | Valores |
|:---:|---|---|
| 5 | `minutes_since_last_tx` | `[-1, 0..1]` — `-1` quando `last_transaction: null` |
| 6 | `km_from_last_tx` | `[-1, 0..1]` — `-1` quando `last_transaction: null` |
| 9 | `is_online` | `0` ou `1` |
| 10 | `card_present` | `0` ou `1` |

A questão é: qual o custo em distância L² de cruzar a fronteira de uma partição?

**Sentinel `-1` (dims 5 e 6):** quando uma transação tem `last_transaction: null`, seus dims 5 e 6 valem `-1`. Qualquer referência com `last_transaction` presente terá esses dims em `[0, 1]`. A diferença mínima é `0 − (−1) = 1` por dim, adicionando pelo menos `1² + 1² = 2,0` à distância quadrática. Como a distância L² esperada entre dois pontos aleatórios em `[0, 1]^14` é ~2,33, um penálti de 2,0 é enorme — vizinhos cruzando essa fronteira seriam raridade.

**Features binárias (dims 9 e 10):** um mismatch em `is_online` ou `card_present` adiciona `1² = 1,0` à distância quadrática (~43% da distância típica entre pontos). Cross-partition neighbors existem, mas são improvável de compor o top-5 quando o candidato na partição correta já está próximo em todas as outras dimensões.

##### Distribuição real do dataset (3M referências)

O estudo foi validado com uma análise completa das 3M referências:

```
last_tx_null  is_online  card_present |  entradas    %   fraude%
--------------------------------------------------------------
false         0          0            |   128.017   4,3%   13,5%
false         0          1            | 1.156.177  38,5%   13,4%
false         1          0            | 1.116.849  37,2%   56,2%
true          0          0            |    32.201   1,1%   13,4%
true          0          1            |   288.785   9,6%   13,4%
true          1          0            |   277.971   9,3%   56,3%
```

Duas descobertas relevantes:

1. **`is_online=1` e `card_present=1` são mutuamente exclusivos** — nenhuma das 3M referências tem os dois ativos ao mesmo tempo (faz sentido: transação online = cartão não está fisicamente presente). Das 8 combinações teóricas possíveis (3 bits), apenas 6 existem de fato.

2. **A taxa de fraude difere drasticamente entre partições:** transações online têm ~56% de fraude vs ~13,4% das presenciais. Isso confirma que os vizinhos de uma transação online raramente estarão na partição offline — a separabilidade é real, não apenas teórica.

##### Speedup esperado

O speedup não é uniforme — cada query vai para a partição que corresponde às suas features. O trabalho médio de scan é a soma ponderada dos tamanhos das partições:

| Partição | Tamanho | P(query cair aqui) | Entradas escaneadas |
|---|:---:|:---:|---:|
| offline, sem card, sem null-tx | 4,3% | 4,3% | ~5.500 |
| offline, com card, sem null-tx | 38,5% | 38,5% | ~445.000 |
| online, sem card, sem null-tx | 37,2% | 37,2% | ~415.000 |
| offline, sem card, null-tx | 1,1% | 1,1% | ~350 |
| offline, com card, null-tx | 9,6% | 9,6% | ~27.700 |
| online, sem card, null-tx | 9,3% | 9,3% | ~25.900 |
| **Média ponderada** | | | **~920.000** |

Brute-force completo: 3.000.000 entradas. Partition: ~920.000 → **speedup médio de ~3,2×**.

O `partition` é a melhor escolha quando se quer resultado **exato dentro da partição** (sem o risco de recall degradado dos métodos ANN) e sem precisar construir e armazenar um arquivo de índice separado. Depende apenas do `references.bin` já gerado pelo preprocess padrão.

#### VP-Tree por bin (`VECTOR_SEARCHER=vptree`)

O adapter `vptree` combina a mesma partição de 6 bins do `partition` com uma **vantage-point tree (VP-Tree)** construída para cada bin no startup. A VP-Tree é uma árvore métrica que, ao buscar, usa a desigualdade triangular para podar ramos inteiros sem avaliar cada ponto individualmente:

```
d(q, p) ≥ |d(q, vp) − d(vp, p)|
```

Se a distância mínima possível de qualquer ponto num ramo ao query for maior que o k-ésimo melhor candidato atual (`tau`), o ramo inteiro é descartado. Isso reduz o número de distâncias calculadas de O(N/bins) para sub-linear — quando a poda funciona.

**Construção:** para cada bin, um vantage point é escolhido aleatoriamente, todos os pontos restantes são ordenados por distância euclidiana a ele e divididos ao meio. A partição esquerda (inner ball, dist ≤ radius) e direita (outer shell, dist > radius) são recursadas até os bins atingirem tamanho ≤ 32 entradas (leaf size), onde a busca é bruta. Custo: O(N log N) por bin.

**Busca:** ao visitar cada nó interno, o query é comparado contra o vantage point e tau é atualizado. A condição de poda:
- `d + tau < radius` → pula o ramo externo (outer shell muito longe)
- `d − tau > radius` → pula o ramo interno (inner ball muito longe)

A eficácia da poda depende da **dimensionalidade intrínseca** dos dados em cada bin. Com dims fixas pelo roteamento (2–4 dimensões constantes por bin), a dimensionalidade efetiva cai para 10–12. No pior caso (distribuição uniforme), a VP-Tree degenera para O(N/bins) como o `partition`. No melhor caso, aproxima-se de O(log N).

**Memória e startup:** mmap compartilhado (~171 MB) + ~16–20 MB heap para as 6 árvores. Build de alguns segundos no startup; sem arquivo de índice extra.

---

#### Ranqueamento de configurações IVF

Com o fix de mmap (`ivf/ivf.go`), `vecFlat8` e `labelFlat` são fatiados diretamente da região mmapeada (`MAP_SHARED`) sem cópia para o heap Go. O perfil de memória por instância passa a ser **~43 MB page cache + ~15 MB heap = ~58 MB RSS** — fora do controle do GC, eliminando a degradação de CPU ao longo do teste. Com `IVF_SQ8=false` (~168 MB heap), o limite de 100 MB por instância é excedido; essa opção só é viável na configuração de 2 instâncias com limite de 165 MB.

A tabela cobre as combinações mais relevantes para N = 3.000.000, D = 14, `IVF_SQ8=true`. A **latência estimada** soma scan de centroides + scan de vetores SQ8 + overhead HTTP/JSON (~250 µs base), com ~2 ns/vetor para o scan inner loop.

| # | `IVF_NLIST` | `IVF_NPROBE` | Vetores/query | % dataset | Speedup vs brute | Recall k=5 (est.) | Latência est. | Build K-means |
|---|-------------|--------------|:-------------:|:---------:|:----------------:|:-----------------:|:-------------:|:-------------:|
| 1 | 4096 | 16 | ~11.700 | 0,39% | ~256× | ~58% | ~275 µs | ~120 s |
| 2 | 4096 | 32 | ~23.400 | 0,78% | ~128× | ~68% | ~297 µs | ~120 s |
| 3 | 2048 | 16 | ~23.400 | 0,78% | ~128× | ~74% | ~297 µs | ~60 s |
| 4 | 4096 | 64 | ~46.900 | 1,56% | ~64× | ~80% | ~344 µs | ~120 s |
| 5 | 2048 | 32 | ~46.900 | 1,56% | ~64× | ~86% | ~344 µs | ~60 s |
| 6 | 1024 | 16 | ~46.900 | 1,56% | ~64× | ~80% | ~344 µs | ~30 s |
| 7 | 512 | 8 | ~46.900 | 1,56% | ~64× | ~70% | ~344 µs | ~15 s |
| 8 | 4096 | 128 | ~93.700 | 3,13% | ~32× | ~91% | ~437 µs | ~120 s |
| **9 ★** | **2048** | **64** | **~93.700** | **3,13%** | **~32×** | **~94%** | **~437 µs** | **~60 s** |
| 10 | 1024 | 32 | ~93.700 | 3,13% | ~32× | ~90% | ~437 µs | ~30 s |
| 11 | 512 | 16 | ~93.700 | 3,13% | ~32× | ~80% | ~437 µs | ~15 s |
| 12 | 4096 | 256 | ~187.500 | 6,25% | ~16× | ~97% | ~625 µs | ~120 s |
| 13 | 2048 | 128 | ~187.500 | 6,25% | ~16× | ~97% | ~625 µs | ~60 s |
| 14 | 1024 | 64 | ~187.500 | 6,25% | ~16× | ~96% | ~625 µs | ~30 s |
| 15 | 512 | 32 | ~187.500 | 6,25% | ~16× | ~87% | ~625 µs | ~15 s |
| 16 | 4096 | 512 | ~375.000 | 12,5% | ~8× | ~99% | ~1000 µs | ~120 s |
| 17 | 2048 | 256 | ~375.000 | 12,5% | ~8× | ~99% | ~1000 µs | ~60 s |
| 18 | 1024 | 128 | ~375.000 | 12,5% | ~8× | ~98% | ~1000 µs | ~30 s |
| 19 | 512 | 64 | ~375.000 | 12,5% | ~8× | ~90% | ~1000 µs | ~15 s |
| 20 | 1024 | 256 | ~750.000 | 25,0% | ~4× | ~99,5% | ~1750 µs | ~30 s |
| 21 | 2048 | 512 | ~750.000 | 25,0% | ~4× | ~99,5% | ~1750 µs | ~60 s |

**Recall:** fração estimada de queries em que o top-5 exato é recuperado. Para mesmo % do dataset, `nlist` maior produz particionamento mais fino → recall ligeiramente superior (ex.: #8 vs #10, #5 vs #6). Com D=14 (dimensionalidade baixa), clusters K-means são mais coesos que em datasets de alta dimensão — as estimativas tendem a ser otimistas; valide com benchmark local.

**nlist=512 (configs #7, #11, #15, #19):** build muito rápido (~15 s), útil para iteração; recall inferior ao equivalente de nlist=1024 ou 2048 por clusters menos coesos. Prefira nlist=1024 em produção quando o tempo de build permitir.

**Build K-means:** estimativa para 8 CPUs com convergência antes de 30 iterações; escala inversamente com CPUs disponíveis.

**Análise por critério da competição:**

| Critério | Config | Justificativa |
|----------|--------|---------------|
| Menor latência absoluta | #1 nlist=4096/nprobe=16 | ~256× vs brute; ~275 µs; recall ~58% — risco de corte de detecção |
| Menor latência com recall seguro (≥88%) | #8 nlist=4096/nprobe=128 | ~32×; ~437 µs; recall ~91% |
| Maior recall mantendo p99 ≤ 1 ms | #16–#18 | 12,5% do dataset; recall ~98–99%; latência ~1000 µs |
| Melhor recall absoluto | #20/#21 | ~99,5%; p99 ~1750 µs — acima do limiar de 1 ms para `p99_score` máximo |
| Build mais rápido com recall aceitável | #10/#11 nlist≤1024 | ≤30 s de build; recall ≥80–90% |
| **Melhor balanço geral ★** | **#9 nlist=2048/nprobe=64** | **~32×; recall ~94%; ~437 µs; seguro contra o corte de detecção** |

> **Corte de detecção:** erros (FP + FN + HTTP 5xx) acima de 15% zeram o `detection_score`. Com recall ~94%, ~6% das queries retornam um top-5 diferente do exato — mas como `fraud_score` é discreto em passos de 0,2 (inteiros/5), a maioria dessas divergências não altera a decisão final. Configs com recall abaixo de ~88% (linhas #1–#7) elevam o risco em transações borderline com score próximo de 0,6.

> **Limiar p99 = 1 ms:** configs até #15 (6,25% do dataset, ~625 µs) operam com folga. Configs #16–#19 (12,5%, ~1000 µs) estão no limite — o scan de vetores sozinho consome ~750 µs, deixando ~250 µs de margem para HTTP e JSON. Configs #20–#21 (25%, ~1750 µs) ultrapassam consistentemente 1 ms; escolha-as apenas quando o ganho de recall justificar a perda em `p99_score`.

> **Impacto do nlist no recall a % fixo:** para 3,13% do dataset escaneado, o recall varia de ~80% (nlist=512) a ~94% (nlist=2048) a ~91% (nlist=4096). O ponto ótimo é nlist=2048 — clusters mais coesos que nlist=512/1024, sem o custo de build de nlist=4096.

### Rodar a aplicação

```bash
go build -o server ./cmd/server

# Apenas references.bin — suficiente para VECTOR_SEARCHER=brute
go run ./cmd/preprocess
./server

# Com índice HNSW pré-construído — startup rápido para VECTOR_SEARCHER=hnsw
BUILD_HNSW=true go run ./cmd/preprocess
VECTOR_SEARCHER=hnsw ./server

# Vamana com R=16 SQ8 (padrão)
BUILD_VAMANA=true go run ./cmd/preprocess
VECTOR_SEARCHER=vamana ./server

# Vamana com beam width maior em runtime (mais recall, sem rebuild)
VECTOR_SEARCHER=vamana VAMANA_L=128 ./server

# Partition-brute: exato por bin, ~3.2× mais rápido que brute, sem arquivo extra
VECTOR_SEARCHER=partition ./server

# VP-Tree: exato por bin com poda triangular, construído no startup (~alguns segundos)
VECTOR_SEARCHER=vptree ./server
```

### Variáveis de ambiente

| Variável | Padrão | Escopo | Descrição |
|----------|--------|--------|-----------|
| `PORT` | `8080` | Runtime | Porta de escuta da instância |
| `LOG_LEVEL` | `info` | Runtime | Nível mínimo de log (`debug`, `info`, `warn`, `error`) |
| `VECTOR_SEARCHER` | `brute` | Runtime | Backend de busca vetorial (`brute`, `partition`, `vptree`, `hnsw`, `ivf` ou `vamana`) |
| `IVF_NPROBE` | `32` | Runtime | Clusters pesquisados por query com `VECTOR_SEARCHER=ivf`; maior = mais recall e mais lento |
| `VAMANA_L` | `64` | Runtime | Beam width na busca com `VECTOR_SEARCHER=vamana`; maior = mais recall e mais lento |
| `BUILD_HNSW` | `false` | Build time | Quando `true`, `cmd/preprocess` gera `references.hnsw`; passado como `ARG` ao Dockerfile |
| `BUILD_IVF` | `false` | Build time | Quando `true`, `cmd/preprocess` gera `references.ivf`; passado como `ARG` ao Dockerfile |
| `IVF_NLIST` | `1024` | Build time | Número de clusters K-means para o índice IVF; passado como `ARG` ao Dockerfile |
| `IVF_SQ8` | `true` | Build time | `true` = uint8 por dim (~45 MB heap); `false` = float32 (~168 MB); gravado no arquivo, lido automaticamente em runtime |
| `BUILD_VAMANA` | `false` | Build time | Quando `true`, `cmd/preprocess` gera `references.vamana`; passado como `ARG` ao Dockerfile |
| `VAMANA_R` | `16` | Build time | Grau máximo por nó no grafo Vamana; R=16 → ~237 MB mmap (SQ8); R=8 → ~141 MB |
| `VAMANA_BUILD_L` | `125` | Build time | Beam width durante a construção do grafo; `0` usa o padrão (125); reduza para 64–75 para cortar o tempo de build ~50% com pequena perda de recall |
| `VAMANA_ALPHA` | `1.2` | Build time | Multiplicador RobustPrune; > 1.0 cria arestas de longo alcance (melhor recall) |
| `VAMANA_SQ8` | `true` | Build time | `true` = uint8 por dim nos vetores (~42 MB); `false` = float32 (~168 MB); gravado no arquivo, lido automaticamente em runtime |

### Logs

O nível de log é controlado pela variável de ambiente `LOG_LEVEL`. O padrão é `info`.

```bash
./server                    # nível info (padrão)
LOG_LEVEL=debug ./server    # habilita logs de debug
LOG_LEVEL=warn  ./server    # apenas warn, error e fatal
LOG_LEVEL=error ./server    # apenas error e fatal
```

A saída é JSON estruturado via **uber-go/zap**, adequada para ingestão em Datadog, Loki, CloudWatch, etc.:

```json
{"level":"info","ts":1741723415.123,"logger":"http","msg":"fraud score request received","transaction_id":"tx-123","amount":384.88,"merchant_id":"MERC-001"}
{"level":"info","ts":1741723415.124,"logger":"usecase","msg":"evaluating transaction","transaction_id":"tx-123"}
{"level":"debug","ts":1741723415.124,"logger":"usecase","msg":"vectorizing request","transaction_id":"tx-123"}
{"level":"debug","ts":1741723415.125,"logger":"vectorizer","msg":"vectorization complete","transaction_id":"tx-123"}
{"level":"debug","ts":1741723415.126,"logger":"knn","msg":"KNN search complete","returned":5}
{"level":"debug","ts":1741723415.127,"logger":"usecase","msg":"fraud score computed","transaction_id":"tx-123","fraud_score":0.2,"approved":true}
{"level":"debug","ts":1741723415.127,"logger":"http","msg":"sending response","transaction_id":"tx-123","approved":true,"fraud_score":0.2}
```

#### Variáveis de ambiente no Docker Compose

O `docker-compose.yml` lê as variáveis do arquivo `.env` na raiz do projeto. As duas variáveis relevantes para o Compose são:

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `VECTOR_SEARCHER` | Runtime env | Backend de busca (`brute`, `partition`, `vptree`, `hnsw`, `ivf` ou `vamana`); injetado em cada instância de API |
| `IVF_NPROBE` | Runtime env | Clusters pesquisados por query com `ivf` (padrão 32); injetado em cada instância |
| `VAMANA_L` | Runtime env | Beam width na busca com `vamana` (padrão 64); injetado em cada instância |
| `BUILD_HNSW` | Build arg (`ARG`) | Quando `true`, `cmd/preprocess` gera `references.hnsw` durante o `docker build` |
| `BUILD_IVF` | Build arg (`ARG`) | Quando `true`, `cmd/preprocess` gera `references.ivf` durante o `docker build` |
| `IVF_NLIST` | Build arg (`ARG`) | Clusters K-means para o índice IVF (padrão 1024) |
| `IVF_SQ8` | Build arg (`ARG`) | `true` (padrão) = SQ8 ativo (~45 MB heap/instância); `false` = float32 (~168 MB) |
| `BUILD_VAMANA` | Build arg (`ARG`) | Quando `true`, `cmd/preprocess` gera `references.vamana` durante o `docker build` |
| `VAMANA_R` | Build arg (`ARG`) | Grau máximo do grafo Vamana (padrão 16); R=8 reduz o arquivo para ~141 MB (SQ8) e acelera o build |
| `VAMANA_BUILD_L` | Build arg (`ARG`) | Beam width na construção (padrão 125); `64` reduz build ~50% com pequena perda de recall |
| `VAMANA_ALPHA` | Build arg (`ARG`) | Multiplicador RobustPrune (padrão 1.2) |
| `VAMANA_SQ8` | Build arg (`ARG`) | `true` (padrão) = vetores uint8 no índice; `false` = float32 |

Para trocar o backend ou habilitar o pré-build, edite o `.env`:

```bash
# .env — IVF-SQ8 com 1024 clusters, 32 probes por query
VECTOR_SEARCHER=ivf
BUILD_IVF=true
IVF_NLIST=1024
IVF_NPROBE=32

# .env — Vamana com R=16 SQ8 (padrão)
# VECTOR_SEARCHER=vamana
# BUILD_VAMANA=true
# VAMANA_L=64
```

Ou passe diretamente na linha de comando sem alterar o arquivo:

```bash
VECTOR_SEARCHER=hnsw docker compose up --build
BUILD_HNSW=true VECTOR_SEARCHER=hnsw docker compose up --build
BUILD_IVF=true VECTOR_SEARCHER=ivf docker compose up --build
BUILD_IVF=true IVF_NLIST=2048 VECTOR_SEARCHER=ivf IVF_NPROBE=64 docker compose up --build
BUILD_VAMANA=true VECTOR_SEARCHER=vamana docker compose up --build
BUILD_VAMANA=true VAMANA_R=8 VECTOR_SEARCHER=vamana VAMANA_L=64 docker compose up --build

# Partition-brute: usa apenas references.bin, sem arquivo extra
VECTOR_SEARCHER=partition docker compose up --build

# VP-Tree: exato por bin, build no startup, sem arquivo extra
VECTOR_SEARCHER=vptree docker compose up --build
```

### Testes

```bash
# Todos os testes
go test ./...

# Com cobertura
go test -cover ./...

# Pacote específico
go test ./internal/adapter/http/
```

## Rodando com Docker

```bash
# Build e sobe com o backend definido em .env
docker compose up --build

# Trocar para HNSW sem editar o .env (usa hnsw.Open se references.hnsw não estiver na imagem)
VECTOR_SEARCHER=hnsw docker compose up --build

# Pré-construir o grafo HNSW na imagem — startup rápido (hnsw.Load)
BUILD_HNSW=true VECTOR_SEARCHER=hnsw docker compose up --build

# IVF com SQ8 (padrão, ~45 MB heap/instância)
BUILD_IVF=true VECTOR_SEARCHER=ivf docker compose up --build

# IVF sem SQ8 (~168 MB heap/instância, distâncias exatas dentro do cluster)
BUILD_IVF=true IVF_SQ8=false VECTOR_SEARCHER=ivf docker compose up --build

# IVF com parâmetros customizados
BUILD_IVF=true IVF_NLIST=2048 VECTOR_SEARCHER=ivf IVF_NPROBE=64 docker compose up --build

# Vamana com R=16 SQ8 (~237 MB mmap, compartilhado entre instâncias)
BUILD_VAMANA=true VECTOR_SEARCHER=vamana docker compose up --build

# Vamana com R=8 SQ8 (~141 MB mmap — mais adequado para o budget de 350 MB)
BUILD_VAMANA=true VAMANA_R=8 VECTOR_SEARCHER=vamana docker compose up --build

# Vamana com beam width maior em runtime (sem rebuild)
BUILD_VAMANA=true VECTOR_SEARCHER=vamana VAMANA_L=128 docker compose up --build

# Partition-brute: exato por bin, sem arquivo de índice extra
VECTOR_SEARCHER=partition docker compose up --build

# A API fica disponível em http://localhost:9999 (via nginx)
# Acesso direto às instâncias: http://localhost:8080 e http://localhost:8081
```

Para rebuild da imagem sem cache:

```bash
docker compose build --no-cache
docker compose up
```

## API

### `GET /ready`

Health check. Retorna `200 OK` quando a API está pronta para receber requisições.

```bash
curl -i http://localhost:8080/ready
```

### `POST /fraud-score`

Avalia uma transação e retorna a decisão de aprovação com o score de fraude.

**Request:**
```json
{
  "id": "tx-123",
  "transaction": {
    "amount": 384.88,
    "installments": 3,
    "requested_at": "2026-03-11T20:23:35Z"
  },
  "customer": {
    "avg_amount": 769.76,
    "tx_count_24h": 3,
    "known_merchants": ["MERC-009", "MERC-001"]
  },
  "merchant": {
    "id": "MERC-001",
    "mcc": "5912",
    "avg_amount": 298.95
  },
  "terminal": {
    "is_online": false,
    "card_present": true,
    "km_from_home": 13.71
  },
  "last_transaction": {
    "timestamp": "2026-03-11T14:58:35Z",
    "km_from_current": 18.86
  }
}
```

O campo `last_transaction` pode ser `null` quando não há transação anterior do cliente.

**Response:**
```json
{
  "approved": true,
  "fraud_score": 0.2
}
```

`approved` é `true` quando `fraud_score < 0.6`. O score é calculado como a proporção de fraudes entre os 5 vizinhos mais próximos na base de referência.

## Arquivos de recursos

Os arquivos em `resources/` são carregados na inicialização da aplicação e não mudam em tempo de execução.

| Arquivo | Descrição |
|---------|-----------|
| `references.json.gz` | Fonte original: 3 milhões de transações rotuladas (`fraud`/`legit`), comprimida em gzip. Não é lida em runtime. |
| `references.bin` | **Gerado por `cmd/preprocess`**. Binário flat SoA com float32 + uint8. Lido via mmap em runtime por todos os modos. |
| `references.hnsw` | **Gerado por `cmd/preprocess`** quando `BUILD_HNSW=true`. Grafo HNSW serializado. Carregado por `hnsw.Load` quando `VECTOR_SEARCHER=hnsw`; elimina o custo de build no startup. |
| `references.ivf` | **Gerado por `cmd/preprocess`** quando `BUILD_IVF=true`. Índice IVF: `[uint32 nlist][uint8 flags][opt. params SQ8][centroides][por cluster: size + vetores + labels]`. ~45 MB com `IVF_SQ8=true`, ~168 MB com `IVF_SQ8=false`. Obrigatório para `VECTOR_SEARCHER=ivf`. |
| `references.vamana` | **Gerado por `cmd/preprocess`** quando `BUILD_VAMANA=true`. Grafo Vamana: `[uint32 N][uint32 R][uint32 medoid][uint32 flags][opt. SQ8 params][N×R uint32 adj.][vetores uint8 ou float32][labels uint8]`. Mmapeado em runtime — compartilhado entre instâncias. ~237 MB para R=16 SQ8; ~141 MB para R=8 SQ8. Obrigatório para `VECTOR_SEARCHER=vamana`. |
| `mcc_risk.json` | Mapa de MCC para score de risco (0.0–1.0); MCCs ausentes usam `0.5` como padrão |
| `normalization.json` | Constantes usadas na normalização dos campos do payload para o vetor |

Exemplos de payloads e vetores de referência para testes locais estão em `resources/example-payloads.json` e `resources/example-references.json`.

## Detecção de fraude

O processo de avaliação de cada transação segue três etapas:

1. **Vectorização** — os campos do payload são normalizados em um vetor de 14 dimensões (valores entre `0.0` e `1.0`, exceto `minutes_since_last_tx` e `km_from_last_tx` que recebem `-1` quando `last_transaction` é `null`).
2. **Busca KNN** — os 5 vetores mais próximos são buscados na base de referência usando distância euclidiana. O backend é selecionado via `VECTOR_SEARCHER`: `brute` (exato, O(N), padrão), `partition` (exato por bin, O(N/bins), ~3,2× mais rápido que brute), `vptree` (VP-Tree exato por bin com poda triangular, sub-linear quando a dimensionalidade intrínseca permite), `hnsw` (aproximado, O(log N)), `ivf` (aproximado IVF-SQ8, O(nprobe·N/nlist)) ou `vamana` (aproximado Vamana/DiskANN, O(L·R)).
3. **Decisão** — `fraud_score = fraudes_entre_os_5 / 5`; `approved = fraud_score < 0.6`.

A especificação completa das 14 dimensões e das fórmulas de normalização está em [`docs/REGRAS_DE_DETECCAO.md`](./docs/REGRAS_DE_DETECCAO.md).
