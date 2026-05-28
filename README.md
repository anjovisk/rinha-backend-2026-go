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
┌──────────────────────┐       ┌──────────────────────────────┐
│   adapter/vector     │       │  adapter/knn  (VECTOR_SEARCHER= │
│   Vectorize()        │       │   brute)  ou  adapter/ann    │
│   (14 dimensões)     │       │   (VECTOR_SEARCHER=hnsw)        │
│   normalization.json │       │                              │
│   mcc_risk.json      │       │   FindNearest() via          │
│                      │       │   brute-force O(N) ou        │
│                      │       │   HNSW aprox. O(log N)       │
└──────────────────────┘       └──────────────────────────────┘
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
| `internal/adapter/ann` | Adapter secundário | Busca aproximada HNSW O(log N) via `github.com/coder/hnsw`; selecionado por `VECTOR_SEARCHER=hnsw` |

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
│   │   └── main.go                    # converte references.json.gz → references.bin + references.hnsw
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
│       └── ann/
│           └── hnsw.go                # implementa NeighborFinder (HNSW aproximado)
├── resources/
│   ├── references.json.gz             # fonte original: 3M vetores rotulados (não lida em runtime)
│   ├── references.bin                 # gerado por cmd/preprocess; lido via mmap em runtime
│   ├── references.hnsw                # gerado por cmd/preprocess quando BUILD_HNSW=true; grafo HNSW serializado
│   ├── mcc_risk.json                  # risco por categoria de merchant (MCC)
│   └── normalization.json             # constantes de normalização
├── docs/                              # documentação em português
├── go.mod
├── .env                               # padrão local: VECTOR_SEARCHER=brute, BUILD_HNSW=false (não versionado)
├── .env.example                       # template comentado de todas as variáveis
├── Dockerfile                         # build multi-stage; ARG BUILD_HNSW controla geração do índice HNSW
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
```

O preprocess produz até dois arquivos:

| Arquivo | Quando gerado | O que é | Usado por |
|---------|--------------|---------|-----------|
| `references.bin` | Sempre | Binário flat SoA: `[uint32 N][N×56 B float32][N×1 B uint8]` | Todos os modos — lido via mmap em runtime |
| `references.hnsw` | `BUILD_HNSW=true` | Grafo HNSW serializado pelo `coder/hnsw` | `VECTOR_SEARCHER=hnsw` com `ann.Load` — startup rápido |

**Por que o binário?**

| | float64 + string (JSON) | float32 + uint8 (binário) |
|---|---|---|
| Tamanho por instância | ~408 MB | ~171 MB |
| Tempo de carga | ~30 s (parsing JSON) | < 1 s (mmap) |

Com `mmap`, as duas instâncias da API compartilham as mesmas páginas físicas no page cache do kernel — o arquivo ocupa ~171 MB na RAM, não ~342 MB. Isso mantém o uso total dentro do budget de 350 MB.

Nenhum dos dois arquivos é versionado no repositório. Em Docker, o preprocess é executado automaticamente durante o build da imagem (ver `Dockerfile`); `BUILD_HNSW` é passado como build arg pelo docker-compose.

### Backend de busca vetorial

O algoritmo usado para encontrar os 5 vizinhos mais próximos é controlado pela variável de ambiente `VECTOR_SEARCHER`:

| `VECTOR_SEARCHER` | Algoritmo | Complexidade de query | Startup | RSS extra por instância | Recall |
|---|---|---|---|---|---|
| `brute` (padrão) | Brute-force exato | O(N) | < 1 s (mmap) | 0 — vetores compartilhados via page cache | 100% |
| `hnsw` + `references.hnsw` presente | HNSW aprox. (`coder/hnsw`, M=4, EfSearch=50) — **Load** | O(log N) | < 1 s (import) | ~170 MB heap (vetores alocados pelo Import) | ~93–97% |
| `hnsw` + `references.hnsw` ausente | HNSW aprox. — **Open** (build from scratch) | O(log N) | dezenas de segundos (O(N log N)) | ~50–100 MB heap (conexões apenas; vetores no mmap) | ~93–97% |

Quando `VECTOR_SEARCHER=hnsw`, o servidor detecta automaticamente se `resources/references.hnsw` existe:
- **Arquivo presente** (`cmd/preprocess` foi executado): usa `ann.Load` — carrega o grafo serializado em O(N), sem custo de build. Vetores ficam em heap (~170 MB por instância).
- **Arquivo ausente**: usa `ann.Open` — constrói o grafo em O(N log N), mais lento para iniciar, mas vetores ficam no mmap (~0 MB heap extra). Um `Warn` é emitido no log.

> **Budget de memória:** com `hnsw` + `Load`, cada instância usa ~170 MB de heap adicional para os vetores. Com duas instâncias, isso soma ~340 MB só para vetores — deixando pouca margem no budget de 350 MB. Use `hnsw` + `Open` se a memória for crítica, ou `hnsw` + `Load` em ambientes com mais RAM.

### Rodar a aplicação

```bash
go build -o server ./cmd/server

# Apenas references.bin — suficiente para VECTOR_SEARCHER=brute
go run ./cmd/preprocess
./server

# Com índice HNSW pré-construído — startup rápido para VECTOR_SEARCHER=hnsw
BUILD_HNSW=true go run ./cmd/preprocess
VECTOR_SEARCHER=hnsw ./server
```

### Variáveis de ambiente

| Variável | Padrão | Escopo | Descrição |
|----------|--------|--------|-----------|
| `PORT` | `8080` | Runtime | Porta de escuta da instância |
| `LOG_LEVEL` | `info` | Runtime | Nível mínimo de log (`debug`, `info`, `warn`, `error`) |
| `VECTOR_SEARCHER` | `brute` | Runtime | Backend de busca vetorial (`brute` ou `hnsw`) |
| `BUILD_HNSW` | `false` | Build time | Quando `true`, `cmd/preprocess` gera `references.hnsw`; passado como `ARG` ao Dockerfile |

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
| `VECTOR_SEARCHER` | Runtime env | Backend de busca (`brute` ou `hnsw`); injetado em cada instância de API |
| `BUILD_HNSW` | Build arg (`ARG`) | Quando `true`, `cmd/preprocess` gera `references.hnsw` durante o `docker build` |

Para trocar o backend ou habilitar o pré-build, edite o `.env`:

```bash
# .env
VECTOR_SEARCHER=hnsw
BUILD_HNSW=true   # gera references.hnsw no build — necessário para startup rápido com hnsw
```

Ou passe diretamente na linha de comando sem alterar o arquivo:

```bash
VECTOR_SEARCHER=hnsw docker compose up --build
BUILD_HNSW=true VECTOR_SEARCHER=hnsw docker compose up --build
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

# Trocar para HNSW sem editar o .env (usa ann.Open se references.hnsw não estiver na imagem)
VECTOR_SEARCHER=hnsw docker compose up --build

# Pré-construir o grafo HNSW na imagem — startup rápido (ann.Load)
BUILD_HNSW=true VECTOR_SEARCHER=hnsw docker compose up --build

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
| `references.hnsw` | **Gerado por `cmd/preprocess`**. Grafo HNSW serializado. Carregado por `ann.Load` quando `VECTOR_SEARCHER=hnsw`; elimina o custo de build no startup. |
| `mcc_risk.json` | Mapa de MCC para score de risco (0.0–1.0); MCCs ausentes usam `0.5` como padrão |
| `normalization.json` | Constantes usadas na normalização dos campos do payload para o vetor |

Exemplos de payloads e vetores de referência para testes locais estão em `resources/example-payloads.json` e `resources/example-references.json`.

## Detecção de fraude

O processo de avaliação de cada transação segue três etapas:

1. **Vectorização** — os campos do payload são normalizados em um vetor de 14 dimensões (valores entre `0.0` e `1.0`, exceto `minutes_since_last_tx` e `km_from_last_tx` que recebem `-1` quando `last_transaction` é `null`).
2. **Busca KNN** — os 5 vetores mais próximos são buscados na base de referência usando distância euclidiana. O backend é selecionado via `VECTOR_SEARCHER`: `brute` (exato, O(N), padrão) ou `hnsw` (aproximado, O(log N)).
3. **Decisão** — `fraud_score = fraudes_entre_os_5 / 5`; `approved = fraud_score < 0.6`.

A especificação completa das 14 dimensões e das fórmulas de normalização está em [`docs/REGRAS_DE_DETECCAO.md`](./docs/REGRAS_DE_DETECCAO.md).
