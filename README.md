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
│   adapter/vector     │       │        adapter/knn           │
│   Vectorize()        │       │        FindNearest()         │
│   (14 dimensões)     │       │  (brute-force sobre os      │
│   normalization.json │       │   3M vetores de referência) │
│   mcc_risk.json      │       │   references.json.gz        │
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
| `internal/adapter/knn` | Adapter secundário | Busca KNN nos vetores de referência (brute-force, substituível por HNSW) |

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
│   │   └── main.go                    # converte references.json.gz → references.bin (build time)
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
│       └── knn/
│           └── searcher.go            # implementa NeighborFinder
├── resources/
│   ├── references.json.gz             # fonte original: 3M vetores rotulados (não lida em runtime)
│   ├── references.bin                 # gerado por cmd/preprocess; lido via mmap em runtime
│   ├── mcc_risk.json                  # risco por categoria de merchant (MCC)
│   └── normalization.json             # constantes de normalização
├── docs/                              # documentação em português
├── go.mod
├── Dockerfile                         # build multi-stage; imagem final distroless/static
├── nginx.conf                         # configuração do load balancer
└── docker-compose.yml                 # nginx (:9999) + api-1 (:8080) + api-2 (:8081)
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

O servidor não lê `references.json.gz` diretamente. Antes de rodar a aplicação pela primeira vez, gere o arquivo binário `references.bin`:

```bash
go run ./cmd/preprocess
# Gera: resources/references.bin
```

Esse passo converte os 3 milhões de vetores do formato JSON para um binário compacto com layout SoA (Struct of Arrays):

```
[uint32 N]  número de entradas
[N × 56 B]  vetores em float32, row-major (14 dimensões por entrada)
[N × 1 B]   labels: 1 = fraud, 0 = legit
```

**Por que o binário?**

| | float64 + string (JSON) | float32 + uint8 (binário) |
|---|---|---|
| Tamanho por instância | ~408 MB | ~171 MB |
| Tempo de carga | ~30 s (parsing JSON) | < 1 s (mmap) |

Com `mmap`, as duas instâncias da API compartilham as mesmas páginas físicas no page cache do kernel — o arquivo ocupa ~171 MB na RAM, não ~342 MB. Isso mantém o uso total dentro do budget de 350 MB.

O `references.bin` não é versionado no repositório. Em Docker, o passo é executado automaticamente durante o `docker compose up --build` (ver `Dockerfile`).

### Rodar a aplicação

```bash
# Gerar o binário de referência (apenas na primeira vez ou após atualizar references.json.gz)
go run ./cmd/preprocess

go build -o server ./cmd/server
./server
# A API fica disponível em http://localhost:8080
```

A porta pode ser alterada via variável de ambiente:

```bash
PORT=8081 ./server
```

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

#### Logs em Docker Compose

Para alterar o nível nas instâncias da API, passe `LOG_LEVEL` como variável de ambiente no `docker-compose.yml`:

```yaml
services:
  api-1:
    environment:
      - LOG_LEVEL=debug
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
# Build e sobe toda a stack (Nginx + 2 instâncias da API)
docker compose up --build

# A API fica disponível em http://localhost:9999 (via nginx)
# Acesso direto às instâncias: http://localhost:8080 e http://localhost:8081
```

Para rebuild da imagem da aplicação sem cache:

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
| `references.bin` | **Gerado por `cmd/preprocess`**. Binário flat SoA com float32 + uint8. Lido via mmap em runtime. |
| `mcc_risk.json` | Mapa de MCC para score de risco (0.0–1.0); MCCs ausentes usam `0.5` como padrão |
| `normalization.json` | Constantes usadas na normalização dos campos do payload para o vetor |

Exemplos de payloads e vetores de referência para testes locais estão em `resources/example-payloads.json` e `resources/example-references.json`.

## Detecção de fraude

O processo de avaliação de cada transação segue três etapas:

1. **Vectorização** — os campos do payload são normalizados em um vetor de 14 dimensões (valores entre `0.0` e `1.0`, exceto `minutes_since_last_tx` e `km_from_last_tx` que recebem `-1` quando `last_transaction` é `null`).
2. **Busca KNN** — os 5 vetores mais próximos são buscados na base de referência usando distância euclidiana (brute force ou índice ANN/VP-Tree).
3. **Decisão** — `fraud_score = fraudes_entre_os_5 / 5`; `approved = fraud_score < 0.6`.

A especificação completa das 14 dimensões e das fórmulas de normalização está em [`docs/REGRAS_DE_DETECCAO.md`](./docs/REGRAS_DE_DETECCAO.md).
