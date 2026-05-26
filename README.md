# Fraud Detection API

API de detecção de fraudes em transações de cartão via busca vetorial. Para cada transação recebida, a API normaliza o payload em um vetor de 14 dimensões, busca os 5 vizinhos mais próximos em uma base de referência pré-indexada e retorna uma decisão de aprovação com score de fraude.

## Arquitetura

```
cliente → load balancer (:9999) → api-1 (:8080)
                                 → api-2 (:8081)
```

O Nginx atua como load balancer e distribui as requisições em round-robin entre as instâncias da API. A configuração é feita via `nginx.conf` montado em volume no container. Nenhuma lógica de negócio é executada no Nginx.

## Stack

- **Go 1.22** — aplicação principal (roteamento com método HTTP nativo via `net/http`)
- **Docker / Docker Compose** — containerização e orquestração
- **Nginx** — load balancer, configurado via `nginx.conf` montado em volume

## Estrutura do repositório

```
.
├── cmd/
│   └── server/
│       └── main.go             # entrypoint; registra rotas e inicia o servidor
├── internal/
│   ├── handler/
│   │   ├── ready.go            # GET /ready
│   │   ├── ready_test.go
│   │   ├── fraud.go            # POST /fraud-score  (a implementar)
│   │   └── fraud_test.go
│   ├── vector/                 # vectorização e normalização (a implementar)
│   └── fraud/                  # lógica de decisão KNN (a implementar)
├── resources/
│   ├── references.json.gz      # 3M vetores rotulados (base de referência)
│   ├── mcc_risk.json           # risco por categoria de merchant (MCC)
│   └── normalization.json      # constantes de normalização
├── docs/                       # documentação em português
├── go.mod
├── Dockerfile                  # build multi-stage; imagem final distroless/static
├── nginx.conf                  # configuração do load balancer
└── docker-compose.yml          # nginx (:9999) + api-1 (:8080) + api-2 (:8081)
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

### Rodar a aplicação

```bash
go build -o server ./cmd/server
./server
# A API fica disponível em http://localhost:8080
```

A porta pode ser alterada via variável de ambiente:

```bash
PORT=8081 ./server
```

### Testes

```bash
# Todos os testes
go test ./...

# Com cobertura
go test -cover ./...

# Pacote específico
go test ./internal/handler/
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
| `references.json.gz` | Base vetorial com 3 milhões de transações rotuladas (`fraud`/`legit`), comprimida em gzip |
| `mcc_risk.json` | Mapa de MCC para score de risco (0.0–1.0); MCCs ausentes usam `0.5` como padrão |
| `normalization.json` | Constantes usadas na normalização dos campos do payload para o vetor |

Exemplos de payloads e vetores de referência para testes locais estão em `resources/example-payloads.json` e `resources/example-references.json`.

## Detecção de fraude

O processo de avaliação de cada transação segue três etapas:

1. **Vectorização** — os campos do payload são normalizados em um vetor de 14 dimensões (valores entre `0.0` e `1.0`, exceto `minutes_since_last_tx` e `km_from_last_tx` que recebem `-1` quando `last_transaction` é `null`).
2. **Busca KNN** — os 5 vetores mais próximos são buscados na base de referência usando distância euclidiana (brute force ou índice ANN/VP-Tree).
3. **Decisão** — `fraud_score = fraudes_entre_os_5 / 5`; `approved = fraud_score < 0.6`.

A especificação completa das 14 dimensões e das fórmulas de normalização está em [`docs/REGRAS_DE_DETECCAO.md`](./docs/REGRAS_DE_DETECCAO.md).
