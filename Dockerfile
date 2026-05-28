# syntax=docker/dockerfile:1

# ── Build ─────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compile the main server binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o server ./cmd/server

# Convert references.json.gz → references.bin.
# When BUILD_HNSW=true, also builds references.hnsw (pre-built HNSW graph for
# fast startup with VECTOR_SEARCHER=hnsw). Adds several minutes to image build time.
ARG BUILD_HNSW=false
RUN BUILD_HNSW=${BUILD_HNSW} CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go run ./cmd/preprocess

# Collect runtime resources into /app/dist/resources/.
# references.hnsw is included only when BUILD_HNSW=true.
RUN mkdir -p /app/dist/resources && \
    cp resources/references.bin \
       resources/mcc_risk.json \
       resources/normalization.json \
       /app/dist/resources/ && \
    if [ -f resources/references.hnsw ]; then \
        cp resources/references.hnsw /app/dist/resources/; \
    fi

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

COPY --from=builder /app/server           ./server
COPY --from=builder /app/dist/resources/  ./resources/

EXPOSE 8080

ENTRYPOINT ["/app/server"]
