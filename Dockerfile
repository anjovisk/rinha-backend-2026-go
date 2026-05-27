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

# Convert references.json.gz → references.bin (flat binary, float32 SoA).
# The .bin file is the only format read at runtime; .json.gz is not shipped.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go run ./cmd/preprocess

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

COPY --from=builder /app/server                        ./server
COPY --from=builder /app/resources/references.bin      ./resources/references.bin
COPY --from=builder /app/resources/mcc_risk.json       ./resources/mcc_risk.json
COPY --from=builder /app/resources/normalization.json  ./resources/normalization.json

EXPOSE 8080

ENTRYPOINT ["/app/server"]
