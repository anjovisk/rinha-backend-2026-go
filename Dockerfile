# syntax=docker/dockerfile:1

# ── Build ─────────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o server ./cmd/server

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /app/server /server
COPY resources/ /resources/

EXPOSE 8080

ENTRYPOINT ["/server"]
