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
# When BUILD_IVF=true, also builds references.ivf (IVF-SQ8 index for
# fast startup with VECTOR_SEARCHER=ivf). IVF_NLIST controls cluster count (default 1024).
# When BUILD_HNSWFLAT=true, builds references.hnswflat (flat mmap HNSW for
# VECTOR_SEARCHER=hnswflat). HNSWFLAT_M=3 SQ8 → ~143 MB mmap, ~158 MB RSS.
ARG BUILD_HNSW=false
ARG BUILD_IVF=false
ARG IVF_NLIST=1024
ARG IVF_SQ8=true
ARG BUILD_VAMANA=false
ARG VAMANA_R=16
ARG VAMANA_BUILD_L=0
ARG VAMANA_ALPHA=1.2
ARG VAMANA_SQ8=true
ARG BUILD_HNSWFLAT=false
ARG HNSWFLAT_M=3
ARG HNSWFLAT_EF_BUILD=100
ARG HNSWFLAT_REFINE=true
ARG HNSWFLAT_SQ8=true
RUN BUILD_HNSW=${BUILD_HNSW} BUILD_IVF=${BUILD_IVF} IVF_NLIST=${IVF_NLIST} IVF_SQ8=${IVF_SQ8} \
    BUILD_VAMANA=${BUILD_VAMANA} VAMANA_R=${VAMANA_R} VAMANA_BUILD_L=${VAMANA_BUILD_L} \
    VAMANA_ALPHA=${VAMANA_ALPHA} VAMANA_SQ8=${VAMANA_SQ8} \
    BUILD_HNSWFLAT=${BUILD_HNSWFLAT} HNSWFLAT_M=${HNSWFLAT_M} \
    HNSWFLAT_EF_BUILD=${HNSWFLAT_EF_BUILD} HNSWFLAT_REFINE=${HNSWFLAT_REFINE} HNSWFLAT_SQ8=${HNSWFLAT_SQ8} \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go run ./cmd/preprocess

# Collect runtime resources into /app/dist/resources/.
# references.hnsw is included only when BUILD_HNSW=true.
# references.ivf is included only when BUILD_IVF=true.
# references.vamana is included only when BUILD_VAMANA=true.
RUN mkdir -p /app/dist/resources && \
    cp resources/references.bin \
       resources/mcc_risk.json \
       resources/normalization.json \
       /app/dist/resources/ && \
    if [ -f resources/references.hnsw ]; then \
        cp resources/references.hnsw /app/dist/resources/; \
    fi && \
    if [ -f resources/references.ivf ]; then \
        cp resources/references.ivf /app/dist/resources/; \
    fi && \
    if [ -f resources/references.vamana ]; then \
        cp resources/references.vamana /app/dist/resources/; \
    fi && \
    if [ -f resources/references.hnswflat ]; then \
        cp resources/references.hnswflat /app/dist/resources/; \
    fi

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 gcr.io/distroless/static-debian12:nonroot

WORKDIR /app

COPY --from=builder /app/server           ./server
COPY --from=builder /app/dist/resources/  ./resources/

EXPOSE 8080

ENTRYPOINT ["/app/server"]
