// Command server starts the HTTP fraud detection API.
//
// At startup it loads three resource files from the resources/ directory,
// wires the adapters and use case, then listens for HTTP traffic.
// The listening port is read from the PORT environment variable; it defaults to 8080.
// Port 9999 is reserved for the Nginx load balancer.
//
// Log verbosity is controlled by the LOG_LEVEL environment variable.
// Accepted values: debug, info (default), warn, error.
//
// The nearest-neighbour backend is selected via VECTOR_SEARCHER:
//
//	VECTOR_SEARCHER=brute      exact brute-force O(N) scan via mmap (default)
//	VECTOR_SEARCHER=hnsw       approximate HNSW O(log N) in-memory graph
//	VECTOR_SEARCHER=ivf        approximate IVF-SQ8, requires resources/references.ivf
//	VECTOR_SEARCHER=vamana     approximate Vamana graph, requires resources/references.vamana
//	VECTOR_SEARCHER=partition  partitioned brute-force; routes each query to one of 6 sub-indexes
//	                           selected by (last_tx_null × is_online × card_present), scanning
//	                           ~31% of the dataset on average (~3.2× speedup over brute).
//	                           Exact within each partition; no separate index file required.
//
// IVF tuning knobs (only when VECTOR_SEARCHER=ivf):
//
//	IVF_NPROBE  number of clusters searched per query (default 32); higher = better recall, slower
//
// Vamana tuning knobs (only when VECTOR_SEARCHER=vamana):
//
//	VAMANA_L  beam width at query time (default 64); higher = better recall, slower
//
// Usage:
//
//	./server
//	PORT=8081 ./server
//	LOG_LEVEL=debug ./server
//	VECTOR_SEARCHER=hnsw ./server
//	VECTOR_SEARCHER=ivf IVF_NPROBE=64 ./server
//	VECTOR_SEARCHER=vamana ./server
//	VECTOR_SEARCHER=vamana VAMANA_L=128 ./server
//	VECTOR_SEARCHER=partition ./server
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	httphandler "anjovisk/fraud-detection/internal/adapter/http"
	"anjovisk/fraud-detection/internal/adapter/hnsw"
	"anjovisk/fraud-detection/internal/adapter/ivf"
	"anjovisk/fraud-detection/internal/adapter/knn"
	"anjovisk/fraud-detection/internal/adapter/partition"
	"anjovisk/fraud-detection/internal/adapter/vamana"
	"anjovisk/fraud-detection/internal/adapter/vector"
	"anjovisk/fraud-detection/internal/adapter/vptree"
	"anjovisk/fraud-detection/internal/port"
	"anjovisk/fraud-detection/internal/usecase"
)

func main() {
	logger := buildLogger()
	defer logger.Sync() //nolint:errcheck

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	norm := loadNormalization("resources/normalization.json", logger)
	logger.Debug("normalization constants loaded",
		zap.Float64("max_amount", norm.MaxAmount),
		zap.Float64("max_km", norm.MaxKm),
		zap.Float64("max_minutes", norm.MaxMinutes),
	)

	mccRisk := loadMCCRisk("resources/mcc_risk.json", logger)
	logger.Debug("MCC risk table loaded", zap.Int("entries", len(mccRisk)))

	nf := buildNeighborFinder(os.Getenv("VECTOR_SEARCHER"), logger.Named("knn"))
	if c, ok := nf.(interface{ Close() error }); ok {
		defer c.Close() //nolint:errcheck
	}

	vec := vector.New(vector.Config{
		MaxAmount:            norm.MaxAmount,
		MaxInstallments:      norm.MaxInstallments,
		AmountVsAvgRatio:     norm.AmountVsAvgRatio,
		MaxMinutes:           norm.MaxMinutes,
		MaxKm:                norm.MaxKm,
		MaxTxCount24h:        norm.MaxTxCount24h,
		MaxMerchantAvgAmount: norm.MaxMerchantAvgAmount,
		MCCRisk:              mccRisk,
	}, logger.Named("vectorizer"))

	fraudUC := usecase.NewFraudScore(vec, nf, logger.Named("usecase"))
	srv := httphandler.NewServer(fraudUC, logger.Named("http"))

	logger.Info("server ready", zap.String("port", port))
	log.Fatal(http.ListenAndServe(":"+port, srv.Routes()))
}

// buildLogger creates a production zap logger with the level configured by LOG_LEVEL.
// The logger outputs JSON to stdout; its level defaults to Info when LOG_LEVEL is absent or invalid.
func buildLogger() *zap.Logger {
	cfg := zap.NewProductionConfig()
	if raw := os.Getenv("LOG_LEVEL"); raw != "" {
		var level zapcore.Level
		if err := level.UnmarshalText([]byte(raw)); err == nil {
			cfg.Level = zap.NewAtomicLevelAt(level)
		}
	}
	logger, err := cfg.Build()
	if err != nil {
		log.Fatalf("build logger: %v", err)
	}
	return logger
}

// normalizationConfig mirrors the JSON schema of resources/normalization.json.
// All fields are divisors used by the vectorizer to normalise raw values to [0, 1].
type normalizationConfig struct {
	// MaxAmount is the maximum expected transaction amount (divisor for dimension 0).
	MaxAmount float64 `json:"max_amount"`
	// MaxInstallments is the maximum number of instalments (divisor for dimension 1).
	MaxInstallments float64 `json:"max_installments"`
	// AmountVsAvgRatio is the scaling factor applied to the amount-vs-customer-average ratio (dimension 2).
	AmountVsAvgRatio float64 `json:"amount_vs_avg_ratio"`
	// MaxMinutes is the maximum minutes since last transaction (divisor for dimension 5).
	MaxMinutes float64 `json:"max_minutes"`
	// MaxKm is the maximum distance in kilometres (divisor for dimensions 6 and 7).
	MaxKm float64 `json:"max_km"`
	// MaxTxCount24h is the maximum 24-hour transaction count (divisor for dimension 8).
	MaxTxCount24h float64 `json:"max_tx_count_24h"`
	// MaxMerchantAvgAmount is the maximum merchant average amount (divisor for dimension 13).
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

// loadNormalization reads and decodes the normalisation constants from path.
// Calls logger.Fatal if the file cannot be opened or parsed.
func loadNormalization(path string, logger *zap.Logger) normalizationConfig {
	f, err := os.Open(path)
	if err != nil {
		logger.Fatal("failed to open normalization file", zap.String("path", path), zap.Error(err))
	}
	defer f.Close()
	var cfg normalizationConfig
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		logger.Fatal("failed to decode normalization file", zap.String("path", path), zap.Error(err))
	}
	return cfg
}

// buildNeighborFinder instantiates the NeighborFinder specified by kind.
// The logger should already be named (e.g. logger.Named("knn")).
//
//   - "hnsw": approximate HNSW graph; loads references.hnsw if present (fast),
//     otherwise builds from references.bin (slow, emits Warn).
//   - "ivf": approximate IVF-SQ8; requires references.ivf (Fatal if absent —
//     run preprocess with BUILD_IVF=true). IVF_NPROBE controls recall vs speed.
//   - "vamana": approximate Vamana graph; requires references.vamana (Fatal if absent —
//     run preprocess with BUILD_VAMANA=true). VAMANA_L controls beam width at query time.
//   - "partition": exact brute-force partitioned by (last_tx_null × is_online × card_present);
//     reads references.bin only — no separate index file required. ~3.2× faster than brute.
//   - "vptree": exact VP-Tree search within each partition bin; same routing as "partition"
//     but with triangle-inequality pruning inside each bin. Built at startup from references.bin.
//     Faster than "partition" when intrinsic dimensionality within bins enables effective pruning.
//   - anything else (including "brute" or ""): exact brute-force O(N) scan.
func buildNeighborFinder(kind string, logger *zap.Logger) port.NeighborFinder {
	const refsPath = "resources/references.bin"
	const hnswPath = "resources/references.hnsw"
	const ivfPath = "resources/references.ivf"
	const vamanaPath = "resources/references.vamana"
	switch kind {
	case "hnsw":
		if _, err := os.Stat(hnswPath); err == nil {
			logger.Info("neighbor finder: HNSW (loading pre-built index)",
				zap.String("graph_path", hnswPath),
			)
			s, err := hnsw.Load(hnswPath, refsPath, logger)
			if err != nil {
				logger.Fatal("failed to load HNSW index", zap.Error(err))
			}
			return s
		}
		logger.Warn("neighbor finder: HNSW (building from scratch — run preprocess to pre-build)",
			zap.String("missing", hnswPath),
		)
		s, err := hnsw.Open(refsPath, logger)
		if err != nil {
			logger.Fatal("failed to build HNSW index", zap.Error(err))
		}
		return s
	case "ivf":
		nprobe := ivf.DefaultNprobe
		if raw := os.Getenv("IVF_NPROBE"); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > 0 {
				nprobe = n
			}
		}
		logger.Info("neighbor finder: IVF-SQ8 (loading pre-built index)",
			zap.String("index_path", ivfPath),
			zap.Int("nprobe", nprobe),
		)
		s, err := ivf.Open(ivfPath, nprobe, logger)
		if err != nil {
			logger.Fatal("failed to load IVF-SQ8 index — run preprocess with BUILD_IVF=true",
				zap.Error(err),
			)
		}
		return s
	case "vamana":
		l := vamana.DefaultL
		if raw := os.Getenv("VAMANA_L"); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > 0 {
				l = n
			}
		}
		logger.Info("neighbor finder: Vamana (loading pre-built index)",
			zap.String("index_path", vamanaPath),
			zap.Int("L", l),
		)
		s, err := vamana.Open(vamanaPath, l, logger)
		if err != nil {
			logger.Fatal("failed to load Vamana index — run preprocess with BUILD_VAMANA=true",
				zap.Error(err),
			)
		}
		return s
	case "partition":
		logger.Info("neighbor finder: partition-brute (exact within partition)",
			zap.String("bin_features", "last_tx_null × is_online × card_present"),
		)
		s, err := partition.Open(refsPath, logger)
		if err != nil {
			logger.Fatal("failed to open partition index", zap.Error(err))
		}
		return s
	case "vptree":
		logger.Info("neighbor finder: VP-Tree (exact within partition)",
			zap.String("bin_features", "last_tx_null × is_online × card_present"),
			zap.Int("leaf_size", 32),
		)
		s, err := vptree.Open(refsPath, logger)
		if err != nil {
			logger.Fatal("failed to build VP-Tree index", zap.Error(err))
		}
		return s
	default:
		logger.Info("neighbor finder: brute-force (exact)", zap.String("vector_searcher", kind))
		s, err := knn.Open(refsPath, logger)
		if err != nil {
			logger.Fatal("failed to open reference index", zap.Error(err))
		}
		return s
	}
}

// loadMCCRisk reads and decodes the MCC-to-risk-score map from path.
// Keys are MCC strings; values are risk scores in [0.0, 1.0].
// Calls logger.Fatal if the file cannot be opened or parsed.
func loadMCCRisk(path string, logger *zap.Logger) map[string]float64 {
	f, err := os.Open(path)
	if err != nil {
		logger.Fatal("failed to open MCC risk file", zap.String("path", path), zap.Error(err))
	}
	defer f.Close()
	var m map[string]float64
	if err := json.NewDecoder(f).Decode(&m); err != nil {
		logger.Fatal("failed to decode MCC risk file", zap.String("path", path), zap.Error(err))
	}
	return m
}

