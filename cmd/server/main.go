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
// Usage:
//
//	./server
//	PORT=8081 ./server
//	LOG_LEVEL=debug ./server
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	httphandler "anjovisk/fraud-detection/internal/adapter/http"
	"anjovisk/fraud-detection/internal/adapter/knn"
	"anjovisk/fraud-detection/internal/adapter/vector"
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

	searcher, err := knn.Open("resources/references.bin", logger.Named("knn"))
	if err != nil {
		logger.Fatal("failed to open reference index", zap.Error(err))
	}
	defer searcher.Close() //nolint:errcheck

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

	fraudUC := usecase.NewFraudScore(vec, searcher, logger.Named("usecase"))
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

