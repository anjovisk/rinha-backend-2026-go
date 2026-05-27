// Package httphandler implements the primary HTTP adapter for the fraud detection API.
// It translates HTTP requests into domain calls via port.FraudScoreUseCase and serialises
// domain results back to JSON. No business logic lives here; all decisions are delegated
// to the use case layer.
package httphandler

import (
	"net/http"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/port"
)

// Server is the primary HTTP adapter. It holds references to use case ports and
// registers all API routes on a ServeMux.
type Server struct {
	fraudUseCase port.FraudScoreUseCase
	logger       *zap.Logger
}

// NewServer creates a Server wired to the given fraud use case and logger.
// fraudUseCase is invoked on every POST /fraud-score request and must not be nil
// when that route is expected to receive traffic.
// logger is used to emit structured logs from all HTTP handlers.
func NewServer(fraudUseCase port.FraudScoreUseCase, logger *zap.Logger) *Server {
	return &Server{fraudUseCase: fraudUseCase, logger: logger}
}

// Routes registers all API endpoints and returns the resulting http.Handler.
// Registered routes:
//
//	GET  /ready        — liveness/readiness health check (see ready.go)
//	POST /fraud-score  — fraud evaluation endpoint (see fraud_score.go)
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("POST /fraud-score", s.handleFraudScore)
	return mux
}
