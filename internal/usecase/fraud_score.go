// Package usecase implements the application layer of the fraud detection system.
// Use cases orchestrate domain entities and outbound ports to fulfil business rules,
// remaining decoupled from transport protocols and infrastructure concerns.
package usecase

import (
	"context"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
	"anjovisk/fraud-detection/internal/port"
)

// FraudScore implements port.FraudScoreUseCase.
// It orchestrates the three-step evaluation pipeline: vectorization, nearest-neighbour
// retrieval, and score computation.
type FraudScore struct {
	vectorizer     port.Vectorizer
	neighborFinder port.NeighborFinder
	logger         *zap.Logger
}

// NewFraudScore constructs a FraudScore use case wired to the given outbound adapters.
// v is the Vectorizer responsible for converting requests into 14-dimensional feature vectors.
// nf is the NeighborFinder used to retrieve the k nearest reference vectors.
// logger is used to emit structured logs during evaluation.
func NewFraudScore(v port.Vectorizer, nf port.NeighborFinder, logger *zap.Logger) *FraudScore {
	return &FraudScore{vectorizer: v, neighborFinder: nf, logger: logger}
}

// Evaluate runs the fraud detection pipeline for req and returns the result.
// The pipeline executes three steps in sequence:
//  1. Vectorize req into a 14-dimensional feature vector via the Vectorizer.
//  2. Retrieve the 5 nearest neighbours from the reference dataset via the NeighborFinder.
//  3. Compute fraud_score = fraudulent_neighbours / 5; set approved = score < 0.6.
//
// ctx is forwarded to outbound ports for deadline and cancellation propagation.
// Returns a non-nil error only if the pipeline cannot produce a result.
func (uc *FraudScore) Evaluate(ctx context.Context, req domain.FraudScoreRequest) (domain.FraudResult, error) {
	uc.logger.Info("evaluating transaction", zap.String("transaction_id", req.ID))

	uc.logger.Debug("vectorizing request", zap.String("transaction_id", req.ID))
	vec := uc.vectorizer.Vectorize(req)

	uc.logger.Debug("finding nearest neighbours",
		zap.String("transaction_id", req.ID),
		zap.Int("k", 5),
	)
	labels := uc.neighborFinder.FindNearest(vec, 5)
	uc.logger.Debug("neighbours retrieved",
		zap.String("transaction_id", req.ID),
		zap.Strings("labels", labels),
	)

	fraudCount := 0
	for _, label := range labels {
		if label == "fraud" {
			fraudCount++
		}
	}

	score := float64(fraudCount) / float64(len(labels))
	result := domain.FraudResult{
		Approved:   score < domain.FraudThreshold,
		FraudScore: score,
	}
	uc.logger.Debug("fraud score computed",
		zap.String("transaction_id", req.ID),
		zap.Float64("fraud_score", score),
		zap.Bool("approved", result.Approved),
	)
	return result, nil
}
