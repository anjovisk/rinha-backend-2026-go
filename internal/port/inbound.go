// Package port defines the interfaces (ports) of the hexagonal architecture.
// Inbound ports are implemented by use cases and invoked by primary adapters (e.g. HTTP handlers).
// Outbound ports are implemented by secondary adapters (e.g. vectorizer, KNN searcher)
// and invoked by use cases. Neither port type carries infrastructure concerns.
package port

import (
	"context"

	"anjovisk/fraud-detection/internal/domain"
)

// FraudScoreUseCase is the inbound port that primary adapters use to trigger fraud evaluation.
// It is implemented by usecase.FraudScore and must be satisfied by any alternative use case
// implementation.
type FraudScoreUseCase interface {
	// Evaluate runs the full fraud detection pipeline for the given transaction request.
	//
	// ctx carries deadline and cancellation signals from the caller; it should be forwarded
	// to any blocking outbound calls.
	// req contains all transaction, customer, merchant, and terminal data needed for evaluation.
	//
	// Returns a FraudResult with the approval decision and numeric score.
	// Returns a non-nil error only if evaluation cannot be completed at all; callers should
	// prefer returning a soft approved=true response over propagating the error as HTTP 5xx.
	Evaluate(ctx context.Context, req domain.FraudScoreRequest) (domain.FraudResult, error)
}
