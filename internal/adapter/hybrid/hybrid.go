// Package hybrid implements port.NeighborFinder by composing a fast approximate
// searcher (hnswflat) with an exact fallback (partition).
//
// Strategy: run the fast searcher first. If the resulting fraud score is strictly
// between 0.0 and 1.0, the query is "borderline" — a single misclassified neighbour
// could shift the decision across the 0.6 threshold — so the exact searcher is used
// instead. Scores 0.0 and 1.0 are safe: flipping the decision would require
// misclassifying at least 3 of 5 neighbours simultaneously, which is negligible at
// typical recall levels.
//
// This gives exact decision accuracy for all but the most extreme neighbour-recall
// failures, while keeping median latency close to the fast searcher.
//
// Enable by setting VECTOR_SEARCHER=hybrid at server startup.
// Requires references.hnswflat (BUILD_HNSWFLAT=true) and references.bin.
package hybrid

import (
	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
	"anjovisk/fraud-detection/internal/port"
)

// Searcher composes a fast approximate finder with an exact fallback.
// For borderline fraud scores the exact finder is used; for unambiguous
// scores (0.0 or 1.0) the fast result is returned directly.
type Searcher struct {
	fast   port.NeighborFinder
	exact  port.NeighborFinder
	logger *zap.Logger
}

// New builds a Searcher that wraps fast (hnswflat) and exact (partition).
// Both finders must already be initialised and ready to serve queries.
func New(fast, exact port.NeighborFinder, logger *zap.Logger) *Searcher {
	return &Searcher{fast: fast, exact: exact, logger: logger}
}

// FindNearest returns the k nearest labels, using the fast searcher when the
// result is unambiguous and the exact searcher for all borderline scores.
// A score is borderline when at least one misclassified neighbour could change
// the approved/rejected decision (i.e. fraud_score ∈ (0.0, 1.0)).
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	labels := s.fast.FindNearest(v, k)

	frauds := 0
	for _, l := range labels {
		if l == "fraud" {
			frauds++
		}
	}

	// score == 0.0 (all legit) or 1.0 (all fraud): safe to return fast result.
	if frauds == 0 || frauds == len(labels) {
		s.logger.Debug("hybrid: fast result accepted (unambiguous score)",
			zap.Int("frauds", frauds), zap.Int("k", len(labels)))
		return labels
	}

	s.logger.Debug("hybrid: borderline score, falling back to exact search",
		zap.Int("frauds", frauds), zap.Int("k", len(labels)))
	return s.exact.FindNearest(v, k)
}
