package port

import "anjovisk/fraud-detection/internal/domain"

// Vectorizer is the outbound port for converting a transaction request into a numerical
// feature vector. It is implemented by adapter/vector.Vectorizer.
type Vectorizer interface {
	// Vectorize maps req into a 14-dimensional feature vector (domain.Vector).
	// All dimensions are normalised to [0.0, 1.0] except indices 5 (minutes_since_last_tx)
	// and 6 (km_from_last_tx), which are set to -1 when req.LastTransaction is nil.
	Vectorize(req domain.FraudScoreRequest) domain.Vector
}

// NeighborFinder is the outbound port for nearest-neighbour lookup over the reference dataset.
// It is implemented by adapter/knn.Searcher. Implementations may use exact brute-force search
// or approximate methods (HNSW, VP-Tree, IVF) for better throughput at the cost of accuracy.
type NeighborFinder interface {
	// FindNearest returns the labels ("fraud" or "legit") of the k vectors in the reference
	// dataset that are closest to v under Euclidean distance, ordered arbitrarily.
	// If the dataset contains fewer than k entries, the returned slice has len < k.
	FindNearest(v domain.Vector, k int) []string
}
