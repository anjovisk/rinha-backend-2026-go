package domain

// FraudThreshold is the exclusive upper bound on FraudScore for a transaction to be approved.
// A transaction is approved when FraudScore < FraudThreshold (i.e. strictly less than 0.6).
const FraudThreshold = 0.6

// FraudResult is the output of the fraud evaluation use case.
// It encodes both the human-readable decision and the underlying numeric score.
type FraudResult struct {
	// Approved is true when FraudScore is strictly below FraudThreshold (0.6).
	Approved bool
	// FraudScore is the proportion of fraudulent labels among the k nearest neighbours,
	// expressed as a float in [0.0, 1.0]. A score of 0.0 means all neighbours are legit;
	// 1.0 means all are fraud.
	FraudScore float64
}
