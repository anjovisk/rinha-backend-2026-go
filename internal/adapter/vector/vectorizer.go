// Package vector implements port.Vectorizer, translating a domain.FraudScoreRequest
// into a 14-dimensional feature vector used for KNN fraud classification.
// Normalisation constants and MCC risk scores are supplied at construction time
// and remain immutable during the lifetime of the Vectorizer.
package vector

import (
	"math"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// Config holds the normalisation constants and risk lookup table required by Vectorizer.
// All values are loaded from resources/normalization.json and resources/mcc_risk.json
// at application startup.
type Config struct {
	// MaxAmount is the divisor used to normalise transaction amounts to [0, 1].
	MaxAmount float64
	// MaxInstallments is the divisor used to normalise the instalment count to [0, 1].
	MaxInstallments float64
	// AmountVsAvgRatio is the scaling factor applied to the amount-vs-customer-average ratio.
	AmountVsAvgRatio float64
	// MaxMinutes is the divisor used to normalise minutes-since-last-transaction to [0, 1].
	MaxMinutes float64
	// MaxKm is the divisor used to normalise distance values (km_from_home, km_from_last) to [0, 1].
	MaxKm float64
	// MaxTxCount24h is the divisor used to normalise the 24-hour transaction count to [0, 1].
	MaxTxCount24h float64
	// MaxMerchantAvgAmount is the divisor used to normalise the merchant's average amount to [0, 1].
	MaxMerchantAvgAmount float64
	// MCCRisk maps a Merchant Category Code to its associated fraud risk score in [0.0, 1.0].
	// MCCs absent from the map default to 0.5.
	MCCRisk map[string]float64
}

// Vectorizer implements port.Vectorizer using the normalisation constants in Config.
type Vectorizer struct {
	cfg    Config
	logger *zap.Logger
}

// New creates a Vectorizer configured with the given constants and logger.
// cfg must be fully populated before the first call to Vectorize.
// logger is used to emit structured logs during vectorization.
func New(cfg Config, logger *zap.Logger) *Vectorizer {
	return &Vectorizer{cfg: cfg, logger: logger}
}

// Vectorize converts req into a 14-dimensional feature vector (domain.Vector).
// Each dimension is described in domain.Vector's documentation. All values are
// clamped to [0.0, 1.0] except dimensions 5 and 6, which are set to -1 when
// req.LastTransaction is nil (sentinel for missing prior transaction).
func (v *Vectorizer) Vectorize(req domain.FraudScoreRequest) domain.Vector {
	v.logger.Debug("starting vectorization", zap.String("transaction_id", req.ID))

	var vec domain.Vector
	t := req.Transaction
	c := req.Customer
	m := req.Merchant
	term := req.Terminal

	vec[0] = clamp(t.Amount / v.cfg.MaxAmount)
	vec[1] = clamp(float64(t.Installments) / v.cfg.MaxInstallments)

	if c.AvgAmount > 0 {
		vec[2] = clamp((t.Amount / c.AvgAmount) / v.cfg.AmountVsAvgRatio)
	} else {
		v.logger.Warn("customer avg_amount is zero, dimension 2 (amount_vs_avg) set to 0.0",
			zap.String("transaction_id", req.ID),
		)
	}

	vec[3] = float64(t.RequestedAt.UTC().Hour()) / 23.0

	// Go's Weekday: Sun=0, Mon=1 … Sat=6. Spec requires Mon=0 … Sun=6.
	wd := int(t.RequestedAt.UTC().Weekday())
	vec[4] = float64((wd+6)%7) / 6.0

	if req.LastTransaction == nil {
		vec[5] = -1
		vec[6] = -1
	} else {
		minutes := t.RequestedAt.Sub(req.LastTransaction.Timestamp).Minutes()
		vec[5] = clamp(minutes / v.cfg.MaxMinutes)
		vec[6] = clamp(req.LastTransaction.KmFromCurrent / v.cfg.MaxKm)
	}

	vec[7] = clamp(term.KmFromHome / v.cfg.MaxKm)
	vec[8] = clamp(float64(c.TxCount24h) / v.cfg.MaxTxCount24h)

	if term.IsOnline {
		vec[9] = 1
	}
	if term.CardPresent {
		vec[10] = 1
	}

	vec[11] = unknownMerchant(m.ID, c.KnownMerchants)

	risk, ok := v.cfg.MCCRisk[m.MCC]
	if !ok {
		v.logger.Debug("MCC not in risk map, using default 0.5",
			zap.String("transaction_id", req.ID),
			zap.String("mcc", m.MCC),
		)
		risk = 0.5
	}
	vec[12] = risk

	vec[13] = clamp(m.AvgAmount / v.cfg.MaxMerchantAvgAmount)

	v.logger.Debug("vectorization complete", zap.String("transaction_id", req.ID))
	return vec
}

// clamp constrains x to the interval [0.0, 1.0].
// Negative values become 0; values above 1 become 1.
func clamp(x float64) float64 {
	return math.Max(0, math.Min(1, x))
}

// unknownMerchant returns 1.0 if id does not appear in known, or 0.0 if it does.
// This encodes whether the merchant is new to the customer's transaction history.
func unknownMerchant(id string, known []string) float64 {
	for _, k := range known {
		if k == id {
			return 0
		}
	}
	return 1
}
