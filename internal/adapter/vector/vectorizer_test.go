package vector_test

import (
	"math"
	"testing"
	"time"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/vector"
	"anjovisk/fraud-detection/internal/domain"
)

// wednesday20h is 2026-03-11T20:00:00Z — a Wednesday at hour 20 UTC.
// Used as a stable timestamp across vectorizer tests.
var wednesday20h = time.Date(2026, 3, 11, 20, 0, 0, 0, time.UTC)

// baseConfig returns the normalisation constants mirroring resources/normalization.json.
func baseConfig() vector.Config {
	return vector.Config{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
		MCCRisk:              map[string]float64{"5411": 0.15, "5812": 0.30},
	}
}

// approxEqual reports whether |a-b| <= tol.
func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// TestVectorize_AllDimensions verifies every dimension index against the formula
// specified in domain.Vector's documentation for a request with no prior transaction.
func TestVectorize_AllDimensions(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	req := domain.FraudScoreRequest{
		Transaction: domain.Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  wednesday20h, // hour=20, weekday=Wed(Go=3 → adjusted=2)
		},
		Customer: domain.Customer{
			AvgAmount:      200,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-KNOWN"},
		},
		Merchant: domain.Merchant{
			ID:        "MERC-UNKNOWN",
			MCC:       "5411",
			AvgAmount: 100,
		},
		Terminal: domain.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  50,
		},
		LastTransaction: nil,
	}

	vec := v.Vectorize(req)

	cases := []struct {
		idx  int
		name string
		want float64
	}{
		{0, "amount", 100.0 / 10000},               // 0.01
		{1, "installments", 1.0 / 12},               // ≈0.0833
		{2, "amount_vs_avg", (100.0 / 200) / 10},   // 0.05
		{3, "hour_of_day", 20.0 / 23},               // ≈0.8696
		{4, "day_of_week", 2.0 / 6},                 // Wed→adjusted=2, ≈0.3333
		{5, "minutes_since_last", -1},                // sentinel: no LastTransaction
		{6, "km_from_last", -1},                      // sentinel: no LastTransaction
		{7, "km_from_home", 50.0 / 1000},            // 0.05
		{8, "tx_count_24h", 3.0 / 20},               // 0.15
		{9, "is_online", 0},
		{10, "card_present", 1},
		{11, "unknown_merchant", 1},                  // MERC-UNKNOWN not in KnownMerchants
		{12, "mcc_risk", 0.15},                       // MCC 5411
		{13, "merchant_avg_amount", 100.0 / 10000},  // 0.01
	}

	for _, c := range cases {
		if !approxEqual(vec[c.idx], c.want, 1e-9) {
			t.Errorf("dim[%d] (%s) = %v, want %v", c.idx, c.name, vec[c.idx], c.want)
		}
	}
}

// TestVectorize_WithLastTransaction verifies that dimensions 5 and 6 contain computed
// values (not -1 sentinels) when LastTransaction is present.
func TestVectorize_WithLastTransaction(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	// last_transaction is 6 hours before requested_at → 360 minutes
	lastTxTime := wednesday20h.Add(-6 * time.Hour)
	req := domain.FraudScoreRequest{
		Transaction: domain.Transaction{Amount: 1, RequestedAt: wednesday20h},
		Customer:    domain.Customer{AvgAmount: 1},
		LastTransaction: &domain.LastTransaction{
			Timestamp:     lastTxTime,
			KmFromCurrent: 50,
		},
	}

	vec := v.Vectorize(req)

	wantMinutes := 360.0 / 1440 // 0.25
	if !approxEqual(vec[5], wantMinutes, 1e-9) {
		t.Errorf("dim[5] minutes_since_last = %v, want %v", vec[5], wantMinutes)
	}
	wantKm := 50.0 / 1000 // 0.05
	if !approxEqual(vec[6], wantKm, 1e-9) {
		t.Errorf("dim[6] km_from_last = %v, want %v", vec[6], wantKm)
	}
}

// TestVectorize_ClampOverflow verifies that values exceeding the normalisation maximum
// are clamped to 1.0 rather than producing values above 1.
func TestVectorize_ClampOverflow(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	req := domain.FraudScoreRequest{
		Transaction: domain.Transaction{
			Amount:       999_999, // far exceeds MaxAmount=10000
			Installments: 999,    // far exceeds MaxInstallments=12
			RequestedAt:  wednesday20h,
		},
		Customer: domain.Customer{AvgAmount: 1},
	}

	vec := v.Vectorize(req)

	if vec[0] != 1.0 {
		t.Errorf("dim[0] amount: overflow should clamp to 1.0, got %v", vec[0])
	}
	if vec[1] != 1.0 {
		t.Errorf("dim[1] installments: overflow should clamp to 1.0, got %v", vec[1])
	}
}

// TestVectorize_KnownMerchant verifies that dim[11] is 0 when the merchant ID
// appears in the customer's known merchant list.
func TestVectorize_KnownMerchant(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	req := domain.FraudScoreRequest{
		Transaction: domain.Transaction{Amount: 1, RequestedAt: wednesday20h},
		Customer: domain.Customer{
			AvgAmount:      1,
			KnownMerchants: []string{"MERC-A", "MERC-B"},
		},
		Merchant: domain.Merchant{ID: "MERC-A"},
	}

	if v.Vectorize(req)[11] != 0.0 {
		t.Error("dim[11] known merchant should be 0.0")
	}
}

// TestVectorize_UnknownMCC_DefaultsToHalf verifies that dim[12] defaults to 0.5
// when the merchant's MCC is not present in the MCCRisk map.
func TestVectorize_UnknownMCC_DefaultsToHalf(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	req := domain.FraudScoreRequest{
		Transaction: domain.Transaction{Amount: 1, RequestedAt: wednesday20h},
		Customer:    domain.Customer{AvgAmount: 1},
		Merchant:    domain.Merchant{MCC: "9999"}, // absent from MCCRisk
	}

	if v.Vectorize(req)[12] != 0.5 {
		t.Errorf("dim[12] unknown MCC should default to 0.5, got %v", v.Vectorize(req)[12])
	}
}

// TestVectorize_Weekday verifies the Mon=0 … Sun=6 mapping required by the spec,
// which differs from Go's native Sunday=0 … Saturday=6 representation.
func TestVectorize_Weekday(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	makeVec := func(ts time.Time) domain.Vector {
		return v.Vectorize(domain.FraudScoreRequest{
			Transaction: domain.Transaction{Amount: 1, RequestedAt: ts},
			Customer:    domain.Customer{AvgAmount: 1},
		})
	}

	// 2026-03-09 is a Monday → spec index 0 → dim[4] = 0/6 = 0.0
	monday := time.Date(2026, 3, 9, 0, 0, 0, 0, time.UTC)
	if makeVec(monday)[4] != 0.0 {
		t.Errorf("Monday day_of_week = %v, want 0.0", makeVec(monday)[4])
	}

	// 2026-03-15 is a Sunday → spec index 6 → dim[4] = 6/6 = 1.0
	sunday := time.Date(2026, 3, 15, 0, 0, 0, 0, time.UTC)
	if makeVec(sunday)[4] != 1.0 {
		t.Errorf("Sunday day_of_week = %v, want 1.0", makeVec(sunday)[4])
	}
}

// TestVectorize_IsOnlineFlag verifies that dim[9] is 1 for online transactions
// and 0 for in-person transactions.
func TestVectorize_IsOnlineFlag(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	makeVec := func(isOnline bool) domain.Vector {
		return v.Vectorize(domain.FraudScoreRequest{
			Transaction: domain.Transaction{Amount: 1, RequestedAt: wednesday20h},
			Customer:    domain.Customer{AvgAmount: 1},
			Terminal:    domain.Terminal{IsOnline: isOnline},
		})
	}

	if makeVec(true)[9] != 1.0 {
		t.Error("dim[9] is_online=true should be 1.0")
	}
	if makeVec(false)[9] != 0.0 {
		t.Error("dim[9] is_online=false should be 0.0")
	}
}

// TestVectorize_CardPresentFlag verifies that dim[10] is 1 when the card is physically
// present and 0 when it is not.
func TestVectorize_CardPresentFlag(t *testing.T) {
	v := vector.New(baseConfig(), zap.NewNop())

	makeVec := func(present bool) domain.Vector {
		return v.Vectorize(domain.FraudScoreRequest{
			Transaction: domain.Transaction{Amount: 1, RequestedAt: wednesday20h},
			Customer:    domain.Customer{AvgAmount: 1},
			Terminal:    domain.Terminal{CardPresent: present},
		})
	}

	if makeVec(true)[10] != 1.0 {
		t.Error("dim[10] card_present=true should be 1.0")
	}
	if makeVec(false)[10] != 0.0 {
		t.Error("dim[10] card_present=false should be 0.0")
	}
}
