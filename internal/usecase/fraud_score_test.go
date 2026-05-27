package usecase_test

import (
	"context"
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
	"anjovisk/fraud-detection/internal/usecase"
)

// stubVectorizer implements port.Vectorizer and always returns a fixed zero vector.
type stubVectorizer struct{}

func (s *stubVectorizer) Vectorize(req domain.FraudScoreRequest) domain.Vector {
	return domain.Vector{}
}

// stubNeighborFinder implements port.NeighborFinder and returns a pre-configured label slice.
type stubNeighborFinder struct {
	labels []string
}

func (s *stubNeighborFinder) FindNearest(v domain.Vector, k int) []string {
	return s.labels
}

func TestFraudScore_Evaluate(t *testing.T) {
	tests := []struct {
		name         string
		labels       []string
		wantScore    float64
		wantApproved bool
	}{
		{
			name:         "all legit neighbours yield score 0 and approval",
			labels:       []string{"legit", "legit", "legit", "legit", "legit"},
			wantScore:    0.0,
			wantApproved: true,
		},
		{
			name:         "all fraud neighbours yield score 1 and rejection",
			labels:       []string{"fraud", "fraud", "fraud", "fraud", "fraud"},
			wantScore:    1.0,
			wantApproved: false,
		},
		{
			name:         "2 fraud 3 legit yields score 0.4 and approval",
			labels:       []string{"fraud", "fraud", "legit", "legit", "legit"},
			wantScore:    0.4,
			wantApproved: true,
		},
		{
			name:         "3 fraud 2 legit yields score 0.6 which is not approved (boundary: score must be strictly less than threshold)",
			labels:       []string{"fraud", "fraud", "fraud", "legit", "legit"},
			wantScore:    0.6,
			wantApproved: false,
		},
		{
			name:         "4 fraud 1 legit yields score 0.8 and rejection",
			labels:       []string{"fraud", "fraud", "fraud", "fraud", "legit"},
			wantScore:    0.8,
			wantApproved: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			uc := usecase.NewFraudScore(
				&stubVectorizer{},
				&stubNeighborFinder{labels: tt.labels},
				zap.NewNop(),
			)

			result, err := uc.Evaluate(context.Background(), domain.FraudScoreRequest{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.FraudScore != tt.wantScore {
				t.Errorf("FraudScore = %v, want %v", result.FraudScore, tt.wantScore)
			}
			if result.Approved != tt.wantApproved {
				t.Errorf("Approved = %v, want %v", result.Approved, tt.wantApproved)
			}
		})
	}
}

// TestFraudScore_Evaluate_PropagatesRequest verifies that the domain request is forwarded
// to the Vectorizer unchanged.
func TestFraudScore_Evaluate_PropagatesRequest(t *testing.T) {
	var gotReq domain.FraudScoreRequest
	fv := &funcVectorizer{fn: func(req domain.FraudScoreRequest) domain.Vector {
		gotReq = req
		return domain.Vector{}
	}}

	want := domain.FraudScoreRequest{ID: "tx-capture-test"}
	uc := usecase.NewFraudScore(
		fv,
		&stubNeighborFinder{labels: []string{"legit", "legit", "legit", "legit", "legit"}},
		zap.NewNop(),
	)
	_, err := uc.Evaluate(context.Background(), want)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotReq.ID != want.ID {
		t.Errorf("Vectorizer received request ID %q, want %q", gotReq.ID, want.ID)
	}
}

// funcVectorizer is a port.Vectorizer backed by an arbitrary function, used in tests
// that need to inspect or control Vectorize behaviour beyond what stubVectorizer offers.
type funcVectorizer struct {
	fn func(domain.FraudScoreRequest) domain.Vector
}

func (f *funcVectorizer) Vectorize(req domain.FraudScoreRequest) domain.Vector {
	return f.fn(req)
}
