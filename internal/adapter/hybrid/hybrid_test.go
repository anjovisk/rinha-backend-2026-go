package hybrid_test

import (
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/hybrid"
	"anjovisk/fraud-detection/internal/domain"
)

// stubFinder is a port.NeighborFinder that always returns a fixed label slice.
type stubFinder struct {
	labels []string
	calls  int
}

func (s *stubFinder) FindNearest(_ domain.Vector, _ int) []string {
	s.calls++
	return s.labels
}

// TestUnambiguousScore_Usesfast verifies that scores of 0.0 and 1.0 (all legit or all
// fraud) are returned from the fast searcher without invoking the exact fallback.
func TestUnambiguousScore_UsesFast(t *testing.T) {
	for _, tc := range []struct {
		name   string
		labels []string
	}{
		{"all legit (score 0.0)", []string{"legit", "legit", "legit", "legit", "legit"}},
		{"all fraud (score 1.0)", []string{"fraud", "fraud", "fraud", "fraud", "fraud"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fast := &stubFinder{labels: tc.labels}
			exact := &stubFinder{labels: []string{"legit", "legit", "legit", "legit", "legit"}}
			s := hybrid.New(fast, exact, zap.NewNop())

			got := s.FindNearest(domain.Vector{}, 5)

			if exact.calls != 0 {
				t.Errorf("exact searcher called %d times, want 0", exact.calls)
			}
			if len(got) != len(tc.labels) {
				t.Errorf("len(labels) = %d, want %d", len(got), len(tc.labels))
			}
		})
	}
}

// TestBorderlineScore_UsesExact verifies that scores strictly between 0.0 and 1.0
// (i.e. 0.2, 0.4, 0.6, 0.8) trigger the exact fallback.
func TestBorderlineScore_UsesExact(t *testing.T) {
	for _, tc := range []struct {
		name       string
		fastLabels []string
	}{
		{"score 0.2 (1 fraud)", []string{"fraud", "legit", "legit", "legit", "legit"}},
		{"score 0.4 (2 frauds)", []string{"fraud", "fraud", "legit", "legit", "legit"}},
		{"score 0.6 (3 frauds)", []string{"fraud", "fraud", "fraud", "legit", "legit"}},
		{"score 0.8 (4 frauds)", []string{"fraud", "fraud", "fraud", "fraud", "legit"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exactLabels := []string{"legit", "legit", "legit", "legit", "legit"}
			fast := &stubFinder{labels: tc.fastLabels}
			exact := &stubFinder{labels: exactLabels}
			s := hybrid.New(fast, exact, zap.NewNop())

			got := s.FindNearest(domain.Vector{}, 5)

			if exact.calls != 1 {
				t.Errorf("exact searcher called %d times, want 1", exact.calls)
			}
			if len(got) != len(exactLabels) {
				t.Errorf("len(labels) = %d, want %d", len(got), len(exactLabels))
			}
			for i, l := range got {
				if l != exactLabels[i] {
					t.Errorf("label[%d] = %q, want %q", i, l, exactLabels[i])
				}
			}
		})
	}
}
