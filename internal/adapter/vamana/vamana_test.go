package vamana_test

import (
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/vamana"
	"anjovisk/fraud-detection/internal/domain"
)

var zeroVec domain.Vector

var oneVec = func() domain.Vector {
	var v domain.Vector
	for i := range v {
		v[i] = 1
	}
	return v
}()

var nearZeroVec = func() domain.Vector {
	var v domain.Vector
	for i := range v {
		v[i] = 0.1
	}
	return v
}()

func refs(pairs ...any) []vamana.Reference {
	out := make([]vamana.Reference, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		out = append(out, vamana.Reference{
			Vector: pairs[i].(domain.Vector),
			Label:  pairs[i+1].(string),
		})
	}
	return out
}

func newSearcher(r []vamana.Reference) *vamana.Searcher {
	return vamana.New(r, vamana.DefaultR, vamana.DefaultL, vamana.DefaultAlpha, vamana.DefaultSQ8, zap.NewNop())
}

// TestVamana_SingleRef verifies that a one-entry dataset returns that entry's label.
func TestVamana_SingleRef(t *testing.T) {
	s := newSearcher(refs(zeroVec, "legit"))

	labels := s.FindNearest(zeroVec, 1)

	if len(labels) != 1 {
		t.Fatalf("expected 1 label, got %d", len(labels))
	}
	if labels[0] != "legit" {
		t.Errorf("expected legit, got %q", labels[0])
	}
}

// TestVamana_ChoosesNearest verifies that k=1 returns the geometrically closest entry.
func TestVamana_ChoosesNearest(t *testing.T) {
	s := newSearcher(refs(zeroVec, "legit", oneVec, "fraud"))

	labels := s.FindNearest(nearZeroVec, 1)

	if len(labels) != 1 || labels[0] != "legit" {
		t.Errorf("nearest to nearZeroVec should be legit (zeroVec), got %v", labels)
	}
}

// TestVamana_ReturnsExactlyKLabels verifies the returned slice has exactly k elements.
func TestVamana_ReturnsExactlyKLabels(t *testing.T) {
	r := make([]vamana.Reference, 10)
	for i := range r {
		r[i] = vamana.Reference{Vector: zeroVec, Label: "legit"}
	}
	s := newSearcher(r)

	labels := s.FindNearest(zeroVec, 5)

	if len(labels) != 5 {
		t.Errorf("expected 5 labels, got %d", len(labels))
	}
}

// TestVamana_KGreaterThanDataset verifies k > dataset size returns all available labels.
func TestVamana_KGreaterThanDataset(t *testing.T) {
	s := newSearcher(refs(zeroVec, "legit", oneVec, "fraud"))

	labels := s.FindNearest(zeroVec, 10)

	if len(labels) != 2 {
		t.Errorf("k > dataset size: expected 2 labels, got %d", len(labels))
	}
}

// TestVamana_Top5AreFraud verifies that when the 5 closest are fraud and 5 distant are
// legit, FindNearest(k=5) returns only fraud labels.
func TestVamana_Top5AreFraud(t *testing.T) {
	r := make([]vamana.Reference, 10)
	for i := 0; i < 5; i++ {
		var v domain.Vector
		for j := range v {
			v[j] = float64(i+1) * 0.001
		}
		r[i] = vamana.Reference{Vector: v, Label: "fraud"}
	}
	for i := 5; i < 10; i++ {
		var v domain.Vector
		for j := range v {
			v[j] = 1.0 - float64(i-4)*0.001
		}
		r[i] = vamana.Reference{Vector: v, Label: "legit"}
	}
	s := newSearcher(r)

	labels := s.FindNearest(zeroVec, 5)

	fraudCount := 0
	for _, l := range labels {
		if l == "fraud" {
			fraudCount++
		}
	}
	if fraudCount != 5 {
		t.Errorf("expected 5 fraud labels among top-5, got %d (%v)", fraudCount, labels)
	}
}

// TestVamana_EmptyDataset verifies FindNearest on an empty Searcher returns nil.
func TestVamana_EmptyDataset(t *testing.T) {
	s := newSearcher(nil)

	labels := s.FindNearest(zeroVec, 5)

	if labels != nil {
		t.Errorf("expected nil for empty dataset, got %v", labels)
	}
}
