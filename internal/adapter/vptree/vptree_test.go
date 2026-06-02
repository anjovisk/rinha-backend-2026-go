package vptree_test

import (
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/vptree"
	"anjovisk/fraud-detection/internal/domain"
)

// makeVec returns a zero vector with the given routing feature values set.
func makeVec(lastNull bool, isOnline, cardPresent int) domain.Vector {
	var v domain.Vector
	if lastNull {
		v[5] = -1
		v[6] = -1
	}
	v[9] = float64(isOnline)
	v[10] = float64(cardPresent)
	return v
}

// TestFindNearest_RoutesToMatchingBin verifies that the VP-Tree only considers
// entries in the matching bin, not entries from other bins.
func TestFindNearest_RoutesToMatchingBin(t *testing.T) {
	refs := []vptree.Reference{
		{Vector: makeVec(false, 1, 0), Label: "legit"}, // same bin as query (online)
		{Vector: makeVec(false, 0, 1), Label: "fraud"}, // different bin (offline+card)
	}
	f := vptree.New(refs, zap.NewNop())

	labels := f.FindNearest(makeVec(false, 1, 0), 1)

	if len(labels) != 1 || labels[0] != "legit" {
		t.Errorf("expected legit (same bin), got %v", labels)
	}
}

// TestFindNearest_LastTxNullRouting verifies correct routing when dims 5/6 are -1.
func TestFindNearest_LastTxNullRouting(t *testing.T) {
	refs := []vptree.Reference{
		{Vector: makeVec(true, 0, 1), Label: "fraud"},  // null-tx, offline+card
		{Vector: makeVec(false, 0, 1), Label: "legit"}, // non-null, offline+card
	}
	f := vptree.New(refs, zap.NewNop())

	labels := f.FindNearest(makeVec(true, 0, 1), 1)

	if len(labels) != 1 || labels[0] != "fraud" {
		t.Errorf("null-tx query should match null-tx ref, got %v", labels)
	}
}

// TestFindNearest_ExactKNN verifies that the VP-Tree returns the geometrically
// nearest entry within its bin, not an arbitrary one.
func TestFindNearest_ExactKNN(t *testing.T) {
	base := makeVec(false, 0, 1) // offline+card bin

	near := base
	near[0] = 0.1 // close to query (dim 0 = 0)

	far := base
	far[0] = 0.9 // far from query

	refs := []vptree.Reference{
		{Vector: far, Label: "fraud"},
		{Vector: near, Label: "legit"},
	}
	f := vptree.New(refs, zap.NewNop())

	labels := f.FindNearest(base, 1)

	if len(labels) != 1 || labels[0] != "legit" {
		t.Errorf("nearest within bin should be legit, got %v", labels)
	}
}

// TestFindNearest_Top5ExactWithLargeBin verifies that the VP-Tree returns the correct
// top-k results when the bin contains more entries than k, including cases where the
// tree must search multiple branches to confirm the k-th best.
func TestFindNearest_Top5ExactWithLargeBin(t *testing.T) {
	base := makeVec(false, 0, 1)

	// 5 entries very close to query; 5 entries far from query.
	refs := make([]vptree.Reference, 10)
	for i := 0; i < 5; i++ {
		v := base
		v[0] = float64(i+1) * 0.001 // dim 0 ≈ 0 → close
		refs[i] = vptree.Reference{Vector: v, Label: "fraud"}
	}
	for i := 5; i < 10; i++ {
		v := base
		v[0] = 1.0 - float64(i-4)*0.001 // dim 0 ≈ 1 → far
		refs[i] = vptree.Reference{Vector: v, Label: "legit"}
	}
	f := vptree.New(refs, zap.NewNop())

	labels := f.FindNearest(base, 5)

	fraudCount := 0
	for _, l := range labels {
		if l == "fraud" {
			fraudCount++
		}
	}
	if fraudCount != 5 {
		t.Errorf("expected 5 fraud in top-5, got %d (%v)", fraudCount, labels)
	}
}

// TestFindNearest_KGreaterThanBin verifies k is clamped to the bin size.
func TestFindNearest_KGreaterThanBin(t *testing.T) {
	refs := []vptree.Reference{
		{Vector: makeVec(false, 1, 0), Label: "legit"},
		{Vector: makeVec(false, 1, 0), Label: "fraud"},
		{Vector: makeVec(false, 0, 1), Label: "legit"}, // different bin — excluded
	}
	f := vptree.New(refs, zap.NewNop())

	labels := f.FindNearest(makeVec(false, 1, 0), 10)

	if len(labels) != 2 {
		t.Errorf("expected 2 labels (bin size), got %d", len(labels))
	}
}

// TestFindNearest_EmptyBinFallback verifies that a query hitting an empty bin
// (online+card, which never appears in the real dataset) falls back to a full scan.
func TestFindNearest_EmptyBinFallback(t *testing.T) {
	refs := []vptree.Reference{
		{Vector: makeVec(false, 0, 1), Label: "legit"},
		{Vector: makeVec(false, 1, 0), Label: "legit"},
	}
	f := vptree.New(refs, zap.NewNop())

	// online+card (bin 3) — not present in refs.
	labels := f.FindNearest(makeVec(false, 1, 1), 1)

	if len(labels) != 1 {
		t.Errorf("fallback should return 1 label, got %d", len(labels))
	}
}

// TestFindNearest_AllSixBins verifies correct routing across all 6 real-world bins.
func TestFindNearest_AllSixBins(t *testing.T) {
	combos := []struct {
		lastNull  bool
		online    int
		card      int
		wantLabel string
	}{
		{false, 0, 0, "legit"},
		{false, 0, 1, "fraud"},
		{false, 1, 0, "legit"},
		{true, 0, 0, "fraud"},
		{true, 0, 1, "legit"},
		{true, 1, 0, "fraud"},
	}

	refs := make([]vptree.Reference, len(combos))
	for i, c := range combos {
		refs[i] = vptree.Reference{
			Vector: makeVec(c.lastNull, c.online, c.card),
			Label:  c.wantLabel,
		}
	}
	f := vptree.New(refs, zap.NewNop())

	for _, c := range combos {
		labels := f.FindNearest(makeVec(c.lastNull, c.online, c.card), 1)
		if len(labels) != 1 || labels[0] != c.wantLabel {
			t.Errorf("bin(lastNull=%v,online=%d,card=%d): want %q, got %v",
				c.lastNull, c.online, c.card, c.wantLabel, labels)
		}
	}
}

// TestFindNearest_PruningCorrectness verifies that tree pruning never misses a true
// nearest neighbour. Creates a bin with many entries spread across different distances
// and confirms the result matches a linear scan.
func TestFindNearest_PruningCorrectness(t *testing.T) {
	base := makeVec(false, 0, 1)
	const n = 200

	refs := make([]vptree.Reference, n)
	// Nearest entry: dim 0 = 0.01, all others zero.
	refs[0] = vptree.Reference{Vector: base, Label: "fraud"}
	refs[0].Vector[0] = 0.01

	for i := 1; i < n; i++ {
		v := base
		v[0] = float64(i) * 0.005
		v[1] = float64(i%7) * 0.1
		v[2] = float64(i%11) * 0.09
		label := "legit"
		if i%3 == 0 {
			label = "fraud"
		}
		refs[i] = vptree.Reference{Vector: v, Label: label}
	}

	f := vptree.New(refs, zap.NewNop())

	query := base // dim 0 = 0; nearest is refs[0] with dim 0 = 0.01
	labels := f.FindNearest(query, 1)

	if len(labels) != 1 || labels[0] != "fraud" {
		t.Errorf("pruning missed nearest neighbour: want fraud, got %v", labels)
	}
}
