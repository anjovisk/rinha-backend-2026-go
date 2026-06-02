package partition_test

import (
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/partition"
	"anjovisk/fraud-detection/internal/domain"
)

// makeVec builds a domain.Vector with the given routing feature values and zero elsewhere.
// lastNull=true sets dims 5 and 6 to -1; isOnline and cardPresent set dims 9 and 10.
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

// TestFindNearest_RoutesToMatchingPartition verifies that a query is matched against
// only its own partition. References in a different partition should not be returned
// even when they are geometrically closer in the other dimensions.
func TestFindNearest_RoutesToMatchingPartition(t *testing.T) {
	// Two references in different partitions.
	// The "legit" entry is in the online partition (bin 2), identical to the query.
	// The "fraud" entry is in the offline+card partition (bin 1), far in routing dims.
	refs := []partition.Reference{
		{Vector: makeVec(false, 1, 0), Label: "legit"}, // same partition as query
		{Vector: makeVec(false, 0, 1), Label: "fraud"}, // different partition
	}
	f := partition.New(refs, zap.NewNop())

	query := makeVec(false, 1, 0)
	labels := f.FindNearest(query, 1)

	if len(labels) != 1 {
		t.Fatalf("expected 1 label, got %d", len(labels))
	}
	if labels[0] != "legit" {
		t.Errorf("expected legit (same partition), got %q", labels[0])
	}
}

// TestFindNearest_LastTxNullRouting verifies that queries with sentinel dims 5/6
// are routed to the null-last-tx bins and do not mix with non-null entries.
func TestFindNearest_LastTxNullRouting(t *testing.T) {
	refs := []partition.Reference{
		{Vector: makeVec(true, 0, 1), Label: "fraud"},  // null-tx, offline+card
		{Vector: makeVec(false, 0, 1), Label: "legit"}, // non-null, offline+card
	}
	f := partition.New(refs, zap.NewNop())

	query := makeVec(true, 0, 1)
	labels := f.FindNearest(query, 1)

	if len(labels) != 1 || labels[0] != "fraud" {
		t.Errorf("null-tx query should match null-tx ref (fraud), got %v", labels)
	}
}

// TestFindNearest_KGreaterThanPartition verifies that k is clamped to the partition
// size when fewer entries exist than requested.
func TestFindNearest_KGreaterThanPartition(t *testing.T) {
	refs := []partition.Reference{
		{Vector: makeVec(false, 1, 0), Label: "legit"},
		{Vector: makeVec(false, 1, 0), Label: "fraud"},
		// Third entry in a different partition — should not be returned.
		{Vector: makeVec(false, 0, 1), Label: "legit"},
	}
	f := partition.New(refs, zap.NewNop())

	query := makeVec(false, 1, 0)
	labels := f.FindNearest(query, 10) // k=10 > partition size of 2

	if len(labels) != 2 {
		t.Errorf("expected 2 labels (partition size), got %d", len(labels))
	}
}

// TestFindNearest_EmptyPartitionFallback verifies that a query with a routing
// combination absent from the reference dataset falls back to a full scan and
// still returns k labels.
func TestFindNearest_EmptyPartitionFallback(t *testing.T) {
	// Dataset has no online+card entries (bin 3, which is empty in real data too).
	refs := []partition.Reference{
		{Vector: makeVec(false, 0, 1), Label: "legit"},
		{Vector: makeVec(false, 1, 0), Label: "legit"},
	}
	f := partition.New(refs, zap.NewNop())

	// Query for online+card (bin 3) — not present in refs.
	query := makeVec(false, 1, 1)
	labels := f.FindNearest(query, 1)

	// Should fall back to full scan and return exactly 1 label.
	if len(labels) != 1 {
		t.Errorf("fallback should return 1 label, got %d", len(labels))
	}
}

// TestFindNearest_ChoosesNearestWithinPartition verifies that within a partition
// the closest reference is returned, not an arbitrary one.
func TestFindNearest_ChoosesNearestWithinPartition(t *testing.T) {
	base := makeVec(false, 0, 1) // offline+card partition

	near := base
	near[0] = 0.1 // close to query

	far := base
	far[0] = 0.9 // far from query

	refs := []partition.Reference{
		{Vector: far, Label: "fraud"},
		{Vector: near, Label: "legit"},
	}
	f := partition.New(refs, zap.NewNop())

	query := base // dim0 = 0
	labels := f.FindNearest(query, 1)

	if len(labels) != 1 || labels[0] != "legit" {
		t.Errorf("nearest within partition should be legit, got %v", labels)
	}
}

// TestFindNearest_AllSixBinsPresent verifies correct routing when all six real-world
// partition bins are populated simultaneously.
func TestFindNearest_AllSixBinsPresent(t *testing.T) {
	combos := []struct {
		lastNull    bool
		online      int
		card        int
		wantLabel   string
	}{
		{false, 0, 0, "legit"},
		{false, 0, 1, "fraud"},
		{false, 1, 0, "legit"},
		{true, 0, 0, "fraud"},
		{true, 0, 1, "legit"},
		{true, 1, 0, "fraud"},
	}

	refs := make([]partition.Reference, len(combos))
	for i, c := range combos {
		refs[i] = partition.Reference{Vector: makeVec(c.lastNull, c.online, c.card), Label: c.wantLabel}
	}
	f := partition.New(refs, zap.NewNop())

	for _, c := range combos {
		query := makeVec(c.lastNull, c.online, c.card)
		labels := f.FindNearest(query, 1)
		if len(labels) != 1 || labels[0] != c.wantLabel {
			t.Errorf("bin(lastNull=%v,online=%d,card=%d): want %q, got %v",
				c.lastNull, c.online, c.card, c.wantLabel, labels)
		}
	}
}
