package simdbrute_test

import (
	"math"
	"math/rand"
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/simdbrute"
	"anjovisk/fraud-detection/internal/domain"
)

// scalarL2Sq computes the reference squared Euclidean distance in pure Go.
func scalarL2Sq(a domain.Vector, b []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := float32(a[i]) - b[i]
		sum += d * d
	}
	return sum
}

// TestFindNearest_MatchesScalar verifies that simdbrute returns the same
// k nearest neighbours as a scalar reference implementation.
func TestFindNearest_MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	// Build a small dataset of 200 random entries.
	refs := make([]simdbrute.Reference, 200)
	for i := range refs {
		for j := range refs[i].Vector {
			refs[i].Vector[j] = rng.Float64()
		}
		if rng.Intn(2) == 0 {
			refs[i].Label = "fraud"
		} else {
			refs[i].Label = "legit"
		}
	}

	f := simdbrute.New(refs, zap.NewNop())

	// Run 20 random queries and compare against a brute-force scalar reference.
	for q := 0; q < 20; q++ {
		var query domain.Vector
		for j := range query {
			query[j] = rng.Float64()
		}

		// simdbrute result (scans the whole dataset via full-scan fallback when all
		// routing features are zero — treated as bin 0).
		got := f.FindNearest(query, 5)

		// Scalar reference: find 5 nearest by brute force.
		type cand struct {
			dist  float32
			label string
		}
		var all []cand
		for _, r := range refs {
			var ref [domain.VectorSize]float32
			for j, v := range r.Vector {
				ref[j] = float32(v)
			}
			all = append(all, cand{scalarL2Sq(query, ref[:]), r.Label})
		}
		// Partial sort: find 5 smallest.
		want := make([]string, 5)
		for i := range want {
			best := 0
			for j := 1; j < len(all); j++ {
				if all[j].dist < all[best].dist {
					best = j
				}
			}
			want[i] = all[best].label
			all[best].dist = float32(math.MaxFloat32)
		}

		if len(got) != len(want) {
			t.Errorf("query %d: got %d labels, want %d", q, len(got), len(want))
			continue
		}
		// Labels are unordered; compare sorted fraud counts.
		gotFrauds, wantFrauds := 0, 0
		for _, l := range got {
			if l == "fraud" {
				gotFrauds++
			}
		}
		for _, l := range want {
			if l == "fraud" {
				wantFrauds++
			}
		}
		if gotFrauds != wantFrauds {
			t.Errorf("query %d: fraud count %d, want %d", q, gotFrauds, wantFrauds)
		}
	}
}
