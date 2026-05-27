package knn_test

import (
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/knn"
	"anjovisk/fraud-detection/internal/domain"
)

// zeroVec is a 14-dimensional zero vector, used as a stable reference point.
var zeroVec domain.Vector

// oneVec is a 14-dimensional vector with 1.0 in every dimension.
var oneVec = func() domain.Vector {
	var v domain.Vector
	for i := range v {
		v[i] = 1
	}
	return v
}()

// nearZeroVec is a 14-dimensional vector with 0.1 in every dimension,
// which is closer to zeroVec than to oneVec under Euclidean distance.
var nearZeroVec = func() domain.Vector {
	var v domain.Vector
	for i := range v {
		v[i] = 0.1
	}
	return v
}()

// TestFindNearest_SingleRef verifies that a dataset of one entry returns that entry's label.
func TestFindNearest_SingleRef(t *testing.T) {
	s := knn.New([]knn.Reference{{Vector: zeroVec, Label: "legit"}}, zap.NewNop())

	labels := s.FindNearest(zeroVec, 1)

	if len(labels) != 1 {
		t.Fatalf("expected 1 label, got %d", len(labels))
	}
	if labels[0] != "legit" {
		t.Errorf("expected label %q, got %q", "legit", labels[0])
	}
}

// TestFindNearest_ChoosesNearest verifies that when k=1 the returned label belongs
// to the reference geometrically closest to the query vector.
func TestFindNearest_ChoosesNearest(t *testing.T) {
	refs := []knn.Reference{
		{Vector: zeroVec, Label: "legit"}, // dist(nearZeroVec, zeroVec) ≈ 0.37
		{Vector: oneVec, Label: "fraud"},  // dist(nearZeroVec, oneVec)  ≈ 3.37
	}
	s := knn.New(refs, zap.NewNop())

	labels := s.FindNearest(nearZeroVec, 1)

	if len(labels) != 1 || labels[0] != "legit" {
		t.Errorf("nearest to nearZeroVec should be legit (zeroVec), got %v", labels)
	}
}

// TestFindNearest_ReturnsExactlyKLabels verifies that the returned slice has exactly
// k elements when the dataset contains more than k references.
func TestFindNearest_ReturnsExactlyKLabels(t *testing.T) {
	refs := make([]knn.Reference, 10)
	for i := range refs {
		refs[i] = knn.Reference{Vector: zeroVec, Label: "legit"}
	}
	s := knn.New(refs, zap.NewNop())

	labels := s.FindNearest(zeroVec, 5)

	if len(labels) != 5 {
		t.Errorf("expected 5 labels, got %d", len(labels))
	}
}

// TestFindNearest_KGreaterThanDataset verifies that when k exceeds the number of
// references the returned slice contains all available labels (not a panic or empty result).
func TestFindNearest_KGreaterThanDataset(t *testing.T) {
	refs := []knn.Reference{
		{Vector: zeroVec, Label: "legit"},
		{Vector: oneVec, Label: "fraud"},
	}
	s := knn.New(refs, zap.NewNop())

	labels := s.FindNearest(zeroVec, 10) // k=10 > len(refs)=2

	if len(labels) != 2 {
		t.Errorf("k > dataset size: expected 2 labels, got %d", len(labels))
	}
}

// TestFindNearest_Top5AreFraud verifies that when the 5 closest references are all
// labelled "fraud" and 5 far references are "legit", FindNearest(k=5) returns only fraud labels.
func TestFindNearest_Top5AreFraud(t *testing.T) {
	refs := make([]knn.Reference, 10)
	// 5 fraud refs very close to zeroVec
	for i := 0; i < 5; i++ {
		var v domain.Vector
		for j := range v {
			v[j] = float64(i+1) * 0.001
		}
		refs[i] = knn.Reference{Vector: v, Label: "fraud"}
	}
	// 5 legit refs very close to oneVec (far from zeroVec)
	for i := 5; i < 10; i++ {
		var v domain.Vector
		for j := range v {
			v[j] = 1.0 - float64(i-4)*0.001
		}
		refs[i] = knn.Reference{Vector: v, Label: "legit"}
	}
	s := knn.New(refs, zap.NewNop())

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

// TestFindNearest_MixedResult verifies a case where 2 of the 5 nearest are fraud,
// producing the label mix expected by the use case scorer.
func TestFindNearest_MixedResult(t *testing.T) {
	refs := []knn.Reference{
		{Vector: nearZeroVec, Label: "fraud"},  // closest group
		{Vector: nearZeroVec, Label: "fraud"},
		{Vector: nearZeroVec, Label: "legit"},
		{Vector: nearZeroVec, Label: "legit"},
		{Vector: nearZeroVec, Label: "legit"},
		{Vector: oneVec, Label: "fraud"},       // far; should not appear in top-5
	}
	s := knn.New(refs, zap.NewNop())

	labels := s.FindNearest(zeroVec, 5)

	if len(labels) != 5 {
		t.Fatalf("expected 5 labels, got %d", len(labels))
	}

	fraudCount := 0
	for _, l := range labels {
		if l == "fraud" {
			fraudCount++
		}
	}
	if fraudCount != 2 {
		t.Errorf("expected 2 fraud labels in top-5, got %d (%v)", fraudCount, labels)
	}
}
