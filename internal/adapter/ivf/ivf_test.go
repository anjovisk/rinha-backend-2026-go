package ivf_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/ivf"
	"anjovisk/fraud-detection/internal/domain"
)

// zeroVec is a 14-dimensional zero vector used as a stable reference point.
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
// closer to zeroVec than to oneVec under Euclidean distance.
var nearZeroVec = func() domain.Vector {
	var v domain.Vector
	for i := range v {
		v[i] = 0.1
	}
	return v
}()

// ref is a helper to build an ivf.Reference.
func ref(v domain.Vector, label string) ivf.Reference {
	return ivf.Reference{Vector: v, Label: label}
}

// forBothModes runs f twice — once with sq8=true, once with sq8=false.
func forBothModes(t *testing.T, f func(t *testing.T, sq8 bool)) {
	t.Helper()
	t.Run("sq8", func(t *testing.T) { f(t, true) })
	t.Run("f32", func(t *testing.T) { f(t, false) })
}

// TestIVF_SingleRef verifies that a one-entry dataset returns that entry's label.
func TestIVF_SingleRef(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		s := ivf.New([]ivf.Reference{ref(zeroVec, "legit")}, 1, 1, sq8, zap.NewNop())

		labels := s.FindNearest(zeroVec, 1)

		if len(labels) != 1 {
			t.Fatalf("expected 1 label, got %d", len(labels))
		}
		if labels[0] != "legit" {
			t.Errorf("expected %q, got %q", "legit", labels[0])
		}
	})
}

// TestIVF_ChoosesNearest verifies that k=1 returns the label of the geometrically closest entry.
func TestIVF_ChoosesNearest(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := []ivf.Reference{
			ref(zeroVec, "legit"),
			ref(oneVec, "fraud"),
		}
		s := ivf.New(refs, 1, 1, sq8, zap.NewNop())

		labels := s.FindNearest(nearZeroVec, 1)

		if len(labels) != 1 || labels[0] != "legit" {
			t.Errorf("nearest to nearZeroVec should be legit (zeroVec), got %v", labels)
		}
	})
}

// TestIVF_ReturnsExactlyKLabels verifies that the returned slice has exactly k elements
// when the dataset contains more references than k.
func TestIVF_ReturnsExactlyKLabels(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := make([]ivf.Reference, 10)
		for i := range refs {
			refs[i] = ref(zeroVec, "legit")
		}
		s := ivf.New(refs, 1, 1, sq8, zap.NewNop())

		labels := s.FindNearest(zeroVec, 5)

		if len(labels) != 5 {
			t.Errorf("expected 5 labels, got %d", len(labels))
		}
	})
}

// TestIVF_KGreaterThanDataset verifies that when k exceeds the dataset size the
// returned slice contains all available labels without panicking.
func TestIVF_KGreaterThanDataset(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := []ivf.Reference{
			ref(zeroVec, "legit"),
			ref(oneVec, "fraud"),
		}
		s := ivf.New(refs, 1, 1, sq8, zap.NewNop())

		labels := s.FindNearest(zeroVec, 10)

		if len(labels) != 2 {
			t.Errorf("k > dataset size: expected 2 labels, got %d", len(labels))
		}
	})
}

// writeBin writes a minimal references.bin to path with the given entries.
// Format: [uint32 N][N×VectorSize×float32 vecs][N×uint8 labels].
func writeBin(t *testing.T, path string, vecs []domain.Vector, labels []uint8) {
	t.Helper()
	n := len(vecs)
	var buf bytes.Buffer
	b4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(b4, uint32(n))
	buf.Write(b4)
	for _, v := range vecs {
		for _, dim := range v {
			binary.LittleEndian.PutUint32(b4, math.Float32bits(float32(dim)))
			buf.Write(b4)
		}
	}
	buf.Write(labels)
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestIVF_BuildOpenRoundtrip verifies that Build writes the flat format and Open reads it back
// correctly via mmap, returning results consistent with the in-memory New constructor.
func TestIVF_BuildOpenRoundtrip(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		dir := t.TempDir()
		binFile := dir + "/refs.bin"
		ivfFile := dir + "/refs.ivf"

		vecs := []domain.Vector{zeroVec, oneVec, nearZeroVec}
		labels := []uint8{0, 1, 0} // legit, fraud, legit

		writeBin(t, binFile, vecs, labels)

		if err := ivf.Build(binFile, ivfFile, 1, sq8, zap.NewNop()); err != nil {
			t.Fatalf("Build: %v", err)
		}

		s, err := ivf.Open(ivfFile, 1, zap.NewNop())
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer s.Close()

		// nearZeroVec is closest to zeroVec (legit), not oneVec (fraud).
		got := s.FindNearest(nearZeroVec, 1)
		if len(got) != 1 || got[0] != "legit" {
			t.Errorf("expected [legit], got %v", got)
		}
	})
}

// TestIVF_Top5AreFraud verifies that when the 5 closest references are all "fraud"
// and 5 distant references are "legit", FindNearest(k=5) returns only fraud labels.
func TestIVF_Top5AreFraud(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := make([]ivf.Reference, 10)
		for i := 0; i < 5; i++ {
			var v domain.Vector
			for j := range v {
				v[j] = float64(i+1) * 0.001
			}
			refs[i] = ref(v, "fraud")
		}
		for i := 5; i < 10; i++ {
			var v domain.Vector
			for j := range v {
				v[j] = 1.0 - float64(i-4)*0.001
			}
			refs[i] = ref(v, "legit")
		}
		s := ivf.New(refs, 1, 1, sq8, zap.NewNop())

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
	})
}
