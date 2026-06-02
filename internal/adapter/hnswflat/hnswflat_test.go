package hnswflat_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/hnswflat"
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

func ref(v domain.Vector, label string) hnswflat.Reference {
	return hnswflat.Reference{Vector: v, Label: label}
}

func forBothModes(t *testing.T, f func(t *testing.T, sq8 bool)) {
	t.Helper()
	t.Run("sq8", func(t *testing.T) { f(t, true) })
	t.Run("f32", func(t *testing.T) { f(t, false) })
}

func TestHNSWFlat_SingleRef(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		s := hnswflat.New([]hnswflat.Reference{ref(zeroVec, "legit")}, 3, 10, 20, sq8, zap.NewNop())
		labels := s.FindNearest(zeroVec, 1)
		if len(labels) != 1 || labels[0] != "legit" {
			t.Errorf("expected [legit], got %v", labels)
		}
	})
}

func TestHNSWFlat_ChoosesNearest(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := []hnswflat.Reference{ref(zeroVec, "legit"), ref(oneVec, "fraud")}
		s := hnswflat.New(refs, 2, 10, 20, sq8, zap.NewNop())
		labels := s.FindNearest(nearZeroVec, 1)
		if len(labels) != 1 || labels[0] != "legit" {
			t.Errorf("expected [legit], got %v", labels)
		}
	})
}

func TestHNSWFlat_ReturnsExactlyK(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		refs := make([]hnswflat.Reference, 20)
		for i := range refs {
			refs[i] = ref(zeroVec, "legit")
		}
		s := hnswflat.New(refs, 3, 10, 30, sq8, zap.NewNop())
		labels := s.FindNearest(zeroVec, 5)
		if len(labels) != 5 {
			t.Errorf("expected 5, got %d", len(labels))
		}
	})
}

// writeBin writes a minimal references.bin for the roundtrip test.
func writeBin(t *testing.T, path string, vecs []domain.Vector, labels []uint8) {
	t.Helper()
	var buf bytes.Buffer
	b4 := make([]byte, 4)
	binary.LittleEndian.PutUint32(b4, uint32(len(vecs)))
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

func TestHNSWFlat_BuildOpenRoundtrip(t *testing.T) {
	forBothModes(t, func(t *testing.T, sq8 bool) {
		dir := t.TempDir()
		binFile := dir + "/refs.bin"
		idxFile := dir + "/refs.hnswflat"

		vecs := []domain.Vector{zeroVec, oneVec, nearZeroVec}
		labels := []uint8{0, 1, 0}

		writeBin(t, binFile, vecs, labels)

		if err := hnswflat.Build(binFile, idxFile, 2, 20, sq8, zap.NewNop()); err != nil {
			t.Fatalf("Build: %v", err)
		}

		s, err := hnswflat.Open(idxFile, 10, zap.NewNop())
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		defer s.Close()

		got := s.FindNearest(nearZeroVec, 1)
		if len(got) != 1 || got[0] != "legit" {
			t.Errorf("expected [legit], got %v", got)
		}
	})
}
