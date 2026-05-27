// Package knn implements port.NeighborFinder using brute-force squared-Euclidean
// nearest-neighbour search over a pre-loaded reference dataset.
//
// The reference dataset is stored in a compact SoA (Struct-of-Arrays) layout:
// all feature vectors as a contiguous []float32 followed by all labels as []uint8.
// At runtime the file is memory-mapped so both API instances share the same physical
// pages in the OS page cache, keeping per-process RSS well within budget.
//
// The search scans all N references in O(N·k) time using a fixed-size candidate buffer
// (no full-dataset sort). Squared Euclidean distance is used throughout; the square root
// is never taken because only relative ordering matters.
package knn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// Index holds the reference dataset in a SoA flat layout suitable for cache-friendly KNN.
// Vectors and labels may be backed by a memory-mapped file or by a plain Go slice.
type Index struct {
	// n is the number of reference entries.
	n int
	// vectors holds all feature vectors contiguously: entry i occupies [i*VectorSize, (i+1)*VectorSize).
	vectors []float32
	// labels holds the fraud/legit classification for each entry: 1=fraud, 0=legit.
	labels []uint8
	// mmapped is non-nil when vectors and labels point into a mmap region.
	// It is held to prevent GC from dropping the backing memory and to allow Close to munmap.
	mmapped []byte
	// logger is the named zap logger injected at construction time.
	logger *zap.Logger
}

// Reference is a labelled dataset entry used to construct an Index from Go values.
// It is intended for tests; production code should use Open.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is the fraud classification: "fraud" or "legit".
	Label string
}

// New builds an Index from a slice of Reference entries.
// This constructor allocates plain Go slices and is used primarily in tests.
// For production use, call Open to memory-map a pre-processed binary file.
func New(refs []Reference, logger *zap.Logger) *Index {
	n := len(refs)
	vectors := make([]float32, n*domain.VectorSize)
	labels := make([]uint8, n)
	for i, r := range refs {
		base := i * domain.VectorSize
		for j, v := range r.Vector {
			vectors[base+j] = float32(v)
		}
		if r.Label == "fraud" {
			labels[i] = 1
		}
	}
	return &Index{n: n, vectors: vectors, labels: labels, logger: logger}
}

// Open memory-maps the binary reference index at path and returns an Index backed by it.
// The binary format is produced by cmd/preprocess:
//
//	[4 bytes uint32 LE]  number of entries N
//	[N × 56 bytes]       float32 LE feature vectors, row-major
//	[N × 1 byte]         uint8 labels, 1=fraud 0=legit
//
// The mmap region is shared (MAP_SHARED | PROT_READ). Both API instances that map the
// same file share physical pages in the OS page cache, avoiding duplication in RAM.
// Call Close when the Index is no longer needed to release the mmap region.
func Open(path string, logger *zap.Logger) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	fileSize := int(info.Size())
	if fileSize < 4 {
		return nil, fmt.Errorf("%s: file too small (%d bytes)", path, fileSize)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	n := int(binary.LittleEndian.Uint32(data[:4]))

	vectorBytes := n * domain.VectorSize * 4
	labelBytes := n
	required := 4 + vectorBytes + labelBytes
	if fileSize < required {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: need %d bytes for %d entries, have %d", path, required, n, fileSize)
	}

	// Reinterpret the vector region as []float32 without copying.
	vectorPtr := (*float32)(unsafe.Pointer(&data[4]))
	vectors := unsafe.Slice(vectorPtr, n*domain.VectorSize)

	labels := data[4+vectorBytes : 4+vectorBytes+labelBytes]

	logger.Debug("reference index opened",
		zap.Int("entries", n),
		zap.Int("file_bytes", fileSize),
	)

	return &Index{
		n:       n,
		vectors: vectors,
		labels:  labels,
		mmapped: data,
		logger:  logger,
	}, nil
}

// Close releases the mmap region backing the Index.
// It is a no-op when the Index was created with New (plain slice).
// Must be called once the Index is no longer in use to avoid a resource leak.
func (idx *Index) Close() error {
	if idx.mmapped == nil {
		return nil
	}
	return syscall.Munmap(idx.mmapped)
}

// FindNearest returns the labels of the k reference entries closest to v under Euclidean distance.
// It scans all N entries in O(N·k) time with a fixed-size candidate buffer.
//
// v is the 14-dimensional query vector produced by port.Vectorizer.
// k is the desired neighbour count; clamped to the dataset size when k > N.
//
// Returned labels are "fraud" or "legit" in unspecified order.
func (idx *Index) FindNearest(v domain.Vector, k int) []string {
	if k > idx.n {
		k = idx.n
	}

	idx.logger.Debug("starting KNN search",
		zap.Int("dataset_size", idx.n),
		zap.Int("k", k),
	)

	type candidate struct {
		dist float32
		idx  int32
	}

	best := make([]candidate, k)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	vecs := idx.vectors
	stride := domain.VectorSize

	for i := 0; i < idx.n; i++ {
		d := squaredEuclidean(v, vecs[i*stride:i*stride+stride])
		if d < worstDist {
			best[worstPos] = candidate{d, int32(i)}
			// Rescan for new worst.
			worstDist = best[0].dist
			worstPos = 0
			for j := 1; j < k; j++ {
				if best[j].dist > worstDist {
					worstDist = best[j].dist
					worstPos = j
				}
			}
		}
	}

	labels := make([]string, k)
	for i, c := range best {
		if idx.labels[c.idx] == 1 {
			labels[i] = "fraud"
		} else {
			labels[i] = "legit"
		}
	}

	idx.logger.Debug("KNN search complete", zap.Int("returned", k))
	return labels
}

// squaredEuclidean computes the squared Euclidean distance between the float64 query vector
// and the float32 reference slice. The square root is intentionally omitted — only relative
// ordering matters for k-NN, and skipping sqrt saves one transcendental per reference entry.
func squaredEuclidean(query domain.Vector, ref []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := float32(query[i]) - ref[i]
		sum += d * d
	}
	return sum
}
