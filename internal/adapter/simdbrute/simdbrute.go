// Package simdbrute implements port.NeighborFinder using the same partition routing
// as adapter/partition but with AVX2-accelerated squared-Euclidean distance computation.
//
// The query vector (float64) is converted to float32 exactly once before the scan loop.
// The inner distance computation then operates entirely on float32 using 256-bit AVX2
// SIMD instructions (8 elements per cycle), yielding roughly 4–8× throughput over
// the scalar implementation for the distance-computation step.
//
// Routing: same 3-bit bin key as partition (last_tx_null × is_online × card_present).
// Search is exact brute-force within each partition bin; cross-partition true neighbours
// are missed at the same rate as partition (negligible given the large L2 penalty for
// mismatched routing features).
//
// Requires AMD64 with AVX2 (Intel Haswell 2013+, AMD Ryzen 2017+).
// Enable with VECTOR_SEARCHER=simdbrute. Requires only resources/references.bin.
package simdbrute

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

// Routing dimensions — identical to adapter/partition.
const (
	dimMinutesSinceLast = 5
	dimKmFromLast       = 6
	dimIsOnline         = 9
	dimCardPresent      = 10
)

const numBins = 8

// binKey encodes (lastTxNull | isOnline | cardPresent) as 3 bits.
type binKey uint8

func binOf(v domain.Vector) binKey {
	var k binKey
	if v[dimMinutesSinceLast] == -1 && v[dimKmFromLast] == -1 {
		k |= 4
	}
	if v[dimIsOnline] == 1 {
		k |= 2
	}
	if v[dimCardPresent] == 1 {
		k |= 1
	}
	return k
}

func binOfRaw(v []float32) binKey {
	var k binKey
	if v[dimMinutesSinceLast] == -1 && v[dimKmFromLast] == -1 {
		k |= 4
	}
	if v[dimIsOnline] == 1 {
		k |= 2
	}
	if v[dimCardPresent] == 1 {
		k |= 1
	}
	return k
}

// Finder routes each query to the matching partition bin and performs exact
// brute-force KNN using AVX2-accelerated distance computation within the bin.
type Finder struct {
	// n is the total number of reference entries.
	n int
	// vectors holds all feature vectors contiguously.
	vectors []float32
	// labels holds the fraud/legit classification for each entry.
	labels []uint8
	// mmapped is non-nil when backed by a memory-mapped file.
	mmapped []byte
	// bins[k] is the sorted list of entry indices in partition bin k.
	bins   [numBins][]uint32
	logger *zap.Logger
}

// Reference is a labelled dataset entry used in tests.
type Reference struct {
	Vector domain.Vector
	Label  string
}

// New builds a Finder from a slice of Reference entries using plain heap allocations.
// Intended for tests; production code should use Open.
func New(refs []Reference, logger *zap.Logger) *Finder {
	n := len(refs)
	vectors := make([]float32, n*domain.VectorSize)
	labels := make([]uint8, n)
	for i, r := range refs {
		base := i * domain.VectorSize
		for j, val := range r.Vector {
			vectors[base+j] = float32(val)
		}
		if r.Label == "fraud" {
			labels[i] = 1
		}
	}
	f := &Finder{n: n, vectors: vectors, labels: labels, logger: logger}
	f.buildBins()
	return f
}

// Open memory-maps the binary reference index at path and partitions it into bins.
// The binary format matches adapter/knn (produced by cmd/preprocess).
// Call Close when done to release the mmap region.
func Open(path string, logger *zap.Logger) (*Finder, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	fileSize := int(info.Size())
	if fileSize < 4 {
		return nil, fmt.Errorf("%s: file too small (%d bytes)", path, fileSize)
	}

	data, err := syscall.Mmap(int(file.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	n := int(binary.LittleEndian.Uint32(data[:4]))
	vectorBytes := n * domain.VectorSize * 4
	required := 4 + vectorBytes + n
	if fileSize < required {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: need %d bytes for %d entries, have %d", path, required, n, fileSize)
	}

	vectorPtr := (*float32)(unsafe.Pointer(&data[4]))
	vectors := unsafe.Slice(vectorPtr, n*domain.VectorSize)
	labels := data[4+vectorBytes : 4+vectorBytes+n]

	f := &Finder{
		n:       n,
		vectors: vectors,
		labels:  labels,
		mmapped: data,
		logger:  logger,
	}
	f.buildBins()

	logger.Info("simdbrute finder ready", zap.Int("total_entries", n))
	return f, nil
}

// buildBins partitions all entries into per-bin index slices.
func (f *Finder) buildBins() {
	stride := domain.VectorSize
	var counts [numBins]int
	for i := 0; i < f.n; i++ {
		counts[binOfRaw(f.vectors[i*stride:i*stride+stride])]++
	}
	for k, c := range counts {
		if c > 0 {
			f.bins[k] = make([]uint32, 0, c)
		}
	}
	for i := 0; i < f.n; i++ {
		k := binOfRaw(f.vectors[i*stride : i*stride+stride])
		f.bins[k] = append(f.bins[k], uint32(i))
	}
	for k, b := range f.bins {
		f.logger.Debug("simdbrute bin built", zap.Int("bin", k), zap.Int("entries", len(b)))
	}
}

// Close releases the mmap region. No-op when created with New.
func (f *Finder) Close() error {
	if f.mmapped == nil {
		return nil
	}
	return syscall.Munmap(f.mmapped)
}

// FindNearest returns the labels of the k nearest entries to v using AVX2-accelerated
// distance computation within the matching partition bin.
func (f *Finder) FindNearest(v domain.Vector, k int) []string {
	key := binOf(v)
	bin := f.bins[key]
	if len(bin) == 0 {
		f.logger.Warn("simdbrute bin empty, falling back to full scan", zap.Int("bin", int(key)))
		return f.scan(v, nil, k)
	}
	return f.scan(v, bin, k)
}

// candidate is a (distance, index) pair used by the top-k buffer in scan.
type candidate struct {
	dist float32
	idx  uint32
}

// scan performs AVX2-accelerated brute-force KNN over the given index slice.
// When indices is nil, all entries are scanned (full-scan fallback).
func (f *Finder) scan(v domain.Vector, indices []uint32, k int) []string {
	// Convert float64 query to float32 once — amortised over all distance calls.
	var qf32 [domain.VectorSize]float32
	for i, x := range v {
		qf32[i] = float32(x)
	}

	n := len(indices)
	fullScan := indices == nil
	if fullScan {
		n = f.n
	}
	if k > n {
		k = n
	}

	best := make([]candidate, k)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	vecs := f.vectors
	stride := domain.VectorSize

	if fullScan {
		for i := 0; i < f.n; i++ {
			d := l2sq14(&qf32[0], &vecs[i*stride])
			if d < worstDist {
				best[worstPos] = candidate{d, uint32(i)}
				worstDist, worstPos = newWorst(best)
			}
		}
	} else {
		for _, entryIdx := range indices {
			d := l2sq14(&qf32[0], &vecs[int(entryIdx)*stride])
			if d < worstDist {
				best[worstPos] = candidate{d, entryIdx}
				worstDist, worstPos = newWorst(best)
			}
		}
	}

	labels := make([]string, k)
	for i, c := range best {
		if f.labels[c.idx] == 1 {
			labels[i] = "fraud"
		} else {
			labels[i] = "legit"
		}
	}
	return labels
}

// newWorst finds the candidate with the highest distance in best and returns its
// distance and position. Called only when the top-k buffer is updated.
func newWorst(best []candidate) (float32, int) {
	w, pos := best[0].dist, 0
	for j := 1; j < len(best); j++ {
		if best[j].dist > w {
			w, pos = best[j].dist, j
		}
	}
	return w, pos
}
