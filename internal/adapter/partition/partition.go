// Package partition implements port.NeighborFinder by routing each query to a
// sub-index selected by three discrete vector features:
//   - lastTxNull:  dimensions 5 and 6 are both -1 (no prior transaction on record)
//   - isOnline:    dimension 9 equals 1
//   - cardPresent: dimension 10 equals 1
//
// The reference dataset is split into up to 8 bins (3 bits). In practice only 6 bins
// are populated because isOnline=1 and cardPresent=1 are mutually exclusive in the
// dataset (an online transaction never has the card physically present).
//
// Expected partition sizes for the 3M reference dataset:
//
//	bin 0  (last_tx=false, online=0, card=0)  ~4.3%   128 K entries
//	bin 1  (last_tx=false, online=0, card=1)  ~38.5%  1.16 M entries
//	bin 2  (last_tx=false, online=1, card=0)  ~37.2%  1.12 M entries
//	bin 4  (last_tx=true,  online=0, card=0)  ~1.1%   32 K entries
//	bin 5  (last_tx=true,  online=0, card=1)  ~9.6%   289 K entries
//	bin 6  (last_tx=true,  online=1, card=0)  ~9.3%   278 K entries
//
// The weighted-average scan is ~31% of a full brute-force pass (~3.2× speedup).
// The search is exact brute-force within each partition; only cross-partition
// true neighbours are missed, which is rare given the large L2 penalty for
// mismatched routing features.
package partition

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

// Routing dimensions within domain.Vector.
const (
	dimMinutesSinceLast = 5  // sentinel -1 when last transaction is absent
	dimKmFromLast       = 6  // sentinel -1 when last transaction is absent
	dimIsOnline         = 9  // 1 = online transaction, 0 = in-person
	dimCardPresent      = 10 // 1 = card physically present, 0 = absent
)

// numBins is the number of partition slots (2^3 routing bits).
// Bins 3 (011) and 7 (111) are unused in practice.
const numBins = 8

// binKey encodes (lastTxNull | isOnline | cardPresent) as 3 bits.
// Bit 2 = lastTxNull, bit 1 = isOnline, bit 0 = cardPresent.
type binKey uint8

// binOf computes the routing bin for a float64 query vector.
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

// binOfRaw computes the routing bin for a float32 reference slice.
// -1.0 is exactly representable as float32, so the sentinel comparison is safe.
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

// Finder routes each FindNearest query to the sub-partition that matches the query's
// discrete routing features and performs exact brute-force KNN within that partition.
//
// When the query's routing features map to an empty bin (unseen combination), the
// search falls back to a full scan of all entries so the caller always gets a result.
type Finder struct {
	// n is the total number of reference entries.
	n int
	// vectors holds all feature vectors contiguously; entry i occupies [i*VectorSize, (i+1)*VectorSize).
	// Backed by a shared mmap when created with Open; a plain heap slice when created with New.
	vectors []float32
	// labels holds the fraud/legit classification for each entry: 1=fraud, 0=legit.
	labels []uint8
	// mmapped is non-nil when vectors and labels point into a mmap region.
	mmapped []byte
	// bins[k] is the sorted list of entry indices belonging to partition bin k.
	bins [numBins][]uint32
	// logger is the named zap logger injected at construction time.
	logger *zap.Logger
}

// Reference is a labelled dataset entry used to construct a Finder in tests.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is the fraud classification: "fraud" or "legit".
	Label string
}

// New builds a Finder from a slice of Reference entries using plain heap allocations.
// This constructor is intended for tests; production code should use Open.
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
//
// The binary format matches adapter/knn:
//
//	[4 bytes uint32 LE]  number of entries N
//	[N × VectorSize × 4 bytes]  float32 LE feature vectors, row-major
//	[N bytes]            uint8 labels (1=fraud, 0=legit)
//
// Both API server instances share the same physical pages via MAP_SHARED,
// keeping the mmap cost in the OS page cache rather than duplicating in RAM.
// Call Close when the Finder is no longer needed.
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

	logger.Info("partition finder ready", zap.Int("total_entries", n))
	return f, nil
}

// buildBins partitions all entries into per-bin index slices.
// Two-pass: count first to pre-allocate, then fill.
// Must be called exactly once after vectors and labels are set.
func (f *Finder) buildBins() {
	stride := domain.VectorSize
	var counts [numBins]int
	for i := 0; i < f.n; i++ {
		k := binOfRaw(f.vectors[i*stride : i*stride+stride])
		counts[k]++
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
		f.logger.Debug("partition bin built",
			zap.Int("bin", k),
			zap.Int("entries", len(b)),
		)
	}
}

// Close releases the mmap region backing the Finder.
// It is a no-op when the Finder was created with New (plain heap slice).
func (f *Finder) Close() error {
	if f.mmapped == nil {
		return nil
	}
	return syscall.Munmap(f.mmapped)
}

// FindNearest returns the labels of the k reference entries closest to v under Euclidean
// distance. Only the partition bin matching v's routing features is searched.
//
// If the matching bin is empty (a combination of routing features not seen in the
// reference dataset), all entries are scanned so the caller always receives k labels.
func (f *Finder) FindNearest(v domain.Vector, k int) []string {
	key := binOf(v)
	bin := f.bins[key]
	if len(bin) == 0 {
		f.logger.Warn("partition bin empty, falling back to full scan",
			zap.Int("bin", int(key)),
		)
		return f.scan(v, fullRange(f.n), k)
	}
	return f.scan(v, bin, k)
}

// scan performs brute-force KNN over the given index slice.
// Each element in indices is an entry index into f.vectors / f.labels.
func (f *Finder) scan(v domain.Vector, indices []uint32, k int) []string {
	n := len(indices)
	if k > n {
		k = n
	}

	type candidate struct {
		dist float32
		idx  uint32
	}

	best := make([]candidate, k)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	vecs := f.vectors
	stride := domain.VectorSize

	for _, entryIdx := range indices {
		base := int(entryIdx) * stride
		d := squaredEuclidean(v, vecs[base:base+stride])
		if d < worstDist {
			best[worstPos] = candidate{d, entryIdx}
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
		if f.labels[c.idx] == 1 {
			labels[i] = "fraud"
		} else {
			labels[i] = "legit"
		}
	}
	return labels
}

// fullRange returns a slice of all entry indices 0..n-1, used for the full-scan fallback.
func fullRange(n int) []uint32 {
	idx := make([]uint32, n)
	for i := range idx {
		idx[i] = uint32(i)
	}
	return idx
}

// squaredEuclidean computes the squared Euclidean distance between a float64 query vector
// and a float32 reference slice. The square root is intentionally omitted — only relative
// ordering matters for k-NN.
func squaredEuclidean(query domain.Vector, ref []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := float32(query[i]) - ref[i]
		sum += d * d
	}
	return sum
}
