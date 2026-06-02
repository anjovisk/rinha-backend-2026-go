// Package ivf implements port.NeighborFinder using an Inverted File Index (IVF)
// with optional 8-bit Scalar Quantization (SQ8) for approximate nearest-neighbour search.
//
// IVF partitions the reference dataset into nlist clusters using K-means.
// At query time, only the nprobe nearest clusters are searched, reducing the
// number of distance computations from N to approximately N×nprobe/nlist.
//
// SQ8 (controlled at build time via IVF_SQ8) compresses each float32 dimension
// to uint8, cutting vector memory by 4× and improving cache locality at the cost
// of a small distance approximation error. When SQ8 is disabled, vectors are
// stored and compared as raw float32, yielding exact distances within each cluster.
//
// Build the index once with Build (cmd/preprocess when BUILD_IVF=true),
// then load it at server startup with Open (VECTOR_SEARCHER=ivf).
// Use IVF_NPROBE to override the default number of clusters probed per query.
package ivf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sync"
	"syscall"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// queryBuf holds pre-allocated scratch space for centroid selection.
// Instances are recycled across FindNearest calls via Searcher.pool to avoid
// the per-query heap allocation of the former indices+dists slices (~12 KB each).
type queryBuf struct {
	// dists holds the squared distances of the current best nprobe centroids.
	dists []float32
	// idxs holds the cluster index for each entry in dists.
	idxs []int
}

// DefaultNprobe is the number of IVF clusters searched per query when IVF_NPROBE is unset.
// With nlist=1024 and nprobe=32, roughly 3% of the dataset is searched per query,
// yielding ~30× speedup over brute force with minimal recall loss.
const DefaultNprobe = 32

// flagSQ8 is the bitmask in the flags byte that signals SQ8 quantization is active.
const flagSQ8 = uint8(1 << 0)

// flagFlat is the bitmask in the flags byte that signals the flat-layout file format.
// Files produced by Build always set this bit. Open rejects files without it so that
// stale per-cluster-layout indexes are detected immediately with a clear error message.
const flagFlat = uint8(1 << 1)

// Searcher holds an IVF index and implements port.NeighborFinder.
// Vectors are stored either as SQ8-quantized uint8 (sq8=true) or raw float32 (sq8=false);
// the mode is determined at build time and recorded in the index file.
type Searcher struct {
	// nlist is the number of K-means clusters in the index.
	nlist int
	// nprobe is the number of clusters searched per query.
	nprobe int
	// sq8 reports whether vectors in this index are SQ8-quantized (uint8 per dimension).
	sq8 bool
	// sq8Min holds the per-dimension minimum value used during SQ8 encoding.
	// Only meaningful when sq8=true.
	sq8Min [domain.VectorSize]float32
	// sq8Scale holds the per-dimension step size (range/255) for SQ8 encoding.
	// Only meaningful when sq8=true.
	sq8Scale [domain.VectorSize]float32
	// centroids holds all cluster centroid vectors in row-major order; cluster c at c*VectorSize.
	centroids []float32
	// vecFlat8 holds SQ8-quantized vectors (1 byte/dim) in cluster order. Non-nil when sq8=true.
	// Cluster c occupies vecFlat8[offsets[c]*VectorSize : (offsets[c]+sizes[c])*VectorSize].
	vecFlat8 []uint8
	// vecFlat32 holds raw float32 vectors (4 bytes/dim) in cluster order. Non-nil when sq8=false.
	// Cluster c occupies vecFlat32[offsets[c]*VectorSize : (offsets[c]+sizes[c])*VectorSize].
	vecFlat32 []float32
	// labelFlat holds labels (1=fraud, 0=legit) in the same cluster order as the vec arrays.
	// Cluster c occupies labelFlat[offsets[c] : offsets[c]+sizes[c]].
	labelFlat []uint8
	// offsets[c] is the index of the first vector belonging to cluster c.
	offsets []int32
	// sizes[c] is the number of vectors in cluster c.
	sizes []int32
	// logger is the named zap logger injected at construction.
	logger *zap.Logger
	// pool recycles queryBuf instances across FindNearest calls, eliminating the
	// per-query allocation of the centroid-search index and distance buffers.
	pool sync.Pool
	// mmapped is non-nil when the Searcher was opened via Open: it holds the mmap
	// region so that vecFlat8 and labelFlat slices pointing into it remain valid.
	// Released by Close. Nil when created with New (plain heap slices).
	mmapped []byte
}

// Open memory-maps the pre-built IVF index at path and returns a Searcher.
// nprobe controls how many clusters are searched per query; values ≤ 0 use DefaultNprobe.
// The file must have been produced by Build (cmd/preprocess with BUILD_IVF=true).
// Whether SQ8 is active is encoded in the file header — no env var needed at serve time.
//
// When sq8=true, vecFlat8 and labelFlat are sliced directly into the mmap region (zero-copy),
// keeping ~45 MB out of Go's GC-managed heap and eliminating the double-buffering that the
// previous os.ReadFile approach caused. Call Close when done to release the mmap.
//
// Binary format (flat layout, flagFlat always set):
//
//	[4B]                    uint32 LE: nlist
//	[4B]                    uint32 LE: N (total vectors)
//	[1B]                    uint8:     flags (bit 0=SQ8, bit 1=flat layout)
//	If SQ8:
//	  [VectorSize×4B]       float32 LE: sq8_min[VectorSize]
//	  [VectorSize×4B]       float32 LE: sq8_scale[VectorSize]
//	[nlist×VectorSize×4B]   float32 LE: centroids, row-major
//	[nlist×4B]              uint32 LE: sizes[nlist] (vectors per cluster)
//	[N×VectorSize B]        uint8 (SQ8) or [N×VectorSize×4B] float32: flat vectors
//	[N×1B]                  uint8: flat labels (1=fraud, 0=legit)
func Open(path string, nprobe int, logger *zap.Logger) (*Searcher, error) {
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
	if fileSize < 9 { // nlist(4) + N(4) + flags(1)
		return nil, fmt.Errorf("%s: file too small", path)
	}

	// The mmap is kept alive in Searcher.mmapped so slices into it remain valid.
	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	p := 0
	nlist := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	n := int(binary.LittleEndian.Uint32(data[p:]))
	p += 4
	flags := data[p]
	p++

	if flags&flagFlat == 0 {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: old per-cluster format detected — regenerate with BUILD_IVF=true", path)
	}
	sq8 := flags&flagSQ8 != 0

	var sq8Min, sq8Scale [domain.VectorSize]float32
	if sq8 {
		sq8ParamBytes := domain.VectorSize * 4 * 2
		if p+sq8ParamBytes > len(data) {
			_ = syscall.Munmap(data)
			return nil, fmt.Errorf("%s: truncated at SQ8 params", path)
		}
		for d := 0; d < domain.VectorSize; d++ {
			sq8Min[d] = math.Float32frombits(binary.LittleEndian.Uint32(data[p:]))
			p += 4
		}
		for d := 0; d < domain.VectorSize; d++ {
			sq8Scale[d] = math.Float32frombits(binary.LittleEndian.Uint32(data[p:]))
			p += 4
		}
	}

	centroidBytes := nlist * domain.VectorSize * 4
	if p+centroidBytes > len(data) {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at centroids", path)
	}
	centroids := make([]float32, nlist*domain.VectorSize)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[p:]))
		p += 4
	}

	sizesBytes := nlist * 4
	if p+sizesBytes > len(data) {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at sizes", path)
	}
	sizes := make([]int32, nlist)
	offsets := make([]int32, nlist)
	var cum int32
	for c := 0; c < nlist; c++ {
		sz := int32(binary.LittleEndian.Uint32(data[p:]))
		p += 4
		offsets[c] = cum
		sizes[c] = sz
		cum += sz
	}

	vecBytesPerVec := domain.VectorSize // SQ8: 1 byte/dim
	if !sq8 {
		vecBytesPerVec = domain.VectorSize * 4
	}
	totalVecBytes := n * vecBytesPerVec
	if p+totalVecBytes+n > len(data) {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at vectors/labels", path)
	}

	var vecFlat8 []uint8
	var vecFlat32 []float32
	if sq8 {
		// Zero-copy: slice directly into the mmap region — no heap allocation.
		vecFlat8 = data[p : p+totalVecBytes]
	} else {
		// float32 region may be unaligned; copy to a properly-aligned heap slice.
		vecFlat32 = make([]float32, n*domain.VectorSize)
		src := data[p : p+totalVecBytes]
		for i := range vecFlat32 {
			vecFlat32[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		}
	}
	p += totalVecBytes

	// Zero-copy: label slice directly into the mmap region.
	labelFlat := data[p : p+n]

	if nprobe <= 0 {
		nprobe = DefaultNprobe
	}
	if nprobe > nlist {
		nprobe = nlist
	}

	logger.Info("IVF index loaded",
		zap.Int("nlist", nlist),
		zap.Int("nprobe", nprobe),
		zap.Bool("sq8", sq8),
		zap.Int("total_vectors", n),
	)

	s := &Searcher{
		nlist:     nlist,
		nprobe:    nprobe,
		sq8:       sq8,
		sq8Min:    sq8Min,
		sq8Scale:  sq8Scale,
		centroids: centroids,
		vecFlat8:  vecFlat8,
		vecFlat32: vecFlat32,
		labelFlat: labelFlat,
		offsets:   offsets,
		sizes:     sizes,
		mmapped:   data,
		logger:    logger,
	}
	s.initPool()
	return s, nil
}

// Close releases the mmap region backing the Searcher's vector and label data.
// It is a no-op when the Searcher was created with New (plain heap slices).
// Must be called once the Searcher is no longer in use to avoid a resource leak.
// Accessing the Searcher after Close results in undefined behaviour.
func (s *Searcher) Close() error {
	if s.mmapped == nil {
		return nil
	}
	return syscall.Munmap(s.mmapped)
}

// initPool initialises the sync.Pool that recycles centroid-search buffers.
// Must be called once after s.nprobe is set.
func (s *Searcher) initPool() {
	nprobe := s.nprobe
	s.pool = sync.Pool{
		New: func() any {
			return &queryBuf{
				dists: make([]float32, nprobe),
				idxs:  make([]int, nprobe),
			}
		},
	}
}

// FindNearest returns the labels ("fraud" or "legit") of up to k approximate nearest
// neighbours of v. It searches the s.nprobe closest clusters; distance is computed
// using SQ8-approximate arithmetic when sq8=true, or exact float32 arithmetic otherwise.
//
// v is the 14-dimensional query vector; k is the desired neighbour count.
// The returned slice may be shorter than k when fewer candidates are available.
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	s.logger.Debug("IVF search starting",
		zap.Int("nprobe", s.nprobe),
		zap.Bool("sq8", s.sq8),
		zap.Int("k", k),
	)

	var query [domain.VectorSize]float32
	for i, f := range v {
		query[i] = float32(f)
	}

	// Borrow a pre-allocated centroid buffer from the pool; return it when done.
	buf := s.pool.Get().(*queryBuf)
	defer s.pool.Put(buf)

	s.nearestClusters(query, buf)

	totalSearched := 0
	for _, ci := range buf.idxs {
		totalSearched += int(s.sizes[ci])
	}
	actualK := k
	if actualK > totalSearched {
		actualK = totalSearched
	}

	type candidate struct {
		dist  float32
		label uint8
	}
	best := make([]candidate, actualK)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	if s.sq8 {
		// Precompute residuals once per query: residual[d] = query[d] - sq8Min[d].
		var residual [domain.VectorSize]float32
		for d := 0; d < domain.VectorSize; d++ {
			residual[d] = query[d] - s.sq8Min[d]
		}
		for _, ci := range buf.idxs {
			off := int(s.offsets[ci])
			sz := int(s.sizes[ci])
			vecs := s.vecFlat8[off*domain.VectorSize : (off+sz)*domain.VectorSize]
			lbls := s.labelFlat[off : off+sz]
			for j := 0; j < sz; j++ {
				d := sq8L2SqEarlyExit(residual[:], s.sq8Scale[:], vecs[j*domain.VectorSize:], worstDist)
				if d < worstDist {
					best[worstPos] = candidate{d, lbls[j]}
					worstDist = best[0].dist
					worstPos = 0
					for r := 1; r < actualK; r++ {
						if best[r].dist > worstDist {
							worstDist = best[r].dist
							worstPos = r
						}
					}
				}
			}
		}
	} else {
		for _, ci := range buf.idxs {
			off := int(s.offsets[ci])
			sz := int(s.sizes[ci])
			vecs := s.vecFlat32[off*domain.VectorSize : (off+sz)*domain.VectorSize]
			lbls := s.labelFlat[off : off+sz]
			for j := 0; j < sz; j++ {
				d := f32L2SqEarlyExit(query[:], vecs[j*domain.VectorSize:], worstDist)
				if d < worstDist {
					best[worstPos] = candidate{d, lbls[j]}
					worstDist = best[0].dist
					worstPos = 0
					for r := 1; r < actualK; r++ {
						if best[r].dist > worstDist {
							worstDist = best[r].dist
							worstPos = r
						}
					}
				}
			}
		}
	}

	out := make([]string, actualK)
	for i, c := range best {
		if c.label == 1 {
			out[i] = "fraud"
		} else {
			out[i] = "legit"
		}
	}

	s.logger.Debug("IVF search complete", zap.Int("returned", len(out)))
	return out
}

// nearestClusters fills buf with the s.nprobe centroid indices closest to query.
//
// Single-pass O(nlist) scan with a fixed-size worst-of-k buffer — the same approach
// used by adapter/knn. Compared to the previous selection sort:
//   - eliminates the per-call allocation of indices+dists slices (~12 KB)
//   - reduces comparisons from O(nlist×nprobe) = 32K to O(nlist + nprobe×hits) ≈ 1K–5K
//   - applies early-exit distance pruning once the buffer has nprobe candidates
func (s *Searcher) nearestClusters(query [domain.VectorSize]float32, buf *queryBuf) {
	best := buf.dists
	idxs := buf.idxs
	for i := range best {
		best[i] = math.MaxFloat32
		idxs[i] = i
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	for c := 0; c < s.nlist; c++ {
		d := f32L2SqEarlyExit(query[:], s.centroids[c*domain.VectorSize:], worstDist)
		if d < worstDist {
			best[worstPos] = d
			idxs[worstPos] = c
			worstDist = best[0]
			worstPos = 0
			for j := 1; j < s.nprobe; j++ {
				if best[j] > worstDist {
					worstDist = best[j]
					worstPos = j
				}
			}
		}
	}
}

// f32L2SqEarlyExit computes the squared L2 distance between two float32 slices,
// aborting and returning the partial sum as soon as it reaches or exceeds threshold.
// Used in hot paths where the partial sum can be compared against the current
// k-th best distance to skip vectors that cannot improve the result set.
func f32L2SqEarlyExit(a, b []float32, threshold float32) float32 {
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := a[d] - b[d]
		sum += diff * diff
		if sum >= threshold {
			return sum
		}
	}
	return sum
}

// sq8L2SqEarlyExit computes the approximate SQ8 squared L2 distance with early exit.
// residual[d] = query[d] - sq8Min[d]; scale[d] maps the uint8 step to the original range.
// Aborts as soon as the partial sum reaches threshold, skipping remaining dimensions.
func sq8L2SqEarlyExit(residual, scale []float32, qvec []uint8, threshold float32) float32 {
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := residual[d] - float32(qvec[d])*scale[d]
		sum += diff * diff
		if sum >= threshold {
			return sum
		}
	}
	return sum
}
