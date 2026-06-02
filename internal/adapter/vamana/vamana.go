// Package vamana implements port.NeighborFinder using the Vamana graph-based
// approximate nearest-neighbour algorithm (the index structure behind DiskANN).
//
// Vamana builds a directed graph where each node has at most R outgoing edges.
// Queries perform a greedy beam search starting from a fixed medoid, visiting at most
// L candidate nodes. Construction runs in two passes with RobustPrune to create
// long-range edges (α > 1.0), which short-circuit the search path and improve recall.
//
// Compared with HNSW: Vamana uses a single flat graph layer (no hierarchy), which
// saves memory and has simpler cache behaviour. For low-dimensional vectors (14 dims)
// the search path is short enough that the hierarchy of HNSW provides little benefit.
//
// Memory layout (mmap-backed at serve time):
//
//	[N×R×4B] uint32: adjacency list (sentinel = N for empty slots)
//	[N×14B]  uint8:  SQ8-quantized vectors (or [N×56B] float32 when SQ8=false)
//	[N×1B]   uint8:  labels (1=fraud, 0=legit)
//
// Enable by setting VECTOR_SEARCHER=vamana at server startup.
// Requires resources/references.vamana (run preprocess with BUILD_VAMANA=true).
package vamana

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

// DefaultR is the maximum number of outgoing edges per node.
// R=16 balances recall (~95-98% for 14-dim vectors) against graph memory (~192 MB for 3M entries).
// Raise to 32 for higher recall at roughly double the graph memory.
const DefaultR = 16

// DefaultL is the beam width (candidate list size) during search.
// Must be ≥ k (neighbours requested). Higher values improve recall at the cost of latency.
const DefaultL = 64

// DefaultAlpha is the RobustPrune distance multiplier.
// Values > 1.0 keep long-range edges that skip across the graph, improving recall.
const DefaultAlpha = float32(1.2)

// DefaultBuildL is the candidate list size used during index construction when
// VAMANA_BUILD_L is not set. Larger than DefaultL to find high-quality edges;
// reduce to 64–75 to cut build time roughly in half at a small recall cost.
const DefaultBuildL = 125

// DefaultSQ8 controls whether vectors are stored as 8-bit integers in the index file.
// true: ~42 MB vectors for 3M entries; false: ~168 MB. The flag is stored in the file header.
const DefaultSQ8 = true

// flagSQ8 is the bitmask in the flags uint32 that signals SQ8 quantization is active.
const flagSQ8 = uint32(1 << 0)

// sentinel is stored in empty adjacency slots. It equals N (the dataset size) and
// is never a valid node index, so iteration stops at the first sentinel.
// The actual sentinel value is stored in the file as the N from the header.

// Searcher holds a Vamana graph over the reference dataset and implements port.NeighborFinder.
type Searcher struct {
	// n is the total number of nodes in the graph.
	n int
	// r is the maximum out-degree of each node.
	r int
	// l is the beam width used during FindNearest queries.
	l int
	// medoid is the index of the node closest to the dataset centroid.
	// All searches start here.
	medoid uint32
	// sq8 reports whether vectors in this index use 8-bit scalar quantization.
	sq8 bool
	// sq8Min and sq8Scale are the per-dimension parameters for SQ8 de-quantization.
	// Only meaningful when sq8=true.
	sq8Min   [domain.VectorSize]float32
	sq8Scale [domain.VectorSize]float32
	// graph is a flat adjacency list: graph[i*r : i*r+r] holds node i's neighbors.
	// Empty slots contain uint32(n) as a sentinel.
	graph []uint32
	// vecs holds float32 vectors when sq8=false: vecs[i*VectorSize:(i+1)*VectorSize].
	vecs []float32
	// vecs8 holds SQ8-encoded vectors when sq8=true: vecs8[i*VectorSize:(i+1)*VectorSize].
	vecs8 []uint8
	// labels holds 1=fraud, 0=legit for each node.
	labels []uint8
	// mmapped is non-nil when the Searcher was loaded via Open.
	// Held alive so graph/vecs/vecs8/labels slices (which point into it) remain valid.
	mmapped []byte
	// logger is the named zap logger injected at construction.
	logger *zap.Logger
}

// cand is a candidate entry in the beam-search list, sorted by distance to the query.
type cand struct {
	dist    float32
	id      uint32
	visited bool // true once this node's out-edges have been expanded
}

// Open memory-maps the Vamana index at path and returns a Searcher.
// l is the beam width used during queries; values ≤ 0 use DefaultL.
//
// The binary format is:
//
//	[4B] uint32 LE: N
//	[4B] uint32 LE: R
//	[4B] uint32 LE: medoid
//	[4B] uint32 LE: flags (bit 0 = SQ8)
//	If SQ8: [VectorSize×4B] float32 LE sq8_min, [VectorSize×4B] float32 LE sq8_scale
//	[N×R×4B] uint32 LE: adjacency (N = sentinel for empty slots)
//	If SQ8: [N×VectorSize B] uint8: quantized vectors
//	Else:   [N×VectorSize×4B] float32 LE: vectors
//	[N×1B] uint8: labels (1=fraud, 0=legit)
func Open(path string, l int, logger *zap.Logger) (*Searcher, error) {
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
	if fileSize < 16 {
		return nil, fmt.Errorf("%s: file too small (%d bytes)", path, fileSize)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	n := int(binary.LittleEndian.Uint32(data[0:]))
	r := int(binary.LittleEndian.Uint32(data[4:]))
	medoid := binary.LittleEndian.Uint32(data[8:])
	flags := binary.LittleEndian.Uint32(data[12:])
	sq8 := flags&flagSQ8 != 0

	p := 16

	var sq8Min, sq8Scale [domain.VectorSize]float32
	if sq8 {
		need := domain.VectorSize * 4 * 2
		if p+need > fileSize {
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

	graphBytes := n * r * 4
	if p+graphBytes > fileSize {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at adjacency list", path)
	}
	graphPtr := (*uint32)(unsafe.Pointer(&data[p]))
	graph := unsafe.Slice(graphPtr, n*r)
	p += graphBytes

	var vecs []float32
	var vecs8 []uint8
	if sq8 {
		vecBytes := n * domain.VectorSize
		if p+vecBytes+n > fileSize {
			_ = syscall.Munmap(data)
			return nil, fmt.Errorf("%s: truncated at vectors", path)
		}
		vecs8 = data[p : p+vecBytes]
		p += vecBytes
	} else {
		vecBytes := n * domain.VectorSize * 4
		if p+vecBytes+n > fileSize {
			_ = syscall.Munmap(data)
			return nil, fmt.Errorf("%s: truncated at vectors", path)
		}
		vecPtr := (*float32)(unsafe.Pointer(&data[p]))
		vecs = unsafe.Slice(vecPtr, n*domain.VectorSize)
		p += vecBytes
	}

	labels := data[p : p+n]

	if l <= 0 {
		l = DefaultL
	}

	logger.Info("Vamana index loaded",
		zap.Int("entries", n),
		zap.Int("R", r),
		zap.Int("L", l),
		zap.Bool("sq8", sq8),
		zap.Uint32("medoid", medoid),
	)

	return &Searcher{
		n:        n,
		r:        r,
		l:        l,
		medoid:   medoid,
		sq8:      sq8,
		sq8Min:   sq8Min,
		sq8Scale: sq8Scale,
		graph:    graph,
		vecs:     vecs,
		vecs8:    vecs8,
		labels:   labels,
		mmapped:  data,
		logger:   logger,
	}, nil
}

// Close releases the mmap region. No-op when the Searcher was created via New.
// Must be called once when the Searcher is no longer in use.
func (s *Searcher) Close() error {
	if s.mmapped == nil {
		return nil
	}
	return syscall.Munmap(s.mmapped)
}

// FindNearest returns the labels ("fraud" or "legit") of up to k approximate nearest
// neighbours of v under Euclidean distance. The greedy beam search starts at the medoid
// and maintains a candidate list of at most s.l entries.
//
// v is the 14-dimensional query vector; k is the desired neighbour count.
// The returned slice may be shorter than k when the index contains fewer entries.
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	s.logger.Debug("Vamana search starting",
		zap.Int("dataset_size", s.n),
		zap.Int("k", k),
		zap.Int("L", s.l),
	)

	if s.n == 0 {
		return nil
	}

	var query [domain.VectorSize]float32
	for i, f := range v {
		query[i] = float32(f)
	}

	cands := s.greedySearch(query, k)

	actualK := k
	if actualK > len(cands) {
		actualK = len(cands)
	}

	out := make([]string, actualK)
	for i, c := range cands[:actualK] {
		if s.labels[c.id] == 1 {
			out[i] = "fraud"
		} else {
			out[i] = "legit"
		}
	}

	s.logger.Debug("Vamana search complete", zap.Int("returned", actualK))
	return out
}

// greedySearch performs a beam search from s.medoid toward query, maintaining at most
// s.l candidates sorted by distance. Returns the candidate list sorted closest-first.
func (s *Searcher) greedySearch(query [domain.VectorSize]float32, k int) []cand {
	beamSize := s.l
	if beamSize < k {
		beamSize = k
	}

	cands := make([]cand, 0, beamSize+s.r)
	inCands := make(map[uint32]struct{}, beamSize*2)

	add := func(id uint32) {
		if _, ok := inCands[id]; ok {
			return
		}
		inCands[id] = struct{}{}
		d := s.nodeDist(id, query)
		// Insert in sorted position (ascending distance).
		pos := len(cands)
		for pos > 0 && cands[pos-1].dist > d {
			pos--
		}
		cands = append(cands, cand{})
		copy(cands[pos+1:], cands[pos:])
		cands[pos] = cand{dist: d, id: id}
		// Evict farthest when over capacity.
		if len(cands) > beamSize {
			delete(inCands, cands[beamSize].id)
			cands = cands[:beamSize]
		}
	}

	add(s.medoid)

	for {
		// Find the closest unvisited candidate.
		p := -1
		for i := range cands {
			if !cands[i].visited {
				p = i
				break
			}
		}
		if p == -1 {
			break
		}
		cands[p].visited = true
		nodeID := cands[p].id

		// Expand neighbors.
		base := int(nodeID) * s.r
		neighbors := s.graph[base : base+s.r]
		sentinel := uint32(s.n)
		for _, nb := range neighbors {
			if nb == sentinel {
				break
			}
			add(nb)
		}
	}

	return cands
}

// nodeDist returns the squared Euclidean distance between node id's stored vector
// and the query. Branches on the sq8 flag.
func (s *Searcher) nodeDist(id uint32, query [domain.VectorSize]float32) float32 {
	if s.sq8 {
		base := int(id) * domain.VectorSize
		v := s.vecs8[base : base+domain.VectorSize]
		var sum float32
		for d := 0; d < domain.VectorSize; d++ {
			decoded := s.sq8Min[d] + float32(v[d])*s.sq8Scale[d]
			diff := query[d] - decoded
			sum += diff * diff
		}
		return sum
	}
	base := int(id) * domain.VectorSize
	v := s.vecs[base : base+domain.VectorSize]
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := query[d] - v[d]
		sum += diff * diff
	}
	return sum
}
