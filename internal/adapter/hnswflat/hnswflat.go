// Package hnswflat implements port.NeighborFinder using a flat-file, memory-mapped
// Hierarchical Navigable Small World (HNSW) approximate nearest-neighbour graph.
//
// Unlike the coder/hnsw-based adapter, this implementation stores the entire graph
// (adjacency lists + SQ8 vectors + labels) in a mmap'd binary file (MAP_SHARED),
// keeping all graph data in the OS page cache rather than Go's GC-managed heap.
// Two API instances mapping the same file share the physical pages.
//
// Memory layout per container (N=3M, M=3, SQ8=true):
//
//	~143 MB page cache (mmap) + ~15 MB Go runtime heap ≈ 158 MB total RSS
//
// Enable by setting VECTOR_SEARCHER=hnswflat at server startup.
// Requires resources/references.hnswflat (run preprocess with BUILD_HNSWFLAT=true).
package hnswflat

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// DefaultM is the maximum number of bidirectional connections per node at upper layers.
// Layer 0 uses 2×M connections. M=3 keeps the total file ~143 MB for 3M vectors (SQ8),
// fitting within the 165 MB Docker budget. Raise to 4 for ~166 MB with better recall.
const DefaultM = 3

// DefaultEfSearch is the size of the dynamic candidate list during graph traversal.
// Higher values improve recall at the cost of latency. Must be ≥ k.
const DefaultEfSearch = 50

// DefaultSQ8 controls SQ8 quantization. true = ~42 MB vectors; false = ~168 MB.
const DefaultSQ8 = true

const flagSQ8 = uint32(1 << 0)
const magic = uint32(0x46484E57) // 'FHNW' — flat HNSW

// emptySlot marks an unused adjacency slot (valid IDs are 0..N-1; N < 2^31).
const emptySlot = uint32(0xFFFFFFFF)

// Searcher holds a memory-mapped HNSW flat index and implements port.NeighborFinder.
type Searcher struct {
	// Zero-copy slices into the mmap region:
	vecs8    []uint8   // [N × VectorSize] SQ8 uint8 vectors; nil when sq8=false
	vecsF32  []float32 // [N × VectorSize] float32 vectors; nil when sq8=true
	labels   []uint8   // [N]
	levels   []uint8  // [N] highest layer for each node (0 = layer-0 only)
	l0adj    []uint32 // [N × 2M] layer-0 adjacency; emptySlot = unused
	upperIDs []uint32 // sorted node IDs that appear in layer 1+
	upperOff []uint32 // [upper_n+1] CSR offsets into upperAdj
	upperAdj []uint32 // packed connections for all upper-layer nodes

	n, m, l  int     // entry count, max degree, number of layers
	entry    uint32  // entry-point node (at the highest layer)
	efSearch int
	sq8      bool
	sq8Min   [domain.VectorSize]float32
	sq8Scale [domain.VectorSize]float32

	mmapped []byte // kept alive so slices into it remain valid; released by Close
	visPool visPool
	logger  *zap.Logger
}

// Open memory-maps the flat HNSW index at path and returns a ready Searcher.
// efSearch controls the beam width during queries; ≤0 uses DefaultEfSearch.
//
// Binary format (all integers little-endian):
//
//	[4B]  uint32 magic
//	[4B]  uint32 N
//	[4B]  uint32 M
//	[4B]  uint32 L (layers including layer 0)
//	[4B]  uint32 entry_node
//	[4B]  uint32 upper_n (nodes in layer 1+)
//	[4B]  uint32 flags (bit 0 = SQ8)
//	[4B]  reserved
//	If SQ8: [VectorSize×4B] sq8_min + [VectorSize×4B] sq8_scale
//	[N × VectorSize B]     uint8 SQ8 vectors (or float32 × 4 without SQ8)
//	[N B]                  uint8 labels
//	[N B]                  uint8 node levels
//	[pad to 4B boundary]
//	[N × 2M × 4B]          uint32 layer-0 adjacency
//	[upper_n × 4B]          uint32 upper node IDs (sorted)
//	[upper_n+1 × 4B]        uint32 upper offsets (CSR)
//	[total_upper_adj × 4B]  uint32 upper adjacency data
func Open(path string, efSearch int, logger *zap.Logger) (*Searcher, error) {
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
	if fileSize < 32 {
		return nil, fmt.Errorf("%s: file too small", path)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}

	p := 0
	readU32 := func() uint32 {
		v := binary.LittleEndian.Uint32(data[p:])
		p += 4
		return v
	}

	if mg := readU32(); mg != magic {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: bad magic 0x%08X (expected 0x%08X) — regenerate with BUILD_HNSWFLAT=true", path, mg, magic)
	}
	n := int(readU32())
	m := int(readU32())
	l := int(readU32())
	entry := readU32()
	upperN := int(readU32())
	flags := readU32()
	_ = readU32() // reserved

	sq8 := flags&flagSQ8 != 0
	var sq8Min, sq8Scale [domain.VectorSize]float32
	if sq8 {
		for d := 0; d < domain.VectorSize; d++ {
			sq8Min[d] = math.Float32frombits(readU32())
		}
		for d := 0; d < domain.VectorSize; d++ {
			sq8Scale[d] = math.Float32frombits(readU32())
		}
	}

	vecBytesPerVec := domain.VectorSize
	if !sq8 {
		vecBytesPerVec = domain.VectorSize * 4
	}

	totalVecBytes := n * vecBytesPerVec
	needed := p + totalVecBytes + n + n // vecs + labels + levels
	if needed > fileSize {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at vecs/labels/levels", path)
	}

	var vecs8 []uint8
	var vecsF32 []float32
	if sq8 {
		vecs8 = data[p : p+totalVecBytes]
	} else {
		ptr := (*float32)(unsafe.Pointer(&data[p]))
		vecsF32 = unsafe.Slice(ptr, n*domain.VectorSize)
	}
	p += totalVecBytes

	labels := data[p : p+n]
	p += n

	levels := data[p : p+n]
	p += n

	// Align to 4-byte boundary.
	if rem := p % 4; rem != 0 {
		p += 4 - rem
	}

	l0Size := n * 2 * m * 4
	if p+l0Size > fileSize {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at layer-0 adjacency", path)
	}
	l0ptr := (*uint32)(unsafe.Pointer(&data[p]))
	l0adj := unsafe.Slice(l0ptr, n*2*m)
	p += l0Size

	upperIDsSize := upperN * 4
	upperOffSize := (upperN + 1) * 4
	if p+upperIDsSize+upperOffSize > fileSize {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at upper-layer tables", path)
	}
	upperIDsPtr := (*uint32)(unsafe.Pointer(&data[p]))
	upperIDs := unsafe.Slice(upperIDsPtr, upperN)
	p += upperIDsSize

	upperOffPtr := (*uint32)(unsafe.Pointer(&data[p]))
	upperOff := unsafe.Slice(upperOffPtr, upperN+1)
	p += upperOffSize

	totalUpperAdj := int(upperOff[upperN])
	if totalUpperAdj > 0 && p+totalUpperAdj*4 > fileSize {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("%s: truncated at upper-layer adjacency", path)
	}
	var upperAdj []uint32
	if totalUpperAdj > 0 {
		upperAdjPtr := (*uint32)(unsafe.Pointer(&data[p]))
		upperAdj = unsafe.Slice(upperAdjPtr, totalUpperAdj)
	}

	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}

	logger.Info("hnswflat index loaded",
		zap.Int("n", n),
		zap.Int("M", m),
		zap.Int("layers", l),
		zap.Bool("sq8", sq8),
		zap.Int("ef_search", efSearch),
	)

	s := &Searcher{
		vecs8:    vecs8,
		vecsF32:  vecsF32,
		labels:   labels,
		levels:   levels,
		l0adj:    l0adj,
		upperIDs: upperIDs,
		upperOff: upperOff,
		upperAdj: upperAdj,
		n:        n,
		m:        m,
		l:        l,
		entry:    entry,
		efSearch: efSearch,
		sq8:      sq8,
		sq8Min:   sq8Min,
		sq8Scale: sq8Scale,
		mmapped:  data,
		logger:   logger,
	}
	s.visPool = newVisPool(n)
	return s, nil
}

// Close releases the mmap region. Accessing the Searcher after Close is undefined.
func (s *Searcher) Close() error {
	if s.mmapped == nil {
		return nil
	}
	return syscall.Munmap(s.mmapped)
}

// FindNearest returns the labels of the k approximate nearest neighbours of v.
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	s.logger.Debug("hnswflat search", zap.Int("k", k), zap.Int("ef_search", s.efSearch))

	var query [domain.VectorSize]float32
	for i, f := range v {
		query[i] = float32(f)
	}

	// Precompute residual = query - sq8Min for SQ8 distance.
	var residual [domain.VectorSize]float32
	if s.sq8 {
		for d := 0; d < domain.VectorSize; d++ {
			residual[d] = query[d] - s.sq8Min[d]
		}
	}

	vis := s.visPool.get()
	defer s.visPool.put(vis)

	ef := s.efSearch
	if ef < k {
		ef = k
	}

	// 1. Greedy descent through layers L-1 … 1 with ef=1.
	cur := s.entry
	for layer := s.l - 1; layer >= 1; layer-- {
		cur = s.greedyLayer(residual, query, cur, layer, vis)
	}

	// 2. Beam search at layer 0.
	results := s.beamSearchL0(residual, query, cur, ef, vis)

	// Return top-k labels.
	if len(results) > k {
		results = results[:k]
	}
	out := make([]string, len(results))
	for i, it := range results {
		if s.labels[it.node] == 1 {
			out[i] = "fraud"
		} else {
			out[i] = "legit"
		}
	}
	s.logger.Debug("hnswflat search done", zap.Int("returned", len(out)))
	return out
}

// greedyLayer performs ef=1 greedy search at layer, starting from entry, and returns
// the nearest node found at that layer to use as entry for the next lower layer.
func (s *Searcher) greedyLayer(residual, query [domain.VectorSize]float32, entry uint32, layer int, vis []bool) uint32 {
	best := entry
	bestDist := s.dist(residual, query, entry)

	changed := true
	for changed {
		changed = false
		for _, nb := range s.upperNeighbors(best, layer) {
			if nb == emptySlot {
				continue
			}
			d := s.dist(residual, query, nb)
			if d < bestDist {
				bestDist = d
				best = nb
				changed = true
			}
		}
	}
	return best
}

// beamSearchL0 performs ef-beam search at layer 0 starting from entry.
// Returns results sorted by distance ascending (closest first).
func (s *Searcher) beamSearchL0(residual, query [domain.VectorSize]float32, entry uint32, ef int, vis []bool) []hItem {
	d0 := s.dist(residual, query, entry)

	cands := make(minHeap, 0, ef*2)
	results := make(maxHeap, 0, ef+1)

	heapPush(&cands, hItem{d0, entry})
	heapPushMax(&results, hItem{d0, entry})
	vis[entry] = true

	for len(cands) > 0 {
		c := heapPop(&cands)
		// If the closest candidate is worse than the ef-th best result, stop.
		if len(results) >= ef && c.dist > results[0].dist {
			break
		}
		base := int(c.node) * 2 * s.m
		for j := 0; j < 2*s.m; j++ {
			nb := s.l0adj[base+j]
			if nb == emptySlot || vis[nb] {
				continue
			}
			vis[nb] = true
			d := s.dist(residual, query, nb)
			if len(results) < ef || d < results[0].dist {
				heapPush(&cands, hItem{d, nb})
				heapPushMax(&results, hItem{d, nb})
				if len(results) > ef {
					heapPopMax(&results)
				}
			}
		}
	}

	// Sort results ascending (closest first).
	out := []hItem(results)
	sort.Slice(out, func(i, j int) bool { return out[i].dist < out[j].dist })
	return out
}

// dist computes the approximate SQ8 (or exact float32) squared L2 distance from the
// precomputed residual/query to node's stored vector.
func (s *Searcher) dist(residual, query [domain.VectorSize]float32, node uint32) float32 {
	if s.sq8 {
		base := int(node) * domain.VectorSize
		vec := s.vecs8[base : base+domain.VectorSize]
		var sum float32
		for d := 0; d < domain.VectorSize; d++ {
			diff := residual[d] - float32(vec[d])*s.sq8Scale[d]
			sum += diff * diff
		}
		return sum
	}
	// Float32 path (sq8=false).
	base := int(node) * domain.VectorSize
	vec := s.vecsF32[base : base+domain.VectorSize]
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := query[d] - vec[d]
		sum += diff * diff
	}
	return sum
}

// upperNeighbors returns the M adjacency slots for node at the given layer (1+).
// Each upper node stores exactly levels[node]×M slots padded with emptySlot,
// so layer l starts at offset (l-1)×M relative to the node's base — O(1) access.
// Returns nil if node is not in layer 1+ or does not reach the requested layer.
func (s *Searcher) upperNeighbors(node uint32, layer int) []uint32 {
	if layer < 1 || layer > int(s.levels[node]) {
		return nil
	}
	idx := sort.Search(len(s.upperIDs), func(i int) bool { return s.upperIDs[i] >= node })
	if idx >= len(s.upperIDs) || s.upperIDs[idx] != node {
		return nil
	}
	base := int(s.upperOff[idx]) + (layer-1)*s.m
	return s.upperAdj[base : base+s.m]
}

// hItem is a (distance, node) pair used in search heaps.
type hItem struct {
	dist float32
	node uint32
}

// minHeap / maxHeap: manual inline heaps to avoid container/heap interface overhead.
type minHeap []hItem
type maxHeap []hItem

func heapPush(h *minHeap, it hItem) {
	*h = append(*h, it)
	i := len(*h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if (*h)[parent].dist <= (*h)[i].dist {
			break
		}
		(*h)[parent], (*h)[i] = (*h)[i], (*h)[parent]
		i = parent
	}
}

func heapPop(h *minHeap) hItem {
	root := (*h)[0]
	last := len(*h) - 1
	(*h)[0] = (*h)[last]
	*h = (*h)[:last]
	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		min := i
		if l < len(*h) && (*h)[l].dist < (*h)[min].dist {
			min = l
		}
		if r < len(*h) && (*h)[r].dist < (*h)[min].dist {
			min = r
		}
		if min == i {
			break
		}
		(*h)[i], (*h)[min] = (*h)[min], (*h)[i]
		i = min
	}
	return root
}

func heapPushMax(h *maxHeap, it hItem) {
	*h = append(*h, it)
	i := len(*h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if (*h)[parent].dist >= (*h)[i].dist {
			break
		}
		(*h)[parent], (*h)[i] = (*h)[i], (*h)[parent]
		i = parent
	}
}

func heapPopMax(h *maxHeap) hItem {
	root := (*h)[0]
	last := len(*h) - 1
	(*h)[0] = (*h)[last]
	*h = (*h)[:last]
	i := 0
	for {
		l, r := 2*i+1, 2*i+2
		max := i
		if l < len(*h) && (*h)[l].dist > (*h)[max].dist {
			max = l
		}
		if r < len(*h) && (*h)[r].dist > (*h)[max].dist {
			max = r
		}
		if max == i {
			break
		}
		(*h)[i], (*h)[max] = (*h)[max], (*h)[i]
		i = max
	}
	return root
}

// visPool manages a sync.Pool of reusable visited-flag slices of length n.
type visPool struct {
	pool sync.Pool
}

func newVisPool(n int) visPool {
	return visPool{
		pool: sync.Pool{New: func() any {
			s := make([]bool, n)
			return &s
		}},
	}
}

func (vp *visPool) get() []bool {
	return *vp.pool.Get().(*[]bool)
}

func (vp *visPool) put(v []bool) {
	for i := range v {
		v[i] = false
	}
	vp.pool.Put(&v)
}
