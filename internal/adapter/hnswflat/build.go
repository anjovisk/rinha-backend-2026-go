// Build-time helpers for the flat HNSW adapter: HNSW construction algorithm,
// SQ8 quantization, and index serialization to the flat binary format.
// These are called only by cmd/preprocess (BUILD_HNSWFLAT=true) and by New (tests).
package hnswflat

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// DefaultEfConstruction is the candidate-list size used during graph construction.
// Larger values build higher-quality graphs at the cost of more build time.
const DefaultEfConstruction = 100

// Reference is a labelled entry for building a Searcher from in-memory data (tests).
type Reference struct {
	Vector domain.Vector
	Label  string
}

// New builds a Searcher entirely in memory from refs. Intended for tests.
func New(refs []Reference, M, efSearch, efConstruction int, sq8 bool, logger *zap.Logger) *Searcher {
	n := len(refs)
	if n == 0 {
		return &Searcher{logger: logger}
	}
	if M <= 0 {
		M = DefaultM
	}
	if efSearch <= 0 {
		efSearch = DefaultEfSearch
	}
	if efConstruction <= 0 {
		efConstruction = DefaultEfConstruction
	}

	vecs := make([]float32, n*domain.VectorSize)
	rawLabels := make([]uint8, n)
	for i, r := range refs {
		for d, f := range r.Vector {
			vecs[i*domain.VectorSize+d] = float32(f)
		}
		if r.Label == "fraud" {
			rawLabels[i] = 1
		}
	}

	g := buildGraph(vecs, rawLabels, n, M, efConstruction, logger)

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecs8 []uint8
	var vecsF32 []float32

	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		vecs8 = quantize(vecs, n, sq8Min, sq8Scale)
	} else {
		vecsF32 = vecs
	}

	s := &Searcher{
		vecs8:    vecs8,
		vecsF32:  vecsF32,
		labels:   rawLabels,
		levels:   g.levels,
		l0adj:    g.l0adj,
		upperIDs: g.upperIDs,
		upperOff: g.upperOff,
		upperAdj: g.upperAdj,
		n:        n,
		m:        M,
		l:        g.numLayers,
		entry:    g.entry,
		efSearch: efSearch,
		sq8:      sq8,
		sq8Min:   sq8Min,
		sq8Scale: sq8Scale,
		logger:   logger,
	}
	s.visPool = newVisPool(n)
	return s
}

// Build constructs a flat HNSW index from the binary reference file at binPath
// and writes it to outPath. M controls the graph degree; efConstruction controls
// the candidate-list size during construction (higher → better recall, slower build).
// sq8 controls SQ8 vector quantization.
func Build(binPath, outPath string, M, efConstruction int, sq8 bool, logger *zap.Logger) error {
	if M <= 0 {
		M = DefaultM
	}
	if efConstruction <= 0 {
		efConstruction = DefaultEfConstruction
	}

	f, err := os.Open(binPath)
	if err != nil {
		return fmt.Errorf("open %s: %w", binPath, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("stat %s: %w", binPath, err)
	}
	fileSize := int(info.Size())
	data, err := syscall.Mmap(int(f.Fd()), 0, fileSize, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fmt.Errorf("mmap %s: %w", binPath, err)
	}
	defer syscall.Munmap(data) //nolint:errcheck

	n := int(binary.LittleEndian.Uint32(data[:4]))
	vecBytes := n * domain.VectorSize * 4
	if fileSize < 4+vecBytes+n {
		return fmt.Errorf("%s: file too small for %d entries", binPath, n)
	}

	vecPtr := (*float32)(unsafe.Pointer(&data[4]))
	vecs := unsafe.Slice(vecPtr, n*domain.VectorSize)
	rawLabels := data[4+vecBytes : 4+vecBytes+n]

	logger.Info("hnswflat build starting",
		zap.Int("N", n),
		zap.Int("M", M),
		zap.Int("ef_construction", efConstruction),
		zap.Bool("sq8", sq8),
		zap.Int("cpus", runtime.NumCPU()),
	)

	g := buildGraph(vecs, rawLabels, n, M, efConstruction, logger)
	logger.Info("hnswflat graph built",
		zap.Int("layers", g.numLayers),
		zap.Uint32("entry", g.entry),
	)

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecs8 []uint8
	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		vecs8 = quantize(vecs, n, sq8Min, sq8Scale)
		logger.Info("hnswflat SQ8 quantization complete")
	}

	return writeIndex(outPath, n, M, g, sq8, sq8Min, sq8Scale, vecs8, vecs, rawLabels, logger)
}

// ---- Graph construction --------------------------------------------------------

// graph holds the in-memory adjacency lists produced during construction.
type graph struct {
	levels    []uint8
	l0adj     []uint32
	upperIDs  []uint32
	upperOff  []uint32
	upperAdj  []uint32
	numLayers int
	entry     uint32
}

// shardLock provides per-node locking via 256 shards to avoid excessive lock contention.
const numShards = 256

type shardedMutex [numShards]sync.Mutex

func (sm *shardedMutex) Lock(id uint32)   { sm[id%numShards].Lock() }
func (sm *shardedMutex) Unlock(id uint32) { sm[id%numShards].Unlock() }

// buildGraph runs the HNSW construction algorithm over vecs and returns the
// adjacency structure ready for serialization.
func buildGraph(vecs []float32, rawLabels []uint8, n, M, efConstruction int, logger *zap.Logger) *graph {
	// Geometric level distribution: P(level ≥ l) = (1/M)^l, mL = 1/ln(M).
	mL := 1.0 / math.Log(float64(M))

	// Pre-assign levels to all nodes using a random source.
	levels := make([]uint8, n)
	maxLevel := 0
	for i := 0; i < n; i++ {
		l := int(math.Floor(-math.Log(rand.Float64()) * mL))
		if l > 15 {
			l = 15 // cap at uint8 max reasonable value
		}
		levels[i] = uint8(l)
		if l > maxLevel {
			maxLevel = l
		}
	}
	numLayers := maxLevel + 1

	// Initialize layer-0 adjacency: N × 2M slots, all emptySlot.
	l0Size := n * 2 * M
	l0adj := make([]uint32, l0Size)
	for i := range l0adj {
		l0adj[i] = emptySlot
	}

	// Upper-layer adjacency stored as dynamic slices (only ~N/M nodes).
	// upperBuild[node] = [][]uint32 per layer (index 0 = layer 1).
	upperBuild := make(map[uint32][][]uint32, n/M)
	var upperMu sync.Mutex

	var sm shardedMutex

	// Sort nodes by level descending to insert higher-level nodes first
	// (improves graph quality without changing correctness).
	order := make([]uint32, n)
	for i := range order {
		order[i] = uint32(i)
	}
	sort.Slice(order, func(i, j int) bool {
		return levels[order[i]] > levels[order[j]]
	})

	// Entry point protected by RW lock.
	var epMu sync.RWMutex
	ep := uint32(0)
	epLevel := -1

	// distFn computes squared L2 between two nodes using float32 vectors.
	distToVec := func(query []float32, b uint32) float32 {
		vb := vecs[int(b)*domain.VectorSize : int(b)*domain.VectorSize+domain.VectorSize]
		var sum float32
		for d := 0; d < domain.VectorSize; d++ {
			diff := query[d] - vb[d]
			sum += diff * diff
		}
		return sum
	}

	// Build visited set from sync.Pool (size N booleans, reused per goroutine).
	type localVis struct{ v []bool }
	visAlloc := sync.Pool{New: func() any { return &localVis{make([]bool, n)} }}

	numCPU := runtime.NumCPU()
	if numCPU > 8 {
		numCPU = 8
	}

	var processed int64
	var wg sync.WaitGroup
	chunkSize := (n + numCPU - 1) / numCPU

	for cpu := 0; cpu < numCPU; cpu++ {
		lo := cpu * chunkSize
		hi := lo + chunkSize
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			lv := visAlloc.Get().(*localVis)
			defer visAlloc.Put(lv)
			vis := lv.v

			for idx := lo; idx < hi; idx++ {
				q := order[idx]
				qLevel := int(levels[q])
				qVec := vecs[int(q)*domain.VectorSize : int(q)*domain.VectorSize+domain.VectorSize]

				// Read current entry point.
				epMu.RLock()
				curEP := ep
				curEPLevel := epLevel
				epMu.RUnlock()

				if curEPLevel < 0 {
					// First node ever: become entry point.
					epMu.Lock()
					if epLevel < 0 {
						ep = q
						epLevel = qLevel
					}
					curEP = ep
					curEPLevel = epLevel
					epMu.Unlock()
					if curEP == q {
						continue // no neighbors to add for the very first node
					}
				}

				// Greedy descent from curEPLevel down to qLevel+1 (ef=1).
				entryNode := curEP
				for layer := curEPLevel; layer > qLevel; layer-- {
					entryNode = greedyUpperBuild(qVec, entryNode, layer, upperBuild, &upperMu, distToVec)
				}

				// Beam search from min(curEPLevel, qLevel) down to 0.
				for layer := min(curEPLevel, qLevel); layer >= 0; layer-- {
					// Search this layer with ef=efConstruction.
					var candidates []hItem
					if layer == 0 {
						candidates = searchLayerBuild(qVec, entryNode, efConstruction, layer,
							l0adj, M, upperBuild, &upperMu, distToVec, vis, n)
					} else {
						candidates = searchLayerBuild(qVec, entryNode, efConstruction, layer,
							l0adj, M, upperBuild, &upperMu, distToVec, vis, n)
					}

					// Clear visited for reuse.
					for _, c := range candidates {
						vis[c.node] = false
					}

					// Select neighbors (heuristic: skip nodes "shadowed" by closer ones).
					neighbors := selectNeighborsHeuristic(qVec, candidates, M*2, vecs)
					if layer > 0 {
						neighbors = selectNeighborsHeuristic(qVec, candidates, M, vecs)
					}
					maxDeg := 2 * M
					if layer > 0 {
						maxDeg = M
					}

					// Add bidirectional connections.
					for _, nb := range neighbors {
						sm.Lock(q)
						addConn(l0adj, upperBuild, &upperMu, q, nb.node, layer, M)
						sm.Unlock(q)

						sm.Lock(nb.node)
						addConn(l0adj, upperBuild, &upperMu, nb.node, q, layer, M)
						// Prune if over capacity.
						pruneConn(l0adj, upperBuild, &upperMu, nb.node, layer, maxDeg, vecs, M)
						sm.Unlock(nb.node)
					}

					if layer > 0 {
						entryNode = candidates[0].node
					}
				}

				// Update global entry point if q has a higher level.
				if qLevel > curEPLevel {
					epMu.Lock()
					if qLevel > epLevel {
						ep = q
						epLevel = qLevel
					}
					epMu.Unlock()
				}

				done := atomic.AddInt64(&processed, 1)
				if done%500_000 == 0 {
					logger.Info("hnswflat build progress", zap.Int64("processed", done), zap.Int("total", n))
				}
			}
		}(lo, hi)
	}
	wg.Wait()

	// Compact upper-layer adjacency into CSR.
	upperIDs, upperOff, upperAdj := compactUpper(upperBuild, levels, numLayers, M)

	return &graph{
		levels:    levels,
		l0adj:     l0adj,
		upperIDs:  upperIDs,
		upperOff:  upperOff,
		upperAdj:  upperAdj,
		numLayers: numLayers,
		entry:     ep,
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// greedyUpperBuild performs greedy ef=1 search at layer in the build graph.
func greedyUpperBuild(qVec []float32, entry uint32, layer int, upper map[uint32][][]uint32, mu *sync.Mutex, distFn func([]float32, uint32) float32) uint32 {
	best := entry
	bestDist := distFn(qVec, entry)
	for {
		mu.Lock()
		nbs := upperNeighborsBuild(upper, best, layer)
		mu.Unlock()
		improved := false
		for _, nb := range nbs {
			if nb == emptySlot {
				continue
			}
			d := distFn(qVec, nb)
			if d < bestDist {
				bestDist = d
				best = nb
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	return best
}

// searchLayerBuild performs ef-beam search at the given layer in the build graph.
// vis is reset for each used node before return.
func searchLayerBuild(qVec []float32, entry uint32, ef, layer int,
	l0adj []uint32, M int,
	upper map[uint32][][]uint32, upperMu *sync.Mutex,
	distFn func([]float32, uint32) float32,
	vis []bool, n int) []hItem {

	d0 := distFn(qVec, entry)
	cands := make(minHeap, 0, ef*2)
	results := make(maxHeap, 0, ef+1)

	heapPush(&cands, hItem{d0, entry})
	heapPushMax(&results, hItem{d0, entry})
	vis[entry] = true

	for len(cands) > 0 {
		c := heapPop(&cands)
		if len(results) >= ef && c.dist > results[0].dist {
			break
		}

		var nbs []uint32
		if layer == 0 {
			base := int(c.node) * 2 * M
			nbs = l0adj[base : base+2*M]
		} else {
			upperMu.Lock()
			nbs = append([]uint32(nil), upperNeighborsBuild(upper, c.node, layer)...)
			upperMu.Unlock()
		}

		for _, nb := range nbs {
			if nb == emptySlot || vis[nb] {
				continue
			}
			vis[nb] = true
			d := distFn(qVec, nb)
			if len(results) < ef || d < results[0].dist {
				heapPush(&cands, hItem{d, nb})
				heapPushMax(&results, hItem{d, nb})
				if len(results) > ef {
					heapPopMax(&results)
				}
			}
		}
	}

	// Return sorted ascending and clear vis.
	out := []hItem(results)
	sort.Slice(out, func(i, j int) bool { return out[i].dist < out[j].dist })
	for _, it := range out {
		vis[it.node] = false
	}
	vis[entry] = false
	return out
}

// selectNeighborsHeuristic selects the best M neighbors from candidates using
// the HNSW diversity heuristic: prefer candidates that are closer to q than
// to any already-selected neighbor.
func selectNeighborsHeuristic(qVec []float32, candidates []hItem, M int, vecs []float32) []hItem {
	if len(candidates) <= M {
		return candidates
	}
	result := make([]hItem, 0, M)
	for _, c := range candidates { // candidates sorted ascending by dist
		if len(result) >= M {
			break
		}
		keep := true
		for _, r := range result {
			// Skip c if it's closer to an existing result than to q.
			va := vecs[int(c.node)*domain.VectorSize : int(c.node)*domain.VectorSize+domain.VectorSize]
			vr := vecs[int(r.node)*domain.VectorSize : int(r.node)*domain.VectorSize+domain.VectorSize]
			var dCR float32
			for d := 0; d < domain.VectorSize; d++ {
				diff := va[d] - vr[d]
				dCR += diff * diff
			}
			if dCR < c.dist {
				keep = false
				break
			}
		}
		if keep {
			result = append(result, c)
		}
	}
	return result
}

// upperNeighborsBuild returns the neighbor slice for node at layer from the build map.
// Caller must hold upperMu.
func upperNeighborsBuild(upper map[uint32][][]uint32, node uint32, layer int) []uint32 {
	layers, ok := upper[node]
	if !ok || layer-1 >= len(layers) {
		return nil
	}
	return layers[layer-1]
}

// addConn adds neighbor to node's adjacency at the given layer.
func addConn(l0adj []uint32, upper map[uint32][][]uint32, mu *sync.Mutex, node, neighbor uint32, layer, M int) {
	if layer == 0 {
		base := int(node) * 2 * M
		for j := 0; j < 2*M; j++ {
			if l0adj[base+j] == emptySlot {
				l0adj[base+j] = neighbor
				return
			}
		}
		return // full; pruneConn will handle
	}
	mu.Lock()
	defer mu.Unlock()
	layers := upper[node]
	need := layer // layers[0] = layer1, layers[layer-1] = layerL
	for len(layers) < need {
		layers = append(layers, nil)
	}
	layers[layer-1] = append(layers[layer-1], neighbor)
	upper[node] = layers
}

// pruneConn trims node's connections at layer to at most maxDeg using the heuristic.
func pruneConn(l0adj []uint32, upper map[uint32][][]uint32, mu *sync.Mutex, node uint32, layer, maxDeg int, vecs []float32, M int) {
	nodeVec := vecs[int(node)*domain.VectorSize : int(node)*domain.VectorSize+domain.VectorSize]

	if layer == 0 {
		base := int(node) * 2 * M
		var cands []hItem
		for j := 0; j < 2*M; j++ {
			nb := l0adj[base+j]
			if nb == emptySlot {
				continue
			}
			d := l2sq(nodeVec, vecs[int(nb)*domain.VectorSize:int(nb)*domain.VectorSize+domain.VectorSize])
			cands = append(cands, hItem{d, nb})
		}
		if len(cands) <= maxDeg {
			return
		}
		sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
		chosen := selectNeighborsHeuristic(nodeVec, cands, maxDeg, vecs)
		// Reset and write chosen.
		for j := 0; j < 2*M; j++ {
			l0adj[base+j] = emptySlot
		}
		for j, c := range chosen {
			l0adj[base+j] = c.node
		}
		return
	}

	mu.Lock()
	defer mu.Unlock()
	layers, ok := upper[node]
	if !ok || layer-1 >= len(layers) {
		return
	}
	nbs := layers[layer-1]
	if len(nbs) <= maxDeg {
		return
	}
	var cands []hItem
	for _, nb := range nbs {
		d := l2sq(nodeVec, vecs[int(nb)*domain.VectorSize:int(nb)*domain.VectorSize+domain.VectorSize])
		cands = append(cands, hItem{d, nb})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	chosen := selectNeighborsHeuristic(nodeVec, cands, maxDeg, vecs)
	newNbs := make([]uint32, len(chosen))
	for i, c := range chosen {
		newNbs[i] = c.node
	}
	layers[layer-1] = newNbs
	upper[node] = layers
}

func l2sq(a, b []float32) float32 {
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := a[d] - b[d]
		sum += diff * diff
	}
	return sum
}

// compactUpper converts the build-time map into sorted CSR arrays.
// Each upper node with max layer k stores exactly k×M uint32 slots in adj,
// padded with emptySlot so that layer l starts at offset (l-1)×M relative
// to the node's base — enabling O(1) random access without extra per-layer offsets.
func compactUpper(upper map[uint32][][]uint32, levels []uint8, numLayers, M int) (ids, offsets, adj []uint32) {
	ids = make([]uint32, 0, len(upper))
	for id := range upper {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })

	offsets = make([]uint32, len(ids)+1)
	var total uint32
	for i, id := range ids {
		offsets[i] = total
		total += uint32(int(levels[id]) * M) // levels[id] = max layer (1-based)
	}
	offsets[len(ids)] = total

	adj = make([]uint32, total)
	for i := range adj {
		adj[i] = emptySlot
	}

	for i, id := range ids {
		layers := upper[id]
		base := int(offsets[i])
		maxL := int(levels[id])
		for l := 1; l <= maxL; l++ {
			layerStart := base + (l-1)*M
			if l-1 < len(layers) {
				for j, nb := range layers[l-1] {
					if j < M {
						adj[layerStart+j] = nb
					}
				}
			}
		}
	}
	return ids, offsets, adj
}

// ---- SQ8 helpers ---------------------------------------------------------------

func computeSQ8Params(vecs []float32, n int) (minV, scale [domain.VectorSize]float32) {
	maxV := [domain.VectorSize]float32{}
	for d := range minV {
		minV[d] = math.MaxFloat32
		maxV[d] = -math.MaxFloat32
	}
	for i := 0; i < n; i++ {
		base := i * domain.VectorSize
		for d := 0; d < domain.VectorSize; d++ {
			v := vecs[base+d]
			if v < minV[d] {
				minV[d] = v
			}
			if v > maxV[d] {
				maxV[d] = v
			}
		}
	}
	for d := range scale {
		r := maxV[d] - minV[d]
		if r == 0 {
			scale[d] = 1
		} else {
			scale[d] = r / 255.0
		}
	}
	return
}

func quantize(vecs []float32, n int, minV, scale [domain.VectorSize]float32) []uint8 {
	out := make([]uint8, n*domain.VectorSize)
	for i := 0; i < n; i++ {
		base := i * domain.VectorSize
		for d := 0; d < domain.VectorSize; d++ {
			f := (vecs[base+d] - minV[d]) / scale[d]
			if f < 0 {
				f = 0
			} else if f > 255 {
				f = 255
			}
			out[base+d] = uint8(f + 0.5)
		}
	}
	return out
}

// ---- Serialization -------------------------------------------------------------

func writeIndex(outPath string, n, M int, g *graph, sq8 bool,
	sq8Min, sq8Scale [domain.VectorSize]float32,
	vecs8 []uint8, vecsF32 []float32, labels []uint8,
	logger *zap.Logger) error {

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20)

	var b4 [4]byte
	u32 := func(v uint32) { binary.LittleEndian.PutUint32(b4[:], v); w.Write(b4[:]) } //nolint:errcheck
	f32 := func(v float32) { u32(math.Float32bits(v)) }

	upperN := uint32(len(g.upperIDs))
	flags := uint32(0)
	if sq8 {
		flags |= flagSQ8
	}

	u32(magic)
	u32(uint32(n))
	u32(uint32(M))
	u32(uint32(g.numLayers))
	u32(g.entry)
	u32(upperN)
	u32(flags)
	u32(0) // reserved

	if sq8 {
		for _, v := range sq8Min {
			f32(v)
		}
		for _, v := range sq8Scale {
			f32(v)
		}
	}

	if sq8 {
		w.Write(vecs8) //nolint:errcheck
	} else {
		for _, v := range vecsF32 {
			f32(v)
		}
	}

	w.Write(labels) //nolint:errcheck
	w.Write(g.levels) //nolint:errcheck

	// Pad to 4-byte boundary.
	headerAndFixed := 32 // magic + 7 uint32 fields = 32 bytes
	if sq8 {
		headerAndFixed += domain.VectorSize * 4 * 2 // sq8 params
	}
	vecBytes := n * domain.VectorSize
	if !sq8 {
		vecBytes = n * domain.VectorSize * 4
	}
	pos := headerAndFixed + vecBytes + n + n // labels + levels
	if rem := pos % 4; rem != 0 {
		pad := make([]byte, 4-rem)
		w.Write(pad) //nolint:errcheck
	}

	for _, v := range g.l0adj {
		u32(v)
	}
	for _, v := range g.upperIDs {
		u32(v)
	}
	for _, v := range g.upperOff {
		u32(v)
	}
	for _, v := range g.upperAdj {
		u32(v)
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush %s: %w", outPath, err)
	}
	logger.Info("hnswflat index written", zap.String("path", outPath), zap.Bool("sq8", sq8))
	return nil
}
