// This file contains the Vamana graph construction algorithm and the Build function
// that writes a ready-to-serve index from references.bin.
//
// Vamana algorithm summary:
//  1. Compute medoid (node closest to the centroid of all vectors).
//  2. Initialise each node with R random outgoing edges.
//  3. Two passes with increasing alpha (1.0, then alpha):
//     for each node p in random order:
//       a. GreedySearch from medoid to p with beam width buildL → candidate set C
//       b. candidates = C ∪ current_neighbors(p)
//       c. RobustPrune(p, candidates, alpha, R)
//       d. For each new neighbor n of p: add back-edge p→n; prune n if over-full.
//
// RobustPrune greedily picks the closest candidate, then removes all candidates r
// for which α·dist(p*,r) ≤ dist(p,r) (i.e. p* "dominates" the path to r).
// This creates long-range edges that compress search paths.
//
// Invoked only by cmd/preprocess (BUILD_VAMANA=true) and by New (tests).
package vamana

import (
	"bufio"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"fmt"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// Reference is a labelled entry used to build a Searcher from in-memory data.
// Intended for tests; production code should call Build then Open.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is the fraud classification: "fraud" or "legit".
	Label string
}

// New builds a Searcher from refs in memory without requiring an on-disk file.
// r is the max degree; l is the query beam width; alpha is the RobustPrune multiplier.
// Build beam width uses DefaultBuildL (appropriate for small test datasets).
// sq8 enables 8-bit scalar quantization; use DefaultSQ8 when unsure. Intended for tests.
func New(refs []Reference, r, l int, alpha float32, sq8 bool, logger *zap.Logger) *Searcher {
	n := len(refs)
	if n == 0 {
		return &Searcher{l: defaultL(l), logger: logger}
	}

	vecs := make([]float32, n*domain.VectorSize)
	labels := make([]uint8, n)
	for i, ref := range refs {
		for d, f := range ref.Vector {
			vecs[i*domain.VectorSize+d] = float32(f)
		}
		if ref.Label == "fraud" {
			labels[i] = 1
		}
	}

	r = clampR(r, n)
	// Pass 0 as buildL to use the package default (capped to n for small datasets).
	graph, medoid := buildVamana(vecs, n, r, defaultBuildL(r, 0), alpha)

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecs8 []uint8
	var vecs32 []float32
	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		vecs8 = quantizeAll(vecs, n, sq8Min, sq8Scale)
	} else {
		vecs32 = vecs
	}

	return &Searcher{
		n:        n,
		r:        r,
		l:        defaultL(l),
		medoid:   medoid,
		sq8:      sq8,
		sq8Min:   sq8Min,
		sq8Scale: sq8Scale,
		graph:    graph,
		vecs:     vecs32,
		vecs8:    vecs8,
		labels:   labels,
		logger:   logger,
	}
}

// Build constructs a Vamana index from the flat binary file at binPath and writes
// it to outPath. r controls max degree; buildL is the beam width used during
// construction (0 means use DefaultBuildL=125); alpha is the RobustPrune multiplier;
// sq8 enables 8-bit quantization of stored vectors.
//
// The SQ8 flag is stored in the file header so Open needs no env var at serve time.
// Memory at serve time (mmap, shared between instances): ~192 MB graph + ~42 MB
// vectors (SQ8) + ~3 MB labels for 3M entries with R=16.
//
// Build time for N=3M: ~20–40 min with multiple CPUs; ~1–2 h with 1 CPU.
// Reduce VAMANA_BUILD_L to 64–75 to halve build time at a small recall cost.
func Build(binPath, outPath string, r, buildL int, alpha float32, sq8 bool, logger *zap.Logger) error {
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

	if fileSize < 4 {
		return fmt.Errorf("%s: file too small", binPath)
	}
	n := int(binary.LittleEndian.Uint32(data[:4]))
	vectorBytes := n * domain.VectorSize * 4
	required := 4 + vectorBytes + n
	if fileSize < required {
		return fmt.Errorf("%s: need %d bytes for %d entries, have %d", binPath, required, n, fileSize)
	}

	vecPtr := (*float32)(unsafe.Pointer(&data[4]))
	vecs := unsafe.Slice(vecPtr, n*domain.VectorSize)
	rawLabels := data[4+vectorBytes : 4+vectorBytes+n]

	r = clampR(r, n)
	buildL = defaultBuildL(r, buildL)

	logger.Info("Vamana build starting",
		zap.Int("entries", n),
		zap.Int("R", r),
		zap.Int("build_L", buildL),
		zap.Float32("alpha", alpha),
		zap.Bool("sq8", sq8),
		zap.Int("cpus", runtime.NumCPU()),
	)

	graph, medoid := buildVamana(vecs, n, r, buildL, alpha)
	logger.Info("Vamana graph built", zap.Uint32("medoid", medoid))

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecs8 []uint8
	var vecs32 []float32
	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		logger.Info("SQ8 parameters computed")
		vecs8 = quantizeAll(vecs, n, sq8Min, sq8Scale)
		logger.Info("SQ8 quantization complete")
	} else {
		// Copy vecs from mmap to heap so we can pass to writeVamana after Munmap.
		vecs32 = make([]float32, n*domain.VectorSize)
		copy(vecs32, vecs)
	}

	labels := make([]uint8, n)
	copy(labels, rawLabels)

	if err := writeVamana(outPath, n, r, medoid, sq8, sq8Min, sq8Scale, graph, vecs8, vecs32, labels); err != nil {
		return err
	}

	logger.Info("Vamana index written",
		zap.String("path", outPath),
		zap.Int("R", r),
		zap.Bool("sq8", sq8),
	)
	return nil
}

// buildVamana runs the Vamana construction algorithm on vecs (n vectors, float32,
// row-major, stride=VectorSize) with max degree r and beam width buildL.
// Returns the flat adjacency list and the medoid index.
//
// The algorithm runs two passes: the first with α=1.0 (exact prune) and the second
// with the caller-supplied alpha, which inserts long-range edges.
func buildVamana(vecs []float32, n, r, buildL int, alpha float32) (graph []uint32, medoid uint32) {
	sentinel := uint32(n)

	// graph[i*r : i*r+r] holds node i's neighbors; empty slots = sentinel.
	graph = make([]uint32, n*r)
	for i := range graph {
		graph[i] = sentinel
	}

	// graphDeg[i] is the current number of valid neighbors for node i.
	graphDeg := make([]uint8, n)

	// Sharded mutexes for concurrent edge updates.
	const numShards = 512
	var shards [numShards]sync.Mutex
	lockNode := func(id int) { shards[id%numShards].Lock() }
	unlockNode := func(id int) { shards[id%numShards].Unlock() }

	// 1. Find medoid: node closest to the centroid of all vectors.
	medoid = findMedoid(vecs, n)

	// 2. Random R-regular initialisation.
	rng := rand.New(rand.NewSource(42))
	perm := rng.Perm(n)
	for i := 0; i < n; i++ {
		neighbors := graph[i*r : i*r+r]
		cnt := 0
		for _, j := range perm {
			if j == i || cnt == r {
				break
			}
			neighbors[cnt] = uint32(j)
			cnt++
		}
		// Re-shuffle perm for next node (Fisher-Yates tail swap keeps it cheap).
		rng.Shuffle(n, func(a, b int) { perm[a], perm[b] = perm[b], perm[a] })
		graphDeg[i] = uint8(cnt)
	}

	numCPU := runtime.NumCPU()

	// 3. Two passes.
	for passIdx, passAlpha := range []float32{1.0, alpha} {
		order := rng.Perm(n)

		// Process nodes in parallel: greedy search is read-heavy; edge updates are locked.
		chunkSize := (n + numCPU - 1) / numCPU
		var wg sync.WaitGroup
		for cpu := 0; cpu < numCPU; cpu++ {
			lo := cpu * chunkSize
			hi := lo + chunkSize
			if hi > n {
				hi = n
			}
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(nodes []int) {
				defer wg.Done()
				// Per-goroutine reusable buffers.
				candBuf := make([]uint32, 0, buildL+r+1)
				for _, p := range nodes {
					query := vecs[p*domain.VectorSize : (p+1)*domain.VectorSize]

					// a. Greedy search from medoid to p's vector.
					found := greedySearchBuild(graph, vecs, n, r, int(medoid), p, query, buildL, sentinel)

					// b. Merge with p's current neighbors.
					lockNode(p)
					deg := int(graphDeg[p])
					candBuf = candBuf[:0]
					// Add greedy results.
					for _, c := range found {
						if uint32(c) != sentinel && c != p {
							candBuf = append(candBuf, uint32(c))
						}
					}
					// Add existing neighbors.
					for _, nb := range graph[p*r : p*r+deg] {
						if nb != sentinel && int(nb) != p {
							candBuf = appendUnique(candBuf, nb)
						}
					}
					unlockNode(p)

					// c. RobustPrune: update p's outgoing edges.
					pruned := robustPrune(vecs, p, candBuf, passAlpha, r, sentinel)

					lockNode(p)
					copy(graph[p*r:p*r+r], pruned)
					for i := len(pruned); i < r; i++ {
						graph[p*r+i] = sentinel
					}
					graphDeg[p] = uint8(len(pruned))
					unlockNode(p)

					// d. Add back-edges: for each new neighbor nb of p, add p to nb's list.
					for _, nb := range pruned {
						if nb == sentinel {
							break
						}
						nbInt := int(nb)
						lockNode(nbInt)
						// Collect nb's current neighbors + p.
						nbDeg := int(graphDeg[nbInt])
						backCands := make([]uint32, nbDeg+1)
						copy(backCands, graph[nbInt*r:nbInt*r+nbDeg])
						backCands[nbDeg] = uint32(p)
						if nbDeg+1 > r {
							// Over capacity — prune nb.
							backPruned := robustPrune(vecs, nbInt, backCands, passAlpha, r, sentinel)
							copy(graph[nbInt*r:nbInt*r+r], backPruned)
							for i := len(backPruned); i < r; i++ {
								graph[nbInt*r+i] = sentinel
							}
							graphDeg[nbInt] = uint8(len(backPruned))
						} else {
							graph[nbInt*r+nbDeg] = uint32(p)
							graphDeg[nbInt] = uint8(nbDeg + 1)
						}
						unlockNode(nbInt)
					}
				}
			}(nodesSlice(order, lo, hi))
		}
		wg.Wait()
		_ = passIdx // suppress unused warning
	}

	return graph, medoid
}

// greedySearchBuild is the greedy beam search used during construction.
// It searches from start toward query (which is node p's own vector) and returns
// up to buildL candidate indices, sorted by distance to query.
// p is excluded from the result (no self-loops).
func greedySearchBuild(graph []uint32, vecs []float32, n, r, start, p int, query []float32, buildL int, sentinel uint32) []int {
	type bc struct {
		dist    float32
		id      int
		visited bool
	}

	cands := make([]bc, 0, buildL+r)
	seen := make(map[int]struct{}, buildL*2)

	addCand := func(id int) {
		if id == p {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		d := vecDistSq(query, vecs[id*domain.VectorSize:(id+1)*domain.VectorSize])
		pos := sort.Search(len(cands), func(i int) bool { return cands[i].dist >= d })
		cands = append(cands, bc{})
		copy(cands[pos+1:], cands[pos:])
		cands[pos] = bc{dist: d, id: id}
		if len(cands) > buildL {
			delete(seen, cands[buildL].id)
			cands = cands[:buildL]
		}
	}

	addCand(start)

	for {
		pivot := -1
		for i := range cands {
			if !cands[i].visited {
				pivot = i
				break
			}
		}
		if pivot == -1 {
			break
		}
		cands[pivot].visited = true
		nodeID := cands[pivot].id
		base := nodeID * r
		for _, nb := range graph[base : base+r] {
			if nb == sentinel {
				break
			}
			addCand(int(nb))
		}
	}

	out := make([]int, len(cands))
	for i, c := range cands {
		out[i] = c.id
	}
	return out
}

// robustPrune selects at most r outgoing edges for node p from candidates.
// candidates is a list of node indices (deduped, p excluded, sentinels excluded).
// Returns the pruned neighbor list sorted by distance to p.
//
// The dominance condition from the paper is α·‖p*−r‖ ≤ ‖p−r‖ (Euclidean distances).
// Since dist² values are stored in items[j].dist, squaring both sides gives
// α²·dist²(p*,r) ≤ dist²(p,r) — note the squared alpha on the left.
func robustPrune(vecs []float32, p int, candidates []uint32, alpha float32, r int, sentinel uint32) []uint32 {
	if len(candidates) == 0 {
		return nil
	}

	type ce struct {
		dist float32
		id   uint32
	}

	// alpha² is used in the squared-distance form of the dominance test.
	alphaSq := alpha * alpha

	pVec := vecs[p*domain.VectorSize : (p+1)*domain.VectorSize]

	items := make([]ce, 0, len(candidates))
	for _, id := range candidates {
		if id == sentinel || int(id) == p {
			continue
		}
		items = append(items, ce{
			dist: vecDistSq(pVec, vecs[int(id)*domain.VectorSize:(int(id)+1)*domain.VectorSize]),
			id:   id,
		})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].dist < items[j].dist })

	out := make([]uint32, 0, r)
	removed := make([]bool, len(items))

	for i := 0; i < len(items) && len(out) < r; i++ {
		if removed[i] {
			continue
		}
		pStar := items[i]
		out = append(out, pStar.id)

		// Remove r if α²·dist²(p*,r) ≤ dist²(p,r), i.e. p* dominates the path to r.
		pStarVec := vecs[int(pStar.id)*domain.VectorSize : (int(pStar.id)+1)*domain.VectorSize]
		for j := i + 1; j < len(items); j++ {
			if removed[j] {
				continue
			}
			rVec := vecs[int(items[j].id)*domain.VectorSize : (int(items[j].id)+1)*domain.VectorSize]
			if alphaSq*vecDistSq(pStarVec, rVec) <= items[j].dist {
				removed[j] = true
			}
		}
	}

	return out
}

// findMedoid returns the index of the vector closest to the centroid of all n vectors.
func findMedoid(vecs []float32, n int) uint32 {
	centroid := make([]float32, domain.VectorSize)
	for i := 0; i < n; i++ {
		for d := 0; d < domain.VectorSize; d++ {
			centroid[d] += vecs[i*domain.VectorSize+d]
		}
	}
	inv := 1.0 / float32(n)
	for d := range centroid {
		centroid[d] *= inv
	}

	var bestDist float32 = math.MaxFloat32
	var best uint32
	for i := 0; i < n; i++ {
		d := vecDistSq(centroid, vecs[i*domain.VectorSize:(i+1)*domain.VectorSize])
		if d < bestDist {
			bestDist = d
			best = uint32(i)
		}
	}
	return best
}

// writeVamana serialises the Vamana index to outPath using a 4 MB buffered writer.
// The binary format is documented in Open's doc comment.
func writeVamana(outPath string, n, r int, medoid uint32, sq8 bool,
	sq8Min, sq8Scale [domain.VectorSize]float32,
	graph []uint32, vecs8 []uint8, vecs32 []float32,
	labels []uint8) error {

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20)

	var u32buf [4]byte
	writeU32 := func(v uint32) {
		binary.LittleEndian.PutUint32(u32buf[:], v)
		w.Write(u32buf[:]) //nolint:errcheck
	}
	writeF32 := func(v float32) { writeU32(math.Float32bits(v)) }

	writeU32(uint32(n))
	writeU32(uint32(r))
	writeU32(medoid)
	var flags uint32
	if sq8 {
		flags |= flagSQ8
	}
	writeU32(flags)

	if sq8 {
		for d := 0; d < domain.VectorSize; d++ {
			writeF32(sq8Min[d])
		}
		for d := 0; d < domain.VectorSize; d++ {
			writeF32(sq8Scale[d])
		}
	}

	for _, nb := range graph {
		writeU32(nb)
	}

	if sq8 {
		w.Write(vecs8) //nolint:errcheck
	} else {
		for _, v := range vecs32 {
			writeF32(v)
		}
	}

	w.Write(labels) //nolint:errcheck

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush %s: %w", outPath, err)
	}
	return nil
}

// computeSQ8Params scans all n vectors and returns per-dimension min and scale.
// scale[d] = (max[d] - min[d]) / 255.
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

// quantizeAll converts n float32 vectors to SQ8-encoded uint8 vectors.
func quantizeAll(vecs []float32, n int, minV, scale [domain.VectorSize]float32) []uint8 {
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

// vecDistSq computes the squared Euclidean distance between two float32 slices of
// length VectorSize. Used in the hot path of both construction and query.
func vecDistSq(a, b []float32) float32 {
	var sum float32
	for d := 0; d < domain.VectorSize; d++ {
		diff := a[d] - b[d]
		sum += diff * diff
	}
	return sum
}

// appendUnique appends id to s only if it is not already present.
func appendUnique(s []uint32, id uint32) []uint32 {
	for _, v := range s {
		if v == id {
			return s
		}
	}
	return append(s, id)
}

// defaultL returns l if l > 0, otherwise DefaultL.
func defaultL(l int) int {
	if l <= 0 {
		return DefaultL
	}
	return l
}

// defaultBuildL returns buildL if it is a valid positive value, otherwise falls
// back to max(2*r, DefaultBuildL). The build beam must be ≥ r to produce
// quality edges; using DefaultBuildL (125) is safe for most configurations.
func defaultBuildL(r, buildL int) int {
	if buildL > 0 {
		if buildL < r {
			return r // never smaller than max degree
		}
		return buildL
	}
	bl := DefaultBuildL
	if 2*r > bl {
		bl = 2 * r
	}
	return bl
}

// clampR clamps r to [1, n-1] so no node has more neighbors than the dataset allows.
func clampR(r, n int) int {
	if r <= 0 {
		r = DefaultR
	}
	if r > n-1 && n > 1 {
		r = n - 1
	}
	if r < 1 {
		r = 1
	}
	return r
}

// nodesSlice returns order[lo:hi] as a plain int slice.
func nodesSlice(order []int, lo, hi int) []int {
	return order[lo:hi]
}
