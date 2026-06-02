// Package vptree implements port.NeighborFinder using vantage-point trees (VP-Trees)
// built independently for each of the 6 partition bins defined by
// (last_tx_null × is_online × card_present) — the same scheme as adapter/partition.
//
// At startup the reference dataset is partitioned and a VP-Tree is built per bin in
// O(N log N) time. At query time the matching bin's tree is traversed: the triangle
// inequality prunes branches whose minimum possible distance exceeds the current k-th
// best distance, yielding sub-linear search when the intrinsic dimensionality of the
// data within the bin is low enough for pruning to take effect.
//
// The search is exact within each partition: every entry that could be a true
// k-nearest-neighbour is guaranteed to be visited. Cross-partition true neighbours are
// not considered, matching the trade-off of adapter/partition. Queries landing on an
// empty bin fall back to a full brute-force scan.
//
// No preprocess step or separate index file is needed. Typical startup is a few seconds.
// Memory: mmap shared between instances (~171 MB page cache) plus tree structure
// (~16–20 MB heap across 6 bins for the 3M reference dataset).
package vptree

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"syscall"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// Routing dimensions within domain.Vector (same as adapter/partition).
const (
	dimMinutesSinceLast = 5  // sentinel -1 when last transaction is absent
	dimKmFromLast       = 6  // sentinel -1 when last transaction is absent
	dimIsOnline         = 9  // 1 = online transaction, 0 = in-person
	dimCardPresent      = 10 // 1 = card physically present, 0 = absent
)

// numBins is the number of routing slots (2^3 bits).
// Bins 3 (011) and 7 (111) are unused in the reference dataset in practice.
const numBins = 8

// leafSize is the maximum number of entries stored in a leaf node.
// Entries at or below this threshold are searched by brute-force without further splitting.
// Larger values reduce tree depth and amortise node-visit overhead at the cost of
// more brute-force work per leaf; smaller values prune earlier but increase overhead.
const leafSize = 32

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

// binOfRaw computes the routing bin for a float32 vector slice.
// -1.0 is exactly representable as float32, so sentinel comparison is safe.
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

// vpNode is a single node in a vantage-point tree.
//
// For internal nodes (leafStart == -1):
//   - vpIdx identifies the vantage-point entry in the shared vectors array.
//   - radius is the Euclidean distance threshold: left subtree holds entries with
//     dist ≤ radius; right subtree holds entries with dist > radius.
//   - left and right are child indices in the owning binTree.nodes slice.
//
// For leaf nodes (leafStart >= 0):
//   - leafStart and leafLen define a range in binTree.leaves.
//   - vpIdx, radius, left, right are unused.
type vpNode struct {
	// vpIdx is the entry index of the vantage point (internal nodes only).
	vpIdx uint32
	// radius is the Euclidean (not squared) distance split threshold.
	radius float32
	// left is the left (inner-ball) child index; -1 if absent.
	left int32
	// right is the right (outer-shell) child index; -1 if absent.
	right int32
	// leafStart is the first index into binTree.leaves; -1 for internal nodes.
	leafStart int32
	// leafLen is the number of entries in the leaf range.
	leafLen uint16
}

// binTree holds the complete VP-Tree for one partition bin.
type binTree struct {
	// nodes is the flat, pre-allocated slice of all tree nodes.
	nodes []vpNode
	// leaves stores the entry indices for all leaf nodes packed contiguously.
	// len(leaves) equals the total number of entries reachable via leaf nodes.
	leaves []uint32
	// root is the index of the root node in nodes; -1 when the tree is empty.
	root int32
	// count is the total number of reference entries in this bin.
	count int
}

// candidate is a KNN result entry used during search.
type candidate struct {
	// dist is the squared Euclidean distance from the query to this entry.
	dist float32
	// idx is the entry index in the reference dataset.
	idx uint32
}

// Finder routes each FindNearest call to the per-bin VP-Tree matching the query's
// routing features and performs exact KNN within that tree.
//
// When the matching bin is empty (an unseen combination of routing features),
// the search falls back to a full brute-force scan so the caller always receives k labels.
type Finder struct {
	// n is the total number of reference entries.
	n int
	// vectors holds all feature vectors: entry i occupies [i*VectorSize, (i+1)*VectorSize).
	// Backed by a shared mmap when opened with Open; a plain heap slice when opened with New.
	vectors []float32
	// labels holds the fraud/legit classification: 1=fraud, 0=legit.
	labels []uint8
	// mmapped is non-nil when vectors and labels point into a mmap region.
	mmapped []byte
	// trees holds the VP-Tree for each partition bin; nil for empty or unused bins.
	trees [numBins]*binTree
	// logger is the named zap logger injected at construction time.
	logger *zap.Logger
}

// Reference is a labelled dataset entry for constructing a Finder in tests.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is "fraud" or "legit".
	Label string
}

// New builds a Finder from a slice of Reference entries using heap allocations.
// Intended for tests; production code should call Open.
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
	f.buildTrees()
	return f
}

// Open memory-maps the binary reference index at path and builds per-bin VP-Trees.
//
// Binary format: [uint32 N][N×VectorSize float32 LE][N uint8 labels] — identical to adapter/knn.
// Both API server instances share physical pages via MAP_SHARED.
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
	f.buildTrees()

	logger.Info("vptree finder ready", zap.Int("total_entries", n))
	return f, nil
}

// buildTrees partitions all entries into bins and constructs a VP-Tree for each.
func (f *Finder) buildTrees() {
	stride := domain.VectorSize

	// Two-pass: count then fill per-bin index slices.
	var counts [numBins]int
	for i := 0; i < f.n; i++ {
		counts[binOfRaw(f.vectors[i*stride:i*stride+stride])]++
	}
	bins := [numBins][]uint32{}
	for k, c := range counts {
		if c > 0 {
			bins[k] = make([]uint32, 0, c)
		}
	}
	for i := 0; i < f.n; i++ {
		k := binOfRaw(f.vectors[i*stride : i*stride+stride])
		bins[k] = append(bins[k], uint32(i))
	}

	for k, bin := range bins {
		if len(bin) == 0 {
			continue
		}
		b := &treeBuilder{
			vectors: f.vectors,
			nodes:   make([]vpNode, 0, 2*len(bin)/leafSize+2),
			leaves:  make([]uint32, 0, len(bin)),
		}
		root := b.build(bin)
		f.trees[k] = &binTree{
			nodes:  b.nodes,
			leaves: b.leaves,
			root:   root,
			count:  len(bin),
		}
		f.logger.Debug("vptree bin built",
			zap.Int("bin", k),
			zap.Int("entries", len(bin)),
			zap.Int("nodes", len(b.nodes)),
		)
	}
}

// treeBuilder accumulates VP-Tree nodes and leaf indices during construction.
type treeBuilder struct {
	// vectors is the shared flat vector array; read-only during build.
	vectors []float32
	// nodes grows as the tree is built; indices are stable positions in this slice.
	nodes []vpNode
	// leaves accumulates all entry indices reachable through leaf nodes.
	leaves []uint32
}

// distEntry pairs an Euclidean distance with an entry index for sorting during build.
type distEntry struct {
	dist float32
	idx  uint32
}

// build recursively constructs a VP-Tree over the given entry indices and returns
// the index of the root node in b.nodes. indices is modified in place during build;
// the caller must not rely on its ordering after build returns.
func (b *treeBuilder) build(indices []uint32) int32 {
	n := len(indices)
	if n == 0 {
		return -1
	}

	nodeIdx := int32(len(b.nodes))
	b.nodes = append(b.nodes, vpNode{left: -1, right: -1, leafStart: -1})

	// Leaf: store all entries and return without selecting a vantage point.
	if n <= leafSize {
		start := int32(len(b.leaves))
		b.leaves = append(b.leaves, indices...)
		b.nodes[nodeIdx].leafStart = start
		b.nodes[nodeIdx].leafLen = uint16(n)
		return nodeIdx
	}

	// Internal: pick a random vantage point and move it to the tail so it is
	// excluded from both child subtrees (it is evaluated at this node during search).
	vpPos := rand.Intn(n)
	vpIdx := indices[vpPos]
	indices[vpPos], indices[n-1] = indices[n-1], indices[vpPos]
	rest := indices[:n-1]

	// Compute Euclidean distances from vp to every remaining entry.
	stride := domain.VectorSize
	vpBase := int(vpIdx) * stride
	vpSlice := b.vectors[vpBase : vpBase+stride]

	entries := make([]distEntry, len(rest))
	for i, idx := range rest {
		base := int(idx) * stride
		entries[i] = distEntry{euclidDist(vpSlice, b.vectors[base:base+stride]), idx}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].dist < entries[j].dist })

	// Split at the median: left (inner ball) gets the closer half,
	// right (outer shell) gets the farther half.
	// radius is placed between the two halves to cleanly separate them.
	mid := len(entries) / 2
	var radius float32
	if mid == 0 {
		radius = 0
	} else {
		radius = (entries[mid-1].dist + entries[mid].dist) / 2.0
	}

	// Restore sorted order into rest before recursing.
	for i, e := range entries {
		rest[i] = e.idx
	}

	b.nodes[nodeIdx].vpIdx = vpIdx
	b.nodes[nodeIdx].radius = radius
	b.nodes[nodeIdx].left = b.build(rest[:mid])
	b.nodes[nodeIdx].right = b.build(rest[mid:])

	return nodeIdx
}

// Close releases the mmap region backing the Finder.
// It is a no-op when the Finder was created with New (heap slice).
func (f *Finder) Close() error {
	if f.mmapped == nil {
		return nil
	}
	return syscall.Munmap(f.mmapped)
}

// FindNearest returns the labels of the k reference entries closest to v under Euclidean
// distance. Only the partition bin matching v's routing features is searched via VP-Tree.
// Falls back to a full brute-force scan if the matching bin is empty.
func (f *Finder) FindNearest(v domain.Vector, k int) []string {
	key := binOf(v)
	tree := f.trees[key]
	if tree == nil {
		f.logger.Warn("vptree bin empty, falling back to full scan",
			zap.Int("bin", int(key)),
		)
		return f.bruteForce(v, fullRange(f.n), k)
	}
	if k > tree.count {
		k = tree.count
	}

	best := make([]candidate, k)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0

	f.searchNode(tree, tree.root, v, best, &worstDist, &worstPos)

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

// searchNode recursively searches the subtree rooted at nodeIdx.
// best, worstDist, and worstPos track the current k-best candidates.
// The triangle inequality prunes branches that cannot improve the k-th best distance.
func (f *Finder) searchNode(tree *binTree, nodeIdx int32, v domain.Vector, best []candidate, worstDist *float32, worstPos *int) {
	if nodeIdx == -1 {
		return
	}
	k := len(best)
	node := &tree.nodes[nodeIdx]

	// Leaf: brute-force over all entries in the leaf range.
	if node.leafStart >= 0 {
		stride := domain.VectorSize
		for _, entryIdx := range tree.leaves[node.leafStart : node.leafStart+int32(node.leafLen)] {
			base := int(entryIdx) * stride
			d := squaredEuclidean(v, f.vectors[base:base+stride])
			if d < *worstDist {
				best[*worstPos] = candidate{d, entryIdx}
				*worstDist, *worstPos = rescan(best, k)
			}
		}
		return
	}

	// Internal: evaluate the vantage point itself.
	stride := domain.VectorSize
	vpBase := int(node.vpIdx) * stride
	dSq := squaredEuclidean(v, f.vectors[vpBase:vpBase+stride])
	if dSq < *worstDist {
		best[*worstPos] = candidate{dSq, node.vpIdx}
		*worstDist, *worstPos = rescan(best, k)
	}

	// Branch decision uses Euclidean (not squared) distance for the triangle inequality.
	// d  = dist(query, vp)
	// tau = dist to current k-th best candidate (prune threshold)
	d := float32(math.Sqrt(float64(dSq)))
	tau := float32(math.Sqrt(float64(*worstDist)))

	if d < node.radius {
		// Query is in the inner ball: search left (closer half) first.
		f.searchNode(tree, node.left, v, best, worstDist, worstPos)
		tau = float32(math.Sqrt(float64(*worstDist)))
		// Visit right only if the outer shell could hold a closer point:
		// the minimum dist from query to any outer point is radius - d;
		// skip if radius - d > tau, i.e., d + tau < radius.
		if d+tau >= node.radius {
			f.searchNode(tree, node.right, v, best, worstDist, worstPos)
		}
	} else {
		// Query is in the outer shell: search right (farther half) first.
		f.searchNode(tree, node.right, v, best, worstDist, worstPos)
		tau = float32(math.Sqrt(float64(*worstDist)))
		// Visit left only if the inner ball could hold a closer point:
		// the minimum dist from query to any inner point is d - radius;
		// skip if d - radius > tau, i.e., d - tau > radius.
		if d-tau <= node.radius {
			f.searchNode(tree, node.left, v, best, worstDist, worstPos)
		}
	}
}

// rescan finds the new worst candidate in best after an insertion and returns
// (worstDist, worstPos). Called only when the buffer changes.
func rescan(best []candidate, k int) (worstDist float32, worstPos int) {
	worstDist = best[0].dist
	worstPos = 0
	for j := 1; j < k; j++ {
		if best[j].dist > worstDist {
			worstDist = best[j].dist
			worstPos = j
		}
	}
	return
}

// bruteForce performs an exact linear scan over the given entry indices.
// Used as fallback when the routing bin is empty.
func (f *Finder) bruteForce(v domain.Vector, indices []uint32, k int) []string {
	n := len(indices)
	if k > n {
		k = n
	}
	best := make([]candidate, k)
	for i := range best {
		best[i].dist = math.MaxFloat32
	}
	worstDist := float32(math.MaxFloat32)
	worstPos := 0
	stride := domain.VectorSize
	for _, entryIdx := range indices {
		base := int(entryIdx) * stride
		d := squaredEuclidean(v, f.vectors[base:base+stride])
		if d < worstDist {
			best[worstPos] = candidate{d, entryIdx}
			worstDist, worstPos = rescan(best, k)
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

// fullRange returns a slice of all entry indices 0..n-1.
func fullRange(n int) []uint32 {
	idx := make([]uint32, n)
	for i := range idx {
		idx[i] = uint32(i)
	}
	return idx
}

// squaredEuclidean computes the squared Euclidean distance between a float64 query
// vector and a float32 reference slice. The square root is omitted — only relative
// ordering matters for k-NN candidate comparison.
func squaredEuclidean(query domain.Vector, ref []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := float32(query[i]) - ref[i]
		sum += d * d
	}
	return sum
}

// euclidDist computes the true Euclidean distance between two float32 vector slices.
// Used during VP-Tree construction (to set node radii) and during search (for pruning
// via the triangle inequality, which requires actual distances, not squared distances).
func euclidDist(a, b []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}
