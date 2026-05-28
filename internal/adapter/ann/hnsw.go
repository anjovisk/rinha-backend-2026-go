// Package ann implements port.NeighborFinder using Hierarchical Navigable Small World (HNSW)
// approximate nearest-neighbour search via github.com/coder/hnsw.
//
// Unlike the brute-force knn adapter, HNSW achieves O(log N) query time at the cost of
// slightly reduced recall and higher startup time (the graph is built in process memory,
// not memory-mapped). Use HNSW when p99 latency is the primary concern and a small
// accuracy trade-off is acceptable.
//
// Enable by setting VECTOR_SEARCHER=hnsw at server startup.
package ann

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"github.com/coder/hnsw"
	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// defaultM is the maximum number of bidirectional connections per node in the HNSW graph.
// M=4 halves connection memory versus M=8 (~50–100 MB saved) at a small recall cost
// (~93–97% vs ~97–99% for 14-dimensional vectors). Raise to 8 if recall matters more than RAM.
const defaultM = 4

// defaultEfSearch is the size of the dynamic candidate list during graph traversal.
// Must be ≥ k (neighbours requested). Higher values improve recall at the cost of latency.
const defaultEfSearch = 50

// Searcher holds an HNSW graph over the reference dataset and implements port.NeighborFinder.
type Searcher struct {
	// graph is the in-memory HNSW index; node keys are indices into labels.
	graph *hnsw.Graph[uint32]
	// labels holds the classification for each entry: 1=fraud, 0=legit.
	// Entry i corresponds to the HNSW node keyed by uint32(i).
	// When mmapped is non-nil, this slice points directly into the mmap region.
	labels []uint8
	// n is the number of reference entries in the index.
	n int
	// mmapped is non-nil when the Searcher was built via Open: it holds the mmap region
	// so that vector slices stored in graph nodes remain valid. Released by Close.
	mmapped []byte
	// logger is the named zap logger injected at construction time.
	logger *zap.Logger
}

// Reference is a labelled entry used to build a Searcher in tests.
// For production use, call Open to build from the pre-processed binary file.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is the fraud classification: "fraud" or "legit".
	Label string
}

// New builds a Searcher from a slice of Reference entries.
// This constructor allocates plain Go slices and is used primarily in tests.
// For production use, call Open to build from the binary reference file.
func New(refs []Reference, logger *zap.Logger) *Searcher {
	n := len(refs)
	allVecs := make([]float32, n*domain.VectorSize)
	labels := make([]uint8, n)

	for i, r := range refs {
		base := i * domain.VectorSize
		for j, f := range r.Vector {
			allVecs[base+j] = float32(f)
		}
		if r.Label == "fraud" {
			labels[i] = 1
		}
	}

	g := newGraph()
	if n > 0 {
		nodes := makeNodes(n, allVecs)
		g.Add(nodes...)
	}

	return &Searcher{graph: g, labels: labels, n: n, logger: logger}
}

// Open memory-maps the binary reference index at path, builds an HNSW graph whose nodes
// hold slices directly into the mmap region (no heap copy of the vectors), and returns a
// Searcher. The binary format is identical to the one produced by cmd/preprocess:
//
//	[4 bytes uint32 LE]  number of entries N
//	[N × 56 bytes]       float32 LE feature vectors, row-major
//	[N × 1 byte]         uint8 labels, 1=fraud 0=legit
//
// The mmap region is kept open for the lifetime of the Searcher so that graph nodes can
// read vectors directly from the OS page cache — saving ~168 MB of heap versus a full copy.
// If the same file is also open by knn.Open (VECTOR_SEARCHER=brute is not set), the OS
// shares the physical pages between both mappings. Call Close when done to release the mmap.
//
// Index construction is O(N log N). With M=4 and no vector copy, total additional RSS is
// roughly the graph connection lists (~50–100 MB for 3 M entries) plus ~3 MB for labels.
func Open(path string, logger *zap.Logger) (*Searcher, error) {
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

	// The mmap is intentionally NOT munmapped here — it is kept alive in Searcher.mmapped
	// so that graph node slices pointing into it remain valid after Open returns.
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

	// Reinterpret the vector region as []float32 directly in the mmap region — no heap copy.
	vecPtr := (*float32)(unsafe.Pointer(&data[4]))
	mmapVecs := unsafe.Slice(vecPtr, n*domain.VectorSize)

	// Labels slice into the mmap region — no heap copy.
	labels := data[4+vectorBytes : 4+vectorBytes+labelBytes]

	logger.Info("building HNSW index",
		zap.Int("entries", n),
		zap.Int("M", defaultM),
		zap.Int("ef_search", defaultEfSearch),
	)

	g := newGraph()
	if n > 0 {
		// Node Value slices point into mmapVecs (which points into data).
		// The mmap must remain open until Close is called.
		nodes := makeNodes(n, mmapVecs)
		g.Add(nodes...)
	}

	logger.Info("HNSW index ready", zap.Int("entries", n))

	return &Searcher{graph: g, labels: labels, n: n, mmapped: data, logger: logger}, nil
}

// Close releases the mmap region backing the Searcher.
// It is a no-op when the Searcher was created with New (plain slices).
// Must be called once the Searcher is no longer in use to avoid a resource leak.
// Accessing the Searcher after Close results in undefined behaviour.
func (s *Searcher) Close() error {
	if s.mmapped == nil {
		return nil
	}
	return syscall.Munmap(s.mmapped)
}

// Export writes the HNSW graph to graphPath using a 4 MB buffered writer.
// Labels are not included — they are read from the binary reference file by Load.
// Intended for use in cmd/preprocess so the server can call Load instead of
// rebuilding the index on every startup.
func (s *Searcher) Export(graphPath string) error {
	f, err := os.Create(graphPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", graphPath, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20)
	if err := s.graph.Export(w); err != nil {
		return fmt.Errorf("export graph: %w", err)
	}
	return w.Flush()
}

// Load imports a pre-built HNSW graph from graphPath and reads labels from binPath.
// Unlike Open, it does not rebuild the graph at startup — O(N) load instead of O(N log N) build.
//
// Trade-off: Import allocates vectors in heap (~168 MB for 3 M entries) because the
// serialised format embeds vector data per node. Open is more memory-efficient (vectors
// stay in the mmap page cache); Load is faster to start.
//
// graphPath is the file produced by Searcher.Export (cmd/preprocess).
// binPath is the binary reference file (resources/references.bin) used only to read labels.
// Call Close on the returned Searcher to release any resources (no-op for Load).
func Load(graphPath, binPath string, logger *zap.Logger) (*Searcher, error) {
	gf, err := os.Open(graphPath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", graphPath, err)
	}
	defer gf.Close()

	g := hnsw.NewGraph[uint32]()
	if err := g.Import(bufio.NewReaderSize(gf, 4<<20)); err != nil {
		return nil, fmt.Errorf("import %s: %w", graphPath, err)
	}
	n := g.Len()

	labels, err := readLabels(binPath, n)
	if err != nil {
		return nil, err
	}

	logger.Info("HNSW index loaded from file",
		zap.Int("entries", n),
		zap.String("graph_path", graphPath),
	)
	return &Searcher{graph: g, labels: labels, n: n, logger: logger}, nil
}

// readLabels mmaps binPath, copies the label bytes for n entries to heap, and unmaps.
// It verifies that the entry count in the binary file matches n.
func readLabels(binPath string, n int) ([]uint8, error) {
	f, err := os.Open(binPath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", binPath, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", binPath, err)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", binPath, err)
	}
	defer syscall.Munmap(data) //nolint:errcheck

	binN := int(binary.LittleEndian.Uint32(data[:4]))
	if binN != n {
		return nil, fmt.Errorf("%s: bin has %d entries but graph has %d — regenerate references.hnsw", binPath, binN, n)
	}

	vectorBytes := n * domain.VectorSize * 4
	labels := make([]uint8, n)
	copy(labels, data[4+vectorBytes:4+vectorBytes+n])
	return labels, nil
}

// FindNearest returns the labels ("fraud" or "legit") of up to k approximate nearest
// neighbours of v under Euclidean distance. The returned slice may be shorter than k
// when the index contains fewer entries or the graph cannot reach k candidates.
//
// v is the 14-dimensional query vector; k is the desired neighbour count.
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	s.logger.Debug("ANN search starting",
		zap.Int("dataset_size", s.n),
		zap.Int("k", k),
	)

	query := make([]float32, domain.VectorSize)
	for i, f := range v {
		query[i] = float32(f)
	}

	results := s.graph.Search(query, k)

	out := make([]string, len(results))
	for i, r := range results {
		if s.labels[r.Key] == 1 {
			out[i] = "fraud"
		} else {
			out[i] = "legit"
		}
	}

	s.logger.Debug("ANN search complete", zap.Int("returned", len(out)))
	return out
}

// newGraph creates and configures an HNSW graph with project defaults.
func newGraph() *hnsw.Graph[uint32] {
	g := hnsw.NewGraph[uint32]()
	g.M = defaultM
	g.EfSearch = defaultEfSearch
	g.Distance = hnsw.EuclideanDistance
	return g
}

// makeNodes builds a slice of HNSW nodes whose Value slices all point into allVecs,
// avoiding per-node allocations.
func makeNodes(n int, allVecs []float32) []hnsw.Node[uint32] {
	nodes := make([]hnsw.Node[uint32], n)
	for i := 0; i < n; i++ {
		nodes[i] = hnsw.MakeNode(uint32(i), allVecs[i*domain.VectorSize:(i+1)*domain.VectorSize])
	}
	return nodes
}
