// This file contains build-time helpers: SQ8 quantization parameter computation,
// K-means clustering, inverted list construction, and the Build function that writes
// a ready-to-serve IVF index from references.bin.
//
// These are invoked only by cmd/preprocess (BUILD_IVF=true) and by New (tests).
// They are not called at server startup.
package ivf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// DefaultNlist is the default number of K-means clusters when IVF_NLIST is unset.
// Tuned for N≈3 million: sqrt(3M)≈1732, rounded down to a power of two for
// alignment convenience. Raise to 2048–4096 for better recall at higher CPU cost.
const DefaultNlist = 1024

// DefaultSQ8 controls whether vectors are compressed to uint8 when IVF_SQ8 is unset.
// true: ~45 MB per instance, small distance approximation error.
// false: ~168 MB per instance, exact distances within each cluster.
const DefaultSQ8 = true

// kmeansMaxIter is the maximum number of K-means assignment+update iterations.
// The loop exits early when no assignments change between iterations.
const kmeansMaxIter = 30

// Reference is a labelled entry used to build a Searcher from in-memory data.
// Intended for tests; production code should call Build then Open.
type Reference struct {
	// Vector is the 14-dimensional feature vector.
	Vector domain.Vector
	// Label is the fraud classification: "fraud" or "legit".
	Label string
}

// New builds a Searcher from refs in memory without requiring an on-disk file.
// It runs K-means with min(nlist, len(refs)) clusters. nprobe controls how many
// clusters are searched per query (≤ 0 uses DefaultNprobe). sq8 enables 8-bit
// scalar quantization; pass DefaultSQ8 when unsure. Intended for tests.
func New(refs []Reference, nlist, nprobe int, sq8 bool, logger *zap.Logger) *Searcher {
	n := len(refs)
	if n == 0 {
		return &Searcher{logger: logger}
	}
	if nlist > n {
		nlist = n
	}

	vecs := make([]float32, n*domain.VectorSize)
	labels := make([]uint8, n)
	for i, r := range refs {
		for d, f := range r.Vector {
			vecs[i*domain.VectorSize+d] = float32(f)
		}
		if r.Label == "fraud" {
			labels[i] = 1
		}
	}

	centroids, assignments := runKMeans(vecs, n, nlist, kmeansMaxIter)
	offsets, sizes := buildOffsets(assignments, nlist, n)

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecFlat8 []uint8
	var vecFlat32 []float32
	var labelFlat []uint8

	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		qvecs := quantizeAll(vecs, n, sq8Min, sq8Scale)
		vecFlat8, labelFlat = buildInvertedListsSQ8(qvecs, labels, assignments, offsets, sizes, nlist, n)
	} else {
		vecFlat32, labelFlat = buildInvertedListsF32(vecs, labels, assignments, offsets, sizes, nlist, n)
	}

	if nprobe <= 0 {
		nprobe = DefaultNprobe
	}
	if nprobe > nlist {
		nprobe = nlist
	}

	return &Searcher{
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
		logger:    logger,
	}
}

// Build constructs an IVF index from the flat binary file at binPath (produced by
// cmd/preprocess) and writes the result to outPath. nlist is the K-means cluster count
// (use DefaultNlist when unsure); sq8 controls whether vectors are SQ8-quantized
// (use DefaultSQ8 when unsure). The logger should be named before passing.
//
// The SQ8 flag is stored in the file header so Open at serve time needs no env var.
// Memory at serve time: ~45 MB when sq8=true, ~168 MB when sq8=false.
func Build(binPath, outPath string, nlist int, sq8 bool, logger *zap.Logger) error {
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

	logger.Info("IVF build starting",
		zap.Int("entries", n),
		zap.Int("nlist", nlist),
		zap.Bool("sq8", sq8),
		zap.Int("kmeans_max_iter", kmeansMaxIter),
		zap.Int("cpus", runtime.NumCPU()),
	)

	centroids, assignments := runKMeans(vecs, n, nlist, kmeansMaxIter)
	logger.Info("K-means complete")

	offsets, sizes := buildOffsets(assignments, nlist, n)

	var sq8Min, sq8Scale [domain.VectorSize]float32
	var vecFlat8 []uint8
	var vecFlat32 []float32
	var labelFlat []uint8

	if sq8 {
		sq8Min, sq8Scale = computeSQ8Params(vecs, n)
		logger.Info("SQ8 parameters computed")
		qvecs := quantizeAll(vecs, n, sq8Min, sq8Scale)
		logger.Info("SQ8 quantization complete", zap.Int("vectors", n))
		vecFlat8, labelFlat = buildInvertedListsSQ8(qvecs, rawLabels, assignments, offsets, sizes, nlist, n)
	} else {
		vecFlat32, labelFlat = buildInvertedListsF32(vecs, rawLabels, assignments, offsets, sizes, nlist, n)
		logger.Info("float32 inverted lists built (no quantization)")
	}

	logger.Info("inverted lists built", zap.Int("clusters", nlist))

	if err := writeIVF(outPath, nlist, sq8, sq8Min, sq8Scale, centroids, offsets, sizes, vecFlat8, vecFlat32, labelFlat); err != nil {
		return err
	}

	logger.Info("IVF index written",
		zap.String("path", outPath),
		zap.Bool("sq8", sq8),
	)
	return nil
}

// computeSQ8Params scans all n vectors and returns per-dimension min and scale.
// scale[d] = (max[d] - min[d]) / 255; sentinel values like -1 (missing last_tx)
// are included in the range so they remain distinguishable after quantization.
func computeSQ8Params(vecs []float32, n int) (min, scale [domain.VectorSize]float32) {
	maxV := [domain.VectorSize]float32{}
	for d := range min {
		min[d] = math.MaxFloat32
		maxV[d] = -math.MaxFloat32
	}
	for i := 0; i < n; i++ {
		base := i * domain.VectorSize
		for d := 0; d < domain.VectorSize; d++ {
			v := vecs[base+d]
			if v < min[d] {
				min[d] = v
			}
			if v > maxV[d] {
				maxV[d] = v
			}
		}
	}
	for d := range scale {
		r := maxV[d] - min[d]
		if r == 0 {
			scale[d] = 1 // constant dimension: all values map to 0
		} else {
			scale[d] = r / 255.0
		}
	}
	return
}

// quantizeAll converts all n float32 vectors to SQ8-encoded uint8 vectors.
// Returns a flat []uint8 of length n*VectorSize; layout mirrors the input.
func quantizeAll(vecs []float32, n int, min, scale [domain.VectorSize]float32) []uint8 {
	out := make([]uint8, n*domain.VectorSize)
	for i := 0; i < n; i++ {
		base := i * domain.VectorSize
		for d := 0; d < domain.VectorSize; d++ {
			f := (vecs[base+d] - min[d]) / scale[d]
			if f < 0 {
				f = 0
			} else if f > 255 {
				f = 255
			}
			out[base+d] = uint8(f + 0.5) // round to nearest
		}
	}
	return out
}

// runKMeans partitions n vectors (float32, row-major, stride=VectorSize) into nlist
// clusters using random initialisation and at most maxIter iterations. Assignment is
// parallelised across all available CPUs; the centroid update step is sequential.
// Returns centroids (nlist×VectorSize float32) and per-vector assignments.
func runKMeans(vecs []float32, n, nlist, maxIter int) (centroids []float32, assignments []int32) {
	const stride = domain.VectorSize
	rng := rand.New(rand.NewSource(42))

	// Random initialisation: sample nlist distinct vectors as starting centroids.
	centroids = make([]float32, nlist*stride)
	perm := rng.Perm(n)
	for c := 0; c < nlist; c++ {
		copy(centroids[c*stride:(c+1)*stride], vecs[perm[c]*stride:(perm[c]+1)*stride])
	}

	assignments = make([]int32, n)
	numCPU := runtime.NumCPU()

	for iter := 0; iter < maxIter; iter++ {
		// --- Parallel assignment step ---
		// Each goroutine owns a disjoint [lo, hi) range of assignments, so no locks
		// are needed on assignments[]. Only the shared changed counter uses atomics.
		var changed int64
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
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					v := vecs[i*stride : (i+1)*stride]
					bestDist := float32(math.MaxFloat32)
					var bestC int32
					for c := 0; c < nlist; c++ {
						d := sliceL2Sq(v, centroids[c*stride:(c+1)*stride])
						if d < bestDist {
							bestDist = d
							bestC = int32(c)
						}
					}
					if assignments[i] != bestC {
						assignments[i] = bestC
						atomic.AddInt64(&changed, 1)
					}
				}
			}(lo, hi)
		}
		wg.Wait()

		if changed == 0 {
			break
		}

		// --- Sequential centroid update step ---
		sums := make([]float64, nlist*stride)
		counts := make([]int32, nlist)
		for i := 0; i < n; i++ {
			c := int(assignments[i])
			counts[c]++
			for d := 0; d < stride; d++ {
				sums[c*stride+d] += float64(vecs[i*stride+d])
			}
		}
		for c := 0; c < nlist; c++ {
			if counts[c] == 0 {
				// Reinitialise empty cluster to avoid degenerate centroids.
				idx := rng.Intn(n)
				copy(centroids[c*stride:(c+1)*stride], vecs[idx*stride:(idx+1)*stride])
				continue
			}
			for d := 0; d < stride; d++ {
				centroids[c*stride+d] = float32(sums[c*stride+d] / float64(counts[c]))
			}
		}
	}
	return centroids, assignments
}

// buildOffsets computes per-cluster offsets and sizes from the assignments array.
// offsets[c] is the index of the first vector belonging to cluster c in the flat layout;
// sizes[c] is the count of vectors in cluster c.
func buildOffsets(assignments []int32, nlist, n int) (offsets, sizes []int32) {
	sizes = make([]int32, nlist)
	for _, c := range assignments {
		sizes[c]++
	}
	offsets = make([]int32, nlist)
	var cum int32
	for c := 0; c < nlist; c++ {
		offsets[c] = cum
		cum += sizes[c]
	}
	return
}

// buildInvertedListsSQ8 reorders SQ8-quantized vectors and labels by cluster assignment.
// Cluster c occupies index range [offsets[c], offsets[c]+sizes[c]) in the returned slices.
func buildInvertedListsSQ8(qvecs, labels []uint8, assignments []int32, offsets, sizes []int32, nlist, n int) (vecFlat, labelFlat []uint8) {
	vecFlat = make([]uint8, n*domain.VectorSize)
	labelFlat = make([]uint8, n)
	pos := make([]int32, nlist)
	for i := 0; i < n; i++ {
		c := int(assignments[i])
		dest := int(offsets[c] + pos[c])
		pos[c]++
		copy(vecFlat[dest*domain.VectorSize:], qvecs[i*domain.VectorSize:(i+1)*domain.VectorSize])
		labelFlat[dest] = labels[i]
	}
	return
}

// buildInvertedListsF32 reorders float32 vectors and labels by cluster assignment.
// Cluster c occupies index range [offsets[c], offsets[c]+sizes[c]) in the returned slices.
func buildInvertedListsF32(vecs []float32, labels []uint8, assignments []int32, offsets, sizes []int32, nlist, n int) (vecFlat []float32, labelFlat []uint8) {
	vecFlat = make([]float32, n*domain.VectorSize)
	labelFlat = make([]uint8, n)
	pos := make([]int32, nlist)
	for i := 0; i < n; i++ {
		c := int(assignments[i])
		dest := int(offsets[c] + pos[c])
		pos[c]++
		copy(vecFlat[dest*domain.VectorSize:], vecs[i*domain.VectorSize:(i+1)*domain.VectorSize])
		labelFlat[dest] = labels[i]
	}
	return
}

// writeIVF serialises all IVF index components to outPath using a 4 MB buffered writer.
// The binary format is described in Open's doc comment.
func writeIVF(outPath string, nlist int, sq8 bool,
	sq8Min, sq8Scale [domain.VectorSize]float32,
	centroids []float32, offsets, sizes []int32,
	vecFlat8 []uint8, vecFlat32 []float32,
	labelFlat []uint8) error {

	f, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20)

	var u32 [4]byte
	writeU32 := func(v uint32) { binary.LittleEndian.PutUint32(u32[:], v); w.Write(u32[:]) } //nolint:errcheck
	writeF32 := func(v float32) { writeU32(math.Float32bits(v)) }

	writeU32(uint32(nlist))

	var flags uint8
	if sq8 {
		flags |= flagSQ8
	}
	w.WriteByte(flags) //nolint:errcheck

	if sq8 {
		for d := 0; d < domain.VectorSize; d++ {
			writeF32(sq8Min[d])
		}
		for d := 0; d < domain.VectorSize; d++ {
			writeF32(sq8Scale[d])
		}
	}

	for _, v := range centroids {
		writeF32(v)
	}

	for c := 0; c < nlist; c++ {
		sz := int(sizes[c])
		writeU32(uint32(sz))
		off := int(offsets[c])
		if sq8 {
			w.Write(vecFlat8[off*domain.VectorSize : (off+sz)*domain.VectorSize]) //nolint:errcheck
		} else {
			for _, v := range vecFlat32[off*domain.VectorSize : (off+sz)*domain.VectorSize] {
				writeF32(v)
			}
		}
		w.Write(labelFlat[off : off+sz]) //nolint:errcheck
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush %s: %w", outPath, err)
	}
	return nil
}

// sliceL2Sq computes the squared L2 distance between two float32 slices of length VectorSize.
// Used during K-means assignment; inline-able by the compiler for the hot path.
func sliceL2Sq(a, b []float32) float32 {
	var sum float32
	for i := 0; i < domain.VectorSize; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
