// Command preprocess converts resources/references.json.gz into resources/references.bin,
// a compact flat-binary file suitable for zero-copy mmap loading at runtime.
//
// Binary layout:
//
//	[4 bytes]        uint32 LE — number of entries N
//	[N × 56 bytes]   float32 LE — feature vectors, row-major (N rows × 14 columns)
//	[N × 1 byte]     uint8 — labels: 1=fraud, 0=legit
//
// Run once at build time via the Dockerfile. The server reads references.bin at startup;
// references.json.gz is not needed at runtime.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"log"
	"math"
	"os"
	"strconv"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/adapter/hnsw"
	"anjovisk/fraud-detection/internal/adapter/hnswflat"
	"anjovisk/fraud-detection/internal/adapter/ivf"
	"anjovisk/fraud-detection/internal/adapter/vamana"
	"anjovisk/fraud-detection/internal/domain"
)

// jsonEntry mirrors one element of the JSON array in references.json.gz.
type jsonEntry struct {
	// Vector holds the 14 pre-computed feature dimensions.
	Vector [domain.VectorSize]float64 `json:"vector"`
	// Label is the fraud classification: "fraud" or "legit".
	Label string `json:"label"`
}

func main() {
	const inPath = "resources/references.json.gz"
	const outPath = "resources/references.bin"

	inFile, err := os.Open(inPath)
	if err != nil {
		log.Fatalf("open %s: %v", inPath, err)
	}
	defer inFile.Close()

	gz, err := gzip.NewReader(inFile)
	if err != nil {
		log.Fatalf("decompress %s: %v", inPath, err)
	}
	defer gz.Close()

	outFile, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("create %s: %v", outPath, err)
	}
	defer outFile.Close()

	w := bufio.NewWriterSize(outFile, 4<<20) // 4 MB write buffer

	// Reserve 4 bytes for the entry count; we'll fill it in at the end.
	if _, err := w.Write(make([]byte, 4)); err != nil {
		log.Fatalf("write header placeholder: %v", err)
	}

	// First pass: stream vectors. Labels are buffered and appended after.
	labels := make([]byte, 0, 3_000_000)

	dec := json.NewDecoder(gz)

	// Consume the opening '['.
	if _, err := dec.Token(); err != nil {
		log.Fatalf("read opening token: %v", err)
	}

	var count uint32
	var buf [4]byte

	for dec.More() {
		var e jsonEntry
		if err := dec.Decode(&e); err != nil {
			log.Fatalf("decode entry %d: %v", count, err)
		}

		// Write 14 float32 values for this vector.
		for _, v := range e.Vector {
			f32 := float32(v)
			if !isFiniteF32(f32) {
				log.Fatalf("entry %d: non-finite value %v", count, v)
			}
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(f32))
			if _, err := w.Write(buf[:]); err != nil {
				log.Fatalf("write vector[%d]: %v", count, err)
			}
		}

		// Buffer the label byte.
		var label byte
		if e.Label == "fraud" {
			label = 1
		}
		labels = append(labels, label)
		count++

		if count%500_000 == 0 {
			log.Printf("processed %d entries…", count)
		}
	}

	// Append the label bytes.
	if _, err := w.Write(labels); err != nil {
		log.Fatalf("write labels: %v", err)
	}

	if err := w.Flush(); err != nil {
		log.Fatalf("flush output: %v", err)
	}

	// Back-fill the entry count at the start of the file.
	var countBuf [4]byte
	binary.LittleEndian.PutUint32(countBuf[:], count)
	if _, err := outFile.WriteAt(countBuf[:], 0); err != nil {
		log.Fatalf("write count header: %v", err)
	}

	log.Printf("done: %d entries → %s", count, outPath)

	if os.Getenv("BUILD_HNSW") == "true" {
		buildHNSW(outPath, count)
	} else {
		log.Printf("skipping HNSW index build (BUILD_HNSW != true)")
	}

	if os.Getenv("BUILD_IVF") == "true" {
		buildIVF(outPath)
	} else {
		log.Printf("skipping IVF-SQ8 index build (BUILD_IVF != true)")
	}

	if os.Getenv("BUILD_VAMANA") == "true" {
		buildVamana(outPath)
	} else {
		log.Printf("skipping Vamana index build (BUILD_VAMANA != true)")
	}

	if os.Getenv("BUILD_HNSWFLAT") == "true" {
		buildHNSWFlat(outPath)
	} else {
		log.Printf("skipping HNSW-flat index build (BUILD_HNSWFLAT != true)")
	}
}

// buildHNSW builds an HNSW approximate index from binPath and exports it to
// resources/references.hnsw. This runs once at build/preprocess time so the
// server can call hnsw.Load instead of rebuilding the index on every startup.
func buildHNSW(binPath string, count uint32) {
	const hnswPath = "resources/references.hnsw"

	log.Printf("building HNSW index (%d entries)…", count)

	s, err := hnsw.Open(binPath, zap.NewNop())
	if err != nil {
		log.Fatalf("build HNSW index: %v", err)
	}
	defer s.Close() //nolint:errcheck

	if err := s.Export(hnswPath); err != nil {
		log.Fatalf("export HNSW index: %v", err)
	}

	log.Printf("done: HNSW index → %s", hnswPath)
}

// buildIVF constructs an IVF approximate index from binPath and exports it to
// resources/references.ivf. Cluster count is read from IVF_NLIST (default 1024);
// SQ8 quantization is read from IVF_SQ8 (default true).
// This runs once at build/preprocess time so the server can call ivf.Open at startup.
func buildIVF(binPath string) {
	const ivfPath = "resources/references.ivf"

	nlist := ivf.DefaultNlist
	if raw := os.Getenv("IVF_NLIST"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			nlist = n
		}
	}

	sq8 := ivf.DefaultSQ8
	if raw := os.Getenv("IVF_SQ8"); raw == "false" {
		sq8 = false
	}

	log.Printf("building IVF index (nlist=%d, sq8=%v)…", nlist, sq8)

	if err := ivf.Build(binPath, ivfPath, nlist, sq8, zap.NewNop()); err != nil {
		log.Fatalf("build IVF index: %v", err)
	}

	log.Printf("done: IVF index → %s", ivfPath)
}

// buildVamana constructs a Vamana approximate index from binPath and exports it to
// resources/references.vamana. Parameters are read from environment variables:
//
//	VAMANA_R          max degree per node (default 16)
//	VAMANA_BUILD_L    beam width used during graph construction (default 125);
//	                  reduce to 64–75 to cut build time ~50% at a small recall cost
//	VAMANA_ALPHA      RobustPrune multiplier (default 1.2)
//	VAMANA_SQ8        "false" to disable 8-bit quantization (default true)
func buildVamana(binPath string) {
	const vamanaPath = "resources/references.vamana"

	r := vamana.DefaultR
	if raw := os.Getenv("VAMANA_R"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			r = n
		}
	}

	// VAMANA_BUILD_L controls the construction beam width (separate from the
	// query-time VAMANA_L). 0 means use the package default (DefaultBuildL=125).
	buildL := 0
	if raw := os.Getenv("VAMANA_BUILD_L"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			buildL = n
		}
	}

	alpha := vamana.DefaultAlpha
	if raw := os.Getenv("VAMANA_ALPHA"); raw != "" {
		if f, err := strconv.ParseFloat(raw, 32); err == nil && f > 0 {
			alpha = float32(f)
		}
	}

	sq8 := vamana.DefaultSQ8
	if os.Getenv("VAMANA_SQ8") == "false" {
		sq8 = false
	}

	log.Printf("building Vamana index (R=%d, build_L=%d, alpha=%.2f, sq8=%v)…", r, buildL, alpha, sq8)

	if err := vamana.Build(binPath, vamanaPath, r, buildL, alpha, sq8, zap.NewNop()); err != nil {
		log.Fatalf("build Vamana index: %v", err)
	}

	log.Printf("done: Vamana index → %s", vamanaPath)
}

// buildHNSWFlat constructs a flat mmap-able HNSW index from binPath and writes it to
// resources/references.hnswflat. Parameters are read from environment variables:
//
//	HNSWFLAT_M            max connections per upper-layer node (default 3, layer 0 = 2×M)
//	HNSWFLAT_EF_BUILD     candidate-list size during construction (default 100)
//	HNSWFLAT_REFINE       "false" to skip the second-pass refinement (default true)
//	HNSWFLAT_SQ8          "false" to disable SQ8 quantization (default true)
//
// Memory at serve time (N=3M, M=3, SQ8=true): ~143 MB mmap + ~15 MB heap ≈ 158 MB RSS.
func buildHNSWFlat(binPath string) {
	const outPath = "resources/references.hnswflat"

	M := hnswflat.DefaultM
	if raw := os.Getenv("HNSWFLAT_M"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			M = n
		}
	}
	efBuild := hnswflat.DefaultEfConstruction
	if raw := os.Getenv("HNSWFLAT_EF_BUILD"); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			efBuild = n
		}
	}
	refine := hnswflat.DefaultRefine
	if os.Getenv("HNSWFLAT_REFINE") == "false" {
		refine = false
	}
	sq8 := hnswflat.DefaultSQ8
	if os.Getenv("HNSWFLAT_SQ8") == "false" {
		sq8 = false
	}

	log.Printf("building HNSW-flat index (M=%d, ef_build=%d, refine=%v, sq8=%v)…", M, efBuild, refine, sq8)

	logger, _ := zap.NewProduction()
	defer logger.Sync() //nolint:errcheck

	if err := hnswflat.Build(binPath, outPath, M, efBuild, refine, sq8, logger); err != nil {
		log.Fatalf("build HNSW-flat index: %v", err)
	}
	log.Printf("done: HNSW-flat index → %s", outPath)
}

// isFiniteF32 reports whether f is neither NaN nor infinite.
// Sentinels like -1.0 (used for missing last_transaction) are finite and accepted.
func isFiniteF32(f float32) bool {
	return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
}
