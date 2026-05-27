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
}

// isFiniteF32 reports whether f is neither NaN nor infinite.
// Sentinels like -1.0 (used for missing last_transaction) are finite and accepted.
func isFiniteF32(f float32) bool {
	return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
}
