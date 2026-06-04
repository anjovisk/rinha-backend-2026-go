// Package simdbrute provides partition-routed exact KNN with AVX2-accelerated
// squared-Euclidean distance computation. Requires AMD64 with AVX2 (Intel Haswell+,
// AMD Ryzen+). The server logs a fatal error at startup if AVX2 is not detected.
package simdbrute

import (
	"log"

	"golang.org/x/sys/cpu"
)

func init() {
	if !cpu.X86.HasAVX2 {
		log.Fatal("simdbrute: AVX2 required (Intel Haswell 2013+ or AMD Ryzen 2017+)")
	}
}

// l2sq14 computes the squared Euclidean distance between two 14-element float32
// arrays using AVX2 (8-wide SIMD). Implemented in distance_amd64.s.
//
//go:noescape
func l2sq14(a, b *float32) float32
