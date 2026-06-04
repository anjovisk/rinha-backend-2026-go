// Package qdrant implements port.NeighborFinder using Qdrant as the vector database.
//
// On the first call to Open, the adapter checks whether the Qdrant collection already
// contains the full reference dataset. If not, it reads resources/references.bin
// (mmap'd), converts every entry to a Qdrant point and upserts them in batches.
// This one-time load is persisted by Qdrant in its storage volume, so subsequent
// server starts skip the load entirely.
//
// The underlying search uses Qdrant's HNSW graph with scalar int8 quantisation
// (≈4× memory reduction). Setting QDRANT_EXACT=true disables HNSW and forces an
// exact brute-force scan inside Qdrant at the cost of higher latency.
//
// Enable by setting VECTOR_SEARCHER=qdrant at server startup.
// Requires a running Qdrant instance reachable at QDRANT_URL (default http://qdrant:6333).
package qdrant

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"syscall"
	"time"
	"unsafe"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// DefaultURL is the Qdrant REST endpoint used when QDRANT_URL is not set.
const DefaultURL = "http://qdrant:6333"

// DefaultCollection is the Qdrant collection name used when QDRANT_COLLECTION is not set.
const DefaultCollection = "references"

// DefaultExact controls whether searches are exact. false = HNSW (fast, approximate).
const DefaultExact = false

// insertBatchSize is the number of points per upsert request.
// Larger batches reduce round-trips; 512 keeps JSON payloads under ~400 KB.
const insertBatchSize = 512

// Searcher queries Qdrant for the k nearest neighbours of a given vector.
type Searcher struct {
	baseURL    string
	collection string
	exact      bool
	client     *http.Client
	logger     *zap.Logger
}

// Open connects to Qdrant, creates the collection if absent, and loads all reference
// vectors from binPath when the collection is empty. Returns a ready-to-use Searcher.
func Open(binPath, url, collection string, exact bool, logger *zap.Logger) (*Searcher, error) {
	s := &Searcher{
		baseURL:    url,
		collection: collection,
		exact:      exact,
		client:     &http.Client{Timeout: 60 * time.Second},
		logger:     logger,
	}

	if err := s.waitReady(60 * time.Second); err != nil {
		return nil, err
	}

	count, err := s.pointCount()
	if err != nil {
		// Collection absent — create it and load data.
		if cerr := s.createCollection(); cerr != nil {
			return nil, fmt.Errorf("create collection: %w", cerr)
		}
		count = 0
	}

	if count == 0 {
		logger.Info("qdrant: collection empty, loading reference vectors",
			zap.String("path", binPath),
		)
		n, err := s.loadFromBin(binPath)
		if err != nil {
			return nil, fmt.Errorf("load vectors: %w", err)
		}
		logger.Info("qdrant: reference vectors loaded", zap.Int("count", n))
	} else {
		logger.Info("qdrant: collection already populated", zap.Int("count", count))
	}

	return s, nil
}

// FindNearest returns the labels of the k nearest reference entries to v.
func (s *Searcher) FindNearest(v domain.Vector, k int) []string {
	// Convert float64 query to float32 for the Qdrant API.
	vec := make([]float32, domain.VectorSize)
	for i, x := range v {
		vec[i] = float32(x)
	}

	type searchParams struct {
		Exact bool `json:"exact"`
	}
	type searchReq struct {
		Vector []float32    `json:"vector"`
		Limit  int          `json:"limit"`
		Params searchParams `json:"params"`
	}
	body, _ := json.Marshal(searchReq{
		Vector: vec,
		Limit:  k,
		Params: searchParams{Exact: s.exact},
	})

	resp, err := s.post(fmt.Sprintf("/collections/%s/points/search", s.collection), body)
	if err != nil {
		s.logger.Error("qdrant search failed", zap.Error(err))
		// Soft fallback: return all legit to avoid HTTP 500.
		labels := make([]string, k)
		for i := range labels {
			labels[i] = "legit"
		}
		return labels
	}
	defer resp.Body.Close()

	var result struct {
		Result []struct {
			Payload struct {
				Fraud int `json:"fraud"`
			} `json:"payload"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		s.logger.Error("qdrant: decode search response", zap.Error(err))
		labels := make([]string, k)
		for i := range labels {
			labels[i] = "legit"
		}
		return labels
	}

	labels := make([]string, len(result.Result))
	for i, r := range result.Result {
		if r.Payload.Fraud == 1 {
			labels[i] = "fraud"
		} else {
			labels[i] = "legit"
		}
	}
	return labels
}

// waitReady polls GET /healthz until Qdrant responds or the deadline is exceeded.
func (s *Searcher) waitReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := s.client.Get(s.baseURL + "/healthz")
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			s.logger.Info("qdrant: service ready")
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		s.logger.Debug("qdrant: waiting for service…")
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("qdrant not ready after %s", timeout)
}

// pointCount returns the number of points in the collection, or an error if it doesn't exist.
func (s *Searcher) pointCount() (int, error) {
	resp, err := s.client.Get(fmt.Sprintf("%s/collections/%s", s.baseURL, s.collection))
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return 0, fmt.Errorf("collection not found")
	}
	var result struct {
		Result struct {
			PointsCount int `json:"points_count"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, err
	}
	return result.Result.PointsCount, nil
}

// createCollection creates the Qdrant collection with Euclid distance and scalar int8
// quantisation to reduce memory usage (~4× vs raw float32).
func (s *Searcher) createCollection() error {
	body, _ := json.Marshal(map[string]any{
		"vectors": map[string]any{
			"size":     domain.VectorSize,
			"distance": "Euclid",
		},
		"quantization_config": map[string]any{
			"scalar": map[string]any{
				"type":       "int8",
				"always_ram": true,
			},
		},
	})

	req, _ := http.NewRequest(http.MethodPut,
		fmt.Sprintf("%s/collections/%s", s.baseURL, s.collection), bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("create collection status %d: %s", resp.StatusCode, b)
	}
	s.logger.Info("qdrant: collection created", zap.String("collection", s.collection))
	return nil
}

// loadFromBin reads references.bin, converts every entry to a Qdrant point and
// upserts them in batches of insertBatchSize. Returns the number of points loaded.
func (s *Searcher) loadFromBin(binPath string) (int, error) {
	f, err := os.Open(binPath)
	if err != nil {
		return 0, fmt.Errorf("open %s: %w", binPath, err)
	}
	defer f.Close()

	info, _ := f.Stat()
	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return 0, fmt.Errorf("mmap %s: %w", binPath, err)
	}
	defer syscall.Munmap(data) //nolint:errcheck

	n := int(binary.LittleEndian.Uint32(data[:4]))
	vectorPtr := (*float32)(unsafe.Pointer(&data[4]))
	vecs := unsafe.Slice(vectorPtr, n*domain.VectorSize)
	labels := data[4+n*domain.VectorSize*4:]

	type point struct {
		ID      int            `json:"id"`
		Vector  []float32      `json:"vector"`
		Payload map[string]int `json:"payload"`
	}

	batch := make([]point, 0, insertBatchSize)
	flush := func() error {
		body, _ := json.Marshal(map[string]any{"points": batch})
		resp, err := s.put(fmt.Sprintf("/collections/%s/points?wait=true", s.collection), body)
		if err != nil {
			return err
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("upsert status %d", resp.StatusCode)
		}
		batch = batch[:0]
		return nil
	}

	for i := 0; i < n; i++ {
		base := i * domain.VectorSize
		vec := make([]float32, domain.VectorSize)
		copy(vec, vecs[base:base+domain.VectorSize])

		fraud := 0
		if labels[i] == 1 {
			fraud = 1
		}
		batch = append(batch, point{
			ID:      i,
			Vector:  vec,
			Payload: map[string]int{"fraud": fraud},
		})

		if len(batch) == insertBatchSize {
			if err := flush(); err != nil {
				return i, err
			}
			if i%100_000 == 0 && i > 0 {
				s.logger.Info("qdrant: loading vectors…", zap.Int("loaded", i))
			}
		}
	}
	if len(batch) > 0 {
		if err := flush(); err != nil {
			return n - len(batch), err
		}
	}
	return n, nil
}

// post sends a JSON POST request and returns the response.
func (s *Searcher) post(path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodPost, s.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return s.client.Do(req)
}

// put sends a JSON PUT request and returns the response.
func (s *Searcher) put(path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodPut, s.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return s.client.Do(req)
}
