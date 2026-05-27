package httphandler_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.uber.org/zap"

	httphandler "anjovisk/fraud-detection/internal/adapter/http"
	"anjovisk/fraud-detection/internal/domain"
)

// mockFraudUseCase implements port.FraudScoreUseCase for testing.
// It records the last request it received and returns a configurable result or error.
type mockFraudUseCase struct {
	result domain.FraudResult
	err    error
	gotReq domain.FraudScoreRequest
}

// Evaluate stores req for later inspection and returns the configured result or error.
func (m *mockFraudUseCase) Evaluate(_ context.Context, req domain.FraudScoreRequest) (domain.FraudResult, error) {
	m.gotReq = req
	return m.result, m.err
}

// postFraudScore sends a POST /fraud-score request with the given JSON body and returns the recorder.
func postFraudScore(t *testing.T, srv *httphandler.Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Routes().ServeHTTP(rec, req)
	return rec
}

// validBody is a well-formed POST /fraud-score payload with last_transaction null.
const validBody = `{
	"id": "tx-test",
	"transaction": {"amount": 100.0, "installments": 1, "requested_at": "2026-03-11T20:00:00Z"},
	"customer": {"avg_amount": 200.0, "tx_count_24h": 2, "known_merchants": ["MERC-1"]},
	"merchant": {"id": "MERC-2", "mcc": "5411", "avg_amount": 50.0},
	"terminal": {"is_online": false, "card_present": true, "km_from_home": 10.0},
	"last_transaction": null
}`

func TestHandleFraudScore_ValidRequest_Returns200WithResult(t *testing.T) {
	mock := &mockFraudUseCase{result: domain.FraudResult{Approved: true, FraudScore: 0.2}}
	rec := postFraudScore(t, httphandler.NewServer(mock, zap.NewNop()), validBody)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Approved {
		t.Errorf("approved = %v, want true", resp.Approved)
	}
	if resp.FraudScore != 0.2 {
		t.Errorf("fraud_score = %v, want 0.2", resp.FraudScore)
	}
}

func TestHandleFraudScore_InvalidJSON_Returns400(t *testing.T) {
	rec := postFraudScore(t, httphandler.NewServer(&mockFraudUseCase{}, zap.NewNop()), "not-json")

	if rec.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", rec.Code)
	}
}

// TestHandleFraudScore_UseCaseError_ReturnsSoftApproval verifies the contest rule:
// on internal errors the handler must return approved=true, fraud_score=0.0 rather than HTTP 5xx.
func TestHandleFraudScore_UseCaseError_ReturnsSoftApproval(t *testing.T) {
	mock := &mockFraudUseCase{err: errors.New("internal error")}
	rec := postFraudScore(t, httphandler.NewServer(mock, zap.NewNop()), validBody)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200 (soft approval), got %d", rec.Code)
	}

	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Approved || resp.FraudScore != 0.0 {
		t.Errorf("expected soft approval {true, 0.0}, got {%v, %v}", resp.Approved, resp.FraudScore)
	}
}

// TestHandleFraudScore_NullLastTransaction verifies that a JSON null last_transaction
// produces a nil LastTransaction in the domain request forwarded to the use case.
func TestHandleFraudScore_NullLastTransaction_MapsToNilDomain(t *testing.T) {
	mock := &mockFraudUseCase{result: domain.FraudResult{Approved: true}}
	postFraudScore(t, httphandler.NewServer(mock, zap.NewNop()), validBody)

	if mock.gotReq.LastTransaction != nil {
		t.Error("expected LastTransaction to be nil when JSON last_transaction is null")
	}
}

// TestHandleFraudScore_WithLastTransaction verifies that a non-null last_transaction
// is correctly mapped into the domain request forwarded to the use case.
func TestHandleFraudScore_WithLastTransaction_MapsFields(t *testing.T) {
	mock := &mockFraudUseCase{result: domain.FraudResult{Approved: true}}
	body := `{
		"id": "tx-with-last",
		"transaction": {"amount": 100.0, "installments": 1, "requested_at": "2026-03-11T20:00:00Z"},
		"customer": {"avg_amount": 200.0, "tx_count_24h": 1, "known_merchants": []},
		"merchant": {"id": "MERC-1", "mcc": "5411", "avg_amount": 50.0},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 0.0},
		"last_transaction": {"timestamp": "2026-03-11T14:00:00Z", "km_from_current": 20.0}
	}`

	postFraudScore(t, httphandler.NewServer(mock, zap.NewNop()), body)

	if mock.gotReq.LastTransaction == nil {
		t.Fatal("expected LastTransaction to be non-nil when provided in JSON")
	}
	if mock.gotReq.LastTransaction.KmFromCurrent != 20.0 {
		t.Errorf("KmFromCurrent = %v, want 20.0", mock.gotReq.LastTransaction.KmFromCurrent)
	}
}

// TestHandleFraudScore_FieldMapping verifies that scalar fields in the JSON body
// are correctly mapped into the domain.FraudScoreRequest forwarded to the use case.
func TestHandleFraudScore_FieldMapping(t *testing.T) {
	mock := &mockFraudUseCase{result: domain.FraudResult{Approved: false, FraudScore: 0.8}}
	postFraudScore(t, httphandler.NewServer(mock, zap.NewNop()), validBody)

	got := mock.gotReq
	if got.ID != "tx-test" {
		t.Errorf("ID = %q, want %q", got.ID, "tx-test")
	}
	if got.Transaction.Amount != 100.0 {
		t.Errorf("Transaction.Amount = %v, want 100.0", got.Transaction.Amount)
	}
	if got.Customer.TxCount24h != 2 {
		t.Errorf("Customer.TxCount24h = %v, want 2", got.Customer.TxCount24h)
	}
	if got.Merchant.MCC != "5411" {
		t.Errorf("Merchant.MCC = %q, want %q", got.Merchant.MCC, "5411")
	}
	if !got.Terminal.CardPresent {
		t.Error("Terminal.CardPresent = false, want true")
	}
}
