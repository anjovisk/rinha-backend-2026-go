package handler_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anjovisk/fraud-detection/internal/handler"
)

// TestReady_returns200 verifica que o handler Ready responde com HTTP 200 OK.
func TestReady_returns200(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()

	handler.Ready(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}
