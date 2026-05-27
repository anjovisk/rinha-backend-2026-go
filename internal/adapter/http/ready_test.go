package httphandler_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"go.uber.org/zap"

	httphandler "anjovisk/fraud-detection/internal/adapter/http"
)

func TestHandleReady_returns200(t *testing.T) {
	srv := httphandler.NewServer(nil, zap.NewNop())
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()

	srv.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, rec.Code)
	}
}
