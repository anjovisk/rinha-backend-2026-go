package httphandler

import "net/http"

// handleReady handles GET /ready health-check requests.
// It returns HTTP 200 OK with no body to signal that the server is ready to accept traffic.
// The load balancer uses this endpoint to decide whether to route requests to this instance.
// Logged at Debug to avoid flooding logs with high-frequency load balancer probes.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	s.logger.Debug("health check")
	w.WriteHeader(http.StatusOK)
}
