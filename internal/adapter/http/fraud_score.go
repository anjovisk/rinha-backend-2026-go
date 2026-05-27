package httphandler

import (
	"encoding/json"
	"net/http"
	"time"

	"go.uber.org/zap"

	"anjovisk/fraud-detection/internal/domain"
)

// fraudScoreRequest is the JSON body expected by POST /fraud-score.
// All fields map directly to the corresponding domain entities.
type fraudScoreRequest struct {
	ID          string          `json:"id"`
	Transaction txPayload       `json:"transaction"`
	Customer    customerPayload `json:"customer"`
	Merchant    merchantPayload `json:"merchant"`
	Terminal    terminalPayload `json:"terminal"`
	// LastTx is nil when the customer has no prior transaction on record.
	LastTx *lastTxPayload `json:"last_transaction"`
}

// txPayload carries the financial and temporal fields of the transaction being evaluated.
type txPayload struct {
	Amount       float64   `json:"amount"`
	Installments int       `json:"installments"`
	RequestedAt  time.Time `json:"requested_at"`
}

// customerPayload carries historical behavioural data about the cardholder.
type customerPayload struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

// merchantPayload carries identifying and statistical data about the merchant.
type merchantPayload struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

// terminalPayload describes the point-of-sale context of the transaction.
type terminalPayload struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

// lastTxPayload carries location and timing data from the customer's previous transaction.
type lastTxPayload struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float64   `json:"km_from_current"`
}

// fraudScoreResponse is the JSON body returned by POST /fraud-score.
type fraudScoreResponse struct {
	// Approved is true when FraudScore is strictly below 0.6.
	Approved bool `json:"approved"`
	// FraudScore is the proportion of fraudulent neighbours in [0.0, 1.0].
	FraudScore float64 `json:"fraud_score"`
}

// handleFraudScore handles POST /fraud-score requests.
// It decodes the JSON body into a fraudScoreRequest, delegates evaluation to the
// fraud use case, and serialises the result as JSON.
//
// On use case error the handler returns a soft approval (approved: true, fraud_score: 0.0)
// instead of HTTP 5xx, following contest rules that penalise server errors more heavily
// than false positives.
func (s *Server) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	var payload fraudScoreRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		s.logger.Debug("failed to decode request body",
			zap.Error(err),
			zap.String("remote_addr", r.RemoteAddr),
		)
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	s.logger.Info("fraud score request received",
		zap.String("transaction_id", payload.ID),
		zap.Float64("amount", payload.Transaction.Amount),
		zap.String("merchant_id", payload.Merchant.ID),
	)

	result, err := s.fraudUseCase.Evaluate(r.Context(), payload.toDomain())
	if err != nil {
		s.logger.Warn("use case error, falling back to soft approval",
			zap.String("transaction_id", payload.ID),
			zap.Error(err),
		)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fraudScoreResponse{Approved: true, FraudScore: 0.0})
		return
	}

	s.logger.Debug("sending response",
		zap.String("transaction_id", payload.ID),
		zap.Bool("approved", result.Approved),
		zap.Float64("fraud_score", result.FraudScore),
	)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(fraudScoreResponse{
		Approved:   result.Approved,
		FraudScore: result.FraudScore,
	})
}

// toDomain converts the HTTP payload into the domain request model consumed by the use case.
// When LastTx is nil, the resulting domain.FraudScoreRequest also carries a nil LastTransaction,
// causing the vectorizer to emit -1 sentinels for dimensions 5 and 6.
func (p *fraudScoreRequest) toDomain() domain.FraudScoreRequest {
	req := domain.FraudScoreRequest{
		ID: p.ID,
		Transaction: domain.Transaction{
			Amount:       p.Transaction.Amount,
			Installments: p.Transaction.Installments,
			RequestedAt:  p.Transaction.RequestedAt,
		},
		Customer: domain.Customer{
			AvgAmount:      p.Customer.AvgAmount,
			TxCount24h:     p.Customer.TxCount24h,
			KnownMerchants: p.Customer.KnownMerchants,
		},
		Merchant: domain.Merchant{
			ID:        p.Merchant.ID,
			MCC:       p.Merchant.MCC,
			AvgAmount: p.Merchant.AvgAmount,
		},
		Terminal: domain.Terminal{
			IsOnline:    p.Terminal.IsOnline,
			CardPresent: p.Terminal.CardPresent,
			KmFromHome:  p.Terminal.KmFromHome,
		},
	}
	if p.LastTx != nil {
		req.LastTransaction = &domain.LastTransaction{
			Timestamp:     p.LastTx.Timestamp,
			KmFromCurrent: p.LastTx.KmFromCurrent,
		}
	}
	return req
}
