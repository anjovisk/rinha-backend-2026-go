// Package domain contains the core business entities of the fraud detection system.
// These types are free of infrastructure dependencies and represent the language
// of the problem: transactions, customers, merchants, terminals, and fraud results.
package domain

import "time"

// Transaction holds the financial details of a card payment request.
type Transaction struct {
	// Amount is the transaction value in the account's currency.
	Amount float64
	// Installments is the number of instalment payments; 1 means a single charge.
	Installments int
	// RequestedAt is the UTC timestamp when the payment was initiated.
	RequestedAt time.Time
}

// Customer holds behavioural context about the cardholder derived from historical data.
type Customer struct {
	// AvgAmount is the customer's historical average transaction amount.
	AvgAmount float64
	// TxCount24h is the number of transactions the customer performed in the last 24 hours.
	TxCount24h int
	// KnownMerchants is the list of merchant IDs the customer has previously transacted with.
	KnownMerchants []string
}

// Merchant holds information about the business accepting the payment.
type Merchant struct {
	// ID uniquely identifies the merchant within the system.
	ID string
	// MCC is the Merchant Category Code (ISO 18245) used to look up the merchant's risk score.
	MCC string
	// AvgAmount is the merchant's historical average transaction amount.
	AvgAmount float64
}

// Terminal describes the point of sale — physical or virtual — where the transaction occurred.
type Terminal struct {
	// IsOnline indicates whether the transaction was processed over the internet.
	IsOnline bool
	// CardPresent indicates whether the physical card was present at the terminal.
	CardPresent bool
	// KmFromHome is the distance in kilometres between the terminal and the customer's home address.
	KmFromHome float64
}

// LastTransaction holds location and timing data from the customer's most recent prior transaction.
// A nil value signals that no prior transaction exists for this customer.
type LastTransaction struct {
	// Timestamp is the UTC time of the previous transaction.
	Timestamp time.Time
	// KmFromCurrent is the distance in kilometres between the previous terminal and the current one.
	KmFromCurrent float64
}

// FraudScoreRequest is the root input to the fraud evaluation pipeline.
// It aggregates all context — transaction details, customer history, merchant info,
// and terminal data — needed to produce a vectorized representation and fraud score.
type FraudScoreRequest struct {
	// ID is the unique identifier of the transaction being evaluated.
	ID          string
	Transaction Transaction
	Customer    Customer
	Merchant    Merchant
	Terminal    Terminal
	// LastTransaction is nil when the customer has no prior transaction on record,
	// which triggers sentinel values (-1) in vector dimensions 5 and 6.
	LastTransaction *LastTransaction
}
