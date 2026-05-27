package domain

// VectorSize is the number of dimensions in the feature vector used for KNN classification.
// The 14 dimensions encode transaction, customer, merchant, and terminal attributes
// as normalised floats in [0.0, 1.0]. Dimensions 5 (minutes_since_last_tx) and
// 6 (km_from_last_tx) are the sole exception: they use -1 as a sentinel value
// when LastTransaction is nil.
const VectorSize = 14

// Vector is the 14-dimensional feature representation of a FraudScoreRequest.
// It is produced by Vectorizer.Vectorize and consumed by NeighborFinder.FindNearest.
//
// Index layout:
//
//	0  amount              clamp(amount / max_amount)
//	1  installments        clamp(installments / max_installments)
//	2  amount_vs_avg       clamp((amount / avg_amount) / ratio)
//	3  hour_of_day         UTC hour / 23
//	4  day_of_week         Mon=0 … Sun=6, divided by 6
//	5  minutes_since_last  clamp(minutes / max_minutes) or -1
//	6  km_from_last        clamp(km / max_km) or -1
//	7  km_from_home        clamp(km / max_km)
//	8  tx_count_24h        clamp(count / max_count)
//	9  is_online           1 or 0
//	10 card_present        1 or 0
//	11 unknown_merchant    1 if merchant not in known list, else 0
//	12 mcc_risk            lookup in mcc_risk.json, default 0.5
//	13 merchant_avg_amount clamp(avg / max_merchant_avg_amount)
type Vector [VectorSize]float64
