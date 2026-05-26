// Package handler implementa os handlers HTTP da API de detecção de fraudes.
// Cada handler corresponde a um endpoint e segue a assinatura padrão
// de [net/http.HandlerFunc]: recebe um [http.ResponseWriter] e um [*http.Request].
package handler

import "net/http"

// Ready responde ao health check da API.
//
// Retorna HTTP 200 OK quando o servidor está pronto para receber requisições.
// É utilizado pelo load balancer e por ferramentas de orquestração para
// determinar se a instância está saudável antes de rotear tráfego.
//
// Rota: GET /ready
func Ready(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}
