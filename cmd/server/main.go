// Command server inicia o servidor HTTP da API de detecção de fraudes.
//
// A porta de escuta é configurada pela variável de ambiente PORT.
// Quando PORT não está definida, o servidor usa a porta 8080 por padrão.
// A porta 9999 é reservada para o load balancer (Nginx).
//
// Uso:
//
//	PORT=8081 ./server
package main

import (
	"log"
	"net/http"
	"os"

	"github.com/anjovisk/fraud-detection/internal/handler"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", handler.Ready)

	log.Printf("server listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
