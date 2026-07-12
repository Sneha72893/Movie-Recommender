package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

const (
	pythonServiceURL = "http://localhost:8000"
	rustServiceURL   = "http://localhost:3000"
	gatewayPort      = ":8080"
)

// HealthInfo for service health aggregation
type HealthInfo struct {
	Status   string `json:"status"`
	Service  string `json:"service"`
	Language string `json:"language"`
}

// AggregatedHealth returned by the gateway
type AggregatedHealth struct {
	Gateway       string       `json:"gateway"`
	Status        string       `json:"status"`
	Uptime        string       `json:"uptime"`
	PyService     *HealthInfo  `json:"python_inference_service,omitempty"`
	RustService   *HealthInfo  `json:"rust_data_engine,omitempty"`
}

var startTime = time.Now()
var client = &http.Client{Timeout: 5 * time.Second}

// fetchJSON is a helper to GET a URL and decode JSON into `target`.
func fetchJSON(url string, target interface{}) error {
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, target)
}

// addCORS adds CORS headers to every response.
func addCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

// healthHandler aggregates the health of all downstream services.
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	resp := AggregatedHealth{
		Gateway:  "Go API Gateway (net/http)",
		Status:   "ok",
		Uptime:   time.Since(startTime).Round(time.Second).String(),
	}

	var pyHealth HealthInfo
	if err := fetchJSON(pythonServiceURL+"/", &pyHealth); err == nil {
		resp.PyService = &pyHealth
	}

	var rustHealth HealthInfo
	if err := fetchJSON(rustServiceURL+"/health", &rustHealth); err == nil {
		resp.RustService = &rustHealth
	}

	json.NewEncoder(w).Encode(resp)
}

// reverseProxy creates a reverse proxy handler to a target URL.
func reverseProxy(target string) http.HandlerFunc {
	targetURL, _ := url.Parse(target)
	proxy := httputil.NewSingleHostReverseProxy(targetURL)
	return func(w http.ResponseWriter, r *http.Request) {
		// Strip the prefix so proxied service sees the right path
		proxy.ServeHTTP(w, r)
	}
}

func main() {
	mux := http.NewServeMux()

	// Gateway health check (aggregates all services)
	mux.HandleFunc("/health", addCORS(healthHandler))

	// Route /api/recommendations/* -> Python FastAPI Inference Service
	mux.HandleFunc("/api/recommendations/", addCORS(func(w http.ResponseWriter, r *http.Request) {
		// Rewrite the path: strip /api prefix
		r.URL.Path = r.URL.Path[4:]
		r.URL.Host = "localhost:8000"
		r.URL.Scheme = "http"
		reverseProxy(pythonServiceURL)(w, r)
	}))

	// Route /api/metrics -> Python FastAPI
	mux.HandleFunc("/api/metrics", addCORS(func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/metrics"
		reverseProxy(pythonServiceURL)(w, r)
	}))

	// Route /api/movies/* -> Rust Data Engine
	mux.HandleFunc("/api/movies", addCORS(func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = "/movies"
		reverseProxy(rustServiceURL)(w, r)
	}))
	mux.HandleFunc("/api/movies/", addCORS(func(w http.ResponseWriter, r *http.Request) {
		r.URL.Path = r.URL.Path[4:] // Strip /api
		reverseProxy(rustServiceURL)(w, r)
	}))

	fmt.Printf("🐹 Go API Gateway listening on http://localhost%s\n", gatewayPort)
	fmt.Println("   Routing:")
	fmt.Printf("   /api/recommendations/* -> Python FastAPI (%s)\n", pythonServiceURL)
	fmt.Printf("   /api/movies/*          -> Rust Data Engine (%s)\n", rustServiceURL)
	fmt.Println("   /health                -> Aggregated health check")

	log.Fatal(http.ListenAndServe(gatewayPort, mux))
}
