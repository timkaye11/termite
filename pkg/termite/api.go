// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.22

//go:generate go tool oapi-codegen --config=cfg.yaml ./openapi.yaml
package termite

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/antfly-go/libaf/embeddings"
	"github.com/antflydb/antfly-go/libaf/s3"
	"github.com/antflydb/antfly-go/libaf/scraping"
	"github.com/antflydb/termite/pkg/termite/lib/ner"
	"github.com/antflydb/termite/pkg/termite/lib/seq2seq"
	"github.com/bytedance/sonic/decoder"
	"github.com/bytedance/sonic/encoder"
	"go.uber.org/zap"
)

// NOTE: SerializeFloatArrays is in codec.go in this package

// TermiteAPI implements the generated ServerInterface
type TermiteAPI struct {
	logger *zap.Logger
	node   *TermiteNode
}

// NewTermiteAPI creates a new HTTP handler for the Termite API using generated code
func NewTermiteAPI(logger *zap.Logger, node *TermiteNode) http.Handler {
	api := &TermiteAPI{
		logger: logger,
		node:   node,
	}
	return HandlerWithOptions(api, StdHTTPServerOptions{
		BaseURL:    "/api",
		BaseRouter: http.NewServeMux(),
	})
}

// GenerateEmbeddings implements ServerInterface
func (t *TermiteAPI) GenerateEmbeddings(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiEmbed(w, r)
}

// ChunkText implements ServerInterface
func (t *TermiteAPI) ChunkText(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiChunk(w, r)
}

// RerankPrompts implements ServerInterface
func (t *TermiteAPI) RerankPrompts(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiRerank(w, r)
}

// RecognizeEntities implements ServerInterface
func (t *TermiteAPI) RecognizeEntities(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiNER(w, r)
}

// GenerateQuestions implements ServerInterface
func (t *TermiteAPI) GenerateQuestions(w http.ResponseWriter, r *http.Request) {
	t.node.handleApiGenerate(w, r)
}

// ListModels implements ServerInterface
func (t *TermiteAPI) ListModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelsResponse{
		Chunkers:      []string{},
		Rerankers:     []string{},
		Embedders:     []string{},
		Recognizers:   []string{},
		Extractors:    []string{},
		Questionators: []string{},
	}

	if t.node.cachedChunker != nil {
		resp.Chunkers = t.node.cachedChunker.ListModels()
	}

	if t.node.embedderProvider != nil {
		resp.Embedders = t.node.embedderProvider.List()
	}

	if t.node.rerankerRegistry != nil {
		resp.Rerankers = t.node.rerankerRegistry.List()
	}

	if t.node.nerRegistry != nil {
		resp.Recognizers = t.node.nerRegistry.List()
		resp.Extractors = t.node.nerRegistry.ListRecognizers()
	}

	if t.node.seq2seqRegistry != nil {
		resp.Questionators = t.node.seq2seqRegistry.List()
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		t.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// GetVersion implements ServerInterface
func (t *TermiteAPI) GetVersion(w http.ResponseWriter, r *http.Request) {
	resp := VersionResponse{
		Version:   Version,
		GitCommit: GitCommit,
		BuildTime: BuildTime,
		GoVersion: runtime.Version(),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		t.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiEmbed handles embedding generation requests using Ollama-compatible API
// with OpenAI-compatible multimodal extension for CLIP models.
func (ln *TermiteNode) handleApiEmbed(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if embedder provider is available
	if ln.embedderProvider == nil {
		http.Error(w, "embedding not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			// Context cancelled
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode the request using generated types
	var req EmbedRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate model
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}

	// Get embedder from provider (lazy loads if needed)
	embedder, err := ln.embedderProvider.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Parse input - supports text strings, arrays, and multimodal content parts
	// Uses scraping package for URL downloads with security config and S3 credentials
	contents, err := parseEmbedInput(r.Context(), req.Input, ln.contentSecurityConfig, ln.s3Credentials)
	if err != nil {
		http.Error(w, fmt.Sprintf("invalid input: %v", err), http.StatusBadRequest)
		return
	}

	if len(contents) == 0 {
		http.Error(w, "input is required", http.StatusBadRequest)
		return
	}

	// Validate MIME types against embedder capabilities
	if err := validateContentTypes(contents, embedder.Capabilities()); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Wrap embedder with caching for deduplicated requests
	cachedEmbedder := ln.embeddingCache.WrapEmbedder(embedder, req.Model)

	// Generate embeddings (with caching and singleflight deduplication)
	embeds, err := cachedEmbedder.Embed(r.Context(), contents)
	if err != nil {
		ln.logger.Error("failed to generate embeddings",
			zap.String("model", req.Model),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("generating embeddings: %v", err), http.StatusInternalServerError)
		return
	}

	// Determine response format based on Accept header
	acceptHeader := r.Header.Get("Accept")

	switch acceptHeader {
	case "application/json":
		// JSON response using Ollama-compatible format
		resp := EmbedResponse{
			Model:      req.Model,
			Embeddings: embeds,
		}
		w.Header().Set("Content-Type", "application/json")
		if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
			ln.logger.Error("encoding JSON response", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

	default:
		// Default: binary serialization (application/octet-stream)
		w.Header().Set("Content-Type", "application/octet-stream")
		if err := SerializeFloatArrays(w, embeds); err != nil {
			ln.logger.Error("serializing embeddings", zap.Error(err))
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

// parseEmbedInput parses the EmbedRequest input which can be:
// - A single text string
// - An array of text strings (Ollama-compatible)
// - An array of ContentPart objects (OpenAI-compatible multimodal)
//
// For image_url content, supports:
// - Data URIs: data:image/png;base64,...
// - HTTP/HTTPS URLs: https://example.com/image.png
// - Local files: file:///path/to/image.png
// - S3 URLs: s3://endpoint/bucket/key
func parseEmbedInput(
	ctx context.Context,
	input EmbedRequest_Input,
	securityConfig *scraping.ContentSecurityConfig,
	s3Creds *s3.Credentials,
) ([][]ai.ContentPart, error) {
	// Try array of strings first (most common case)
	if arr, err := input.AsEmbedRequestInput1(); err == nil && len(arr) > 0 {
		contents := make([][]ai.ContentPart, len(arr))
		for i, t := range arr {
			contents[i] = []ai.ContentPart{ai.TextContent{Text: t}}
		}
		return contents, nil
	}

	// Try single string
	if str, err := input.AsEmbedRequestInput0(); err == nil && str != "" {
		return [][]ai.ContentPart{{ai.TextContent{Text: str}}}, nil
	}

	// Try multimodal content parts (OpenAI-compatible)
	if parts, err := input.AsEmbedRequestInput2(); err == nil && len(parts) > 0 {
		contents := make([][]ai.ContentPart, len(parts))
		for i, part := range parts {
			// Try text content
			if textPart, err := part.AsTextContentPart(); err == nil {
				contents[i] = []ai.ContentPart{ai.TextContent{Text: textPart.Text}}
				continue
			}

			// Try image URL content
			if imgPart, err := part.AsImageURLContentPart(); err == nil {
				// Use scraping package - handles data:, http://, https://, file://, s3://
				mimeType, data, err := scraping.DownloadContent(ctx, imgPart.ImageUrl.Url, securityConfig, s3Creds)
				if err != nil {
					return nil, fmt.Errorf("downloading image at index %d: %w", i, err)
				}
				contents[i] = []ai.ContentPart{ai.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				}}
				continue
			}

			return nil, fmt.Errorf("unknown content type at index %d", i)
		}
		return contents, nil
	}

	return nil, errors.New("input must be a string, array of strings, or array of content parts")
}

// validateContentTypes checks that all content types in the input are supported
// by the embedder's capabilities.
func validateContentTypes(contents [][]ai.ContentPart, caps embeddings.EmbedderCapabilities) error {
	// Build set of supported MIME types
	supported := make(map[string]bool)
	for _, m := range caps.SupportedMIMETypes {
		supported[m.MIMEType] = true
	}

	// Check each content part
	for i, parts := range contents {
		for _, part := range parts {
			switch p := part.(type) {
			case ai.TextContent:
				// Text is always supported via text/plain
				if !supported["text/plain"] {
					return fmt.Errorf("model does not support text input")
				}
			case ai.BinaryContent:
				if !supported[p.MIMEType] {
					return fmt.Errorf("unsupported MIME type at index %d: %s (model supports: %v)",
						i, p.MIMEType, getMIMETypeList(caps))
				}
			}
		}
	}

	return nil
}

// getMIMETypeList returns a list of supported MIME types for error messages.
func getMIMETypeList(caps embeddings.EmbedderCapabilities) []string {
	types := make([]string, len(caps.SupportedMIMETypes))
	for i, m := range caps.SupportedMIMETypes {
		types[i] = m.MIMEType
	}
	return types
}

// handleApiChunk handles text chunking requests
func (ln *TermiteNode) handleApiChunk(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	var req ChunkRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate the request
	if req.Text == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	// Convert ChunkConfig to internal chunkConfig type
	internalConfig := chunkConfig{
		Model:         req.Config.Model,
		TargetTokens:  req.Config.TargetTokens,
		OverlapTokens: req.Config.OverlapTokens,
		Separator:     req.Config.Separator,
		MaxChunks:     req.Config.MaxChunks,
		Threshold:     req.Config.Threshold,
	}

	// Use cached chunker to process the request
	chunks, cacheHit, err := ln.cachedChunker.Chunk(r.Context(), req.Text, internalConfig)
	if err != nil {
		ln.logger.Error("chunking failed", zap.Error(err))
		http.Error(w, fmt.Sprintf("chunking text: %v", err), http.StatusInternalServerError)
		return
	}

	// Record metrics
	modelUsed := internalConfig.Model
	if modelUsed == "" {
		modelUsed = "default"
	}
	RecordChunkerRequest(modelUsed)
	RecordChunkCreation(modelUsed, len(chunks))

	// Build response
	resp := ChunkResponse{
		Chunks:   chunks,
		Model:    internalConfig.Model,
		CacheHit: cacheHit,
	}

	// Return response
	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiRerank handles reranking requests
func (ln *TermiteNode) handleApiRerank(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if reranking is available
	if ln.rerankerRegistry == nil || len(ln.rerankerRegistry.List()) == 0 {
		http.Error(w, "reranking not available", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model   string   `json:"model"`   // Model name to use (required)
		Query   string   `json:"query"`   // Query text
		Prompts []string `json:"prompts"` // Pre-rendered document texts to rerank
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if len(req.Prompts) == 0 {
		http.Error(w, "prompts are required", http.StatusBadRequest)
		return
	}

	// Get model from registry
	reranker, err := ln.rerankerRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Wrap reranker with caching for deduplicated requests
	cachedReranker := ln.rerankingCache.WrapReranker(reranker, req.Model)

	// Rerank prompts (with caching and singleflight deduplication)
	scores, err := cachedReranker.Rerank(r.Context(), req.Query, req.Prompts)
	if err != nil {
		ln.logger.Error("reranking failed",
			zap.String("model", req.Model),
			zap.String("query", req.Query),
			zap.Int("num_prompts", len(req.Prompts)),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("reranking failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Record metrics
	RecordRerankerRequest(req.Model)
	RecordRerankingCreation(req.Model, len(req.Prompts))

	// Validate response
	if len(scores) != len(req.Prompts) {
		http.Error(w,
			fmt.Sprintf("expected %d scores, got %d", len(req.Prompts), len(scores)),
			http.StatusInternalServerError)
		return
	}

	ln.logger.Info("reranking request completed",
		zap.String("model", req.Model),
		zap.String("query", req.Query),
		zap.Int("num_prompts", len(req.Prompts)),
		zap.Int("num_scores", len(scores)))

	// Send response
	resp := struct {
		Model  string    `json:"model"`
		Scores []float32 `json:"scores"`
	}{
		Model:  req.Model,
		Scores: scores,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiNER handles NER (Named Entity Recognition) requests
func (ln *TermiteNode) handleApiNER(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if NER is available
	if ln.nerRegistry == nil || len(ln.nerRegistry.List()) == 0 {
		http.Error(w, "NER not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model  string   `json:"model"`  // Model name to use (required)
		Texts  []string `json:"texts"`  // Texts to extract entities from
		Labels []string `json:"labels"` // Custom labels for GLiNER models (optional)
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		http.Error(w, "texts are required", http.StatusBadRequest)
		return
	}

	var entities [][]ner.Entity

	// Check if this is a zero-shot Recognizer with custom labels
	if len(req.Labels) > 0 && ln.nerRegistry.IsRecognizer(req.Model) {
		// Use Recognizer with custom labels (zero-shot NER)
		recognizer, err := ln.nerRegistry.GetRecognizer(req.Model)
		if err != nil {
			http.Error(w, fmt.Sprintf("Recognizer not found: %s", req.Model), http.StatusNotFound)
			return
		}

		// Recognize with custom labels
		entities, err = recognizer.RecognizeWithLabels(r.Context(), req.Texts, req.Labels)
		if err != nil {
			ln.logger.Error("Recognition failed",
				zap.String("model", req.Model),
				zap.Strings("labels", req.Labels),
				zap.Int("num_texts", len(req.Texts)),
				zap.Error(err))
			http.Error(w, fmt.Sprintf("Recognition failed: %v", err), http.StatusInternalServerError)
			return
		}
	} else {
		// Get standard NER model from registry
		model, err := ln.nerRegistry.Get(req.Model)
		if err != nil {
			http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
			return
		}

		// Wrap model with caching for deduplicated requests
		cachedModel := ln.nerCache.WrapModel(model, req.Model)

		// Recognize entities (with caching and singleflight deduplication)
		entities, err = cachedModel.Recognize(r.Context(), req.Texts)
		if err != nil {
			ln.logger.Error("NER failed",
				zap.String("model", req.Model),
				zap.Int("num_texts", len(req.Texts)),
				zap.Error(err))
			http.Error(w, fmt.Sprintf("NER failed: %v", err), http.StatusInternalServerError)
			return
		}
	}

	// Record metrics
	RecordNERRequest(req.Model)
	totalEntities := 0
	for _, textEntities := range entities {
		totalEntities += len(textEntities)
	}
	RecordNERCreation(req.Model, totalEntities)

	ln.logger.Info("NER request completed",
		zap.String("model", req.Model),
		zap.Int("num_texts", len(req.Texts)),
		zap.Int("total_entities", totalEntities))

	// Convert internal Entity type to API response type
	apiEntities := make([][]RecognizeEntity, len(entities))
	for i, textEntities := range entities {
		apiEntities[i] = make([]RecognizeEntity, len(textEntities))
		for j, e := range textEntities {
			apiEntities[i][j] = RecognizeEntity{
				Text:  e.Text,
				Label: e.Label,
				Start: e.Start,
				End:   e.End,
				Score: e.Score,
			}
		}
	}

	// Send response
	nerResp := RecognizeResponse{
		Model:    req.Model,
		Entities: apiEntities,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(nerResp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiGenerate handles Seq2Seq text generation requests
func (ln *TermiteNode) handleApiGenerate(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if generation is available
	if ln.seq2seqRegistry == nil || len(ln.seq2seqRegistry.List()) == 0 {
		http.Error(w, "generation not available: no models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request
	var req struct {
		Model  string   `json:"model"`  // Model name to use (required)
		Inputs []string `json:"inputs"` // Input texts to generate from
	}
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Inputs) == 0 {
		http.Error(w, "inputs are required", http.StatusBadRequest)
		return
	}

	// Get model from registry
	model, err := ln.seq2seqRegistry.Get(req.Model)
	if err != nil {
		http.Error(w, fmt.Sprintf("model not found: %s", req.Model), http.StatusNotFound)
		return
	}

	// Generate text
	output, err := model.Generate(r.Context(), req.Inputs)
	if err != nil {
		ln.logger.Error("generation failed",
			zap.String("model", req.Model),
			zap.Int("num_inputs", len(req.Inputs)),
			zap.Error(err))
		http.Error(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
		return
	}

	ln.logger.Info("generation request completed",
		zap.String("model", req.Model),
		zap.Int("num_inputs", len(req.Inputs)),
		zap.Int("num_outputs", len(output.Texts)))

	// Send response
	resp := QuestionGenerateResponse{
		Model: req.Model,
		Texts: output.Texts,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(resp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// handleApiKnowledgeGraph handles knowledge graph building requests
func (ln *TermiteNode) handleApiKnowledgeGraph(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()

	// Check if any KG-capable models are available (NER or REBEL)
	hasNER := ln.nerRegistry != nil && len(ln.nerRegistry.List()) > 0
	hasREBEL := ln.relRegistry != nil && len(ln.relRegistry.List()) > 0
	if !hasNER && !hasREBEL {
		http.Error(w, "knowledge graph not available: no NER or relation extraction models configured", http.StatusServiceUnavailable)
		return
	}

	// Apply backpressure via request queue
	release, err := ln.requestQueue.Acquire(r.Context())
	if err != nil {
		switch err {
		case ErrQueueFull:
			RecordQueueRejection()
			WriteQueueFullResponse(w, 5*time.Second)
		case ErrRequestTimeout:
			RecordQueueTimeout()
			WriteTimeoutResponse(w)
		default:
			http.Error(w, "request cancelled", http.StatusRequestTimeout)
		}
		return
	}
	defer release()

	// Update queue metrics
	UpdateQueueMetrics(ln.requestQueue.Stats())

	// Decode request using generated types
	var req KnowledgeGraphRequest
	if err := decoder.NewStreamDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decoding request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.Model == "" {
		http.Error(w, "model is required", http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		http.Error(w, "texts are required", http.StatusBadRequest)
		return
	}

	var entities [][]ner.Entity
	var relations [][]ner.Relation

	// Try REBEL model first (for relation extraction)
	if ln.relRegistry != nil {
		rebelModel, err := ln.relRegistry.Get(req.Model)
		if err == nil {
			ln.logger.Info("Using REBEL model for KG extraction",
				zap.String("model", req.Model),
				zap.Int("num_texts", len(req.Texts)))
			// Use REBEL for relation extraction
			entities, relations, err = ln.extractWithREBEL(r.Context(), rebelModel, req.Texts)
			if err != nil {
				ln.logger.Error("REBEL relation extraction failed",
					zap.String("model", req.Model),
					zap.Error(err))
				http.Error(w, fmt.Sprintf("relation extraction failed: %v", err), http.StatusInternalServerError)
				return
			}
			// Log extraction results
			totalEntities := 0
			totalRelations := 0
			for i := range entities {
				totalEntities += len(entities[i])
				totalRelations += len(relations[i])
			}
			ln.logger.Info("REBEL extraction results",
				zap.Int("total_entities", totalEntities),
				zap.Int("total_relations", totalRelations))
			goto buildKG
		}
	}

	// Fall back to GLiNER model with relation extraction support
	{
		glinerModel, err := ln.nerRegistry.GetGLiNER(req.Model)
		if err != nil {
			http.Error(w, fmt.Sprintf("model not found: %s (tried REBEL and GLiNER registries)", req.Model), http.StatusNotFound)
			return
		}

		// Cast to RelationExtractionModel to access relation extraction methods
		relModel, ok := glinerModel.(ner.RelationExtractionModel)
		if !ok {
			http.Error(w, fmt.Sprintf("model %s does not support relation extraction", req.Model), http.StatusBadRequest)
			return
		}

		// Get entity and relation labels (use defaults if not provided)
		entityLabels := req.EntityLabels
		if len(entityLabels) == 0 {
			entityLabels = glinerModel.Labels()
		}

		relationLabels := req.RelationLabels
		if len(relationLabels) == 0 {
			relationLabels = relModel.RelationLabels()
		}

		// Extract entities and relations
		entities, relations, err = relModel.RecognizeWithRelations(r.Context(), req.Texts, entityLabels, relationLabels)
		if err != nil {
			ln.logger.Error("GLiNER recognition with relations failed",
				zap.String("model", req.Model),
				zap.Error(err))
			http.Error(w, fmt.Sprintf("entity/relation extraction failed: %v", err), http.StatusInternalServerError)
			return
		}
	}

buildKG:

	// Build KG config
	kgConfig := ner.DefaultKGBuilderConfig()
	// Apply config overrides if non-zero
	if req.Config.SimilarityThreshold != 0 {
		kgConfig.EntityResolver.SimilarityThreshold = req.Config.SimilarityThreshold
	}
	// TypeMustMatch - since bool defaults to false, we need to check if it was explicitly set
	// For now, always use the value from request (false means don't require type match)
	kgConfig.EntityResolver.TypeMustMatch = req.Config.TypeMustMatch
	if req.Config.MinEntityConfidence != 0 {
		kgConfig.MinEntityConfidence = req.Config.MinEntityConfidence
	}
	if req.Config.MinRelationConfidence != 0 {
		kgConfig.MinRelationConfidence = req.Config.MinRelationConfidence
	}
	// DeduplicateRelations - bool, use the value directly
	kgConfig.DeduplicateRelations = req.Config.DeduplicateRelations
	// TrackProvenance - bool, use the value directly
	kgConfig.TrackProvenance = req.Config.TrackProvenance

	// Build extractions for each text
	extractions := make([]ner.ExtractionInput, len(req.Texts))
	for i, text := range req.Texts {
		extractions[i] = ner.ExtractionInput{
			DocumentID:     fmt.Sprintf("text-%d", i),
			Entities:       entities[i],
			Relations:      relations[i],
			ExtractorModel: req.Model,
			ExtractionTime: time.Now(),
		}
		_ = text // Used for document context
	}

	// Build knowledge graph
	kg := ner.BuildKnowledgeGraphFromMultiple(extractions, kgConfig)

	// Record metrics
	RecordNERRequest(req.Model)
	totalEntities := 0
	for _, textEntities := range entities {
		totalEntities += len(textEntities)
	}
	RecordNERCreation(req.Model, totalEntities)

	ln.logger.Info("knowledge graph built",
		zap.String("model", req.Model),
		zap.Int("num_texts", len(req.Texts)),
		zap.Int("nodes", kg.NodeCount()),
		zap.Int("edges", kg.EdgeCount()))

	// Convert to API response types
	meta := kg.Metadata()
	apiMeta := KGMetadata{
		Name:          meta.Name,
		Description:   meta.Description,
		CreatedAt:     meta.CreatedAt,
		UpdatedAt:     meta.UpdatedAt,
		NodeCount:     meta.NodeCount,
		EdgeCount:     meta.EdgeCount,
		DocumentCount: meta.DocumentCount,
		DocumentIds:   meta.DocumentIDs,
	}

	apiNodes := make([]KGNode, 0, kg.NodeCount())
	for _, node := range kg.Nodes() {
		apiNode := KGNode{
			Id:            node.ID,
			CanonicalName: node.CanonicalName,
			Type:          node.Type,
			Confidence:    node.Confidence,
			CreatedAt:     node.CreatedAt,
			UpdatedAt:     node.UpdatedAt,
			Mentions:      node.Mentions,
		}
		if len(node.Properties) > 0 {
			apiNode.Properties = make(map[string]interface{})
			for k, v := range node.Properties {
				apiNode.Properties[k] = v
			}
		}
		if len(node.Provenance) > 0 {
			apiProv := make([]KGProvenance, len(node.Provenance))
			for j, p := range node.Provenance {
				apiProv[j] = KGProvenance{
					SourceDocument:      p.SourceDocument,
					SourceUrl:           p.SourceURL,
					SourceText:          p.SourceText,
					CharOffsetStart:     p.CharOffsetStart,
					CharOffsetEnd:       p.CharOffsetEnd,
					ExtractorModel:      p.ExtractorModel,
					ExtractionTimestamp: p.ExtractionTimestamp,
					ExtractorConfidence: p.ExtractorConfidence,
				}
			}
			apiNode.Provenance = apiProv
		}
		apiNodes = append(apiNodes, apiNode)
	}

	apiEdges := make([]KGEdge, 0, kg.EdgeCount())
	for _, edge := range kg.Edges() {
		apiEdge := KGEdge{
			Id:         edge.ID,
			SourceId:   edge.SourceID,
			TargetId:   edge.TargetID,
			Type:       edge.Type,
			Confidence: edge.Confidence,
			CreatedAt:  edge.CreatedAt,
			UpdatedAt:  edge.UpdatedAt,
		}
		if len(edge.Properties) > 0 {
			apiEdge.Properties = make(map[string]interface{})
			for k, v := range edge.Properties {
				apiEdge.Properties[k] = v
			}
		}
		if len(edge.Provenance) > 0 {
			apiProv := make([]KGProvenance, len(edge.Provenance))
			for j, p := range edge.Provenance {
				apiProv[j] = KGProvenance{
					SourceDocument:      p.SourceDocument,
					SourceUrl:           p.SourceURL,
					SourceText:          p.SourceText,
					CharOffsetStart:     p.CharOffsetStart,
					CharOffsetEnd:       p.CharOffsetEnd,
					ExtractorModel:      p.ExtractorModel,
					ExtractionTimestamp: p.ExtractionTimestamp,
					ExtractorConfidence: p.ExtractorConfidence,
				}
			}
			apiEdge.Provenance = apiProv
		}
		apiEdges = append(apiEdges, apiEdge)
	}

	// Send response
	kgResp := KnowledgeGraphResponse{
		Model:    req.Model,
		Metadata: apiMeta,
		Nodes:    apiNodes,
		Edges:    apiEdges,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := encoder.NewStreamEncoder(w).Encode(kgResp); err != nil {
		ln.logger.Error("encoding response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// extractWithREBEL uses a REBEL model to extract entities and relations from text.
// REBEL outputs triplets (subject, object, relation) which we convert to ner.Entity and ner.Relation format.
func (ln *TermiteNode) extractWithREBEL(ctx context.Context, model seq2seq.RelationExtractor, texts []string) ([][]ner.Entity, [][]ner.Relation, error) {
	output, err := model.ExtractRelations(ctx, texts)
	if err != nil {
		return nil, nil, fmt.Errorf("REBEL extraction: %w", err)
	}

	entities := make([][]ner.Entity, len(texts))
	relations := make([][]ner.Relation, len(texts))

	for i, triplets := range output.Triplets {
		text := texts[i]
		entityMap := make(map[string]int) // entity text -> entity index
		textEntities := []ner.Entity{}
		textRelations := []ner.Relation{}

		for _, triplet := range triplets {
			// Create or find subject entity
			subjectIdx, ok := entityMap[triplet.Subject]
			if !ok {
				subjectIdx = len(textEntities)
				entityMap[triplet.Subject] = subjectIdx
				start, end := findSpan(text, triplet.Subject)
				textEntities = append(textEntities, ner.Entity{
					Text:  triplet.Subject,
					Label: "ENTITY", // REBEL doesn't provide entity types
					Start: start,
					End:   end,
					Score: triplet.Score,
				})
			}

			// Create or find object entity
			objectIdx, ok := entityMap[triplet.Object]
			if !ok {
				objectIdx = len(textEntities)
				entityMap[triplet.Object] = objectIdx
				start, end := findSpan(text, triplet.Object)
				textEntities = append(textEntities, ner.Entity{
					Text:  triplet.Object,
					Label: "ENTITY", // REBEL doesn't provide entity types
					Start: start,
					End:   end,
					Score: triplet.Score,
				})
			}

			// Create relation between subject and object
			subjectEntity := textEntities[subjectIdx]
			objectEntity := textEntities[objectIdx]
			textRelations = append(textRelations, ner.Relation{
				HeadEntity: subjectEntity,
				TailEntity: objectEntity,
				Label:      triplet.Relation,
				Score:      triplet.Score,
			})
		}

		entities[i] = textEntities
		relations[i] = textRelations
	}

	return entities, relations, nil
}

// findSpan finds the character offsets of a substring in text.
// Returns (0, len(substring)) if not found (to avoid nil spans).
func findSpan(text, substring string) (int, int) {
	// Case-insensitive search
	idx := strings.Index(strings.ToLower(text), strings.ToLower(substring))
	if idx >= 0 {
		return idx, idx + len(substring)
	}
	// Not found - return placeholder span
	return 0, len(substring)
}
