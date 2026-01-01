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

package graph

import (
	"context"
	"fmt"
	"math"
	"sync"
)

// EmbedderFunc is a function type that computes embeddings for a batch of texts.
// It allows injecting any embedder implementation (e.g., from termite's embedding service).
// The returned embeddings should be normalized for cosine similarity to work correctly.
type EmbedderFunc func(ctx context.Context, texts []string) ([][]float32, error)

// VectorEntityResolver performs entity resolution using vector embeddings.
// It computes cosine similarity between entity text embeddings to determine
// if entities refer to the same real-world object.
//
// The resolver maintains a thread-safe cache of embeddings to avoid
// redundant embedding computations for previously seen entity texts.
type VectorEntityResolver struct {
	embedder EmbedderFunc

	// cache stores computed embeddings keyed by entity text
	cache map[string][]float32
	mu    sync.RWMutex

	// similarityThreshold is the minimum cosine similarity for entities
	// to be considered the same. Default: 0.85
	similarityThreshold float32
}

// VectorResolverOption is a functional option for configuring VectorEntityResolver.
type VectorResolverOption func(*VectorEntityResolver)

// WithSimilarityThreshold sets the minimum cosine similarity threshold
// for entities to be considered matches. Default is 0.85.
func WithSimilarityThreshold(threshold float32) VectorResolverOption {
	return func(v *VectorEntityResolver) {
		v.similarityThreshold = threshold
	}
}

// NewVectorEntityResolver creates a new VectorEntityResolver with the given embedder function.
// The embedder function will be called to compute embeddings for entity texts.
//
// Example usage:
//
//	embedder := func(ctx context.Context, texts []string) ([][]float32, error) {
//	    // Call your embedding service here
//	    return embeddingService.Embed(ctx, texts)
//	}
//	resolver := NewVectorEntityResolver(embedder)
func NewVectorEntityResolver(embedder EmbedderFunc, opts ...VectorResolverOption) *VectorEntityResolver {
	v := &VectorEntityResolver{
		embedder:            embedder,
		cache:               make(map[string][]float32),
		similarityThreshold: 0.85,
	}

	for _, opt := range opts {
		opt(v)
	}

	return v
}

// SimilarityThreshold returns the configured similarity threshold.
func (v *VectorEntityResolver) SimilarityThreshold() float32 {
	return v.similarityThreshold
}

// SetSimilarityThreshold updates the similarity threshold.
func (v *VectorEntityResolver) SetSimilarityThreshold(threshold float32) {
	v.similarityThreshold = threshold
}

// GetEmbedding retrieves the cached embedding for a text, or computes and caches it.
// This method is thread-safe.
func (v *VectorEntityResolver) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	// Try to get from cache first (read lock)
	v.mu.RLock()
	if embedding, ok := v.cache[text]; ok {
		v.mu.RUnlock()
		return embedding, nil
	}
	v.mu.RUnlock()

	// Not in cache, compute embedding
	embeddings, err := v.embedder(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("computing embedding for %q: %w", text, err)
	}

	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return nil, fmt.Errorf("embedder returned empty embedding for %q", text)
	}

	embedding := embeddings[0]

	// Cache the result (write lock)
	v.mu.Lock()
	v.cache[text] = embedding
	v.mu.Unlock()

	return embedding, nil
}

// GetEmbeddings retrieves cached embeddings for multiple texts, computing any missing ones.
// This method batches the embedding computation for efficiency.
// This method is thread-safe.
func (v *VectorEntityResolver) GetEmbeddings(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	result := make([][]float32, len(texts))
	var textsToEmbed []string
	var textsToEmbedIndices []int

	// Check cache for existing embeddings
	v.mu.RLock()
	for i, text := range texts {
		if embedding, ok := v.cache[text]; ok {
			result[i] = embedding
		} else {
			textsToEmbed = append(textsToEmbed, text)
			textsToEmbedIndices = append(textsToEmbedIndices, i)
		}
	}
	v.mu.RUnlock()

	// Compute missing embeddings
	if len(textsToEmbed) > 0 {
		embeddings, err := v.embedder(ctx, textsToEmbed)
		if err != nil {
			return nil, fmt.Errorf("computing embeddings: %w", err)
		}

		if len(embeddings) != len(textsToEmbed) {
			return nil, fmt.Errorf("embedder returned %d embeddings for %d texts",
				len(embeddings), len(textsToEmbed))
		}

		// Cache and add to result
		v.mu.Lock()
		for i, embedding := range embeddings {
			text := textsToEmbed[i]
			v.cache[text] = embedding
			result[textsToEmbedIndices[i]] = embedding
		}
		v.mu.Unlock()
	}

	return result, nil
}

// ComputeSimilarity computes the cosine similarity between two entity texts.
// It uses cached embeddings when available.
// Returns a value between -1.0 and 1.0, where 1.0 means identical.
func (v *VectorEntityResolver) ComputeSimilarity(ctx context.Context, text1, text2 string) (float32, error) {
	// Get embeddings (batch for efficiency)
	embeddings, err := v.GetEmbeddings(ctx, []string{text1, text2})
	if err != nil {
		return 0, err
	}

	return CosineSimilarity(embeddings[0], embeddings[1]), nil
}

// FindBestMatch finds the best matching text from candidates based on vector similarity.
// Returns the best match text, its similarity score, and the index in candidates.
// Returns ("", 0, -1) if no match meets the threshold.
func (v *VectorEntityResolver) FindBestMatch(ctx context.Context, query string, candidates []string) (string, float32, int, error) {
	if len(candidates) == 0 {
		return "", 0, -1, nil
	}

	// Get query embedding
	queryEmb, err := v.GetEmbedding(ctx, query)
	if err != nil {
		return "", 0, -1, err
	}

	// Get candidate embeddings
	candidateEmbs, err := v.GetEmbeddings(ctx, candidates)
	if err != nil {
		return "", 0, -1, err
	}

	// Find best match
	var bestMatch string
	var bestScore float32
	bestIdx := -1

	for i, candEmb := range candidateEmbs {
		similarity := CosineSimilarity(queryEmb, candEmb)
		if similarity >= v.similarityThreshold && similarity > bestScore {
			bestScore = similarity
			bestMatch = candidates[i]
			bestIdx = i
		}
	}

	return bestMatch, bestScore, bestIdx, nil
}

// ClearCache clears the embedding cache.
// This is useful when the embedder configuration changes or for memory management.
func (v *VectorEntityResolver) ClearCache() {
	v.mu.Lock()
	v.cache = make(map[string][]float32)
	v.mu.Unlock()
}

// CacheSize returns the number of cached embeddings.
func (v *VectorEntityResolver) CacheSize() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return len(v.cache)
}

// =============================================================================
// Vector Similarity Functions
// =============================================================================

// CosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns a value between -1.0 and 1.0, where 1.0 means identical direction,
// 0 means orthogonal, and -1.0 means opposite direction.
//
// For normalized vectors (magnitude = 1), this is equivalent to the dot product.
// If either vector is zero-length or has zero magnitude, returns 0.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	// Handle mismatched lengths by using the shorter length
	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var dotProduct float64
	var magnitudeA float64
	var magnitudeB float64

	for i := 0; i < n; i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		magnitudeA += float64(a[i]) * float64(a[i])
		magnitudeB += float64(b[i]) * float64(b[i])
	}

	// Handle zero magnitude vectors
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}

	similarity := dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB))

	// Clamp to [-1, 1] to handle floating point errors
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}

	return float32(similarity)
}

// DotProduct computes the dot product of two float32 vectors.
// If vectors have different lengths, uses the shorter length.
func DotProduct(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var result float64
	for i := 0; i < n; i++ {
		result += float64(a[i]) * float64(b[i])
	}

	return float32(result)
}

// EuclideanDistance computes the Euclidean distance between two float32 vectors.
// Smaller values indicate more similar vectors.
func EuclideanDistance(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var sum float64
	for i := 0; i < n; i++ {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}

	return float32(math.Sqrt(sum))
}

// NormalizeVector normalizes a vector to unit length (magnitude = 1).
// Returns a new normalized vector. If the input has zero magnitude, returns a zero vector.
func NormalizeVector(v []float32) []float32 {
	if len(v) == 0 {
		return nil
	}

	var magnitude float64
	for _, val := range v {
		magnitude += float64(val) * float64(val)
	}

	magnitude = math.Sqrt(magnitude)
	if magnitude == 0 {
		result := make([]float32, len(v))
		return result
	}

	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(float64(val) / magnitude)
	}

	return result
}
