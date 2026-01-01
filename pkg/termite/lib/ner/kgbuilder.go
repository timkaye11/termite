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

package ner

import (
	"context"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"
)

// =============================================================================
// Entity Resolution Configuration
// =============================================================================

// EntityResolverConfig configures entity resolution behavior
type EntityResolverConfig struct {
	// SimilarityThreshold is the minimum similarity score (0.0-1.0) for entities
	// to be considered the same. Default: 0.85
	SimilarityThreshold float32

	// TypeMustMatch requires entities to have the same type to be merged.
	// Default: true
	TypeMustMatch bool

	// CaseSensitive enables case-sensitive matching. Default: false
	CaseSensitive bool

	// UseJaroWinkler uses Jaro-Winkler similarity instead of simple overlap.
	// Default: true (only used when Strategy is StringSimilarity)
	UseJaroWinkler bool

	// MergeConfidenceStrategy determines how to combine confidence scores
	// when merging entities. Default: MaxConfidence
	MergeConfidenceStrategy ConfidenceStrategy

	// Strategy defines the resolution method to use.
	// Default: StringSimilarity (maintains backward compatibility)
	Strategy ResolutionStrategy

	// HybridLowerThreshold is the lower bound for ambiguous string similarity
	// in HybridSimilarity mode. Matches below this use string similarity only.
	// Default: 0.7
	HybridLowerThreshold float32

	// HybridUpperThreshold is the upper bound for ambiguous string similarity
	// in HybridSimilarity mode. Matches above this use string similarity only.
	// Default: 0.9
	HybridUpperThreshold float32
}

// ConfidenceStrategy defines how to combine confidence scores
type ConfidenceStrategy int

const (
	// MaxConfidence uses the maximum confidence score
	MaxConfidence ConfidenceStrategy = iota
	// AverageConfidence uses the average of all confidence scores
	AverageConfidence
	// WeightedConfidence weights by number of mentions
	WeightedConfidence
)

// ResolutionStrategy defines the method used for entity resolution
type ResolutionStrategy int

const (
	// StringSimilarity uses string-based similarity (Jaro-Winkler or token overlap).
	// This is the default and maintains backward compatibility.
	StringSimilarity ResolutionStrategy = iota
	// VectorSimilarity uses embedding-based cosine similarity for resolution.
	// Requires a VectorEntityResolver to be set on the KGBuilder.
	VectorSimilarity
	// HybridSimilarity uses string similarity first, falling back to vector
	// similarity for ambiguous cases (string similarity between 0.7-0.9).
	// Requires a VectorEntityResolver to be set on the KGBuilder.
	HybridSimilarity
)

// DefaultEntityResolverConfig returns the default entity resolution configuration
func DefaultEntityResolverConfig() EntityResolverConfig {
	return EntityResolverConfig{
		SimilarityThreshold:     0.85,
		TypeMustMatch:           true,
		CaseSensitive:           false,
		UseJaroWinkler:          true,
		MergeConfidenceStrategy: MaxConfidence,
		Strategy:                StringSimilarity,
		HybridLowerThreshold:    0.7,
		HybridUpperThreshold:    0.9,
	}
}

// =============================================================================
// Knowledge Graph Builder
// =============================================================================

// KGBuilderConfig configures the knowledge graph builder
type KGBuilderConfig struct {
	// EntityResolver configures entity resolution
	EntityResolver EntityResolverConfig

	// MinEntityConfidence is the minimum confidence for entities to be added.
	// Default: 0.0 (include all)
	MinEntityConfidence float32

	// MinRelationConfidence is the minimum confidence for relations to be added.
	// Default: 0.0 (include all)
	MinRelationConfidence float32

	// DeduplicateRelations removes duplicate relations between the same entities.
	// Default: true
	DeduplicateRelations bool

	// TrackProvenance enables provenance tracking for all extractions.
	// Default: true
	TrackProvenance bool
}

// DefaultKGBuilderConfig returns the default builder configuration
func DefaultKGBuilderConfig() KGBuilderConfig {
	return KGBuilderConfig{
		EntityResolver:        DefaultEntityResolverConfig(),
		MinEntityConfidence:   0.0,
		MinRelationConfidence: 0.0,
		DeduplicateRelations:  true,
		TrackProvenance:       true,
	}
}

// KGBuilder builds knowledge graphs from extracted entities and relations
type KGBuilder struct {
	config         KGBuilderConfig
	graph          *KnowledgeGraph
	vectorResolver *VectorEntityResolver
}

// NewKGBuilder creates a new knowledge graph builder
func NewKGBuilder(config KGBuilderConfig) *KGBuilder {
	return &KGBuilder{
		config: config,
		graph:  NewKnowledgeGraph(),
	}
}

// NewKGBuilderWithGraph creates a builder that adds to an existing graph
func NewKGBuilderWithGraph(config KGBuilderConfig, graph *KnowledgeGraph) *KGBuilder {
	return &KGBuilder{
		config: config,
		graph:  graph,
	}
}

// Graph returns the underlying knowledge graph
func (b *KGBuilder) Graph() *KnowledgeGraph {
	return b.graph
}

// SetVectorResolver sets the vector entity resolver for vector-based resolution.
// This is required when using VectorSimilarity or HybridSimilarity resolution strategies.
// The resolver will be used to compute embedding-based similarity between entities.
//
// Example usage:
//
//	embedder := func(ctx context.Context, texts []string) ([][]float32, error) {
//	    return embeddingService.Embed(ctx, texts)
//	}
//	resolver := NewVectorEntityResolver(embedder)
//	builder.SetVectorResolver(resolver)
func (b *KGBuilder) SetVectorResolver(resolver *VectorEntityResolver) {
	b.vectorResolver = resolver
}

// VectorResolver returns the current vector entity resolver, if set.
func (b *KGBuilder) VectorResolver() *VectorEntityResolver {
	return b.vectorResolver
}

// =============================================================================
// Building from Entities and Relations
// =============================================================================

// ExtractionInput represents extracted entities and relations from a document
type ExtractionInput struct {
	// DocumentID identifies the source document
	DocumentID string

	// DocumentURL is the optional URL of the source
	DocumentURL string

	// Entities are the extracted entities
	Entities []Entity

	// Relations are the extracted relationships
	Relations []Relation

	// ExtractorModel is the model used for extraction
	ExtractorModel string

	// ExtractionTime is when the extraction occurred
	ExtractionTime time.Time
}

// AddExtraction adds entities and relations from an extraction to the graph.
// This is a convenience method that uses context.Background().
// For vector-based resolution strategies, use AddExtractionWithContext instead.
func (b *KGBuilder) AddExtraction(input ExtractionInput) error {
	return b.AddExtractionWithContext(context.Background(), input)
}

// AddExtractionWithContext adds entities and relations from an extraction to the graph.
// The context is used for vector embedding operations when using VectorSimilarity
// or HybridSimilarity resolution strategies.
func (b *KGBuilder) AddExtractionWithContext(ctx context.Context, input ExtractionInput) error {
	if input.ExtractionTime.IsZero() {
		input.ExtractionTime = time.Now()
	}

	// Map from original entity key to resolved node ID
	entityToNode := make(map[string]string)

	// First pass: add/resolve all entities
	for _, entity := range input.Entities {
		if entity.Score < b.config.MinEntityConfidence {
			continue
		}

		nodeID, err := b.addOrResolveEntity(ctx, entity, input)
		if err != nil {
			return err
		}
		entityToNode[entityKey(entity)] = nodeID
	}

	// Second pass: add relations
	for _, relation := range input.Relations {
		if relation.Score < b.config.MinRelationConfidence {
			continue
		}

		// Find or create nodes for head and tail entities
		headNodeID, ok := entityToNode[entityKey(relation.HeadEntity)]
		if !ok {
			// Entity wasn't added (maybe below threshold), add it now
			nodeID, err := b.addOrResolveEntity(ctx, relation.HeadEntity, input)
			if err != nil {
				return err
			}
			headNodeID = nodeID
			entityToNode[entityKey(relation.HeadEntity)] = headNodeID
		}

		tailNodeID, ok := entityToNode[entityKey(relation.TailEntity)]
		if !ok {
			nodeID, err := b.addOrResolveEntity(ctx, relation.TailEntity, input)
			if err != nil {
				return err
			}
			tailNodeID = nodeID
			entityToNode[entityKey(relation.TailEntity)] = tailNodeID
		}

		// Add the relation as an edge
		if err := b.addRelation(relation, headNodeID, tailNodeID, input); err != nil {
			return err
		}
	}

	// Update document tracking
	b.graph.mu.Lock()
	b.graph.metadata.DocumentCount++
	if input.DocumentID != "" {
		b.graph.metadata.DocumentIDs = append(b.graph.metadata.DocumentIDs, input.DocumentID)
	}
	b.graph.mu.Unlock()

	return nil
}

// addOrResolveEntity adds an entity or resolves it to an existing node
func (b *KGBuilder) addOrResolveEntity(ctx context.Context, entity Entity, input ExtractionInput) (string, error) {
	// Try to find an existing node that matches this entity
	existingNode, err := b.findMatchingNode(ctx, entity)
	if err != nil {
		return "", err
	}

	provenance := Provenance{
		SourceDocument:      input.DocumentID,
		SourceURL:           input.DocumentURL,
		SourceText:          entity.Text,
		CharOffsetStart:     entity.Start,
		CharOffsetEnd:       entity.End,
		ExtractorModel:      input.ExtractorModel,
		ExtractionTimestamp: input.ExtractionTime,
		ExtractorConfidence: entity.Score,
	}

	if existingNode != nil {
		// Merge with existing node
		return b.mergeEntityIntoNode(entity, existingNode, provenance)
	}

	// Create new node
	node := &KGNode{
		ID:            uuid.New().String(),
		CanonicalName: entity.Text,
		Type:          entity.Label,
		Mentions:      []string{entity.Text},
		Confidence:    entity.Score,
	}

	if b.config.TrackProvenance {
		node.Provenance = []Provenance{provenance}
	}

	if err := b.graph.AddNode(node); err != nil {
		return "", err
	}

	return node.ID, nil
}

// findMatchingNode finds an existing node that matches the entity using the configured
// resolution strategy (StringSimilarity, VectorSimilarity, or HybridSimilarity).
func (b *KGBuilder) findMatchingNode(ctx context.Context, entity Entity) (*KGNode, error) {
	config := b.config.EntityResolver

	// Gather candidates from multiple sources
	candidateMap := make(map[string]*KGNode)

	// First try exact match by mention
	for _, node := range b.graph.FindNodesByMention(entity.Text) {
		candidateMap[node.ID] = node
	}

	// Also try by canonical name
	for _, node := range b.graph.GetNodesByName(entity.Text) {
		candidateMap[node.ID] = node
	}

	// If no exact matches, search all nodes of the same type for similar names
	// This ensures we catch cases like "John Smith" vs "John" vs "J. Smith"
	if len(candidateMap) == 0 {
		for _, node := range b.graph.GetNodesByType(entity.Label) {
			candidateMap[node.ID] = node
		}
	}

	// Filter candidates by type if required
	candidates := make([]*KGNode, 0, len(candidateMap))
	for _, candidate := range candidateMap {
		if config.TypeMustMatch && normalizeType(candidate.Type) != normalizeType(entity.Label) {
			continue
		}
		candidates = append(candidates, candidate)
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	// Use the appropriate resolution strategy
	switch config.Strategy {
	case VectorSimilarity:
		return b.findMatchingNodeVector(ctx, entity, candidates)
	case HybridSimilarity:
		return b.findMatchingNodeHybrid(ctx, entity, candidates)
	default: // StringSimilarity
		return b.findMatchingNodeString(entity, candidates), nil
	}
}

// findMatchingNodeString finds the best matching node using string similarity.
func (b *KGBuilder) findMatchingNodeString(entity Entity, candidates []*KGNode) *KGNode {
	config := b.config.EntityResolver

	var bestMatch *KGNode
	var bestScore float32

	for _, candidate := range candidates {
		similarity := b.computeStringSimilarity(entity.Text, candidate, config)

		if similarity >= config.SimilarityThreshold && similarity > bestScore {
			bestScore = similarity
			bestMatch = candidate
		}
	}

	return bestMatch
}

// findMatchingNodeVector finds the best matching node using vector similarity.
// Returns an error if the vector resolver is not configured.
func (b *KGBuilder) findMatchingNodeVector(ctx context.Context, entity Entity, candidates []*KGNode) (*KGNode, error) {
	if b.vectorResolver == nil {
		return nil, nil // Fall back to no match if resolver not configured
	}

	config := b.config.EntityResolver

	var bestMatch *KGNode
	var bestScore float32

	for _, candidate := range candidates {
		similarity, err := b.computeVectorSimilarity(ctx, entity.Text, candidate)
		if err != nil {
			return nil, err
		}

		if similarity >= config.SimilarityThreshold && similarity > bestScore {
			bestScore = similarity
			bestMatch = candidate
		}
	}

	return bestMatch, nil
}

// findMatchingNodeHybrid uses string similarity first, falling back to vector
// similarity for ambiguous cases (string similarity between hybrid thresholds).
func (b *KGBuilder) findMatchingNodeHybrid(ctx context.Context, entity Entity, candidates []*KGNode) (*KGNode, error) {
	config := b.config.EntityResolver

	var bestMatch *KGNode
	var bestScore float32
	var ambiguousCandidates []*KGNode

	// First pass: use string similarity
	for _, candidate := range candidates {
		similarity := b.computeStringSimilarity(entity.Text, candidate, config)

		// Strong match - use string similarity
		if similarity >= config.HybridUpperThreshold {
			if similarity > bestScore {
				bestScore = similarity
				bestMatch = candidate
			}
		} else if similarity >= config.HybridLowerThreshold {
			// Ambiguous range - collect for vector comparison
			ambiguousCandidates = append(ambiguousCandidates, candidate)
		}
		// Below lower threshold - not a match
	}

	// If we have a strong match, return it
	if bestMatch != nil {
		return bestMatch, nil
	}

	// If no strong matches and we have ambiguous candidates, use vector similarity
	if len(ambiguousCandidates) > 0 && b.vectorResolver != nil {
		return b.findMatchingNodeVector(ctx, entity, ambiguousCandidates)
	}

	return nil, nil
}

// computeStringSimilarity computes the string similarity between entity text and a candidate node.
// It checks both the canonical name and all mentions, returning the highest score.
func (b *KGBuilder) computeStringSimilarity(entityText string, candidate *KGNode, config EntityResolverConfig) float32 {
	var similarity float32
	if config.UseJaroWinkler {
		similarity = jaroWinklerSimilarity(entityText, candidate.CanonicalName, config.CaseSensitive)
	} else {
		similarity = tokenOverlapSimilarity(entityText, candidate.CanonicalName, config.CaseSensitive)
	}

	// Also check against all mentions
	for _, mention := range candidate.Mentions {
		var mentionSim float32
		if config.UseJaroWinkler {
			mentionSim = jaroWinklerSimilarity(entityText, mention, config.CaseSensitive)
		} else {
			mentionSim = tokenOverlapSimilarity(entityText, mention, config.CaseSensitive)
		}
		if mentionSim > similarity {
			similarity = mentionSim
		}
	}

	return similarity
}

// computeVectorSimilarity computes the vector similarity between entity text and a candidate node.
// It checks both the canonical name and all mentions, returning the highest score.
func (b *KGBuilder) computeVectorSimilarity(ctx context.Context, entityText string, candidate *KGNode) (float32, error) {
	if b.vectorResolver == nil {
		return 0, nil
	}

	// Compare against canonical name
	similarity, err := b.vectorResolver.ComputeSimilarity(ctx, entityText, candidate.CanonicalName)
	if err != nil {
		return 0, err
	}

	// Also check against all mentions
	for _, mention := range candidate.Mentions {
		mentionSim, err := b.vectorResolver.ComputeSimilarity(ctx, entityText, mention)
		if err != nil {
			return 0, err
		}
		if mentionSim > similarity {
			similarity = mentionSim
		}
	}

	return similarity, nil
}

// mergeEntityIntoNode merges an entity into an existing node
func (b *KGBuilder) mergeEntityIntoNode(entity Entity, node *KGNode, provenance Provenance) (string, error) {
	config := b.config.EntityResolver

	// Check if mention already exists
	mentionExists := false
	normalizedMention := normalizeName(entity.Text)
	for _, m := range node.Mentions {
		if normalizeName(m) == normalizedMention {
			mentionExists = true
			break
		}
	}

	if !mentionExists {
		node.Mentions = append(node.Mentions, entity.Text)
	}

	// Update confidence based on strategy
	switch config.MergeConfidenceStrategy {
	case MaxConfidence:
		if entity.Score > node.Confidence {
			node.Confidence = entity.Score
		}
	case AverageConfidence:
		// Simple average for now
		node.Confidence = (node.Confidence + entity.Score) / 2
	case WeightedConfidence:
		// Weight by number of provenance records
		n := float32(len(node.Provenance))
		node.Confidence = (node.Confidence*n + entity.Score) / (n + 1)
	}

	// Add provenance
	if b.config.TrackProvenance {
		node.Provenance = append(node.Provenance, provenance)
	}

	// Update the node
	return node.ID, b.graph.UpdateNode(node)
}

// addRelation adds a relation as an edge between two nodes
func (b *KGBuilder) addRelation(relation Relation, sourceID, targetID string, input ExtractionInput) error {
	// Check for existing duplicate relation
	if b.config.DeduplicateRelations {
		existing := b.graph.GetEdgeBetween(sourceID, targetID, relation.Label)
		if existing != nil {
			// Update existing edge with new provenance
			if b.config.TrackProvenance {
				prov := Provenance{
					SourceDocument:      input.DocumentID,
					SourceURL:           input.DocumentURL,
					ExtractorModel:      input.ExtractorModel,
					ExtractionTimestamp: input.ExtractionTime,
					ExtractorConfidence: relation.Score,
				}
				existing.Provenance = append(existing.Provenance, prov)
			}

			// Update confidence
			if relation.Score > existing.Confidence {
				existing.Confidence = relation.Score
			}

			return b.graph.UpdateEdge(existing)
		}
	}

	// Create new edge
	edge := &KGEdge{
		ID:         uuid.New().String(),
		SourceID:   sourceID,
		TargetID:   targetID,
		Type:       relation.Label,
		Confidence: relation.Score,
	}

	if b.config.TrackProvenance {
		edge.Provenance = []Provenance{{
			SourceDocument:      input.DocumentID,
			SourceURL:           input.DocumentURL,
			ExtractorModel:      input.ExtractorModel,
			ExtractionTimestamp: input.ExtractionTime,
			ExtractorConfidence: relation.Score,
		}}
	}

	return b.graph.AddEdge(edge)
}

// entityKey creates a unique key for an entity based on text, label, and position
func entityKey(entity Entity) string {
	return entity.Text + "|" + entity.Label + "|" + strconv.Itoa(entity.Start) + "|" + strconv.Itoa(entity.End)
}

// =============================================================================
// Convenience Functions
// =============================================================================

// BuildKnowledgeGraph is a convenience function to build a knowledge graph
// from entities and relations with default configuration
func BuildKnowledgeGraph(entities []Entity, relations []Relation) (*KnowledgeGraph, error) {
	return BuildKnowledgeGraphWithConfig(entities, relations, DefaultKGBuilderConfig())
}

// BuildKnowledgeGraphWithConfig builds a knowledge graph with custom configuration
func BuildKnowledgeGraphWithConfig(entities []Entity, relations []Relation, config KGBuilderConfig) (*KnowledgeGraph, error) {
	builder := NewKGBuilder(config)

	input := ExtractionInput{
		Entities:       entities,
		Relations:      relations,
		ExtractionTime: time.Now(),
	}

	if err := builder.AddExtraction(input); err != nil {
		return nil, err
	}

	return builder.Graph(), nil
}

// BuildKnowledgeGraphFromMultiple builds a knowledge graph from multiple extraction results
func BuildKnowledgeGraphFromMultiple(extractions []ExtractionInput, config KGBuilderConfig) (*KnowledgeGraph, error) {
	builder := NewKGBuilder(config)

	for _, extraction := range extractions {
		if err := builder.AddExtraction(extraction); err != nil {
			return nil, err
		}
	}

	return builder.Graph(), nil
}

// =============================================================================
// String Similarity Functions
// =============================================================================

// jaroWinklerSimilarity calculates Jaro-Winkler similarity between two strings
// Returns a value between 0.0 (no similarity) and 1.0 (identical)
func jaroWinklerSimilarity(s1, s2 string, caseSensitive bool) float32 {
	// Handle empty strings first
	if len(s1) == 0 || len(s2) == 0 {
		return 0.0
	}

	if !caseSensitive {
		s1 = strings.ToLower(s1)
		s2 = strings.ToLower(s2)
	}

	if s1 == s2 {
		return 1.0
	}

	// Calculate Jaro similarity first
	jaroSim := jaroSimilarity(s1, s2)

	// Calculate common prefix length (up to 4 characters)
	prefixLen := 0
	maxPrefix := min(4, min(len(s1), len(s2)))
	for i := 0; i < maxPrefix; i++ {
		if s1[i] == s2[i] {
			prefixLen++
		} else {
			break
		}
	}

	// Winkler modification: boost similarity for common prefixes
	// Using standard scaling factor of 0.1
	return float32(jaroSim + float64(prefixLen)*0.1*(1.0-jaroSim))
}

// jaroSimilarity calculates Jaro similarity between two strings
func jaroSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	len1, len2 := len(s1), len(s2)
	if len1 == 0 || len2 == 0 {
		return 0.0
	}

	// Match window
	matchWindow := max(len1, len2)/2 - 1
	if matchWindow < 0 {
		matchWindow = 0
	}

	s1Matches := make([]bool, len1)
	s2Matches := make([]bool, len2)

	matches := 0
	transpositions := 0

	// Find matches
	for i := 0; i < len1; i++ {
		start := max(0, i-matchWindow)
		end := min(i+matchWindow+1, len2)

		for j := start; j < end; j++ {
			if s2Matches[j] || s1[i] != s2[j] {
				continue
			}
			s1Matches[i] = true
			s2Matches[j] = true
			matches++
			break
		}
	}

	if matches == 0 {
		return 0.0
	}

	// Count transpositions
	k := 0
	for i := 0; i < len1; i++ {
		if !s1Matches[i] {
			continue
		}
		for !s2Matches[k] {
			k++
		}
		if s1[i] != s2[k] {
			transpositions++
		}
		k++
	}

	return (float64(matches)/float64(len1) +
		float64(matches)/float64(len2) +
		float64(matches-transpositions/2)/float64(matches)) / 3.0
}

// tokenOverlapSimilarity calculates token overlap (Jaccard) similarity
func tokenOverlapSimilarity(s1, s2 string, caseSensitive bool) float32 {
	if !caseSensitive {
		s1 = strings.ToLower(s1)
		s2 = strings.ToLower(s2)
	}

	tokens1 := tokenize(s1)
	tokens2 := tokenize(s2)

	if len(tokens1) == 0 || len(tokens2) == 0 {
		return 0.0
	}

	// Build set for tokens1
	set1 := make(map[string]struct{})
	for _, t := range tokens1 {
		set1[t] = struct{}{}
	}

	// Count intersection
	intersection := 0
	set2 := make(map[string]struct{})
	for _, t := range tokens2 {
		set2[t] = struct{}{}
		if _, exists := set1[t]; exists {
			intersection++
		}
	}

	// Union size
	union := len(set1) + len(set2) - intersection

	if union == 0 {
		return 0.0
	}

	return float32(intersection) / float32(union)
}

// tokenize splits a string into tokens (words)
func tokenize(s string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}
