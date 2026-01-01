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
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"
)

// =============================================================================
// Entity and Relation Types (local definitions to avoid import cycle)
// =============================================================================

// Entity represents a named entity extracted from text.
// This mirrors ner.Entity but is defined here to avoid import cycles.
type Entity struct {
	// Text is the entity text (e.g., "John Smith")
	Text string `json:"text"`
	// Label is the entity type (e.g., "person", "organization", "location")
	Label string `json:"label"`
	// Start is the character offset where the entity begins
	Start int `json:"start"`
	// End is the character offset where the entity ends (exclusive)
	End int `json:"end"`
	// Score is the confidence score (0.0 to 1.0)
	Score float32 `json:"score"`
}

// Relation represents a relationship between two entities.
// This mirrors ner.Relation but is defined here to avoid import cycles.
type Relation struct {
	// HeadEntity is the source entity in the relationship
	HeadEntity Entity `json:"head"`
	// TailEntity is the target entity in the relationship
	TailEntity Entity `json:"tail"`
	// Label is the relationship type (e.g., "founded", "works_at", "located_in")
	Label string `json:"label"`
	// Score is the model's confidence in this relationship (0.0-1.0)
	Score float32 `json:"score"`
}

// =============================================================================
// Extractor Interfaces
// =============================================================================

// EntityExtractor defines the interface for extracting entities from text.
// This can be implemented by the Termite API layer to call GLiNER or other models.
type EntityExtractor interface {
	// Extract extracts entities from the given texts using the specified labels.
	// Returns a slice of entities for each input text.
	Extract(ctx context.Context, texts []string, labels []string) ([][]Entity, error)
}

// RelationExtractor defines the interface for extracting relations from text.
// This can be implemented by the Termite API layer to call REBEL or other models.
type RelationExtractor interface {
	// Extract extracts relations from the given texts.
	// Returns a slice of relations for each input text.
	Extract(ctx context.Context, texts []string) ([][]Relation, error)
}

// TextChunker defines the interface for splitting text into chunks.
type TextChunker interface {
	// Chunk splits text into overlapping chunks suitable for extraction.
	// Returns the chunks and their character offsets in the original text.
	Chunk(text string, chunkSize int, overlap int) []TextChunk
}

// TextChunk represents a chunk of text with its position in the original document.
type TextChunk struct {
	// Text is the chunk content
	Text string
	// StartChar is the character offset where the chunk begins in the original text
	StartChar int
	// EndChar is the character offset where the chunk ends (exclusive)
	EndChar int
}

// =============================================================================
// Knowledge Graph Types (local definitions to avoid import cycle)
// =============================================================================

// KGNode represents an entity node in the knowledge graph.
type KGNode struct {
	// ID is the unique identifier for this node
	ID string `json:"id"`
	// CanonicalName is the primary/preferred name for this entity
	CanonicalName string `json:"canonical_name"`
	// Type is the entity type (e.g., "person", "organization", "location")
	Type string `json:"type"`
	// Mentions are all surface forms that refer to this entity
	Mentions []string `json:"mentions,omitempty"`
	// Properties are arbitrary key-value attributes
	Properties map[string]any `json:"properties,omitempty"`
	// Confidence is the aggregated confidence score (0.0-1.0)
	Confidence float32 `json:"confidence"`
	// Provenance tracks the sources of this entity
	Provenance []Provenance `json:"provenance,omitempty"`
	// CreatedAt is when the node was first created
	CreatedAt time.Time `json:"created_at"`
	// UpdatedAt is when the node was last modified
	UpdatedAt time.Time `json:"updated_at"`
}

// KGEdge represents a relationship edge between two nodes.
type KGEdge struct {
	// ID is the unique identifier for this edge
	ID string `json:"id"`
	// SourceID is the ID of the source node (head entity)
	SourceID string `json:"source_id"`
	// TargetID is the ID of the target node (tail entity)
	TargetID string `json:"target_id"`
	// Type is the relationship type (e.g., "founded", "works_at", "located_in")
	Type string `json:"type"`
	// Properties are arbitrary key-value attributes
	Properties map[string]any `json:"properties,omitempty"`
	// Confidence is the confidence score (0.0-1.0)
	Confidence float32 `json:"confidence"`
	// Provenance tracks the sources of this relationship
	Provenance []Provenance `json:"provenance,omitempty"`
	// CreatedAt is when the edge was first created
	CreatedAt time.Time `json:"created_at"`
	// UpdatedAt is when the edge was last modified
	UpdatedAt time.Time `json:"updated_at"`
}

// Provenance records the origin of extracted information.
type Provenance struct {
	// SourceDocument is the document ID or path
	SourceDocument string `json:"source_document,omitempty"`
	// SourceURL is the URL of the source (if applicable)
	SourceURL string `json:"source_url,omitempty"`
	// SourceText is the original text span that yielded this extraction
	SourceText string `json:"source_text,omitempty"`
	// CharOffsetStart is the character offset where the extraction begins
	CharOffsetStart int `json:"char_offset_start,omitempty"`
	// CharOffsetEnd is the character offset where the extraction ends
	CharOffsetEnd int `json:"char_offset_end,omitempty"`
	// ExtractorModel is the model used for extraction
	ExtractorModel string `json:"extractor_model,omitempty"`
	// ExtractionTimestamp is when the extraction occurred
	ExtractionTimestamp time.Time `json:"extraction_timestamp"`
	// ExtractorConfidence is the raw confidence from the extraction model
	ExtractorConfidence float32 `json:"extractor_confidence"`
}

// KnowledgeGraph represents a property graph for storing entities and relationships.
// This is a simplified version that works with the builder without import cycles.
type KnowledgeGraph struct {
	// Nodes indexed by ID
	nodes map[string]*KGNode
	// Edges indexed by ID
	edges map[string]*KGEdge
	// Indexes for fast lookup
	nodesByType   map[string]map[string]struct{}
	nodesByName   map[string]map[string]struct{}
	edgesByType   map[string]map[string]struct{}
	outgoingEdges map[string]map[string]struct{}
	incomingEdges map[string]map[string]struct{}
	// Metadata
	documentCount int
	documentIDs   []string
	createdAt     time.Time
	updatedAt     time.Time
	// Concurrency control
	mu sync.RWMutex
}

// NewKnowledgeGraph creates a new empty knowledge graph.
func NewKnowledgeGraph() *KnowledgeGraph {
	now := time.Now()
	return &KnowledgeGraph{
		nodes:         make(map[string]*KGNode),
		edges:         make(map[string]*KGEdge),
		nodesByType:   make(map[string]map[string]struct{}),
		nodesByName:   make(map[string]map[string]struct{}),
		edgesByType:   make(map[string]map[string]struct{}),
		outgoingEdges: make(map[string]map[string]struct{}),
		incomingEdges: make(map[string]map[string]struct{}),
		createdAt:     now,
		updatedAt:     now,
	}
}

// AddNode adds a new node to the graph.
func (kg *KnowledgeGraph) AddNode(node *KGNode) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	if _, exists := kg.nodes[node.ID]; exists {
		return fmt.Errorf("node with ID %s already exists", node.ID)
	}

	now := time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = now
	}
	node.UpdatedAt = now

	kg.nodes[node.ID] = node
	kg.indexNode(node)
	kg.updatedAt = now

	return nil
}

// GetNode retrieves a node by ID.
func (kg *KnowledgeGraph) GetNode(id string) *KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return kg.nodes[id]
}

// AddEdge adds a new edge to the graph.
func (kg *KnowledgeGraph) AddEdge(edge *KGEdge) error {
	if edge == nil {
		return fmt.Errorf("edge cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	if _, exists := kg.nodes[edge.SourceID]; !exists {
		return fmt.Errorf("source node %s not found", edge.SourceID)
	}
	if _, exists := kg.nodes[edge.TargetID]; !exists {
		return fmt.Errorf("target node %s not found", edge.TargetID)
	}

	if _, exists := kg.edges[edge.ID]; exists {
		return fmt.Errorf("edge with ID %s already exists", edge.ID)
	}

	now := time.Now()
	if edge.CreatedAt.IsZero() {
		edge.CreatedAt = now
	}
	edge.UpdatedAt = now

	kg.edges[edge.ID] = edge
	kg.indexEdge(edge)
	kg.updatedAt = now

	return nil
}

// GetEdge retrieves an edge by ID.
func (kg *KnowledgeGraph) GetEdge(id string) *KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return kg.edges[id]
}

// FindNodesByMention finds nodes that have a given mention.
func (kg *KnowledgeGraph) FindNodesByMention(mention string) []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	normalized := strings.ToLower(strings.TrimSpace(mention))
	ids, exists := kg.nodesByName[normalized]
	if !exists {
		return nil
	}

	nodes := make([]*KGNode, 0, len(ids))
	for id := range ids {
		if node, ok := kg.nodes[id]; ok {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// GetEdgeBetween finds an edge between two nodes with a specific type.
func (kg *KnowledgeGraph) GetEdgeBetween(sourceID, targetID, edgeType string) *KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.outgoingEdges[sourceID]
	if !exists {
		return nil
	}

	normalizedType := strings.ToLower(strings.TrimSpace(edgeType))
	for id := range ids {
		if edge, ok := kg.edges[id]; ok {
			if edge.TargetID == targetID && strings.ToLower(edge.Type) == normalizedType {
				return edge
			}
		}
	}
	return nil
}

// UpdateNode updates an existing node.
func (kg *KnowledgeGraph) UpdateNode(node *KGNode) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	existing, exists := kg.nodes[node.ID]
	if !exists {
		return fmt.Errorf("node with ID %s not found", node.ID)
	}

	kg.unindexNode(existing)
	node.UpdatedAt = time.Now()
	node.CreatedAt = existing.CreatedAt
	kg.nodes[node.ID] = node
	kg.indexNode(node)
	kg.updatedAt = node.UpdatedAt

	return nil
}

// UpdateEdge updates an existing edge.
func (kg *KnowledgeGraph) UpdateEdge(edge *KGEdge) error {
	if edge == nil {
		return fmt.Errorf("edge cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	existing, exists := kg.edges[edge.ID]
	if !exists {
		return fmt.Errorf("edge with ID %s not found", edge.ID)
	}

	kg.unindexEdge(existing)
	edge.UpdatedAt = time.Now()
	edge.CreatedAt = existing.CreatedAt
	kg.edges[edge.ID] = edge
	kg.indexEdge(edge)
	kg.updatedAt = edge.UpdatedAt

	return nil
}

// Nodes returns all nodes in the graph.
func (kg *KnowledgeGraph) Nodes() []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	nodes := make([]*KGNode, 0, len(kg.nodes))
	for _, node := range kg.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

// Edges returns all edges in the graph.
func (kg *KnowledgeGraph) Edges() []*KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	edges := make([]*KGEdge, 0, len(kg.edges))
	for _, edge := range kg.edges {
		edges = append(edges, edge)
	}
	return edges
}

// NodeCount returns the number of nodes.
func (kg *KnowledgeGraph) NodeCount() int {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return len(kg.nodes)
}

// EdgeCount returns the number of edges.
func (kg *KnowledgeGraph) EdgeCount() int {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return len(kg.edges)
}

// NodeTypes returns all unique node types.
func (kg *KnowledgeGraph) NodeTypes() []string {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	types := make([]string, 0, len(kg.nodesByType))
	for t := range kg.nodesByType {
		types = append(types, t)
	}
	return types
}

// EdgeTypes returns all unique edge types.
func (kg *KnowledgeGraph) EdgeTypes() []string {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	types := make([]string, 0, len(kg.edgesByType))
	for t := range kg.edgesByType {
		types = append(types, t)
	}
	return types
}

// indexNode adds a node to all indexes (must hold lock).
func (kg *KnowledgeGraph) indexNode(node *KGNode) {
	normalizedType := strings.ToLower(strings.TrimSpace(node.Type))
	if kg.nodesByType[normalizedType] == nil {
		kg.nodesByType[normalizedType] = make(map[string]struct{})
	}
	kg.nodesByType[normalizedType][node.ID] = struct{}{}

	normalizedName := strings.ToLower(strings.TrimSpace(node.CanonicalName))
	if kg.nodesByName[normalizedName] == nil {
		kg.nodesByName[normalizedName] = make(map[string]struct{})
	}
	kg.nodesByName[normalizedName][node.ID] = struct{}{}

	for _, mention := range node.Mentions {
		normalizedMention := strings.ToLower(strings.TrimSpace(mention))
		if kg.nodesByName[normalizedMention] == nil {
			kg.nodesByName[normalizedMention] = make(map[string]struct{})
		}
		kg.nodesByName[normalizedMention][node.ID] = struct{}{}
	}
}

// unindexNode removes a node from all indexes (must hold lock).
func (kg *KnowledgeGraph) unindexNode(node *KGNode) {
	normalizedType := strings.ToLower(strings.TrimSpace(node.Type))
	if typeSet, ok := kg.nodesByType[normalizedType]; ok {
		delete(typeSet, node.ID)
		if len(typeSet) == 0 {
			delete(kg.nodesByType, normalizedType)
		}
	}

	normalizedName := strings.ToLower(strings.TrimSpace(node.CanonicalName))
	if nameSet, ok := kg.nodesByName[normalizedName]; ok {
		delete(nameSet, node.ID)
		if len(nameSet) == 0 {
			delete(kg.nodesByName, normalizedName)
		}
	}

	for _, mention := range node.Mentions {
		normalizedMention := strings.ToLower(strings.TrimSpace(mention))
		if mentionSet, ok := kg.nodesByName[normalizedMention]; ok {
			delete(mentionSet, node.ID)
			if len(mentionSet) == 0 {
				delete(kg.nodesByName, normalizedMention)
			}
		}
	}
}

// indexEdge adds an edge to all indexes (must hold lock).
func (kg *KnowledgeGraph) indexEdge(edge *KGEdge) {
	normalizedType := strings.ToLower(strings.TrimSpace(edge.Type))
	if kg.edgesByType[normalizedType] == nil {
		kg.edgesByType[normalizedType] = make(map[string]struct{})
	}
	kg.edgesByType[normalizedType][edge.ID] = struct{}{}

	if kg.outgoingEdges[edge.SourceID] == nil {
		kg.outgoingEdges[edge.SourceID] = make(map[string]struct{})
	}
	kg.outgoingEdges[edge.SourceID][edge.ID] = struct{}{}

	if kg.incomingEdges[edge.TargetID] == nil {
		kg.incomingEdges[edge.TargetID] = make(map[string]struct{})
	}
	kg.incomingEdges[edge.TargetID][edge.ID] = struct{}{}
}

// unindexEdge removes an edge from all indexes (must hold lock).
func (kg *KnowledgeGraph) unindexEdge(edge *KGEdge) {
	normalizedType := strings.ToLower(strings.TrimSpace(edge.Type))
	if typeSet, ok := kg.edgesByType[normalizedType]; ok {
		delete(typeSet, edge.ID)
		if len(typeSet) == 0 {
			delete(kg.edgesByType, normalizedType)
		}
	}

	if outSet, ok := kg.outgoingEdges[edge.SourceID]; ok {
		delete(outSet, edge.ID)
		if len(outSet) == 0 {
			delete(kg.outgoingEdges, edge.SourceID)
		}
	}

	if inSet, ok := kg.incomingEdges[edge.TargetID]; ok {
		delete(inSet, edge.ID)
		if len(inSet) == 0 {
			delete(kg.incomingEdges, edge.TargetID)
		}
	}
}

// ToJSON serializes the knowledge graph to JSON.
func (kg *KnowledgeGraph) ToJSON() ([]byte, error) {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	export := struct {
		Nodes         []*KGNode `json:"nodes"`
		Edges         []*KGEdge `json:"edges"`
		DocumentCount int       `json:"document_count"`
		DocumentIDs   []string  `json:"document_ids,omitempty"`
		CreatedAt     time.Time `json:"created_at"`
		UpdatedAt     time.Time `json:"updated_at"`
	}{
		Nodes:         make([]*KGNode, 0, len(kg.nodes)),
		Edges:         make([]*KGEdge, 0, len(kg.edges)),
		DocumentCount: kg.documentCount,
		DocumentIDs:   kg.documentIDs,
		CreatedAt:     kg.createdAt,
		UpdatedAt:     kg.updatedAt,
	}

	for _, node := range kg.nodes {
		export.Nodes = append(export.Nodes, node)
	}
	for _, edge := range kg.edges {
		export.Edges = append(export.Edges, edge)
	}

	return json.MarshalIndent(export, "", "  ")
}

// =============================================================================
// Entity Resolution
// =============================================================================

// ResolutionStrategy defines how to resolve duplicate entities.
type ResolutionStrategy int

const (
	// ResolutionNone performs no entity resolution
	ResolutionNone ResolutionStrategy = iota

	// ResolutionExact merges only exact text matches (case-insensitive)
	ResolutionExact

	// ResolutionFuzzy uses string similarity matching (Jaro-Winkler)
	ResolutionFuzzy

	// ResolutionVector uses embedding similarity for resolution
	ResolutionVector
)

// String returns the string representation of the resolution strategy.
func (rs ResolutionStrategy) String() string {
	switch rs {
	case ResolutionNone:
		return "none"
	case ResolutionExact:
		return "exact"
	case ResolutionFuzzy:
		return "fuzzy"
	case ResolutionVector:
		return "vector"
	default:
		return "unknown"
	}
}

// =============================================================================
// Rejected Triple for Builder (extends the existing RejectedTriple)
// =============================================================================

// BuilderRejectedTriple represents a relation that was rejected during validation,
// with full entity information for debugging.
type BuilderRejectedTriple struct {
	// HeadEntity is the source entity
	HeadEntity Entity `json:"head_entity"`

	// TailEntity is the target entity
	TailEntity Entity `json:"tail_entity"`

	// RelationType is the relation type
	RelationType string `json:"relation_type"`

	// Reason is why the triple was rejected
	Reason string `json:"reason"`

	// DocumentID is the source document
	DocumentID string `json:"document_id,omitempty"`

	// Timestamp is when the rejection occurred
	Timestamp time.Time `json:"timestamp"`
}

// =============================================================================
// Document Type
// =============================================================================

// Document represents a document to be processed for knowledge graph extraction.
type Document struct {
	// ID is the unique identifier for this document
	ID string

	// URL is the optional source URL of the document
	URL string

	// Text is the document content
	Text string
}

// =============================================================================
// GraphBuilder Configuration
// =============================================================================

// GraphBuilderConfig configures the GraphBuilder behavior.
type GraphBuilderConfig struct {
	// Schema defines allowed entity and relation types (optional)
	Schema *SchemaDefinition

	// Models
	EntityModel   string // GLiNER model name for entity extraction
	RelationModel string // REBEL model name for relation extraction
	EmbedModel    string // Embedding model for vector resolution (optional)

	// Entity Resolution
	ResolutionStrategy  ResolutionStrategy
	SimilarityThreshold float32 // For fuzzy/vector resolution (default: 0.85)

	// Processing
	NumWorkers   int // Worker pool size (default: runtime.NumCPU())
	ChunkSize    int // Target tokens per chunk (default: 512)
	ChunkOverlap int // Token overlap between chunks (default: 100)

	// Filtering
	MinEntityConfidence   float32 // Minimum entity confidence (default: 0.0)
	MinRelationConfidence float32 // Minimum relation confidence (default: 0.0)

	// KG Builder Config
	DeduplicateRelations bool // Remove duplicate relations (default: true)
	TrackProvenance      bool // Track extraction provenance (default: true)
}

// DefaultGraphBuilderConfig returns sensible defaults for the GraphBuilder.
func DefaultGraphBuilderConfig() GraphBuilderConfig {
	return GraphBuilderConfig{
		ResolutionStrategy:    ResolutionFuzzy,
		SimilarityThreshold:   0.85,
		NumWorkers:            runtime.NumCPU(),
		ChunkSize:             512,
		ChunkOverlap:          100,
		MinEntityConfidence:   0.0,
		MinRelationConfidence: 0.0,
		DeduplicateRelations:  true,
		TrackProvenance:       true,
	}
}

// =============================================================================
// GraphBuilder
// =============================================================================

// GraphBuilder orchestrates the full knowledge graph generation pipeline.
// It processes documents through entity and relation extraction, validates
// against an optional schema, resolves duplicate entities, and builds
// a thread-safe knowledge graph.
type GraphBuilder struct {
	config    GraphBuilderConfig
	schema    *SchemaDefinition
	validator *SchemaValidator
	resolver  *VectorEntityResolver // optional, for vector-based resolution

	// Extractors (injected via SetExtractors)
	entityExtractor   EntityExtractor
	relationExtractor RelationExtractor
	chunker           TextChunker

	// Internal state
	graph    *KnowledgeGraph
	rejected []BuilderRejectedTriple

	// Concurrency control
	mu sync.RWMutex
}

// NewGraphBuilder creates a new GraphBuilder with the given configuration.
func NewGraphBuilder(config GraphBuilderConfig) *GraphBuilder {
	// Apply defaults
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 512
	}
	if config.ChunkOverlap < 0 {
		config.ChunkOverlap = 100
	}
	if config.SimilarityThreshold <= 0 {
		config.SimilarityThreshold = 0.85
	}

	// Build validator if schema provided
	var validator *SchemaValidator
	if config.Schema != nil {
		var err error
		validator, err = NewSchemaValidator(config.Schema)
		if err != nil {
			// Log error but continue without validation
			validator = nil
		}
	}

	return &GraphBuilder{
		config:    config,
		schema:    config.Schema,
		validator: validator,
		graph:     NewKnowledgeGraph(),
		rejected:  make([]BuilderRejectedTriple, 0),
	}
}

// SetExtractors sets the entity and relation extractors to use.
// This must be called before processing documents.
func (gb *GraphBuilder) SetExtractors(entity EntityExtractor, relation RelationExtractor) {
	gb.mu.Lock()
	defer gb.mu.Unlock()
	gb.entityExtractor = entity
	gb.relationExtractor = relation
}

// SetChunker sets the text chunker to use.
// If not set, a default simple chunker will be used.
func (gb *GraphBuilder) SetChunker(chunker TextChunker) {
	gb.mu.Lock()
	defer gb.mu.Unlock()
	gb.chunker = chunker
}

// SetVectorResolver sets the vector-based entity resolver.
// This is required when using ResolutionVector strategy.
func (gb *GraphBuilder) SetVectorResolver(resolver *VectorEntityResolver) {
	gb.mu.Lock()
	defer gb.mu.Unlock()
	gb.resolver = resolver
}

// =============================================================================
// Document Processing
// =============================================================================

// documentResult holds the extraction results for a single document.
type documentResult struct {
	DocID     string
	DocURL    string
	Entities  []Entity
	Relations []Relation
	Rejected  []BuilderRejectedTriple
	Error     error
}

// AddDocument processes a single document and adds its entities and relations
// to the knowledge graph.
func (gb *GraphBuilder) AddDocument(ctx context.Context, docID string, text string) error {
	doc := Document{
		ID:   docID,
		Text: text,
	}
	return gb.AddDocuments(ctx, []Document{doc})
}

// AddDocuments processes multiple documents concurrently and adds their
// entities and relations to the knowledge graph.
func (gb *GraphBuilder) AddDocuments(ctx context.Context, docs []Document) error {
	if len(docs) == 0 {
		return nil
	}

	gb.mu.RLock()
	if gb.entityExtractor == nil {
		gb.mu.RUnlock()
		return errors.New("entity extractor not set; call SetExtractors first")
	}
	if gb.relationExtractor == nil {
		gb.mu.RUnlock()
		return errors.New("relation extractor not set; call SetExtractors first")
	}
	gb.mu.RUnlock()

	// Create worker pool
	numWorkers := gb.config.NumWorkers
	if numWorkers > len(docs) {
		numWorkers = len(docs)
	}

	docChan := make(chan Document, len(docs))
	resultChan := make(chan documentResult, len(docs))

	// Context with cancellation
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup

	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			gb.documentWorker(ctx, docChan, resultChan)
		}()
	}

	// Send documents to workers
	go func() {
		for _, doc := range docs {
			select {
			case docChan <- doc:
			case <-ctx.Done():
				return
			}
		}
		close(docChan)
	}()

	// Wait for workers and close result channel
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	var firstErr error
	for result := range resultChan {
		if result.Error != nil {
			if firstErr == nil {
				firstErr = result.Error
			}
			continue
		}

		// Add to knowledge graph (thread-safe)
		if err := gb.addResultToGraph(result); err != nil {
			if firstErr == nil {
				firstErr = err
			}
		}
	}

	return firstErr
}

// documentWorker processes documents from the input channel.
func (gb *GraphBuilder) documentWorker(ctx context.Context, docs <-chan Document, results chan<- documentResult) {
	for doc := range docs {
		select {
		case <-ctx.Done():
			results <- documentResult{
				DocID: doc.ID,
				Error: ctx.Err(),
			}
			return
		default:
		}

		result := gb.processDocument(ctx, doc)
		results <- result
	}
}

// processDocument processes a single document through the extraction pipeline.
func (gb *GraphBuilder) processDocument(ctx context.Context, doc Document) documentResult {
	result := documentResult{
		DocID:     doc.ID,
		DocURL:    doc.URL,
		Entities:  make([]Entity, 0),
		Relations: make([]Relation, 0),
		Rejected:  make([]BuilderRejectedTriple, 0),
	}

	// Step 1: Chunk the text
	chunks := gb.chunkText(doc.Text)

	// Step 2: Extract entities and relations from each chunk
	for _, chunk := range chunks {
		select {
		case <-ctx.Done():
			result.Error = ctx.Err()
			return result
		default:
		}

		// Extract entities
		entities, err := gb.extractEntities(ctx, chunk)
		if err != nil {
			result.Error = fmt.Errorf("entity extraction failed for doc %s: %w", doc.ID, err)
			return result
		}

		// Extract relations
		relations, err := gb.extractRelations(ctx, chunk)
		if err != nil {
			result.Error = fmt.Errorf("relation extraction failed for doc %s: %w", doc.ID, err)
			return result
		}

		// Adjust character offsets based on chunk position
		for i := range entities {
			entities[i].Start += chunk.StartChar
			entities[i].End += chunk.StartChar
		}
		for i := range relations {
			relations[i].HeadEntity.Start += chunk.StartChar
			relations[i].HeadEntity.End += chunk.StartChar
			relations[i].TailEntity.Start += chunk.StartChar
			relations[i].TailEntity.End += chunk.StartChar
		}

		// Step 3: Validate against schema and filter
		validEntities, validRelations, rejected := gb.validateExtractions(entities, relations, doc.ID)

		result.Entities = append(result.Entities, validEntities...)
		result.Relations = append(result.Relations, validRelations...)
		result.Rejected = append(result.Rejected, rejected...)
	}

	return result
}

// chunkText splits text into overlapping chunks for processing.
func (gb *GraphBuilder) chunkText(text string) []TextChunk {
	if gb.chunker != nil {
		return gb.chunker.Chunk(text, gb.config.ChunkSize, gb.config.ChunkOverlap)
	}

	// Default simple chunking by paragraphs or sentence boundaries
	return simpleChunk(text, gb.config.ChunkSize, gb.config.ChunkOverlap)
}

// simpleChunk provides a basic text chunking implementation.
func simpleChunk(text string, targetSize int, overlap int) []TextChunk {
	if text == "" {
		return nil
	}

	// For simplicity, split on double newlines (paragraphs)
	paragraphs := strings.Split(text, "\n\n")
	if len(paragraphs) <= 1 {
		// Try single newlines
		paragraphs = strings.Split(text, "\n")
	}

	var chunks []TextChunk
	currentChunk := strings.Builder{}
	currentStart := 0
	charPos := 0

	for i, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			charPos += 2 // Account for \n\n
			continue
		}

		// Estimate tokens (rough: ~4 chars per token)
		estimatedTokens := (currentChunk.Len() + len(para)) / 4

		if estimatedTokens > targetSize && currentChunk.Len() > 0 {
			// Finalize current chunk
			chunkText := strings.TrimSpace(currentChunk.String())
			if chunkText != "" {
				chunks = append(chunks, TextChunk{
					Text:      chunkText,
					StartChar: currentStart,
					EndChar:   charPos,
				})
			}

			// Start new chunk with overlap
			currentChunk.Reset()
			overlapStart := charPos

			// Add overlap from previous chunk if available
			if overlap > 0 && len(chunks) > 0 {
				lastChunk := chunks[len(chunks)-1]
				overlapChars := overlap * 4 // Rough estimate
				if overlapChars > len(lastChunk.Text) {
					overlapChars = len(lastChunk.Text)
				}
				overlapText := lastChunk.Text[len(lastChunk.Text)-overlapChars:]
				currentChunk.WriteString(overlapText)
				currentChunk.WriteString(" ")
				overlapStart = lastChunk.EndChar - overlapChars
			}
			currentStart = overlapStart
		}

		if currentChunk.Len() > 0 {
			currentChunk.WriteString(" ")
		}
		currentChunk.WriteString(para)

		// Update position
		if i < len(paragraphs)-1 {
			charPos += len(para) + 2 // +2 for \n\n separator
		} else {
			charPos += len(para)
		}
	}

	// Add final chunk
	chunkText := strings.TrimSpace(currentChunk.String())
	if chunkText != "" {
		chunks = append(chunks, TextChunk{
			Text:      chunkText,
			StartChar: currentStart,
			EndChar:   charPos,
		})
	}

	// If no chunks created, return the whole text as one chunk
	if len(chunks) == 0 && text != "" {
		chunks = append(chunks, TextChunk{
			Text:      strings.TrimSpace(text),
			StartChar: 0,
			EndChar:   len(text),
		})
	}

	return chunks
}

// extractEntities calls the entity extractor on a text chunk.
func (gb *GraphBuilder) extractEntities(ctx context.Context, chunk TextChunk) ([]Entity, error) {
	// Get entity labels from schema if available
	var labels []string
	if gb.validator != nil {
		labels = gb.validator.NodeTypes()
	}

	results, err := gb.entityExtractor.Extract(ctx, []string{chunk.Text}, labels)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, nil
	}

	// Filter by confidence
	var filtered []Entity
	for _, entity := range results[0] {
		if entity.Score >= gb.config.MinEntityConfidence {
			filtered = append(filtered, entity)
		}
	}

	return filtered, nil
}

// extractRelations calls the relation extractor on a text chunk.
func (gb *GraphBuilder) extractRelations(ctx context.Context, chunk TextChunk) ([]Relation, error) {
	results, err := gb.relationExtractor.Extract(ctx, []string{chunk.Text})
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, nil
	}

	// Filter by confidence
	var filtered []Relation
	for _, relation := range results[0] {
		if relation.Score >= gb.config.MinRelationConfidence {
			filtered = append(filtered, relation)
		}
	}

	return filtered, nil
}

// validateExtractions validates entities and relations against the schema.
func (gb *GraphBuilder) validateExtractions(
	entities []Entity,
	relations []Relation,
	docID string,
) ([]Entity, []Relation, []BuilderRejectedTriple) {
	validEntities := make([]Entity, 0, len(entities))
	validRelations := make([]Relation, 0, len(relations))
	rejected := make([]BuilderRejectedTriple, 0)

	// Validate entities
	for _, entity := range entities {
		if gb.validator != nil {
			if !gb.validator.IsValidNodeType(entity.Label) {
				// Entity type not valid, skip it
				continue
			}
		}
		validEntities = append(validEntities, entity)
	}

	// Validate relations
	for _, relation := range relations {
		if gb.validator != nil {
			rejectedTriple := gb.validator.ValidateTriple(
				relation.HeadEntity.Label,
				relation.HeadEntity.Text,
				relation.Label,
				relation.TailEntity.Label,
				relation.TailEntity.Text,
			)
			if rejectedTriple != nil {
				rejected = append(rejected, BuilderRejectedTriple{
					HeadEntity:   relation.HeadEntity,
					TailEntity:   relation.TailEntity,
					RelationType: relation.Label,
					Reason:       rejectedTriple.Reason,
					DocumentID:   docID,
					Timestamp:    time.Now(),
				})
				continue
			}
		}
		validRelations = append(validRelations, relation)
	}

	return validEntities, validRelations, rejected
}

// addResultToGraph adds extraction results to the knowledge graph.
func (gb *GraphBuilder) addResultToGraph(result documentResult) error {
	gb.mu.Lock()
	defer gb.mu.Unlock()

	// Add rejected triples
	gb.rejected = append(gb.rejected, result.Rejected...)

	// Map from entity key to node ID for relation linking
	entityToNode := make(map[string]string)

	// Add entities as nodes
	for _, entity := range result.Entities {
		nodeID, err := gb.addOrResolveEntity(entity, result.DocID, result.DocURL)
		if err != nil {
			return err
		}
		entityToNode[entityKey(entity)] = nodeID
	}

	// Add relations as edges
	for _, relation := range result.Relations {
		// Get or create nodes for head and tail entities
		headNodeID, ok := entityToNode[entityKey(relation.HeadEntity)]
		if !ok {
			var err error
			headNodeID, err = gb.addOrResolveEntity(relation.HeadEntity, result.DocID, result.DocURL)
			if err != nil {
				return err
			}
			entityToNode[entityKey(relation.HeadEntity)] = headNodeID
		}

		tailNodeID, ok := entityToNode[entityKey(relation.TailEntity)]
		if !ok {
			var err error
			tailNodeID, err = gb.addOrResolveEntity(relation.TailEntity, result.DocID, result.DocURL)
			if err != nil {
				return err
			}
			entityToNode[entityKey(relation.TailEntity)] = tailNodeID
		}

		// Add the relation as an edge
		if err := gb.addRelation(relation, headNodeID, tailNodeID, result.DocID, result.DocURL); err != nil {
			return err
		}
	}

	// Update document tracking
	gb.graph.documentCount++
	if result.DocID != "" {
		gb.graph.documentIDs = append(gb.graph.documentIDs, result.DocID)
	}

	return nil
}

// addOrResolveEntity adds an entity or resolves it to an existing node.
func (gb *GraphBuilder) addOrResolveEntity(entity Entity, docID, docURL string) (string, error) {
	// Try to find an existing matching node
	existingNode := gb.findMatchingNode(entity)

	provenance := Provenance{
		SourceDocument:      docID,
		SourceURL:           docURL,
		SourceText:          entity.Text,
		CharOffsetStart:     entity.Start,
		CharOffsetEnd:       entity.End,
		ExtractorModel:      gb.config.EntityModel,
		ExtractionTimestamp: time.Now(),
		ExtractorConfidence: entity.Score,
	}

	if existingNode != nil {
		// Merge with existing node
		return gb.mergeEntityIntoNode(entity, existingNode, provenance)
	}

	// Create new node
	node := &KGNode{
		ID:            generateID(),
		CanonicalName: entity.Text,
		Type:          entity.Label,
		Mentions:      []string{entity.Text},
		Confidence:    entity.Score,
	}

	if gb.config.TrackProvenance {
		node.Provenance = []Provenance{provenance}
	}

	if err := gb.graph.AddNode(node); err != nil {
		return "", err
	}

	return node.ID, nil
}

// findMatchingNode finds an existing node that matches the entity.
func (gb *GraphBuilder) findMatchingNode(entity Entity) *KGNode {
	if gb.config.ResolutionStrategy == ResolutionNone {
		return nil
	}

	// Find candidates by mention
	candidates := gb.graph.FindNodesByMention(entity.Text)

	for _, node := range candidates {
		// Check type match
		if strings.ToLower(node.Type) != strings.ToLower(entity.Label) {
			continue
		}

		switch gb.config.ResolutionStrategy {
		case ResolutionExact:
			// Exact case-insensitive match
			if strings.EqualFold(node.CanonicalName, entity.Text) {
				return node
			}
			for _, mention := range node.Mentions {
				if strings.EqualFold(mention, entity.Text) {
					return node
				}
			}

		case ResolutionFuzzy:
			// Use Jaro-Winkler similarity
			similarity := jaroWinklerSimilarity(entity.Text, node.CanonicalName)
			if similarity >= gb.config.SimilarityThreshold {
				return node
			}
			for _, mention := range node.Mentions {
				similarity = jaroWinklerSimilarity(entity.Text, mention)
				if similarity >= gb.config.SimilarityThreshold {
					return node
				}
			}

		case ResolutionVector:
			// Vector resolution is handled in Build()
			// For now, fall back to exact matching
			if strings.EqualFold(node.CanonicalName, entity.Text) {
				return node
			}
		}
	}

	return nil
}

// mergeEntityIntoNode merges an entity into an existing node.
func (gb *GraphBuilder) mergeEntityIntoNode(entity Entity, node *KGNode, provenance Provenance) (string, error) {
	// Check if mention already exists
	mentionExists := false
	normalizedMention := strings.ToLower(strings.TrimSpace(entity.Text))
	for _, m := range node.Mentions {
		if strings.ToLower(strings.TrimSpace(m)) == normalizedMention {
			mentionExists = true
			break
		}
	}

	if !mentionExists {
		node.Mentions = append(node.Mentions, entity.Text)
	}

	// Update confidence (use max)
	if entity.Score > node.Confidence {
		node.Confidence = entity.Score
	}

	// Add provenance
	if gb.config.TrackProvenance {
		node.Provenance = append(node.Provenance, provenance)
	}

	return node.ID, gb.graph.UpdateNode(node)
}

// addRelation adds a relation as an edge between two nodes.
func (gb *GraphBuilder) addRelation(relation Relation, sourceID, targetID, docID, docURL string) error {
	// Check for existing duplicate relation
	if gb.config.DeduplicateRelations {
		existing := gb.graph.GetEdgeBetween(sourceID, targetID, relation.Label)
		if existing != nil {
			// Update existing edge with new provenance
			if gb.config.TrackProvenance {
				prov := Provenance{
					SourceDocument:      docID,
					SourceURL:           docURL,
					ExtractorModel:      gb.config.RelationModel,
					ExtractionTimestamp: time.Now(),
					ExtractorConfidence: relation.Score,
				}
				existing.Provenance = append(existing.Provenance, prov)
			}

			// Update confidence
			if relation.Score > existing.Confidence {
				existing.Confidence = relation.Score
			}

			return gb.graph.UpdateEdge(existing)
		}
	}

	// Create new edge
	edge := &KGEdge{
		ID:         generateID(),
		SourceID:   sourceID,
		TargetID:   targetID,
		Type:       relation.Label,
		Confidence: relation.Score,
	}

	if gb.config.TrackProvenance {
		edge.Provenance = []Provenance{{
			SourceDocument:      docID,
			SourceURL:           docURL,
			ExtractorModel:      gb.config.RelationModel,
			ExtractionTimestamp: time.Now(),
			ExtractorConfidence: relation.Score,
		}}
	}

	return gb.graph.AddEdge(edge)
}

// entityKey creates a unique key for an entity based on text, label, and position.
func entityKey(entity Entity) string {
	return fmt.Sprintf("%s|%s|%d|%d", entity.Text, entity.Label, entity.Start, entity.End)
}

// generateID generates a unique ID for nodes and edges.
func generateID() string {
	// Simple UUID-like generation
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), time.Now().UnixNano()%1000000)
}

// =============================================================================
// Graph Finalization and Export
// =============================================================================

// Build finalizes the knowledge graph and returns it.
// This performs any final entity resolution and cleanup.
func (gb *GraphBuilder) Build() (*KnowledgeGraph, error) {
	gb.mu.Lock()
	defer gb.mu.Unlock()

	// Perform vector-based resolution if configured
	if gb.config.ResolutionStrategy == ResolutionVector && gb.resolver != nil {
		if err := gb.resolveEntitiesVector(); err != nil {
			return nil, fmt.Errorf("vector entity resolution failed: %w", err)
		}
	}

	return gb.graph, nil
}

// resolveEntitiesVector performs vector-based entity resolution.
func (gb *GraphBuilder) resolveEntitiesVector() error {
	if gb.resolver == nil {
		return nil
	}

	// TODO: Implement full vector resolution
	// This would:
	// 1. Get all unique entity mentions
	// 2. Compute embeddings for each
	// 3. Find clusters of similar entities
	// 4. Merge nodes within each cluster

	return nil
}

// GetRejectedTriples returns all triples that were rejected during validation.
func (gb *GraphBuilder) GetRejectedTriples() []BuilderRejectedTriple {
	gb.mu.RLock()
	defer gb.mu.RUnlock()

	// Return a copy to avoid race conditions
	result := make([]BuilderRejectedTriple, len(gb.rejected))
	copy(result, gb.rejected)
	return result
}

// Export exports the knowledge graph to the specified format.
// Supported formats: "json", "jsonld", "ntriples", "cypher"
func (gb *GraphBuilder) Export(format string) ([]byte, error) {
	gb.mu.RLock()
	defer gb.mu.RUnlock()

	switch strings.ToLower(format) {
	case "json":
		return gb.graph.ToJSON()

	case "jsonld":
		return gb.exportJSONLD()

	case "ntriples", "nt":
		return gb.exportNTriples()

	case "cypher":
		return gb.exportCypher()

	default:
		return nil, fmt.Errorf("unsupported export format: %s", format)
	}
}

// exportJSONLD exports the graph in JSON-LD format.
func (gb *GraphBuilder) exportJSONLD() ([]byte, error) {
	nodes := gb.graph.Nodes()
	edges := gb.graph.Edges()

	type jsonLDNode struct {
		ID         string   `json:"@id"`
		Type       string   `json:"@type"`
		Name       string   `json:"schema:name"`
		Mentions   []string `json:"schema:alternateName,omitempty"`
		Confidence float32  `json:"schema:confidence,omitempty"`
	}

	type jsonLDEdge struct {
		Type       string  `json:"@type"`
		Source     string  `json:"schema:source"`
		Target     string  `json:"schema:target"`
		Confidence float32 `json:"schema:confidence,omitempty"`
	}

	type jsonLDGraph struct {
		Context map[string]string `json:"@context"`
		Graph   []interface{}     `json:"@graph"`
	}

	graph := jsonLDGraph{
		Context: map[string]string{
			"schema": "https://schema.org/",
		},
		Graph: make([]interface{}, 0, len(nodes)+len(edges)),
	}

	for _, node := range nodes {
		jNode := jsonLDNode{
			ID:         node.ID,
			Type:       node.Type,
			Name:       node.CanonicalName,
			Mentions:   node.Mentions,
			Confidence: node.Confidence,
		}
		graph.Graph = append(graph.Graph, jNode)
	}

	for _, edge := range edges {
		jEdge := jsonLDEdge{
			Type:       edge.Type,
			Source:     edge.SourceID,
			Target:     edge.TargetID,
			Confidence: edge.Confidence,
		}
		graph.Graph = append(graph.Graph, jEdge)
	}

	return json.MarshalIndent(graph, "", "  ")
}

// exportNTriples exports the graph in N-Triples format.
func (gb *GraphBuilder) exportNTriples() ([]byte, error) {
	nodes := gb.graph.Nodes()
	edges := gb.graph.Edges()

	var sb strings.Builder

	for _, node := range nodes {
		fmt.Fprintf(&sb, "<%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <%s> .\n",
			node.ID, node.Type)
		fmt.Fprintf(&sb, "<%s> <http://www.w3.org/2000/01/rdf-schema#label> \"%s\" .\n",
			node.ID, escapeNTriples(node.CanonicalName))
	}

	for _, edge := range edges {
		fmt.Fprintf(&sb, "<%s> <%s> <%s> .\n",
			edge.SourceID, edge.Type, edge.TargetID)
	}

	return []byte(sb.String()), nil
}

// escapeNTriples escapes a string for N-Triples format.
func escapeNTriples(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return s
}

// exportCypher exports the graph as Cypher statements for Neo4j.
func (gb *GraphBuilder) exportCypher() ([]byte, error) {
	nodes := gb.graph.Nodes()
	edges := gb.graph.Edges()

	var sb strings.Builder

	for _, node := range nodes {
		fmt.Fprintf(&sb, "CREATE (n%s:%s {id: '%s', name: '%s', confidence: %.4f});\n",
			strings.ReplaceAll(node.ID, "-", ""),
			escapeCypher(node.Type),
			node.ID,
			escapeCypher(node.CanonicalName),
			node.Confidence)
	}

	sb.WriteString("\n")

	for _, edge := range edges {
		fmt.Fprintf(&sb, "MATCH (a {id: '%s'}), (b {id: '%s'}) CREATE (a)-[:%s {confidence: %.4f}]->(b);\n",
			edge.SourceID,
			edge.TargetID,
			strings.ToUpper(strings.ReplaceAll(escapeCypher(edge.Type), " ", "_")),
			edge.Confidence)
	}

	return []byte(sb.String()), nil
}

// escapeCypher escapes a string for Cypher.
func escapeCypher(s string) string {
	s = strings.ReplaceAll(s, "'", "\\'")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	return s
}

// =============================================================================
// Statistics and Inspection
// =============================================================================

// Stats returns statistics about the current graph state.
func (gb *GraphBuilder) Stats() GraphStats {
	gb.mu.RLock()
	defer gb.mu.RUnlock()

	return GraphStats{
		NodeCount:     gb.graph.NodeCount(),
		EdgeCount:     gb.graph.EdgeCount(),
		NodeTypes:     gb.graph.NodeTypes(),
		EdgeTypes:     gb.graph.EdgeTypes(),
		RejectedCount: len(gb.rejected),
	}
}

// GraphStats contains statistics about the knowledge graph.
type GraphStats struct {
	NodeCount     int      `json:"node_count"`
	EdgeCount     int      `json:"edge_count"`
	NodeTypes     []string `json:"node_types"`
	EdgeTypes     []string `json:"edge_types"`
	RejectedCount int      `json:"rejected_count"`
}

// Graph returns a read-only view of the current graph.
// Note: Modifications to the returned graph are not thread-safe.
func (gb *GraphBuilder) Graph() *KnowledgeGraph {
	gb.mu.RLock()
	defer gb.mu.RUnlock()
	return gb.graph
}

// Reset clears the graph and rejected triples, allowing reuse of the builder.
func (gb *GraphBuilder) Reset() {
	gb.mu.Lock()
	defer gb.mu.Unlock()

	gb.graph = NewKnowledgeGraph()
	gb.rejected = make([]BuilderRejectedTriple, 0)
}

// =============================================================================
// String Similarity (Jaro-Winkler)
// =============================================================================

// jaroWinklerSimilarity calculates Jaro-Winkler similarity between two strings.
func jaroWinklerSimilarity(s1, s2 string) float32 {
	s1 = strings.ToLower(s1)
	s2 = strings.ToLower(s2)

	if s1 == s2 {
		return 1.0
	}

	if len(s1) == 0 || len(s2) == 0 {
		return 0.0
	}

	jaroSim := jaroSimilarity(s1, s2)

	// Calculate common prefix length (up to 4 characters)
	prefixLen := 0
	maxPrefix := 4
	if len(s1) < maxPrefix {
		maxPrefix = len(s1)
	}
	if len(s2) < maxPrefix {
		maxPrefix = len(s2)
	}
	for i := 0; i < maxPrefix; i++ {
		if s1[i] == s2[i] {
			prefixLen++
		} else {
			break
		}
	}

	return float32(jaroSim + float64(prefixLen)*0.1*(1.0-jaroSim))
}

// jaroSimilarity calculates Jaro similarity between two strings.
func jaroSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	len1, len2 := len(s1), len(s2)
	if len1 == 0 || len2 == 0 {
		return 0.0
	}

	matchWindow := len1
	if len2 > len1 {
		matchWindow = len2
	}
	matchWindow = matchWindow/2 - 1
	if matchWindow < 0 {
		matchWindow = 0
	}

	s1Matches := make([]bool, len1)
	s2Matches := make([]bool, len2)

	matches := 0
	transpositions := 0

	for i := 0; i < len1; i++ {
		start := i - matchWindow
		if start < 0 {
			start = 0
		}
		end := i + matchWindow + 1
		if end > len2 {
			end = len2
		}

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
