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
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// =============================================================================
// Core Knowledge Graph Types
// =============================================================================

// KnowledgeGraph represents a property graph for storing entities and relationships
// extracted from text. It uses an adjacency list representation for efficient
// traversal and supports incremental updates with entity resolution.
type KnowledgeGraph struct {
	// Nodes (entities) indexed by ID
	nodes map[string]*KGNode

	// Edges (relationships) indexed by ID
	edges map[string]*KGEdge

	// Indexes for fast lookup
	nodesByType    map[string]map[string]struct{} // type -> set of node IDs
	nodesByName    map[string]map[string]struct{} // canonical name -> set of node IDs
	edgesByType    map[string]map[string]struct{} // type -> set of edge IDs
	outgoingEdges  map[string]map[string]struct{} // source node ID -> set of edge IDs
	incomingEdges  map[string]map[string]struct{} // target node ID -> set of edge IDs
	nodesByMention map[string]map[string]struct{} // mention text -> set of node IDs

	// Metadata
	metadata KGMetadata

	// Concurrency control
	mu sync.RWMutex
}

// KGNode represents an entity node in the knowledge graph
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

// KGEdge represents a relationship edge between two nodes
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

// Provenance records the origin of extracted information
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

	// ExtractorModel is the model used for extraction (e.g., "gliner_small")
	ExtractorModel string `json:"extractor_model,omitempty"`

	// ExtractionTimestamp is when the extraction occurred
	ExtractionTimestamp time.Time `json:"extraction_timestamp"`

	// ExtractorConfidence is the raw confidence from the extraction model
	ExtractorConfidence float32 `json:"extractor_confidence"`
}

// KGMetadata contains graph-level metadata
type KGMetadata struct {
	// Name is an optional name for the knowledge graph
	Name string `json:"name,omitempty"`

	// Description is an optional description
	Description string `json:"description,omitempty"`

	// CreatedAt is when the graph was created
	CreatedAt time.Time `json:"created_at"`

	// UpdatedAt is when the graph was last modified
	UpdatedAt time.Time `json:"updated_at"`

	// NodeCount is the total number of nodes
	NodeCount int `json:"node_count"`

	// EdgeCount is the total number of edges
	EdgeCount int `json:"edge_count"`

	// DocumentCount is the number of documents processed
	DocumentCount int `json:"document_count"`

	// DocumentIDs are the IDs of processed documents
	DocumentIDs []string `json:"document_ids,omitempty"`
}

// =============================================================================
// Knowledge Graph Construction
// =============================================================================

// NewKnowledgeGraph creates a new empty knowledge graph
func NewKnowledgeGraph() *KnowledgeGraph {
	now := time.Now()
	return &KnowledgeGraph{
		nodes:          make(map[string]*KGNode),
		edges:          make(map[string]*KGEdge),
		nodesByType:    make(map[string]map[string]struct{}),
		nodesByName:    make(map[string]map[string]struct{}),
		edgesByType:    make(map[string]map[string]struct{}),
		outgoingEdges:  make(map[string]map[string]struct{}),
		incomingEdges:  make(map[string]map[string]struct{}),
		nodesByMention: make(map[string]map[string]struct{}),
		metadata: KGMetadata{
			CreatedAt: now,
			UpdatedAt: now,
		},
	}
}

// NewKnowledgeGraphWithName creates a new knowledge graph with a name
func NewKnowledgeGraphWithName(name, description string) *KnowledgeGraph {
	kg := NewKnowledgeGraph()
	kg.metadata.Name = name
	kg.metadata.Description = description
	return kg
}

// =============================================================================
// Node Operations
// =============================================================================

// AddNode adds a new node to the graph
func (kg *KnowledgeGraph) AddNode(node *KGNode) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	// Generate ID if not provided
	if node.ID == "" {
		node.ID = uuid.New().String()
	}

	// Check for duplicate
	if _, exists := kg.nodes[node.ID]; exists {
		return fmt.Errorf("node with ID %s already exists", node.ID)
	}

	// Set timestamps
	now := time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = now
	}
	node.UpdatedAt = now

	// Store node
	kg.nodes[node.ID] = node

	// Update indexes
	kg.indexNode(node)

	// Update metadata
	kg.metadata.NodeCount++
	kg.metadata.UpdatedAt = now

	return nil
}

// GetNode retrieves a node by ID
func (kg *KnowledgeGraph) GetNode(id string) *KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return kg.nodes[id]
}

// GetNodesByType retrieves all nodes of a given type
func (kg *KnowledgeGraph) GetNodesByType(nodeType string) []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.nodesByType[normalizeType(nodeType)]
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

// GetNodesByName retrieves nodes by canonical name (case-insensitive)
func (kg *KnowledgeGraph) GetNodesByName(name string) []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.nodesByName[normalizeName(name)]
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

// FindNodesByMention finds nodes that have a given mention
func (kg *KnowledgeGraph) FindNodesByMention(mention string) []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.nodesByMention[normalizeName(mention)]
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

// UpdateNode updates an existing node
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

	// Remove old indexes
	kg.unindexNode(existing)

	// Update timestamp
	node.UpdatedAt = time.Now()
	node.CreatedAt = existing.CreatedAt // Preserve creation time

	// Store updated node
	kg.nodes[node.ID] = node

	// Add new indexes
	kg.indexNode(node)

	kg.metadata.UpdatedAt = node.UpdatedAt

	return nil
}

// RemoveNode removes a node and all its connected edges
func (kg *KnowledgeGraph) RemoveNode(id string) error {
	kg.mu.Lock()
	defer kg.mu.Unlock()

	node, exists := kg.nodes[id]
	if !exists {
		return fmt.Errorf("node with ID %s not found", id)
	}

	// Remove all connected edges
	if outgoing, ok := kg.outgoingEdges[id]; ok {
		for edgeID := range outgoing {
			kg.removeEdgeUnsafe(edgeID)
		}
	}
	if incoming, ok := kg.incomingEdges[id]; ok {
		for edgeID := range incoming {
			kg.removeEdgeUnsafe(edgeID)
		}
	}

	// Remove from indexes
	kg.unindexNode(node)

	// Remove node
	delete(kg.nodes, id)
	delete(kg.outgoingEdges, id)
	delete(kg.incomingEdges, id)

	// Update metadata
	kg.metadata.NodeCount--
	kg.metadata.UpdatedAt = time.Now()

	return nil
}

// indexNode adds a node to all indexes (must hold lock)
func (kg *KnowledgeGraph) indexNode(node *KGNode) {
	// Index by type
	normalizedType := normalizeType(node.Type)
	if kg.nodesByType[normalizedType] == nil {
		kg.nodesByType[normalizedType] = make(map[string]struct{})
	}
	kg.nodesByType[normalizedType][node.ID] = struct{}{}

	// Index by canonical name
	normalizedName := normalizeName(node.CanonicalName)
	if kg.nodesByName[normalizedName] == nil {
		kg.nodesByName[normalizedName] = make(map[string]struct{})
	}
	kg.nodesByName[normalizedName][node.ID] = struct{}{}

	// Index by mentions
	for _, mention := range node.Mentions {
		normalizedMention := normalizeName(mention)
		if kg.nodesByMention[normalizedMention] == nil {
			kg.nodesByMention[normalizedMention] = make(map[string]struct{})
		}
		kg.nodesByMention[normalizedMention][node.ID] = struct{}{}
	}
}

// unindexNode removes a node from all indexes (must hold lock)
func (kg *KnowledgeGraph) unindexNode(node *KGNode) {
	// Remove from type index
	normalizedType := normalizeType(node.Type)
	if typeSet, ok := kg.nodesByType[normalizedType]; ok {
		delete(typeSet, node.ID)
		if len(typeSet) == 0 {
			delete(kg.nodesByType, normalizedType)
		}
	}

	// Remove from name index
	normalizedName := normalizeName(node.CanonicalName)
	if nameSet, ok := kg.nodesByName[normalizedName]; ok {
		delete(nameSet, node.ID)
		if len(nameSet) == 0 {
			delete(kg.nodesByName, normalizedName)
		}
	}

	// Remove from mention indexes
	for _, mention := range node.Mentions {
		normalizedMention := normalizeName(mention)
		if mentionSet, ok := kg.nodesByMention[normalizedMention]; ok {
			delete(mentionSet, node.ID)
			if len(mentionSet) == 0 {
				delete(kg.nodesByMention, normalizedMention)
			}
		}
	}
}

// =============================================================================
// Edge Operations
// =============================================================================

// AddEdge adds a new edge to the graph
func (kg *KnowledgeGraph) AddEdge(edge *KGEdge) error {
	if edge == nil {
		return fmt.Errorf("edge cannot be nil")
	}

	kg.mu.Lock()
	defer kg.mu.Unlock()

	// Validate source and target exist
	if _, exists := kg.nodes[edge.SourceID]; !exists {
		return fmt.Errorf("source node %s not found", edge.SourceID)
	}
	if _, exists := kg.nodes[edge.TargetID]; !exists {
		return fmt.Errorf("target node %s not found", edge.TargetID)
	}

	// Generate ID if not provided
	if edge.ID == "" {
		edge.ID = uuid.New().String()
	}

	// Check for duplicate
	if _, exists := kg.edges[edge.ID]; exists {
		return fmt.Errorf("edge with ID %s already exists", edge.ID)
	}

	// Set timestamps
	now := time.Now()
	if edge.CreatedAt.IsZero() {
		edge.CreatedAt = now
	}
	edge.UpdatedAt = now

	// Store edge
	kg.edges[edge.ID] = edge

	// Update indexes
	kg.indexEdge(edge)

	// Update metadata
	kg.metadata.EdgeCount++
	kg.metadata.UpdatedAt = now

	return nil
}

// GetEdge retrieves an edge by ID
func (kg *KnowledgeGraph) GetEdge(id string) *KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return kg.edges[id]
}

// GetEdgesByType retrieves all edges of a given type
func (kg *KnowledgeGraph) GetEdgesByType(edgeType string) []*KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.edgesByType[normalizeType(edgeType)]
	if !exists {
		return nil
	}

	edges := make([]*KGEdge, 0, len(ids))
	for id := range ids {
		if edge, ok := kg.edges[id]; ok {
			edges = append(edges, edge)
		}
	}
	return edges
}

// GetOutgoingEdges retrieves all edges originating from a node
func (kg *KnowledgeGraph) GetOutgoingEdges(nodeID string) []*KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.outgoingEdges[nodeID]
	if !exists {
		return nil
	}

	edges := make([]*KGEdge, 0, len(ids))
	for id := range ids {
		if edge, ok := kg.edges[id]; ok {
			edges = append(edges, edge)
		}
	}
	return edges
}

// GetIncomingEdges retrieves all edges pointing to a node
func (kg *KnowledgeGraph) GetIncomingEdges(nodeID string) []*KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.incomingEdges[nodeID]
	if !exists {
		return nil
	}

	edges := make([]*KGEdge, 0, len(ids))
	for id := range ids {
		if edge, ok := kg.edges[id]; ok {
			edges = append(edges, edge)
		}
	}
	return edges
}

// GetEdgeBetween finds an edge between two nodes with a specific type
func (kg *KnowledgeGraph) GetEdgeBetween(sourceID, targetID, edgeType string) *KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	ids, exists := kg.outgoingEdges[sourceID]
	if !exists {
		return nil
	}

	normalizedType := normalizeType(edgeType)
	for id := range ids {
		if edge, ok := kg.edges[id]; ok {
			if edge.TargetID == targetID && normalizeType(edge.Type) == normalizedType {
				return edge
			}
		}
	}
	return nil
}

// RemoveEdge removes an edge from the graph
func (kg *KnowledgeGraph) RemoveEdge(id string) error {
	kg.mu.Lock()
	defer kg.mu.Unlock()
	return kg.removeEdgeUnsafe(id)
}

// UpdateEdge updates an existing edge
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

	// Remove old indexes
	kg.unindexEdge(existing)

	// Update timestamp
	edge.UpdatedAt = time.Now()
	edge.CreatedAt = existing.CreatedAt // Preserve creation time

	// Store updated edge
	kg.edges[edge.ID] = edge

	// Add new indexes
	kg.indexEdge(edge)

	kg.metadata.UpdatedAt = edge.UpdatedAt

	return nil
}

// removeEdgeUnsafe removes an edge without locking (must hold lock)
func (kg *KnowledgeGraph) removeEdgeUnsafe(id string) error {
	edge, exists := kg.edges[id]
	if !exists {
		return fmt.Errorf("edge with ID %s not found", id)
	}

	// Remove from indexes
	kg.unindexEdge(edge)

	// Remove edge
	delete(kg.edges, id)

	// Update metadata
	kg.metadata.EdgeCount--
	kg.metadata.UpdatedAt = time.Now()

	return nil
}

// indexEdge adds an edge to all indexes (must hold lock)
func (kg *KnowledgeGraph) indexEdge(edge *KGEdge) {
	// Index by type
	normalizedType := normalizeType(edge.Type)
	if kg.edgesByType[normalizedType] == nil {
		kg.edgesByType[normalizedType] = make(map[string]struct{})
	}
	kg.edgesByType[normalizedType][edge.ID] = struct{}{}

	// Index by source (outgoing)
	if kg.outgoingEdges[edge.SourceID] == nil {
		kg.outgoingEdges[edge.SourceID] = make(map[string]struct{})
	}
	kg.outgoingEdges[edge.SourceID][edge.ID] = struct{}{}

	// Index by target (incoming)
	if kg.incomingEdges[edge.TargetID] == nil {
		kg.incomingEdges[edge.TargetID] = make(map[string]struct{})
	}
	kg.incomingEdges[edge.TargetID][edge.ID] = struct{}{}
}

// unindexEdge removes an edge from all indexes (must hold lock)
func (kg *KnowledgeGraph) unindexEdge(edge *KGEdge) {
	// Remove from type index
	normalizedType := normalizeType(edge.Type)
	if typeSet, ok := kg.edgesByType[normalizedType]; ok {
		delete(typeSet, edge.ID)
		if len(typeSet) == 0 {
			delete(kg.edgesByType, normalizedType)
		}
	}

	// Remove from outgoing index
	if outSet, ok := kg.outgoingEdges[edge.SourceID]; ok {
		delete(outSet, edge.ID)
		if len(outSet) == 0 {
			delete(kg.outgoingEdges, edge.SourceID)
		}
	}

	// Remove from incoming index
	if inSet, ok := kg.incomingEdges[edge.TargetID]; ok {
		delete(inSet, edge.ID)
		if len(inSet) == 0 {
			delete(kg.incomingEdges, edge.TargetID)
		}
	}
}

// =============================================================================
// Graph Queries
// =============================================================================

// Nodes returns all nodes in the graph
func (kg *KnowledgeGraph) Nodes() []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	nodes := make([]*KGNode, 0, len(kg.nodes))
	for _, node := range kg.nodes {
		nodes = append(nodes, node)
	}
	return nodes
}

// Edges returns all edges in the graph
func (kg *KnowledgeGraph) Edges() []*KGEdge {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	edges := make([]*KGEdge, 0, len(kg.edges))
	for _, edge := range kg.edges {
		edges = append(edges, edge)
	}
	return edges
}

// NodeCount returns the number of nodes
func (kg *KnowledgeGraph) NodeCount() int {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return len(kg.nodes)
}

// EdgeCount returns the number of edges
func (kg *KnowledgeGraph) EdgeCount() int {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return len(kg.edges)
}

// Metadata returns graph metadata
func (kg *KnowledgeGraph) Metadata() KGMetadata {
	kg.mu.RLock()
	defer kg.mu.RUnlock()
	return kg.metadata
}

// NodeTypes returns all unique node types in the graph
func (kg *KnowledgeGraph) NodeTypes() []string {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	types := make([]string, 0, len(kg.nodesByType))
	for t := range kg.nodesByType {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}

// EdgeTypes returns all unique edge types in the graph
func (kg *KnowledgeGraph) EdgeTypes() []string {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	types := make([]string, 0, len(kg.edgesByType))
	for t := range kg.edgesByType {
		types = append(types, t)
	}
	sort.Strings(types)
	return types
}

// GetNeighbors returns all nodes connected to a given node (both directions)
func (kg *KnowledgeGraph) GetNeighbors(nodeID string) []*KGNode {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	neighborIDs := make(map[string]struct{})

	// Add outgoing neighbors
	if outgoing, ok := kg.outgoingEdges[nodeID]; ok {
		for edgeID := range outgoing {
			if edge, ok := kg.edges[edgeID]; ok {
				neighborIDs[edge.TargetID] = struct{}{}
			}
		}
	}

	// Add incoming neighbors
	if incoming, ok := kg.incomingEdges[nodeID]; ok {
		for edgeID := range incoming {
			if edge, ok := kg.edges[edgeID]; ok {
				neighborIDs[edge.SourceID] = struct{}{}
			}
		}
	}

	neighbors := make([]*KGNode, 0, len(neighborIDs))
	for id := range neighborIDs {
		if node, ok := kg.nodes[id]; ok {
			neighbors = append(neighbors, node)
		}
	}
	return neighbors
}

// =============================================================================
// Serialization
// =============================================================================

// KGExport is the serializable representation of a knowledge graph
type KGExport struct {
	Metadata KGMetadata `json:"metadata"`
	Nodes    []*KGNode  `json:"nodes"`
	Edges    []*KGEdge  `json:"edges"`
}

// ToJSON serializes the knowledge graph to JSON
func (kg *KnowledgeGraph) ToJSON() ([]byte, error) {
	kg.mu.RLock()
	defer kg.mu.RUnlock()

	export := KGExport{
		Metadata: kg.metadata,
		Nodes:    make([]*KGNode, 0, len(kg.nodes)),
		Edges:    make([]*KGEdge, 0, len(kg.edges)),
	}

	for _, node := range kg.nodes {
		export.Nodes = append(export.Nodes, node)
	}
	for _, edge := range kg.edges {
		export.Edges = append(export.Edges, edge)
	}

	// Sort for consistent output
	sort.Slice(export.Nodes, func(i, j int) bool {
		return export.Nodes[i].ID < export.Nodes[j].ID
	})
	sort.Slice(export.Edges, func(i, j int) bool {
		return export.Edges[i].ID < export.Edges[j].ID
	})

	return json.MarshalIndent(export, "", "  ")
}

// FromJSON deserializes a knowledge graph from JSON
func FromJSON(data []byte) (*KnowledgeGraph, error) {
	var export KGExport
	if err := json.Unmarshal(data, &export); err != nil {
		return nil, fmt.Errorf("unmarshaling JSON: %w", err)
	}

	kg := NewKnowledgeGraph()

	// Add nodes first (AddNode will update NodeCount)
	for _, node := range export.Nodes {
		if err := kg.AddNode(node); err != nil {
			return nil, fmt.Errorf("adding node %s: %w", node.ID, err)
		}
	}

	// Then add edges (AddEdge will update EdgeCount)
	for _, edge := range export.Edges {
		if err := kg.AddEdge(edge); err != nil {
			return nil, fmt.Errorf("adding edge %s: %w", edge.ID, err)
		}
	}

	// Restore metadata fields that aren't calculated by AddNode/AddEdge
	// NodeCount and EdgeCount are already set correctly by the Add* methods
	kg.metadata.Name = export.Metadata.Name
	kg.metadata.Description = export.Metadata.Description
	kg.metadata.DocumentCount = export.Metadata.DocumentCount
	kg.metadata.DocumentIDs = export.Metadata.DocumentIDs
	// Preserve original timestamps if they were set
	if !export.Metadata.CreatedAt.IsZero() {
		kg.metadata.CreatedAt = export.Metadata.CreatedAt
	}

	return kg, nil
}

// =============================================================================
// Utility Functions
// =============================================================================

// normalizeType normalizes a type string for consistent indexing
func normalizeType(t string) string {
	return strings.ToLower(strings.TrimSpace(t))
}

// normalizeName normalizes a name string for consistent indexing
func normalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}
