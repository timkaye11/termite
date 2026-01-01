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

// Package graph provides graph-based utilities for knowledge graph operations,
// including export functionality for Knowledge Graphs to various graph database formats.
//
// # Cypher Export
//
// The package provides a CypherExporter that generates valid Cypher statements
// compatible with Neo4j, FalkorDB, and other Cypher-based graph databases.
//
// Basic usage:
//
//	exporter := graph.NewCypherExporter(nil)
//	cypherBytes, err := exporter.Export(kg)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(string(cypherBytes))
//
// With custom options:
//
//	opts := &graph.CypherExportOptions{
//	    IncludeProvenance: true,
//	    BatchSize:         100,
//	    UseUnwind:         true,
//	}
//	exporter := graph.NewCypherExporter(opts)
//	cypherBytes, err := exporter.Export(kg)
//
// # Output Formats
//
// The exporter supports two output modes:
//
// 1. Standard CREATE statements (default):
//
//	CREATE (n1:Person {id: "uuid-1", name: "Elon Musk", confidence: 0.95})
//	CREATE (n2:Company {id: "uuid-2", name: "Tesla", confidence: 0.98})
//	CREATE (n1)-[:CEO_OF {id: "edge-1", confidence: 0.92}]->(n2)
//
// 2. UNWIND mode (more efficient for bulk imports):
//
//	UNWIND $nodes AS node
//	CREATE (n:Entity {id: node.id, name: node.name, type: node.type})
package graph

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// =============================================================================
// Knowledge Graph Interface (to avoid import cycles with ner package)
// =============================================================================

// KGNodeData represents the data needed from a knowledge graph node for export.
// This interface allows the exporter to work with any knowledge graph implementation.
type KGNodeData interface {
	// GetID returns the unique identifier for this node
	GetID() string
	// GetCanonicalName returns the primary/preferred name for this entity
	GetCanonicalName() string
	// GetType returns the entity type (e.g., "person", "organization", "location")
	GetType() string
	// GetMentions returns all surface forms that refer to this entity
	GetMentions() []string
	// GetProperties returns arbitrary key-value attributes
	GetProperties() map[string]any
	// GetConfidence returns the aggregated confidence score (0.0-1.0)
	GetConfidence() float32
	// GetProvenance returns the provenance records for this node
	GetProvenance() []ProvenanceData
}

// KGEdgeData represents the data needed from a knowledge graph edge for export.
type KGEdgeData interface {
	// GetID returns the unique identifier for this edge
	GetID() string
	// GetSourceID returns the ID of the source node
	GetSourceID() string
	// GetTargetID returns the ID of the target node
	GetTargetID() string
	// GetType returns the relationship type (e.g., "founded", "works_at")
	GetType() string
	// GetProperties returns arbitrary key-value attributes
	GetProperties() map[string]any
	// GetConfidence returns the confidence score (0.0-1.0)
	GetConfidence() float32
	// GetProvenance returns the provenance records for this edge
	GetProvenance() []ProvenanceData
}

// ProvenanceData represents provenance information for export.
type ProvenanceData interface {
	// GetSourceDocument returns the document ID or path
	GetSourceDocument() string
	// GetSourceURL returns the URL of the source (if applicable)
	GetSourceURL() string
	// GetSourceText returns the original text span that yielded this extraction
	GetSourceText() string
	// GetExtractorModel returns the model used for extraction
	GetExtractorModel() string
}

// KnowledgeGraphData represents the data needed from a knowledge graph for export.
type KnowledgeGraphData interface {
	// Nodes returns all nodes in the graph
	Nodes() []KGNodeData
	// Edges returns all edges in the graph
	Edges() []KGEdgeData
}

// =============================================================================
// Exporter Interface
// =============================================================================

// Exporter defines the interface for exporting Knowledge Graphs to various formats.
// Implementations should handle the conversion of nodes and edges to the target format,
// including proper escaping of special characters and formatting of properties.
type Exporter interface {
	// Export converts a KnowledgeGraph to the target format and returns the result
	// as a byte slice. Returns an error if the conversion fails.
	Export(kg KnowledgeGraphData) ([]byte, error)
}

// =============================================================================
// Simple Node/Edge implementations for adapter use
// =============================================================================

// SimpleNode is a simple implementation of KGNodeData for use as an adapter.
type SimpleNode struct {
	ID            string
	CanonicalName string
	Type          string
	Mentions      []string
	Properties    map[string]any
	Confidence    float32
	Provenance    []SimpleProvenance
}

func (n *SimpleNode) GetID() string                  { return n.ID }
func (n *SimpleNode) GetCanonicalName() string       { return n.CanonicalName }
func (n *SimpleNode) GetType() string                { return n.Type }
func (n *SimpleNode) GetMentions() []string          { return n.Mentions }
func (n *SimpleNode) GetProperties() map[string]any  { return n.Properties }
func (n *SimpleNode) GetConfidence() float32         { return n.Confidence }
func (n *SimpleNode) GetProvenance() []ProvenanceData {
	result := make([]ProvenanceData, len(n.Provenance))
	for i := range n.Provenance {
		result[i] = &n.Provenance[i]
	}
	return result
}

// SimpleEdge is a simple implementation of KGEdgeData for use as an adapter.
type SimpleEdge struct {
	ID         string
	SourceID   string
	TargetID   string
	Type       string
	Properties map[string]any
	Confidence float32
	Provenance []SimpleProvenance
}

func (e *SimpleEdge) GetID() string                  { return e.ID }
func (e *SimpleEdge) GetSourceID() string            { return e.SourceID }
func (e *SimpleEdge) GetTargetID() string            { return e.TargetID }
func (e *SimpleEdge) GetType() string                { return e.Type }
func (e *SimpleEdge) GetProperties() map[string]any  { return e.Properties }
func (e *SimpleEdge) GetConfidence() float32         { return e.Confidence }
func (e *SimpleEdge) GetProvenance() []ProvenanceData {
	result := make([]ProvenanceData, len(e.Provenance))
	for i := range e.Provenance {
		result[i] = &e.Provenance[i]
	}
	return result
}

// SimpleProvenance is a simple implementation of ProvenanceData.
type SimpleProvenance struct {
	SourceDocument      string
	SourceURL           string
	SourceText          string
	CharOffsetStart     int
	CharOffsetEnd       int
	ExtractorModel      string
	ExtractionTimestamp time.Time
	ExtractorConfidence float32
}

func (p *SimpleProvenance) GetSourceDocument() string { return p.SourceDocument }
func (p *SimpleProvenance) GetSourceURL() string      { return p.SourceURL }
func (p *SimpleProvenance) GetSourceText() string     { return p.SourceText }
func (p *SimpleProvenance) GetExtractorModel() string { return p.ExtractorModel }

// SimpleKnowledgeGraph is a simple implementation of KnowledgeGraphData for use as an adapter.
type SimpleKnowledgeGraph struct {
	NodeList []KGNodeData
	EdgeList []KGEdgeData
}

func (kg *SimpleKnowledgeGraph) Nodes() []KGNodeData { return kg.NodeList }
func (kg *SimpleKnowledgeGraph) Edges() []KGEdgeData { return kg.EdgeList }

// =============================================================================
// Cypher Exporter
// =============================================================================

// CypherExportOptions configures the behavior of the CypherExporter.
type CypherExportOptions struct {
	// IncludeProvenance includes source document information as nested properties
	// on nodes and edges. This adds provenance_source, provenance_text, and
	// provenance_model properties when available.
	IncludeProvenance bool

	// BatchSize groups CREATE statements into batches of this size.
	// Useful for large graphs to avoid overwhelming the database with a single
	// transaction. A value of 0 means no batching (all statements in one block).
	BatchSize int

	// UseUnwind generates UNWIND-based bulk import statements instead of individual
	// CREATE statements. This is more efficient for large imports but requires
	// passing the data as parameters ($nodes, $edges).
	UseUnwind bool

	// NodeLabelPrefix is an optional prefix to add to all node labels.
	// For example, with prefix "KG_", a "Person" type becomes "KG_Person".
	NodeLabelPrefix string

	// RelationshipTypePrefix is an optional prefix to add to all relationship types.
	// For example, with prefix "REL_", a "CEO_OF" type becomes "REL_CEO_OF".
	RelationshipTypePrefix string
}

// DefaultCypherExportOptions returns the default export options.
func DefaultCypherExportOptions() *CypherExportOptions {
	return &CypherExportOptions{
		IncludeProvenance:      false,
		BatchSize:              0,
		UseUnwind:              false,
		NodeLabelPrefix:        "",
		RelationshipTypePrefix: "",
	}
}

// CypherExporter exports Knowledge Graphs to Cypher format, compatible with
// Neo4j, FalkorDB, and other Cypher-based graph databases.
//
// The exporter supports two output modes:
//
// 1. Standard CREATE statements (default):
//
//	CREATE (n1:Person {id: "uuid-1", name: "Elon Musk", confidence: 0.95})
//	CREATE (n2:Company {id: "uuid-2", name: "Tesla", confidence: 0.98})
//	CREATE (n1)-[:CEO_OF {id: "edge-1", confidence: 0.92}]->(n2)
//
// 2. UNWIND mode (more efficient for bulk imports):
//
//	UNWIND $nodes AS node
//	CREATE (n:Entity {id: node.id, name: node.name, type: node.type})
//
// Node properties include: id, name, type, confidence, and mentions (as array).
// Edge properties include: id, type, confidence, sourceId, and targetId.
type CypherExporter struct {
	opts *CypherExportOptions
}

// NewCypherExporter creates a new CypherExporter with the given options.
// If opts is nil, default options are used.
//
// Example usage with default options:
//
//	exporter := NewCypherExporter(nil)
//	cypher, err := exporter.Export(kg)
//
// Example usage with custom options:
//
//	opts := &CypherExportOptions{
//	    IncludeProvenance: true,
//	    BatchSize:         100,
//	    UseUnwind:         true,
//	}
//	exporter := NewCypherExporter(opts)
//	cypher, err := exporter.Export(kg)
func NewCypherExporter(opts *CypherExportOptions) *CypherExporter {
	if opts == nil {
		opts = DefaultCypherExportOptions()
	}
	return &CypherExporter{opts: opts}
}

// Export converts a KnowledgeGraph to Cypher statements.
// It generates CREATE statements for nodes and relationships with all their properties.
// The output is valid Cypher syntax that can be executed against Neo4j, FalkorDB,
// or other Cypher-compatible databases.
//
// For empty graphs, Export returns an empty byte slice with no error.
//
// Node labels are derived from the entity type (e.g., "Person", "Organization").
// Relationship types are derived from the edge type (e.g., "CEO_OF", "WORKS_AT").
func (e *CypherExporter) Export(kg KnowledgeGraphData) ([]byte, error) {
	if kg == nil {
		return nil, fmt.Errorf("knowledge graph cannot be nil")
	}

	nodes := kg.Nodes()
	edges := kg.Edges()

	// Handle empty graph
	if len(nodes) == 0 && len(edges) == 0 {
		return []byte{}, nil
	}

	if e.opts.UseUnwind {
		return e.exportUnwind(nodes, edges)
	}

	return e.exportCreate(nodes, edges)
}

// exportCreate generates individual CREATE statements for nodes and edges.
func (e *CypherExporter) exportCreate(nodes []KGNodeData, edges []KGEdgeData) ([]byte, error) {
	var buf bytes.Buffer

	// Sort nodes by ID for consistent output
	sortedNodes := make([]KGNodeData, len(nodes))
	copy(sortedNodes, nodes)
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].GetID() < sortedNodes[j].GetID()
	})

	// Sort edges by ID for consistent output
	sortedEdges := make([]KGEdgeData, len(edges))
	copy(sortedEdges, edges)
	sort.Slice(sortedEdges, func(i, j int) bool {
		return sortedEdges[i].GetID() < sortedEdges[j].GetID()
	})

	// Build node ID to variable name mapping
	nodeVars := make(map[string]string)
	for i, node := range sortedNodes {
		nodeVars[node.GetID()] = fmt.Sprintf("n%d", i)
	}

	// Write header comment
	buf.WriteString("// Generated Cypher statements for Knowledge Graph import\n")
	buf.WriteString("// Nodes: " + fmt.Sprintf("%d", len(nodes)) + ", Edges: " + fmt.Sprintf("%d", len(edges)) + "\n\n")

	// Generate node CREATE statements
	if len(sortedNodes) > 0 {
		buf.WriteString("// Nodes\n")

		batchCount := 0
		for _, node := range sortedNodes {
			varName := nodeVars[node.GetID()]
			stmt, err := e.nodeToCreate(varName, node)
			if err != nil {
				return nil, fmt.Errorf("generating CREATE for node %s: %w", node.GetID(), err)
			}
			buf.WriteString(stmt)
			buf.WriteString("\n")

			batchCount++
			if e.opts.BatchSize > 0 && batchCount >= e.opts.BatchSize {
				buf.WriteString("\n// --- Batch separator ---\n\n")
				batchCount = 0
			}
		}
	}

	// Generate edge CREATE statements
	if len(sortedEdges) > 0 {
		buf.WriteString("\n// Relationships\n")

		batchCount := 0
		for _, edge := range sortedEdges {
			sourceVar, sourceExists := nodeVars[edge.GetSourceID()]
			targetVar, targetExists := nodeVars[edge.GetTargetID()]

			if !sourceExists || !targetExists {
				// Skip edges with missing nodes (shouldn't happen with valid KG)
				continue
			}

			stmt, err := e.edgeToCreate(sourceVar, targetVar, edge)
			if err != nil {
				return nil, fmt.Errorf("generating CREATE for edge %s: %w", edge.GetID(), err)
			}
			buf.WriteString(stmt)
			buf.WriteString("\n")

			batchCount++
			if e.opts.BatchSize > 0 && batchCount >= e.opts.BatchSize {
				buf.WriteString("\n// --- Batch separator ---\n\n")
				batchCount = 0
			}
		}
	}

	return buf.Bytes(), nil
}

// exportUnwind generates UNWIND-based bulk import statements.
func (e *CypherExporter) exportUnwind(nodes []KGNodeData, edges []KGEdgeData) ([]byte, error) {
	var buf bytes.Buffer

	// Write header comment
	buf.WriteString("// Generated Cypher UNWIND statements for Knowledge Graph bulk import\n")
	buf.WriteString("// Nodes: " + fmt.Sprintf("%d", len(nodes)) + ", Edges: " + fmt.Sprintf("%d", len(edges)) + "\n")
	buf.WriteString("// Usage: Pass 'nodes' and 'edges' as query parameters\n\n")

	// Generate node data and UNWIND statement
	if len(nodes) > 0 {
		buf.WriteString("// Node data (pass as $nodes parameter)\n")
		nodeData, err := e.nodesToJSON(nodes)
		if err != nil {
			return nil, fmt.Errorf("serializing nodes: %w", err)
		}
		buf.WriteString("// $nodes = ")
		buf.Write(nodeData)
		buf.WriteString("\n\n")

		buf.WriteString("// Create nodes using UNWIND\n")
		buf.WriteString("UNWIND $nodes AS node\n")
		buf.WriteString("CALL apoc.create.node([node.label], node.properties) YIELD node AS n\n")
		buf.WriteString("RETURN count(n) AS nodesCreated;\n\n")

		// Alternative without APOC (using MERGE pattern)
		buf.WriteString("// Alternative without APOC (creates Entity nodes with type property):\n")
		buf.WriteString("// UNWIND $nodes AS node\n")
		buf.WriteString("// CREATE (n:Entity {id: node.id, name: node.name, type: node.type, confidence: node.confidence, mentions: node.mentions})\n")
		buf.WriteString("// RETURN count(n) AS nodesCreated;\n\n")
	}

	// Generate edge data and UNWIND statement
	if len(edges) > 0 {
		buf.WriteString("// Edge data (pass as $edges parameter)\n")
		edgeData, err := e.edgesToJSON(edges)
		if err != nil {
			return nil, fmt.Errorf("serializing edges: %w", err)
		}
		buf.WriteString("// $edges = ")
		buf.Write(edgeData)
		buf.WriteString("\n\n")

		buf.WriteString("// Create relationships using UNWIND\n")
		buf.WriteString("UNWIND $edges AS edge\n")
		buf.WriteString("MATCH (a {id: edge.sourceId}), (b {id: edge.targetId})\n")
		buf.WriteString("CALL apoc.create.relationship(a, edge.type, edge.properties, b) YIELD rel\n")
		buf.WriteString("RETURN count(rel) AS relationshipsCreated;\n\n")

		// Alternative without APOC
		buf.WriteString("// Alternative without APOC (creates RELATION type with type property):\n")
		buf.WriteString("// UNWIND $edges AS edge\n")
		buf.WriteString("// MATCH (a {id: edge.sourceId}), (b {id: edge.targetId})\n")
		buf.WriteString("// CREATE (a)-[r:RELATION {id: edge.id, type: edge.type, confidence: edge.confidence}]->(b)\n")
		buf.WriteString("// RETURN count(r) AS relationshipsCreated;\n")
	}

	return buf.Bytes(), nil
}

// nodeToCreate generates a CREATE statement for a single node.
func (e *CypherExporter) nodeToCreate(varName string, node KGNodeData) (string, error) {
	label := e.formatLabel(node.GetType())
	props := e.nodeProperties(node)

	propsStr, err := formatCypherProperties(props)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("CREATE (%s:%s %s)", varName, label, propsStr), nil
}

// edgeToCreate generates a CREATE statement for a single edge.
func (e *CypherExporter) edgeToCreate(sourceVar, targetVar string, edge KGEdgeData) (string, error) {
	relType := e.formatRelationType(edge.GetType())
	props := e.edgeProperties(edge)

	propsStr, err := formatCypherProperties(props)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("CREATE (%s)-[:%s %s]->(%s)", sourceVar, relType, propsStr, targetVar), nil
}

// nodeProperties builds the property map for a node.
func (e *CypherExporter) nodeProperties(node KGNodeData) map[string]any {
	props := map[string]any{
		"id":         node.GetID(),
		"name":       node.GetCanonicalName(),
		"type":       node.GetType(),
		"confidence": node.GetConfidence(),
	}

	mentions := node.GetMentions()
	if len(mentions) > 0 {
		props["mentions"] = mentions
	}

	// Add provenance if enabled
	provenance := node.GetProvenance()
	if e.opts.IncludeProvenance && len(provenance) > 0 {
		prov := provenance[0] // Use first provenance entry
		if prov.GetSourceDocument() != "" {
			props["provenance_source"] = prov.GetSourceDocument()
		}
		if prov.GetSourceText() != "" {
			props["provenance_text"] = prov.GetSourceText()
		}
		if prov.GetExtractorModel() != "" {
			props["provenance_model"] = prov.GetExtractorModel()
		}
		if prov.GetSourceURL() != "" {
			props["provenance_url"] = prov.GetSourceURL()
		}
	}

	// Add custom properties from the node
	for k, v := range node.GetProperties() {
		// Avoid overwriting core properties
		if _, exists := props[k]; !exists {
			props[k] = v
		}
	}

	return props
}

// edgeProperties builds the property map for an edge.
func (e *CypherExporter) edgeProperties(edge KGEdgeData) map[string]any {
	props := map[string]any{
		"id":         edge.GetID(),
		"type":       edge.GetType(),
		"confidence": edge.GetConfidence(),
		"sourceId":   edge.GetSourceID(),
		"targetId":   edge.GetTargetID(),
	}

	// Add provenance if enabled
	provenance := edge.GetProvenance()
	if e.opts.IncludeProvenance && len(provenance) > 0 {
		prov := provenance[0] // Use first provenance entry
		if prov.GetSourceDocument() != "" {
			props["provenance_source"] = prov.GetSourceDocument()
		}
		if prov.GetSourceText() != "" {
			props["provenance_text"] = prov.GetSourceText()
		}
		if prov.GetExtractorModel() != "" {
			props["provenance_model"] = prov.GetExtractorModel()
		}
		if prov.GetSourceURL() != "" {
			props["provenance_url"] = prov.GetSourceURL()
		}
	}

	// Add custom properties from the edge
	for k, v := range edge.GetProperties() {
		// Avoid overwriting core properties
		if _, exists := props[k]; !exists {
			props[k] = v
		}
	}

	return props
}

// formatLabel converts an entity type to a valid Cypher label.
// It capitalizes the first letter and replaces spaces/dashes with underscores.
func (e *CypherExporter) formatLabel(entityType string) string {
	label := ToCypherIdentifier(entityType)
	if label == "" {
		label = "Entity"
	}
	return e.opts.NodeLabelPrefix + label
}

// formatRelationType converts a relationship type to a valid Cypher relationship type.
// It converts to uppercase and replaces spaces/dashes with underscores.
func (e *CypherExporter) formatRelationType(relType string) string {
	rt := ToCypherRelationType(relType)
	if rt == "" {
		rt = "RELATED_TO"
	}
	return e.opts.RelationshipTypePrefix + rt
}

// nodesToJSON serializes nodes to JSON for UNWIND mode.
func (e *CypherExporter) nodesToJSON(nodes []KGNodeData) ([]byte, error) {
	var data []map[string]any

	for _, node := range nodes {
		item := map[string]any{
			"id":         node.GetID(),
			"label":      e.formatLabel(node.GetType()),
			"properties": e.nodeProperties(node),
		}
		data = append(data, item)
	}

	return json.MarshalIndent(data, "", "  ")
}

// edgesToJSON serializes edges to JSON for UNWIND mode.
func (e *CypherExporter) edgesToJSON(edges []KGEdgeData) ([]byte, error) {
	var data []map[string]any

	for _, edge := range edges {
		item := map[string]any{
			"id":         edge.GetID(),
			"sourceId":   edge.GetSourceID(),
			"targetId":   edge.GetTargetID(),
			"type":       e.formatRelationType(edge.GetType()),
			"properties": e.edgeProperties(edge),
		}
		data = append(data, item)
	}

	return json.MarshalIndent(data, "", "  ")
}

// =============================================================================
// Cypher Formatting Utilities
// =============================================================================

// formatCypherProperties formats a property map as a Cypher property string.
// Example: {id: "uuid", name: "John", confidence: 0.95}
func formatCypherProperties(props map[string]any) (string, error) {
	if len(props) == 0 {
		return "{}", nil
	}

	// Sort keys for consistent output
	keys := make([]string, 0, len(props))
	for k := range props {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		v := props[k]
		formatted, err := formatCypherValue(v)
		if err != nil {
			return "", fmt.Errorf("formatting property %s: %w", k, err)
		}
		parts = append(parts, fmt.Sprintf("%s: %s", k, formatted))
	}

	return "{" + strings.Join(parts, ", ") + "}", nil
}

// formatCypherValue formats a single value for Cypher syntax.
func formatCypherValue(v any) (string, error) {
	switch val := v.(type) {
	case string:
		return FormatCypherString(val), nil
	case bool:
		if val {
			return "true", nil
		}
		return "false", nil
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", val), nil
	case float32:
		return fmt.Sprintf("%g", val), nil
	case float64:
		return fmt.Sprintf("%g", val), nil
	case []string:
		return FormatCypherStringArray(val), nil
	case []any:
		return formatCypherAnyArray(val)
	case nil:
		return "null", nil
	default:
		// For complex types, serialize to JSON string
		jsonBytes, err := json.Marshal(val)
		if err != nil {
			return "", fmt.Errorf("marshaling value: %w", err)
		}
		return FormatCypherString(string(jsonBytes)), nil
	}
}

// FormatCypherString escapes and quotes a string for Cypher.
func FormatCypherString(s string) string {
	// Escape special characters
	escaped := strings.ReplaceAll(s, "\\", "\\\\") // Backslash first
	escaped = strings.ReplaceAll(escaped, "'", "\\'")
	escaped = strings.ReplaceAll(escaped, "\"", "\\\"")
	escaped = strings.ReplaceAll(escaped, "\n", "\\n")
	escaped = strings.ReplaceAll(escaped, "\r", "\\r")
	escaped = strings.ReplaceAll(escaped, "\t", "\\t")
	return "\"" + escaped + "\""
}

// FormatCypherStringArray formats a string slice as a Cypher array.
func FormatCypherStringArray(arr []string) string {
	if len(arr) == 0 {
		return "[]"
	}

	parts := make([]string, len(arr))
	for i, s := range arr {
		parts[i] = FormatCypherString(s)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// formatCypherAnyArray formats an interface slice as a Cypher array.
func formatCypherAnyArray(arr []any) (string, error) {
	if len(arr) == 0 {
		return "[]", nil
	}

	parts := make([]string, len(arr))
	for i, v := range arr {
		formatted, err := formatCypherValue(v)
		if err != nil {
			return "", err
		}
		parts[i] = formatted
	}
	return "[" + strings.Join(parts, ", ") + "]", nil
}

// ToCypherIdentifier converts a string to a valid Cypher identifier (label).
// It capitalizes the first letter of each word and removes invalid characters.
func ToCypherIdentifier(s string) string {
	if s == "" {
		return ""
	}

	// Replace common separators with spaces for splitting
	s = strings.ReplaceAll(s, "-", " ")
	s = strings.ReplaceAll(s, "_", " ")

	words := strings.Fields(s)
	for i, word := range words {
		if len(word) > 0 {
			// Capitalize first letter, lowercase rest
			words[i] = strings.ToUpper(string(word[0])) + strings.ToLower(word[1:])
		}
	}

	result := strings.Join(words, "")

	// Remove any remaining invalid characters (keep only alphanumeric and underscore)
	var clean strings.Builder
	for i, r := range result {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9' && i > 0) || r == '_' {
			clean.WriteRune(r)
		}
	}

	return clean.String()
}

// ToCypherRelationType converts a string to a valid Cypher relationship type.
// It converts to uppercase and replaces spaces/dashes with underscores.
func ToCypherRelationType(s string) string {
	if s == "" {
		return ""
	}

	// Replace common separators with underscores
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "-", "_")

	// Convert to uppercase
	s = strings.ToUpper(s)

	// Remove any invalid characters (keep only alphanumeric and underscore)
	var clean strings.Builder
	for i, r := range s {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9' && i > 0) || r == '_' {
			clean.WriteRune(r)
		}
	}

	return clean.String()
}
