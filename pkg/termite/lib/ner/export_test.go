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
	"strings"
	"testing"

	"github.com/antflydb/termite/pkg/termite/lib/graph"
)

func TestKnowledgeGraph_AsExportable(t *testing.T) {
	kg := NewKnowledgeGraph()

	// Add nodes
	node1 := &KGNode{
		ID:            "node-1",
		CanonicalName: "Elon Musk",
		Type:          "person",
		Confidence:    0.95,
		Mentions:      []string{"Elon Musk", "Musk"},
	}
	node2 := &KGNode{
		ID:            "node-2",
		CanonicalName: "Tesla",
		Type:          "company",
		Confidence:    0.98,
		Mentions:      []string{"Tesla"},
	}

	if err := kg.AddNode(node1); err != nil {
		t.Fatalf("failed to add node1: %v", err)
	}
	if err := kg.AddNode(node2); err != nil {
		t.Fatalf("failed to add node2: %v", err)
	}

	// Add edge
	edge := &KGEdge{
		ID:         "edge-1",
		SourceID:   "node-1",
		TargetID:   "node-2",
		Type:       "ceo of",
		Confidence: 0.92,
	}
	if err := kg.AddEdge(edge); err != nil {
		t.Fatalf("failed to add edge: %v", err)
	}

	// Get exportable interface
	exportable := kg.AsExportable()

	// Verify nodes
	nodes := exportable.Nodes()
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(nodes))
	}

	// Verify edges
	edges := exportable.Edges()
	if len(edges) != 1 {
		t.Errorf("expected 1 edge, got %d", len(edges))
	}

	// Verify node data is accessible through interface
	for _, node := range nodes {
		if node.GetID() == "" {
			t.Error("node ID should not be empty")
		}
		if node.GetCanonicalName() == "" {
			t.Error("node canonical name should not be empty")
		}
		if node.GetType() == "" {
			t.Error("node type should not be empty")
		}
	}

	// Verify edge data is accessible through interface
	for _, edge := range edges {
		if edge.GetID() == "" {
			t.Error("edge ID should not be empty")
		}
		if edge.GetSourceID() == "" {
			t.Error("edge source ID should not be empty")
		}
		if edge.GetTargetID() == "" {
			t.Error("edge target ID should not be empty")
		}
	}
}

func TestKnowledgeGraph_ExportToCypher(t *testing.T) {
	kg := NewKnowledgeGraph()

	// Add nodes
	node1 := &KGNode{
		ID:            "node-1",
		CanonicalName: "Elon Musk",
		Type:          "person",
		Confidence:    0.95,
		Mentions:      []string{"Elon Musk", "Musk"},
	}
	node2 := &KGNode{
		ID:            "node-2",
		CanonicalName: "Tesla",
		Type:          "company",
		Confidence:    0.98,
		Mentions:      []string{"Tesla"},
	}

	if err := kg.AddNode(node1); err != nil {
		t.Fatalf("failed to add node1: %v", err)
	}
	if err := kg.AddNode(node2); err != nil {
		t.Fatalf("failed to add node2: %v", err)
	}

	// Add edge
	edge := &KGEdge{
		ID:         "edge-1",
		SourceID:   "node-1",
		TargetID:   "node-2",
		Type:       "ceo of",
		Confidence: 0.92,
	}
	if err := kg.AddEdge(edge); err != nil {
		t.Fatalf("failed to add edge: %v", err)
	}

	// Export to Cypher with default options
	cypher, err := kg.ExportToCypher(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypherStr := string(cypher)

	// Verify node CREATE statements
	if !strings.Contains(cypherStr, "CREATE (n0:Person") {
		t.Error("missing Person node CREATE statement")
	}
	if !strings.Contains(cypherStr, "CREATE (n1:Company") {
		t.Error("missing Company node CREATE statement")
	}

	// Verify edge CREATE statement
	if !strings.Contains(cypherStr, "[:CEO_OF") {
		t.Error("missing CEO_OF relationship CREATE statement")
	}

	// Verify properties
	if !strings.Contains(cypherStr, `"Elon Musk"`) {
		t.Error("missing Elon Musk name")
	}
	if !strings.Contains(cypherStr, `"Tesla"`) {
		t.Error("missing Tesla name")
	}
}

func TestKnowledgeGraph_ExportToCypher_WithOptions(t *testing.T) {
	kg := NewKnowledgeGraph()

	node := &KGNode{
		ID:            "node-1",
		CanonicalName: "Test Entity",
		Type:          "entity",
		Confidence:    0.9,
		Provenance: []Provenance{
			{
				SourceDocument: "doc-123",
				SourceText:     "Test Entity mentioned here",
				ExtractorModel: "gliner_small",
			},
		},
	}
	if err := kg.AddNode(node); err != nil {
		t.Fatalf("failed to add node: %v", err)
	}

	// Export with provenance enabled
	opts := &graph.CypherExportOptions{
		IncludeProvenance: true,
	}
	cypher, err := kg.ExportToCypher(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypherStr := string(cypher)

	if !strings.Contains(cypherStr, "provenance_source") {
		t.Error("provenance_source should be included when enabled")
	}
	if !strings.Contains(cypherStr, "doc-123") {
		t.Error("provenance source document should be included")
	}
}

func TestKnowledgeGraph_ExportToCypher_EmptyGraph(t *testing.T) {
	kg := NewKnowledgeGraph()

	cypher, err := kg.ExportToCypher(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cypher) != 0 {
		t.Errorf("expected empty result for empty graph, got: %s", string(cypher))
	}
}

func TestKnowledgeGraph_ExportToCypher_UnwindMode(t *testing.T) {
	kg := NewKnowledgeGraph()

	node := &KGNode{
		ID:            "node-1",
		CanonicalName: "Test",
		Type:          "entity",
		Confidence:    0.9,
	}
	if err := kg.AddNode(node); err != nil {
		t.Fatalf("failed to add node: %v", err)
	}

	opts := &graph.CypherExportOptions{
		UseUnwind: true,
	}
	cypher, err := kg.ExportToCypher(opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypherStr := string(cypher)

	if !strings.Contains(cypherStr, "UNWIND $nodes") {
		t.Error("missing UNWIND statement for nodes")
	}
}

func TestKGNodeAdapter_GetProvenance(t *testing.T) {
	node := &KGNode{
		ID:            "test-id",
		CanonicalName: "Test",
		Type:          "entity",
		Confidence:    0.9,
		Provenance: []Provenance{
			{
				SourceDocument: "doc-1",
				SourceURL:      "http://example.com",
				SourceText:     "text here",
				ExtractorModel: "model-1",
			},
			{
				SourceDocument: "doc-2",
				ExtractorModel: "model-2",
			},
		},
	}

	adapter := &KGNodeAdapter{node}
	provenance := adapter.GetProvenance()

	if len(provenance) != 2 {
		t.Fatalf("expected 2 provenance entries, got %d", len(provenance))
	}

	if provenance[0].GetSourceDocument() != "doc-1" {
		t.Error("first provenance source document mismatch")
	}
	if provenance[0].GetSourceURL() != "http://example.com" {
		t.Error("first provenance source URL mismatch")
	}
	if provenance[0].GetSourceText() != "text here" {
		t.Error("first provenance source text mismatch")
	}
	if provenance[0].GetExtractorModel() != "model-1" {
		t.Error("first provenance extractor model mismatch")
	}

	if provenance[1].GetSourceDocument() != "doc-2" {
		t.Error("second provenance source document mismatch")
	}
}

func TestKGEdgeAdapter_GetProvenance(t *testing.T) {
	edge := &KGEdge{
		ID:         "edge-id",
		SourceID:   "source-id",
		TargetID:   "target-id",
		Type:       "relation",
		Confidence: 0.85,
		Provenance: []Provenance{
			{
				SourceDocument: "doc-1",
				ExtractorModel: "model-1",
			},
		},
	}

	adapter := &KGEdgeAdapter{edge}
	provenance := adapter.GetProvenance()

	if len(provenance) != 1 {
		t.Fatalf("expected 1 provenance entry, got %d", len(provenance))
	}

	if provenance[0].GetSourceDocument() != "doc-1" {
		t.Error("provenance source document mismatch")
	}
	if provenance[0].GetExtractorModel() != "model-1" {
		t.Error("provenance extractor model mismatch")
	}
}
