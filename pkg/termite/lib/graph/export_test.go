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
	"strings"
	"testing"
)

func TestCypherExporter_Export_EmptyGraph(t *testing.T) {
	kg := &SimpleKnowledgeGraph{
		NodeList: nil,
		EdgeList: nil,
	}
	exporter := NewCypherExporter(nil)

	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result) != 0 {
		t.Errorf("expected empty result for empty graph, got: %s", string(result))
	}
}

func TestCypherExporter_Export_NilGraph(t *testing.T) {
	exporter := NewCypherExporter(nil)

	_, err := exporter.Export(nil)
	if err == nil {
		t.Error("expected error for nil graph")
	}
}

func TestCypherExporter_Export_SimpleGraph(t *testing.T) {
	node1 := &SimpleNode{
		ID:            "node-1",
		CanonicalName: "Elon Musk",
		Type:          "person",
		Confidence:    0.95,
		Mentions:      []string{"Elon Musk", "Musk"},
	}
	node2 := &SimpleNode{
		ID:            "node-2",
		CanonicalName: "Tesla",
		Type:          "company",
		Confidence:    0.98,
		Mentions:      []string{"Tesla"},
	}
	edge := &SimpleEdge{
		ID:         "edge-1",
		SourceID:   "node-1",
		TargetID:   "node-2",
		Type:       "ceo of",
		Confidence: 0.92,
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node1, node2},
		EdgeList: []KGEdgeData{edge},
	}

	exporter := NewCypherExporter(nil)
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypher := string(result)

	// Verify node CREATE statements
	if !strings.Contains(cypher, "CREATE (n0:Person") {
		t.Error("missing Person node CREATE statement")
	}
	if !strings.Contains(cypher, "CREATE (n1:Company") {
		t.Error("missing Company node CREATE statement")
	}

	// Verify edge CREATE statement
	if !strings.Contains(cypher, "[:CEO_OF") {
		t.Error("missing CEO_OF relationship CREATE statement")
	}

	// Verify properties
	if !strings.Contains(cypher, `"Elon Musk"`) {
		t.Error("missing Elon Musk name")
	}
	if !strings.Contains(cypher, `"Tesla"`) {
		t.Error("missing Tesla name")
	}
}

func TestCypherExporter_Export_WithProvenance(t *testing.T) {
	node := &SimpleNode{
		ID:            "node-1",
		CanonicalName: "Test Entity",
		Type:          "entity",
		Confidence:    0.9,
		Provenance: []SimpleProvenance{
			{
				SourceDocument: "doc-123",
				SourceText:     "Test Entity mentioned here",
				ExtractorModel: "gliner_small",
			},
		},
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node},
		EdgeList: nil,
	}

	// Test with provenance disabled (default)
	exporter := NewCypherExporter(nil)
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if strings.Contains(string(result), "provenance_source") {
		t.Error("provenance should not be included by default")
	}

	// Test with provenance enabled
	exporter = NewCypherExporter(&CypherExportOptions{IncludeProvenance: true})
	result, err = exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(string(result), "provenance_source") {
		t.Error("provenance_source should be included when enabled")
	}
	if !strings.Contains(string(result), "doc-123") {
		t.Error("provenance source document should be included")
	}
}

func TestCypherExporter_Export_UnwindMode(t *testing.T) {
	node := &SimpleNode{
		ID:            "node-1",
		CanonicalName: "Test",
		Type:          "entity",
		Confidence:    0.9,
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node},
		EdgeList: nil,
	}

	exporter := NewCypherExporter(&CypherExportOptions{UseUnwind: true})
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypher := string(result)

	if !strings.Contains(cypher, "UNWIND $nodes") {
		t.Error("missing UNWIND statement for nodes")
	}
	if !strings.Contains(cypher, "$nodes = [") {
		t.Error("missing $nodes JSON data")
	}
}

func TestCypherExporter_Export_BatchSize(t *testing.T) {
	var nodes []KGNodeData
	for i := 0; i < 5; i++ {
		nodes = append(nodes, &SimpleNode{
			ID:            string(rune('a' + i)),
			CanonicalName: string(rune('A' + i)),
			Type:          "entity",
			Confidence:    0.9,
		})
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: nodes,
		EdgeList: nil,
	}

	exporter := NewCypherExporter(&CypherExportOptions{BatchSize: 2})
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypher := string(result)

	// Should have batch separators
	if !strings.Contains(cypher, "Batch separator") {
		t.Error("missing batch separators")
	}
}

func TestFormatCypherString_SpecialCharacters(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`simple`, `"simple"`},
		{`with "quotes"`, `"with \"quotes\""`},
		{`with 'single'`, `"with \'single\'"`},
		{`with\backslash`, `"with\\backslash"`},
		{"with\nnewline", `"with\nnewline"`},
		{"with\ttab", `"with\ttab"`},
	}

	for _, tt := range tests {
		result := FormatCypherString(tt.input)
		if result != tt.expected {
			t.Errorf("FormatCypherString(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestToCypherIdentifier(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"person", "Person"},
		{"PERSON", "Person"},
		{"Person", "Person"},
		{"organization", "Organization"},
		{"geo-location", "GeoLocation"},
		{"geo_location", "GeoLocation"},
		{"geo location", "GeoLocation"},
		{"", ""},
	}

	for _, tt := range tests {
		result := ToCypherIdentifier(tt.input)
		if result != tt.expected {
			t.Errorf("ToCypherIdentifier(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestToCypherRelationType(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"ceo of", "CEO_OF"},
		{"CEO_OF", "CEO_OF"},
		{"works-at", "WORKS_AT"},
		{"located in", "LOCATED_IN"},
		{"", ""},
	}

	for _, tt := range tests {
		result := ToCypherRelationType(tt.input)
		if result != tt.expected {
			t.Errorf("ToCypherRelationType(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestFormatCypherProperties(t *testing.T) {
	props := map[string]any{
		"id":         "test-id",
		"confidence": float32(0.95),
		"count":      42,
		"active":     true,
	}

	result, err := formatCypherProperties(props)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Properties should be in alphabetical order
	if !strings.Contains(result, `active: true`) {
		t.Error("missing active property")
	}
	if !strings.Contains(result, `confidence: 0.95`) {
		t.Error("missing confidence property")
	}
	if !strings.Contains(result, `count: 42`) {
		t.Error("missing count property")
	}
	if !strings.Contains(result, `id: "test-id"`) {
		t.Error("missing id property")
	}
}

func TestFormatCypherStringArray(t *testing.T) {
	tests := []struct {
		input    []string
		expected string
	}{
		{nil, "[]"},
		{[]string{}, "[]"},
		{[]string{"one"}, `["one"]`},
		{[]string{"one", "two"}, `["one", "two"]`},
		{[]string{`with "quotes"`}, `["with \"quotes\""]`},
	}

	for _, tt := range tests {
		result := FormatCypherStringArray(tt.input)
		if result != tt.expected {
			t.Errorf("FormatCypherStringArray(%v) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestCypherExporter_LabelPrefix(t *testing.T) {
	node := &SimpleNode{
		ID:            "node-1",
		CanonicalName: "Test",
		Type:          "person",
		Confidence:    0.9,
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node},
		EdgeList: nil,
	}

	exporter := NewCypherExporter(&CypherExportOptions{
		NodeLabelPrefix: "KG_",
	})
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(string(result), ":KG_Person") {
		t.Error("missing KG_ prefix on label")
	}
}

func TestCypherExporter_RelationshipTypePrefix(t *testing.T) {
	node1 := &SimpleNode{ID: "n1", CanonicalName: "A", Type: "entity", Confidence: 0.9}
	node2 := &SimpleNode{ID: "n2", CanonicalName: "B", Type: "entity", Confidence: 0.9}
	edge := &SimpleEdge{
		ID:         "e1",
		SourceID:   "n1",
		TargetID:   "n2",
		Type:       "related to",
		Confidence: 0.8,
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node1, node2},
		EdgeList: []KGEdgeData{edge},
	}

	exporter := NewCypherExporter(&CypherExportOptions{
		RelationshipTypePrefix: "REL_",
	})
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(string(result), ":REL_RELATED_TO") {
		t.Error("missing REL_ prefix on relationship type")
	}
}

func TestCypherExporter_CustomProperties(t *testing.T) {
	node := &SimpleNode{
		ID:            "node-1",
		CanonicalName: "Test Entity",
		Type:          "entity",
		Confidence:    0.9,
		Properties: map[string]any{
			"custom_field": "custom_value",
			"priority":     1,
		},
	}

	kg := &SimpleKnowledgeGraph{
		NodeList: []KGNodeData{node},
		EdgeList: nil,
	}

	exporter := NewCypherExporter(nil)
	result, err := exporter.Export(kg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cypher := string(result)

	if !strings.Contains(cypher, "custom_field") {
		t.Error("missing custom_field property")
	}
	if !strings.Contains(cypher, "custom_value") {
		t.Error("missing custom_value")
	}
	if !strings.Contains(cypher, "priority: 1") {
		t.Error("missing priority property")
	}
}

func TestDefaultCypherExportOptions(t *testing.T) {
	opts := DefaultCypherExportOptions()

	if opts.IncludeProvenance {
		t.Error("IncludeProvenance should be false by default")
	}
	if opts.BatchSize != 0 {
		t.Error("BatchSize should be 0 by default")
	}
	if opts.UseUnwind {
		t.Error("UseUnwind should be false by default")
	}
	if opts.NodeLabelPrefix != "" {
		t.Error("NodeLabelPrefix should be empty by default")
	}
	if opts.RelationshipTypePrefix != "" {
		t.Error("RelationshipTypePrefix should be empty by default")
	}
}

func TestSimpleNode_Interface(t *testing.T) {
	node := &SimpleNode{
		ID:            "test-id",
		CanonicalName: "Test Name",
		Type:          "test-type",
		Confidence:    0.99,
		Mentions:      []string{"mention1", "mention2"},
		Properties:    map[string]any{"key": "value"},
		Provenance: []SimpleProvenance{
			{SourceDocument: "doc1", SourceText: "text1"},
		},
	}

	// Verify interface methods work
	if node.GetID() != "test-id" {
		t.Error("GetID mismatch")
	}
	if node.GetCanonicalName() != "Test Name" {
		t.Error("GetCanonicalName mismatch")
	}
	if node.GetType() != "test-type" {
		t.Error("GetType mismatch")
	}
	if node.GetConfidence() != 0.99 {
		t.Error("GetConfidence mismatch")
	}
	if len(node.GetMentions()) != 2 {
		t.Error("GetMentions mismatch")
	}
	if len(node.GetProperties()) != 1 {
		t.Error("GetProperties mismatch")
	}
	if len(node.GetProvenance()) != 1 {
		t.Error("GetProvenance mismatch")
	}
}

func TestSimpleEdge_Interface(t *testing.T) {
	edge := &SimpleEdge{
		ID:         "edge-id",
		SourceID:   "source-id",
		TargetID:   "target-id",
		Type:       "edge-type",
		Confidence: 0.85,
		Properties: map[string]any{"key": "value"},
		Provenance: []SimpleProvenance{
			{SourceDocument: "doc1"},
		},
	}

	// Verify interface methods work
	if edge.GetID() != "edge-id" {
		t.Error("GetID mismatch")
	}
	if edge.GetSourceID() != "source-id" {
		t.Error("GetSourceID mismatch")
	}
	if edge.GetTargetID() != "target-id" {
		t.Error("GetTargetID mismatch")
	}
	if edge.GetType() != "edge-type" {
		t.Error("GetType mismatch")
	}
	if edge.GetConfidence() != 0.85 {
		t.Error("GetConfidence mismatch")
	}
	if len(edge.GetProperties()) != 1 {
		t.Error("GetProperties mismatch")
	}
	if len(edge.GetProvenance()) != 1 {
		t.Error("GetProvenance mismatch")
	}
}
