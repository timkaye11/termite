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
	"testing"
	"time"
)

func TestNewKnowledgeGraph(t *testing.T) {
	kg := NewKnowledgeGraph()

	if kg == nil {
		t.Fatal("NewKnowledgeGraph returned nil")
	}

	if kg.NodeCount() != 0 {
		t.Errorf("expected 0 nodes, got %d", kg.NodeCount())
	}

	if kg.EdgeCount() != 0 {
		t.Errorf("expected 0 edges, got %d", kg.EdgeCount())
	}
}

func TestKnowledgeGraph_AddNode(t *testing.T) {
	kg := NewKnowledgeGraph()

	node := &KGNode{
		ID:            "node1",
		CanonicalName: "John Smith",
		Type:          "person",
		Mentions:      []string{"John", "J. Smith"},
		Confidence:    0.95,
	}

	err := kg.AddNode(node)
	if err != nil {
		t.Fatalf("AddNode failed: %v", err)
	}

	if kg.NodeCount() != 1 {
		t.Errorf("expected 1 node, got %d", kg.NodeCount())
	}

	// Retrieve the node
	retrieved := kg.GetNode("node1")
	if retrieved == nil {
		t.Fatal("GetNode returned nil")
	}

	if retrieved.CanonicalName != "John Smith" {
		t.Errorf("expected canonical name 'John Smith', got '%s'", retrieved.CanonicalName)
	}

	// Test duplicate rejection
	err = kg.AddNode(&KGNode{ID: "node1", CanonicalName: "Duplicate"})
	if err == nil {
		t.Error("expected error when adding duplicate node")
	}
}

func TestKnowledgeGraph_AddEdge(t *testing.T) {
	kg := NewKnowledgeGraph()

	// Add nodes first
	kg.AddNode(&KGNode{ID: "person1", CanonicalName: "Elon Musk", Type: "person"})
	kg.AddNode(&KGNode{ID: "company1", CanonicalName: "SpaceX", Type: "organization"})

	edge := &KGEdge{
		ID:         "edge1",
		SourceID:   "person1",
		TargetID:   "company1",
		Type:       "founded",
		Confidence: 0.9,
	}

	err := kg.AddEdge(edge)
	if err != nil {
		t.Fatalf("AddEdge failed: %v", err)
	}

	if kg.EdgeCount() != 1 {
		t.Errorf("expected 1 edge, got %d", kg.EdgeCount())
	}

	// Test edge with non-existent source
	err = kg.AddEdge(&KGEdge{
		ID:       "edge2",
		SourceID: "nonexistent",
		TargetID: "company1",
		Type:     "works_at",
	})
	if err == nil {
		t.Error("expected error when adding edge with non-existent source")
	}
}

func TestKnowledgeGraph_GetNodesByType(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John", Type: "person"})
	kg.AddNode(&KGNode{ID: "p2", CanonicalName: "Jane", Type: "person"})
	kg.AddNode(&KGNode{ID: "o1", CanonicalName: "Google", Type: "organization"})

	people := kg.GetNodesByType("person")
	if len(people) != 2 {
		t.Errorf("expected 2 person nodes, got %d", len(people))
	}

	orgs := kg.GetNodesByType("organization")
	if len(orgs) != 1 {
		t.Errorf("expected 1 organization node, got %d", len(orgs))
	}

	// Case insensitive
	people2 := kg.GetNodesByType("PERSON")
	if len(people2) != 2 {
		t.Errorf("expected type matching to be case-insensitive, got %d nodes", len(people2))
	}
}

func TestKnowledgeGraph_GetNodesByName(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John Smith", Type: "person"})
	kg.AddNode(&KGNode{ID: "p2", CanonicalName: "John Smith", Type: "person"}) // Allow duplicate names

	nodes := kg.GetNodesByName("John Smith")
	if len(nodes) != 2 {
		t.Errorf("expected 2 nodes with name 'John Smith', got %d", len(nodes))
	}

	// Case insensitive
	nodes2 := kg.GetNodesByName("john smith")
	if len(nodes2) != 2 {
		t.Errorf("expected name matching to be case-insensitive, got %d nodes", len(nodes2))
	}
}

func TestKnowledgeGraph_FindNodesByMention(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{
		ID:            "p1",
		CanonicalName: "John Smith",
		Type:          "person",
		Mentions:      []string{"John", "J. Smith", "Mr. Smith"},
	})

	// Find by mention
	nodes := kg.FindNodesByMention("John")
	if len(nodes) != 1 {
		t.Errorf("expected 1 node with mention 'John', got %d", len(nodes))
	}

	nodes = kg.FindNodesByMention("Mr. Smith")
	if len(nodes) != 1 {
		t.Errorf("expected 1 node with mention 'Mr. Smith', got %d", len(nodes))
	}

	nodes = kg.FindNodesByMention("Unknown")
	if len(nodes) != 0 {
		t.Errorf("expected 0 nodes with mention 'Unknown', got %d", len(nodes))
	}
}

func TestKnowledgeGraph_GetOutgoingAndIncomingEdges(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "Elon Musk", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Tesla", Type: "organization"})
	kg.AddNode(&KGNode{ID: "c2", CanonicalName: "SpaceX", Type: "organization"})

	kg.AddEdge(&KGEdge{ID: "e1", SourceID: "p1", TargetID: "c1", Type: "ceo_of"})
	kg.AddEdge(&KGEdge{ID: "e2", SourceID: "p1", TargetID: "c2", Type: "founded"})

	outgoing := kg.GetOutgoingEdges("p1")
	if len(outgoing) != 2 {
		t.Errorf("expected 2 outgoing edges from p1, got %d", len(outgoing))
	}

	incoming := kg.GetIncomingEdges("c1")
	if len(incoming) != 1 {
		t.Errorf("expected 1 incoming edge to c1, got %d", len(incoming))
	}
}

func TestKnowledgeGraph_GetNeighbors(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "Elon Musk", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Tesla", Type: "organization"})
	kg.AddNode(&KGNode{ID: "c2", CanonicalName: "SpaceX", Type: "organization"})
	kg.AddNode(&KGNode{ID: "l1", CanonicalName: "California", Type: "location"})

	kg.AddEdge(&KGEdge{ID: "e1", SourceID: "p1", TargetID: "c1", Type: "ceo_of"})
	kg.AddEdge(&KGEdge{ID: "e2", SourceID: "p1", TargetID: "c2", Type: "founded"})
	kg.AddEdge(&KGEdge{ID: "e3", SourceID: "c1", TargetID: "l1", Type: "located_in"})

	neighbors := kg.GetNeighbors("p1")
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors for p1, got %d", len(neighbors))
	}

	neighbors = kg.GetNeighbors("c1")
	if len(neighbors) != 2 { // p1 (incoming) and l1 (outgoing)
		t.Errorf("expected 2 neighbors for c1, got %d", len(neighbors))
	}
}

func TestKnowledgeGraph_RemoveNode(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Google", Type: "organization"})
	kg.AddEdge(&KGEdge{ID: "e1", SourceID: "p1", TargetID: "c1", Type: "works_at"})

	// Remove node should also remove connected edges
	err := kg.RemoveNode("p1")
	if err != nil {
		t.Fatalf("RemoveNode failed: %v", err)
	}

	if kg.NodeCount() != 1 {
		t.Errorf("expected 1 node after removal, got %d", kg.NodeCount())
	}

	if kg.EdgeCount() != 0 {
		t.Errorf("expected 0 edges after removing connected node, got %d", kg.EdgeCount())
	}
}

func TestKnowledgeGraph_Serialization(t *testing.T) {
	kg := NewKnowledgeGraphWithName("test-graph", "A test knowledge graph")

	kg.AddNode(&KGNode{
		ID:            "p1",
		CanonicalName: "John Smith",
		Type:          "person",
		Mentions:      []string{"John", "J. Smith"},
		Confidence:    0.95,
		Properties:    map[string]any{"age": 35},
	})
	kg.AddNode(&KGNode{
		ID:            "c1",
		CanonicalName: "Google",
		Type:          "organization",
		Confidence:    0.99,
	})
	kg.AddEdge(&KGEdge{
		ID:         "e1",
		SourceID:   "p1",
		TargetID:   "c1",
		Type:       "works_at",
		Confidence: 0.88,
	})

	// Serialize to JSON
	jsonData, err := kg.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}

	// Deserialize
	kg2, err := FromJSON(jsonData)
	if err != nil {
		t.Fatalf("FromJSON failed: %v", err)
	}

	// Verify
	if kg2.NodeCount() != 2 {
		t.Errorf("expected 2 nodes after deserialization, got %d", kg2.NodeCount())
	}
	if kg2.EdgeCount() != 1 {
		t.Errorf("expected 1 edge after deserialization, got %d", kg2.EdgeCount())
	}

	node := kg2.GetNode("p1")
	if node == nil {
		t.Fatal("node p1 not found after deserialization")
	}
	if node.CanonicalName != "John Smith" {
		t.Errorf("expected canonical name 'John Smith', got '%s'", node.CanonicalName)
	}
	if len(node.Mentions) != 2 {
		t.Errorf("expected 2 mentions, got %d", len(node.Mentions))
	}
}

func TestKnowledgeGraph_NodeTypes(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Google", Type: "organization"})
	kg.AddNode(&KGNode{ID: "l1", CanonicalName: "New York", Type: "location"})

	types := kg.NodeTypes()
	if len(types) != 3 {
		t.Errorf("expected 3 node types, got %d", len(types))
	}
}

func TestKnowledgeGraph_EdgeTypes(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Google", Type: "organization"})
	kg.AddNode(&KGNode{ID: "l1", CanonicalName: "New York", Type: "location"})

	kg.AddEdge(&KGEdge{ID: "e1", SourceID: "p1", TargetID: "c1", Type: "works_at"})
	kg.AddEdge(&KGEdge{ID: "e2", SourceID: "c1", TargetID: "l1", Type: "located_in"})

	types := kg.EdgeTypes()
	if len(types) != 2 {
		t.Errorf("expected 2 edge types, got %d", len(types))
	}
}

func TestKnowledgeGraph_GetEdgeBetween(t *testing.T) {
	kg := NewKnowledgeGraph()

	kg.AddNode(&KGNode{ID: "p1", CanonicalName: "John", Type: "person"})
	kg.AddNode(&KGNode{ID: "c1", CanonicalName: "Google", Type: "organization"})

	kg.AddEdge(&KGEdge{ID: "e1", SourceID: "p1", TargetID: "c1", Type: "works_at"})

	edge := kg.GetEdgeBetween("p1", "c1", "works_at")
	if edge == nil {
		t.Fatal("GetEdgeBetween returned nil for existing edge")
	}
	if edge.ID != "e1" {
		t.Errorf("expected edge ID 'e1', got '%s'", edge.ID)
	}

	// Non-existent edge
	edge = kg.GetEdgeBetween("p1", "c1", "ceo_of")
	if edge != nil {
		t.Error("GetEdgeBetween should return nil for non-existent edge type")
	}
}

func TestKGBuilder_BuildSimple(t *testing.T) {
	entities := []Entity{
		{Text: "John Smith", Label: "person", Start: 0, End: 10, Score: 0.95},
		{Text: "Google", Label: "organization", Start: 20, End: 26, Score: 0.98},
	}

	relations := []Relation{
		{
			HeadEntity: entities[0],
			TailEntity: entities[1],
			Label:      "works_at",
			Score:      0.88,
		},
	}

	kg := BuildKnowledgeGraph(entities, relations)

	if kg.NodeCount() != 2 {
		t.Errorf("expected 2 nodes, got %d", kg.NodeCount())
	}
	if kg.EdgeCount() != 1 {
		t.Errorf("expected 1 edge, got %d", kg.EdgeCount())
	}

	// Verify node types
	people := kg.GetNodesByType("person")
	if len(people) != 1 {
		t.Errorf("expected 1 person node, got %d", len(people))
	}
	if people[0].CanonicalName != "John Smith" {
		t.Errorf("expected canonical name 'John Smith', got '%s'", people[0].CanonicalName)
	}
}

func TestKGBuilder_EntityResolution(t *testing.T) {
	entities := []Entity{
		{Text: "John Smith", Label: "person", Start: 0, End: 10, Score: 0.95},
		{Text: "John", Label: "person", Start: 50, End: 54, Score: 0.85},       // Should merge
		{Text: "J. Smith", Label: "person", Start: 100, End: 108, Score: 0.80}, // Should merge
		{Text: "Google", Label: "organization", Start: 20, End: 26, Score: 0.98},
	}

	config := DefaultKGBuilderConfig()
	config.EntityResolver.SimilarityThreshold = 0.7 // Lower threshold for testing

	kg := BuildKnowledgeGraphWithConfig(entities, nil, config)

	// Should have merged similar person entities
	people := kg.GetNodesByType("person")
	if len(people) != 1 {
		t.Errorf("expected 1 person node after merging, got %d", len(people))
	}

	if len(people) > 0 {
		// Check mentions were collected
		if len(people[0].Mentions) < 2 {
			t.Errorf("expected multiple mentions after merge, got %d", len(people[0].Mentions))
		}
	}
}

func TestKGBuilder_MultipleDocuments(t *testing.T) {
	config := DefaultKGBuilderConfig()
	builder := NewKGBuilder(config)

	// First document
	extraction1 := ExtractionInput{
		DocumentID: "doc1",
		Entities: []Entity{
			{Text: "Elon Musk", Label: "person", Start: 0, End: 9, Score: 0.95},
			{Text: "SpaceX", Label: "organization", Start: 20, End: 26, Score: 0.98},
		},
		Relations: []Relation{
			{
				HeadEntity: Entity{Text: "Elon Musk", Label: "person"},
				TailEntity: Entity{Text: "SpaceX", Label: "organization"},
				Label:      "founded",
				Score:      0.9,
			},
		},
		ExtractorModel: "gliner_small",
		ExtractionTime: time.Now(),
	}

	// Second document
	extraction2 := ExtractionInput{
		DocumentID: "doc2",
		Entities: []Entity{
			{Text: "Elon Musk", Label: "person", Start: 0, End: 9, Score: 0.92}, // Same person
			{Text: "Tesla", Label: "organization", Start: 30, End: 35, Score: 0.97},
		},
		Relations: []Relation{
			{
				HeadEntity: Entity{Text: "Elon Musk", Label: "person"},
				TailEntity: Entity{Text: "Tesla", Label: "organization"},
				Label:      "ceo_of",
				Score:      0.88,
			},
		},
		ExtractorModel: "gliner_small",
		ExtractionTime: time.Now(),
	}

	builder.AddExtraction(extraction1)
	builder.AddExtraction(extraction2)

	kg := builder.Graph()

	// Should have 3 nodes (Elon Musk merged, SpaceX, Tesla)
	if kg.NodeCount() != 3 {
		t.Errorf("expected 3 nodes, got %d", kg.NodeCount())
	}

	// Should have 2 edges
	if kg.EdgeCount() != 2 {
		t.Errorf("expected 2 edges, got %d", kg.EdgeCount())
	}

	// Check Elon Musk has multiple provenance records
	people := kg.GetNodesByType("person")
	if len(people) != 1 {
		t.Errorf("expected 1 person node, got %d", len(people))
	} else if len(people[0].Provenance) < 2 {
		t.Errorf("expected multiple provenance records for merged entity, got %d", len(people[0].Provenance))
	}

	// Check metadata
	meta := kg.Metadata()
	if meta.DocumentCount != 2 {
		t.Errorf("expected document count 2, got %d", meta.DocumentCount)
	}
}

func TestKGBuilder_ConfidenceStrategies(t *testing.T) {
	entities := []Entity{
		{Text: "John Smith", Label: "person", Start: 0, End: 10, Score: 0.95},
		{Text: "John Smith", Label: "person", Start: 50, End: 60, Score: 0.75}, // Same entity, lower score
	}

	// Test MaxConfidence strategy
	config := DefaultKGBuilderConfig()
	config.EntityResolver.MergeConfidenceStrategy = MaxConfidence
	kg := BuildKnowledgeGraphWithConfig(entities, nil, config)
	people := kg.GetNodesByType("person")
	if len(people) != 1 {
		t.Fatalf("expected 1 person node, got %d", len(people))
	}
	if people[0].Confidence != 0.95 {
		t.Errorf("expected max confidence 0.95, got %f", people[0].Confidence)
	}
}

func TestJaroWinklerSimilarity(t *testing.T) {
	tests := []struct {
		s1, s2   string
		expected float32
	}{
		{"John", "John", 1.0},
		{"John", "john", 1.0}, // Case insensitive by default
		{"", "", 0.0},
		{"John", "", 0.0},
		{"Martha", "Marhta", 0.96}, // Classic Jaro-Winkler example (approx)
		{"John Smith", "J. Smith", 0.85},
	}

	for _, tt := range tests {
		result := jaroWinklerSimilarity(tt.s1, tt.s2, false)
		// Allow some tolerance for floating point
		if result < tt.expected-0.1 || result > tt.expected+0.1 {
			t.Errorf("jaroWinklerSimilarity(%q, %q) = %f, expected ~%f", tt.s1, tt.s2, result, tt.expected)
		}
	}
}

func TestTokenOverlapSimilarity(t *testing.T) {
	tests := []struct {
		s1, s2   string
		expected float32
	}{
		{"hello world", "hello world", 1.0},
		{"hello world", "world hello", 1.0},
		{"hello", "world", 0.0},
		{"John Smith", "John", 0.5}, // 1 common / 2 total
	}

	for _, tt := range tests {
		result := tokenOverlapSimilarity(tt.s1, tt.s2, false)
		if result < tt.expected-0.01 || result > tt.expected+0.01 {
			t.Errorf("tokenOverlapSimilarity(%q, %q) = %f, expected %f", tt.s1, tt.s2, result, tt.expected)
		}
	}
}

func TestKGExport_JSON(t *testing.T) {
	export := KGExport{
		Metadata: KGMetadata{
			Name:      "test",
			NodeCount: 1,
			EdgeCount: 0,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		},
		Nodes: []*KGNode{
			{
				ID:            "n1",
				CanonicalName: "Test",
				Type:          "test",
				Confidence:    0.9,
			},
		},
		Edges: []*KGEdge{},
	}

	data, err := json.Marshal(export)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var export2 KGExport
	if err := json.Unmarshal(data, &export2); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if export2.Metadata.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", export2.Metadata.Name)
	}
}
