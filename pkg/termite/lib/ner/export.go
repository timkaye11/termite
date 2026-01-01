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
	"github.com/antflydb/termite/pkg/termite/lib/graph"
)

// =============================================================================
// Export Adapters for ner.KnowledgeGraph
// =============================================================================

// KGNodeAdapter adapts a KGNode to implement graph.KGNodeData.
type KGNodeAdapter struct {
	*KGNode
}

// GetID returns the unique identifier for this node.
func (a *KGNodeAdapter) GetID() string { return a.ID }

// GetCanonicalName returns the primary/preferred name for this entity.
func (a *KGNodeAdapter) GetCanonicalName() string { return a.CanonicalName }

// GetType returns the entity type.
func (a *KGNodeAdapter) GetType() string { return a.Type }

// GetMentions returns all surface forms that refer to this entity.
func (a *KGNodeAdapter) GetMentions() []string { return a.Mentions }

// GetProperties returns arbitrary key-value attributes.
func (a *KGNodeAdapter) GetProperties() map[string]any { return a.Properties }

// GetConfidence returns the aggregated confidence score (0.0-1.0).
func (a *KGNodeAdapter) GetConfidence() float32 { return a.Confidence }

// GetProvenance returns the provenance records for this node.
func (a *KGNodeAdapter) GetProvenance() []graph.ProvenanceData {
	result := make([]graph.ProvenanceData, len(a.Provenance))
	for i := range a.Provenance {
		result[i] = &ProvenanceAdapter{&a.Provenance[i]}
	}
	return result
}

// KGEdgeAdapter adapts a KGEdge to implement graph.KGEdgeData.
type KGEdgeAdapter struct {
	*KGEdge
}

// GetID returns the unique identifier for this edge.
func (a *KGEdgeAdapter) GetID() string { return a.ID }

// GetSourceID returns the ID of the source node.
func (a *KGEdgeAdapter) GetSourceID() string { return a.SourceID }

// GetTargetID returns the ID of the target node.
func (a *KGEdgeAdapter) GetTargetID() string { return a.TargetID }

// GetType returns the relationship type.
func (a *KGEdgeAdapter) GetType() string { return a.Type }

// GetProperties returns arbitrary key-value attributes.
func (a *KGEdgeAdapter) GetProperties() map[string]any { return a.Properties }

// GetConfidence returns the confidence score (0.0-1.0).
func (a *KGEdgeAdapter) GetConfidence() float32 { return a.Confidence }

// GetProvenance returns the provenance records for this edge.
func (a *KGEdgeAdapter) GetProvenance() []graph.ProvenanceData {
	result := make([]graph.ProvenanceData, len(a.Provenance))
	for i := range a.Provenance {
		result[i] = &ProvenanceAdapter{&a.Provenance[i]}
	}
	return result
}

// ProvenanceAdapter adapts a Provenance to implement graph.ProvenanceData.
type ProvenanceAdapter struct {
	*Provenance
}

// GetSourceDocument returns the document ID or path.
func (a *ProvenanceAdapter) GetSourceDocument() string { return a.SourceDocument }

// GetSourceURL returns the URL of the source.
func (a *ProvenanceAdapter) GetSourceURL() string { return a.SourceURL }

// GetSourceText returns the original text span.
func (a *ProvenanceAdapter) GetSourceText() string { return a.SourceText }

// GetExtractorModel returns the model used for extraction.
func (a *ProvenanceAdapter) GetExtractorModel() string { return a.ExtractorModel }

// KnowledgeGraphAdapter adapts a KnowledgeGraph to implement graph.KnowledgeGraphData.
type KnowledgeGraphAdapter struct {
	*KnowledgeGraph
}

// Nodes returns all nodes in the graph as graph.KGNodeData.
func (a *KnowledgeGraphAdapter) Nodes() []graph.KGNodeData {
	nodes := a.KnowledgeGraph.Nodes()
	result := make([]graph.KGNodeData, len(nodes))
	for i, node := range nodes {
		result[i] = &KGNodeAdapter{node}
	}
	return result
}

// Edges returns all edges in the graph as graph.KGEdgeData.
func (a *KnowledgeGraphAdapter) Edges() []graph.KGEdgeData {
	edges := a.KnowledgeGraph.Edges()
	result := make([]graph.KGEdgeData, len(edges))
	for i, edge := range edges {
		result[i] = &KGEdgeAdapter{edge}
	}
	return result
}

// AsExportable returns a KnowledgeGraph wrapped in an adapter that implements
// graph.KnowledgeGraphData, allowing it to be used with graph.Exporter implementations.
//
// Example usage:
//
//	kg := ner.NewKnowledgeGraph()
//	// ... populate the graph ...
//
//	exporter := graph.NewCypherExporter(nil)
//	cypher, err := exporter.Export(kg.AsExportable())
func (kg *KnowledgeGraph) AsExportable() graph.KnowledgeGraphData {
	return &KnowledgeGraphAdapter{kg}
}

// ExportToCypher is a convenience method that exports the knowledge graph to Cypher format.
// This is equivalent to creating a CypherExporter and calling Export on the graph.
//
// Example usage:
//
//	kg := ner.NewKnowledgeGraph()
//	// ... populate the graph ...
//
//	cypher, err := kg.ExportToCypher(nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(string(cypher))
func (kg *KnowledgeGraph) ExportToCypher(opts *graph.CypherExportOptions) ([]byte, error) {
	exporter := graph.NewCypherExporter(opts)
	return exporter.Export(kg.AsExportable())
}
