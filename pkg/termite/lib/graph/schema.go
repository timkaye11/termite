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

// Package graph provides schema types and validation for Knowledge Graph generation.
package graph

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// =============================================================================
// Schema Types
// =============================================================================

// SchemaDefinition defines the valid node types, edge types, and constraints
// for a knowledge graph. It provides a declarative way to specify what entity
// and relation types are allowed, and how they can be connected.
type SchemaDefinition struct {
	// NodeTypes is the list of valid entity types (e.g., "Person", "Company", "Location").
	// Node type matching is case-insensitive.
	NodeTypes []string `json:"node_types"`

	// EdgeTypes is the list of valid relation types (e.g., "WORKS_FOR", "ACQUIRED", "LOCATED_IN").
	// Edge type matching is case-insensitive.
	EdgeTypes []string `json:"edge_types"`

	// Constraints defines optional rules for valid triples.
	// If empty, any combination of valid node and edge types is allowed.
	// If specified, only triples matching a constraint are valid.
	Constraints []SchemaConstraint `json:"constraints,omitempty"`
}

// SchemaConstraint defines a valid triple pattern: source_type -> relation -> target_type.
// For example, "Person" -> "WORKS_FOR" -> "Company" means a Person node can have
// a WORKS_FOR edge pointing to a Company node.
type SchemaConstraint struct {
	// SourceType is the required type for the source (subject) node
	SourceType string `json:"source_type"`

	// RelationType is the required type for the edge (predicate)
	RelationType string `json:"relation_type"`

	// TargetType is the required type for the target (object) node
	TargetType string `json:"target_type"`
}

// =============================================================================
// Rejected Triple
// =============================================================================

// RejectedTriple represents a triple that failed schema validation,
// along with the reason for rejection.
type RejectedTriple struct {
	// SubjectType is the entity type of the subject (source) node
	SubjectType string `json:"subject_type"`

	// SubjectText is the text of the subject entity
	SubjectText string `json:"subject_text"`

	// Relation is the relation type (predicate)
	Relation string `json:"relation"`

	// ObjectType is the entity type of the object (target) node
	ObjectType string `json:"object_type"`

	// ObjectText is the text of the object entity
	ObjectText string `json:"object_text"`

	// Reason explains why the triple was rejected
	Reason string `json:"reason"`
}

// RejectionReason constants for common rejection types
const (
	ReasonInvalidSourceType   = "invalid source entity type"
	ReasonInvalidTargetType   = "invalid target entity type"
	ReasonInvalidRelationType = "invalid relation type"
	ReasonConstraintViolation = "constraint violation: triple pattern not allowed"
)

// =============================================================================
// Schema Validator
// =============================================================================

// SchemaValidator validates entities and relations against a schema definition.
// It is safe for concurrent use.
type SchemaValidator struct {
	schema *SchemaDefinition

	// Normalized lookup sets for O(1) validation
	nodeTypes map[string]struct{}
	edgeTypes map[string]struct{}

	// Constraint lookup: "sourceType|relationType|targetType" -> true
	// Only populated if constraints are defined
	constraints map[string]struct{}

	// Whether constraints are enforced (true if schema has constraints)
	hasConstraints bool

	mu sync.RWMutex
}

// NewSchemaValidator creates a new validator for the given schema.
// The schema must have at least one node type and one edge type.
func NewSchemaValidator(schema *SchemaDefinition) (*SchemaValidator, error) {
	if schema == nil {
		return nil, fmt.Errorf("schema cannot be nil")
	}
	if len(schema.NodeTypes) == 0 {
		return nil, fmt.Errorf("schema must define at least one node type")
	}
	if len(schema.EdgeTypes) == 0 {
		return nil, fmt.Errorf("schema must define at least one edge type")
	}

	v := &SchemaValidator{
		schema:         schema,
		nodeTypes:      make(map[string]struct{}, len(schema.NodeTypes)),
		edgeTypes:      make(map[string]struct{}, len(schema.EdgeTypes)),
		constraints:    make(map[string]struct{}),
		hasConstraints: len(schema.Constraints) > 0,
	}

	// Build normalized node type lookup
	for _, nt := range schema.NodeTypes {
		v.nodeTypes[normalizeType(nt)] = struct{}{}
	}

	// Build normalized edge type lookup
	for _, et := range schema.EdgeTypes {
		v.edgeTypes[normalizeType(et)] = struct{}{}
	}

	// Build constraint lookup if constraints are defined
	for _, c := range schema.Constraints {
		key := constraintKey(c.SourceType, c.RelationType, c.TargetType)
		v.constraints[key] = struct{}{}
	}

	return v, nil
}

// IsValidNodeType checks if the given entity type is allowed by the schema.
func (v *SchemaValidator) IsValidNodeType(nodeType string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()

	_, ok := v.nodeTypes[normalizeType(nodeType)]
	return ok
}

// IsValidEdgeType checks if the given relation type is allowed by the schema.
func (v *SchemaValidator) IsValidEdgeType(edgeType string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()

	_, ok := v.edgeTypes[normalizeType(edgeType)]
	return ok
}

// IsValidTriple checks if a triple (subject_type, relation, object_type) is valid
// according to the schema. Returns true if the triple is valid, false otherwise.
// If the schema has no constraints, any combination of valid types is allowed.
func (v *SchemaValidator) IsValidTriple(subjectType, relation, objectType string) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()

	// First check if all types are valid
	if _, ok := v.nodeTypes[normalizeType(subjectType)]; !ok {
		return false
	}
	if _, ok := v.edgeTypes[normalizeType(relation)]; !ok {
		return false
	}
	if _, ok := v.nodeTypes[normalizeType(objectType)]; !ok {
		return false
	}

	// If no constraints defined, any valid type combination is allowed
	if !v.hasConstraints {
		return true
	}

	// Check against constraints
	key := constraintKey(subjectType, relation, objectType)
	_, ok := v.constraints[key]
	return ok
}

// ValidateTriple validates a triple and returns detailed rejection information
// if the triple is invalid. Returns nil if the triple is valid.
func (v *SchemaValidator) ValidateTriple(subjectType, subjectText, relation, objectType, objectText string) *RejectedTriple {
	v.mu.RLock()
	defer v.mu.RUnlock()

	// Check subject type
	if _, ok := v.nodeTypes[normalizeType(subjectType)]; !ok {
		return &RejectedTriple{
			SubjectType: subjectType,
			SubjectText: subjectText,
			Relation:    relation,
			ObjectType:  objectType,
			ObjectText:  objectText,
			Reason:      ReasonInvalidSourceType,
		}
	}

	// Check relation type
	if _, ok := v.edgeTypes[normalizeType(relation)]; !ok {
		return &RejectedTriple{
			SubjectType: subjectType,
			SubjectText: subjectText,
			Relation:    relation,
			ObjectType:  objectType,
			ObjectText:  objectText,
			Reason:      ReasonInvalidRelationType,
		}
	}

	// Check object type
	if _, ok := v.nodeTypes[normalizeType(objectType)]; !ok {
		return &RejectedTriple{
			SubjectType: subjectType,
			SubjectText: subjectText,
			Relation:    relation,
			ObjectType:  objectType,
			ObjectText:  objectText,
			Reason:      ReasonInvalidTargetType,
		}
	}

	// If no constraints, triple is valid
	if !v.hasConstraints {
		return nil
	}

	// Check constraint
	key := constraintKey(subjectType, relation, objectType)
	if _, ok := v.constraints[key]; !ok {
		return &RejectedTriple{
			SubjectType: subjectType,
			SubjectText: subjectText,
			Relation:    relation,
			ObjectType:  objectType,
			ObjectText:  objectText,
			Reason:      ReasonConstraintViolation,
		}
	}

	return nil
}

// Schema returns the underlying schema definition.
func (v *SchemaValidator) Schema() *SchemaDefinition {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.schema
}

// NodeTypes returns all valid node types from the schema.
func (v *SchemaValidator) NodeTypes() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	result := make([]string, len(v.schema.NodeTypes))
	copy(result, v.schema.NodeTypes)
	return result
}

// EdgeTypes returns all valid edge types from the schema.
func (v *SchemaValidator) EdgeTypes() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	result := make([]string, len(v.schema.EdgeTypes))
	copy(result, v.schema.EdgeTypes)
	return result
}

// HasConstraints returns true if the schema has explicit constraints defined.
func (v *SchemaValidator) HasConstraints() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.hasConstraints
}

// =============================================================================
// Helper Functions
// =============================================================================

// LoadSchemaFromJSON parses a schema definition from JSON data.
func LoadSchemaFromJSON(data []byte) (*SchemaDefinition, error) {
	var schema SchemaDefinition
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("parsing schema JSON: %w", err)
	}

	// Validate the schema has required fields
	if len(schema.NodeTypes) == 0 {
		return nil, fmt.Errorf("schema must define at least one node type")
	}
	if len(schema.EdgeTypes) == 0 {
		return nil, fmt.Errorf("schema must define at least one edge type")
	}

	// Validate constraints reference valid types
	nodeTypeSet := make(map[string]struct{}, len(schema.NodeTypes))
	for _, nt := range schema.NodeTypes {
		nodeTypeSet[normalizeType(nt)] = struct{}{}
	}
	edgeTypeSet := make(map[string]struct{}, len(schema.EdgeTypes))
	for _, et := range schema.EdgeTypes {
		edgeTypeSet[normalizeType(et)] = struct{}{}
	}

	for i, c := range schema.Constraints {
		if _, ok := nodeTypeSet[normalizeType(c.SourceType)]; !ok {
			return nil, fmt.Errorf("constraint %d references unknown source type: %s", i, c.SourceType)
		}
		if _, ok := edgeTypeSet[normalizeType(c.RelationType)]; !ok {
			return nil, fmt.Errorf("constraint %d references unknown relation type: %s", i, c.RelationType)
		}
		if _, ok := nodeTypeSet[normalizeType(c.TargetType)]; !ok {
			return nil, fmt.Errorf("constraint %d references unknown target type: %s", i, c.TargetType)
		}
	}

	return &schema, nil
}

// normalizeType normalizes a type string for case-insensitive matching.
func normalizeType(t string) string {
	return strings.ToLower(strings.TrimSpace(t))
}

// constraintKey creates a normalized lookup key for a constraint.
func constraintKey(sourceType, relationType, targetType string) string {
	return normalizeType(sourceType) + "|" + normalizeType(relationType) + "|" + normalizeType(targetType)
}
