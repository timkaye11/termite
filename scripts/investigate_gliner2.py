#!/usr/bin/env python3
"""
GLiNER2 Model Investigation Script

This script analyzes the GLiNER2 model structure to understand:
1. What components are available for each task (NER, relations, classification)
2. How labels are encoded for different tasks
3. The input/output format for each task type

Usage:
    python investigate_gliner2.py [model_id]
    python investigate_gliner2.py fastino/gliner2-base-v1
"""

import argparse
import json
import sys
from typing import Any, Dict, List


def analyze_model_structure(model) -> Dict[str, Any]:
    """Analyze the GLiNER2 model structure."""
    structure = {
        "type": type(model).__name__,
        "attributes": {},
        "methods": [],
        "modules": {},
        "config": {},
    }

    # Get all attributes
    for attr in dir(model):
        if attr.startswith('_'):
            continue
        try:
            obj = getattr(model, attr)
            if callable(obj):
                structure["methods"].append(attr)
            elif hasattr(obj, 'parameters'):
                # It's a PyTorch module
                structure["modules"][attr] = {
                    "type": type(obj).__name__,
                    "params": sum(p.numel() for p in obj.parameters()),
                }
            else:
                structure["attributes"][attr] = str(type(obj).__name__)
        except Exception as e:
            structure["attributes"][attr] = f"Error: {e}"

    return structure


def analyze_encoder(model) -> Dict[str, Any]:
    """Analyze the encoder component."""
    if not hasattr(model, 'encoder'):
        return {"error": "No encoder found"}

    encoder = model.encoder
    info = {
        "type": type(encoder).__name__,
        "has_config": hasattr(encoder, 'config'),
    }

    if hasattr(encoder, 'config'):
        config = encoder.config
        info["config"] = {
            "hidden_size": getattr(config, 'hidden_size', None),
            "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
            "num_attention_heads": getattr(config, 'num_attention_heads', None),
            "vocab_size": getattr(config, 'vocab_size', None),
            "model_type": getattr(config, 'model_type', None),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
        }

    return info


def analyze_span_rep(model) -> Dict[str, Any]:
    """Analyze the span representation component."""
    if not hasattr(model, 'span_rep'):
        return {"error": "No span_rep found"}

    span_rep = model.span_rep
    info = {
        "type": type(span_rep).__name__,
        "submodules": {},
    }

    for name, module in span_rep.named_modules():
        if name:
            info["submodules"][name] = type(module).__name__

    # Check for specific layers
    if hasattr(span_rep, 'span_rep_layer'):
        layer = span_rep.span_rep_layer
        info["span_rep_layer"] = {
            "type": type(layer).__name__,
            "has_project_start": hasattr(layer, 'project_start'),
            "has_project_end": hasattr(layer, 'project_end'),
            "has_out_project": hasattr(layer, 'out_project'),
        }

        # Get projection dimensions
        if hasattr(layer, 'project_start'):
            proj = layer.project_start
            if hasattr(proj, 'weight'):
                info["span_rep_layer"]["project_start_shape"] = list(proj.weight.shape)

    return info


def analyze_classifier(model) -> Dict[str, Any]:
    """Analyze the classifier head."""
    if not hasattr(model, 'classifier'):
        return {"error": "No classifier found"}

    classifier = model.classifier
    info = {
        "type": type(classifier).__name__,
        "submodules": {},
    }

    for name, module in classifier.named_modules():
        if name:
            info["submodules"][name] = {
                "type": type(module).__name__,
            }
            if hasattr(module, 'weight'):
                info["submodules"][name]["weight_shape"] = list(module.weight.shape)

    return info


def analyze_relation_components(model) -> Dict[str, Any]:
    """Check for relation-specific components."""
    info = {
        "has_rel_classifier": hasattr(model, 'rel_classifier'),
        "has_rel_head": hasattr(model, 'rel_head'),
        "has_pair_scorer": hasattr(model, 'pair_scorer'),
        "has_biaffine": hasattr(model, 'biaffine'),
        "has_relation_rep": hasattr(model, 'relation_rep'),
    }

    # Check for relation-related attributes
    relation_attrs = []
    for attr in dir(model):
        if 'rel' in attr.lower() or 'pair' in attr.lower():
            relation_attrs.append(attr)
    info["relation_attributes"] = relation_attrs

    return info


def analyze_classification_components(model) -> Dict[str, Any]:
    """Check for classification-specific components."""
    info = {
        "has_cls_head": hasattr(model, 'cls_head'),
        "has_classification_head": hasattr(model, 'classification_head'),
        "has_pooler": hasattr(model, 'pooler'),
    }

    # Check for classification-related attributes
    cls_attrs = []
    for attr in dir(model):
        if 'cls' in attr.lower() or 'class' in attr.lower() or 'pool' in attr.lower():
            if not attr.startswith('_'):
                cls_attrs.append(attr)
    info["classification_attributes"] = cls_attrs

    return info


def analyze_tokenizer(model) -> Dict[str, Any]:
    """Analyze the tokenizer/processor."""
    info = {}

    if hasattr(model, 'processor'):
        processor = model.processor
        info["processor_type"] = type(processor).__name__

        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            info["tokenizer_type"] = type(tokenizer).__name__

            # Check for special tokens
            if hasattr(tokenizer, 'special_tokens_map'):
                info["special_tokens"] = tokenizer.special_tokens_map

            # Check for added tokens
            if hasattr(tokenizer, 'added_tokens_encoder'):
                info["added_tokens"] = list(tokenizer.added_tokens_encoder.keys())[:20]

    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        info["tokenizer_type"] = type(tokenizer).__name__

    return info


def test_inference(model, text: str, labels: List[str]) -> Dict[str, Any]:
    """Test inference with the model."""
    info = {}

    # Test entity extraction
    try:
        entities = model.extract_entities(text, labels)
        info["extract_entities"] = {
            "success": True,
            "output_type": type(entities).__name__,
            "sample_output": str(entities)[:500],
        }
    except Exception as e:
        info["extract_entities"] = {"success": False, "error": str(e)}

    # Test relation extraction
    try:
        relations = model.extract_relations(text, ["works_for", "located_in"])
        info["extract_relations"] = {
            "success": True,
            "output_type": type(relations).__name__,
            "sample_output": str(relations)[:500],
        }
    except Exception as e:
        info["extract_relations"] = {"success": False, "error": str(e)}

    # Test classification
    try:
        classification = model.classify_text(text, {"sentiment": ["positive", "negative", "neutral"]})
        info["classify_text"] = {
            "success": True,
            "output_type": type(classification).__name__,
            "sample_output": str(classification)[:500],
        }
    except Exception as e:
        info["classify_text"] = {"success": False, "error": str(e)}

    return info


def analyze_internal_forward(model) -> Dict[str, Any]:
    """Analyze the internal forward pass to understand data flow."""
    import torch
    import inspect

    info = {}

    # Get the forward method signature if available
    if hasattr(model, 'forward'):
        sig = inspect.signature(model.forward)
        info["forward_signature"] = str(sig)

    # Check for different processing methods
    processing_methods = [
        'encode', 'predict', 'extract', 'process',
        'get_span_representations', 'compute_logits',
    ]

    found_methods = {}
    for method in processing_methods:
        if hasattr(model, method):
            func = getattr(model, method)
            if callable(func):
                try:
                    sig = inspect.signature(func)
                    found_methods[method] = str(sig)
                except:
                    found_methods[method] = "callable"

    info["processing_methods"] = found_methods

    return info


def main():
    parser = argparse.ArgumentParser(description="Investigate GLiNER2 model structure")
    parser.add_argument("model_id", nargs="?", default="fastino/gliner2-base-v1",
                        help="HuggingFace model ID")
    parser.add_argument("--test", action="store_true",
                        help="Run inference tests")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"Loading GLiNER2 model: {args.model_id}")

    try:
        from gliner2 import GLiNER2
    except ImportError:
        print("ERROR: gliner2 package not installed. Run: pip install gliner2")
        sys.exit(1)

    model = GLiNER2.from_pretrained(args.model_id)

    print("Analyzing model structure...")

    results = {
        "model_id": args.model_id,
        "structure": analyze_model_structure(model),
        "encoder": analyze_encoder(model),
        "span_rep": analyze_span_rep(model),
        "classifier": analyze_classifier(model),
        "relation_components": analyze_relation_components(model),
        "classification_components": analyze_classification_components(model),
        "tokenizer": analyze_tokenizer(model),
        "internal_forward": analyze_internal_forward(model),
    }

    if args.test:
        print("Running inference tests...")
        results["inference_tests"] = test_inference(
            model,
            "John Smith works at Google in Mountain View. The company was founded by Larry Page.",
            ["person", "organization", "location"]
        )

    # Print results
    print("\n" + "=" * 80)
    print("GLiNER2 MODEL ANALYSIS")
    print("=" * 80)

    print("\n## Model Type")
    print(f"  Type: {results['structure']['type']}")

    print("\n## Key Attributes")
    for attr, typ in list(results['structure']['attributes'].items())[:20]:
        print(f"  {attr}: {typ}")

    print("\n## Modules (Trainable Components)")
    for name, info in results['structure']['modules'].items():
        print(f"  {name}: {info['type']} ({info['params']:,} params)")

    print("\n## Key Methods")
    for method in results['structure']['methods'][:30]:
        print(f"  - {method}")

    print("\n## Encoder")
    if "config" in results['encoder']:
        for k, v in results['encoder']['config'].items():
            print(f"  {k}: {v}")

    print("\n## Span Representation")
    if "span_rep_layer" in results['span_rep']:
        for k, v in results['span_rep']['span_rep_layer'].items():
            print(f"  {k}: {v}")

    print("\n## Relation Components")
    for k, v in results['relation_components'].items():
        print(f"  {k}: {v}")

    print("\n## Classification Components")
    for k, v in results['classification_components'].items():
        print(f"  {k}: {v}")

    print("\n## Tokenizer")
    for k, v in results['tokenizer'].items():
        if isinstance(v, list):
            print(f"  {k}: {v[:10]}...")
        else:
            print(f"  {k}: {v}")

    if args.test:
        print("\n## Inference Tests")
        for test_name, test_result in results['inference_tests'].items():
            status = "✓" if test_result.get("success") else "✗"
            print(f"  {status} {test_name}")
            if test_result.get("success"):
                print(f"      Output: {test_result.get('sample_output', '')[:200]}...")
            else:
                print(f"      Error: {test_result.get('error', 'Unknown')}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
