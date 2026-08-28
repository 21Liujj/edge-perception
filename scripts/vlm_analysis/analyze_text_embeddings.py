#!/usr/bin/env python3
"""Stage F: Text Embedding Geometry Mini-Study.

Extract real text embeddings from NanoOWL for selected categories + prompt variants,
compute cosine similarity matrix, analyze prompt sensitivity from embedding space.
"""
import sys
import os
import json
import numpy as np
import torch

# Add third_party paths
ROOT = "/home/nvidia/jetson-deploy-lab"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party/nanoowl"))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from nanoowl.owl_predictor import OwlPredictor


def main():
    # Selected 5 categories from E4 manifest, representative of E4 findings
    # person: high-freq, semantic grounding test
    # chair: common object, multi-word possible
    # hair drier: E2 showed low AP, compound word
    # bird: small object
    # car: common vehicle
    
    categories = ["person", "chair", "hair drier", "bird", "car"]
    
    # Prompt templates per category
    prompt_variants = {
        "person": [
            "person",
            "a person", 
            "photo of a person",
            "human"
        ],
        "chair": [
            "chair",
            "a chair",
            "photo of a chair",
            "seat"
        ],
        "hair drier": [
            "hair drier",
            "a hair drier",
            "photo of a hair drier",
            "hairdryer"
        ],
        "bird": [
            "bird",
            "a bird",
            "photo of a bird",
            "flying bird"
        ],
        "car": [
            "car",
            "a car",
            "photo of a car",
            "automobile"
        ]
    }
    
    print("Loading NanoOWL predictor (text-side only)...")
    model_name = "google/owlvit-base-patch32"
    predictor = OwlPredictor(model_name=model_name, device="cuda")
    
    print(f"Extracting text embeddings for {len(categories)} categories...")
    
    embeddings = {}
    all_prompts = []
    
    for cat in categories:
        prompts = prompt_variants[cat]
        with torch.no_grad():
            text_output = predictor.encode_text(prompts)
        text_embeds = text_output.text_embeds.cpu().numpy()
        
        embeddings[cat] = {
            "prompts": prompts,
            "embeddings": text_embeds,
            "shape": list(text_embeds.shape),
            "dtype": str(text_embeds.dtype)
        }
        
        for p in prompts:
            all_prompts.append({"category": cat, "prompt": p})
    
    # Compute global cosine similarity matrix (all prompts vs all prompts)
    all_embeds = np.concatenate([embeddings[cat]["embeddings"] for cat in categories], axis=0)
    
    # Normalize
    all_embeds_norm = all_embeds / (np.linalg.norm(all_embeds, axis=1, keepdims=True) + 1e-6)
    
    # Cosine similarity = normalized dot product
    cos_sim_matrix = all_embeds_norm @ all_embeds_norm.T
    
    # Write CSV
    output_dir = os.path.join(ROOT, "results/stage_f_vlm_algorithm")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "text_embedding_similarity.csv")
    with open(csv_path, "w") as f:
        # Header
        f.write("prompt_i,category_i,prompt_j,category_j,cosine_similarity\n")
        
        n = len(all_prompts)
        for i in range(n):
            for j in range(n):
                pi = all_prompts[i]
                pj = all_prompts[j]
                sim = float(cos_sim_matrix[i, j])
                f.write(f'"{pi["prompt"]}",{pi["category"]},"{pj["prompt"]}",{pj["category"]},{sim:.6f}\n')
    
    print(f"Wrote {csv_path}")
    
    # Compute within-category and cross-category statistics
    within_cat_sims = []
    cross_cat_sims = []
    
    for i in range(len(all_prompts)):
        for j in range(i+1, len(all_prompts)):
            sim = cos_sim_matrix[i, j]
            if all_prompts[i]["category"] == all_prompts[j]["category"]:
                within_cat_sims.append(sim)
            else:
                cross_cat_sims.append(sim)
    
    within_cat_sims = np.array(within_cat_sims)
    cross_cat_sims = np.array(cross_cat_sims)
    
    # Per-category intra-similarity
    per_category_stats = {}
    offset = 0
    for cat in categories:
        n_variants = len(prompt_variants[cat])
        cat_embeds = all_embeds_norm[offset:offset+n_variants]
        cat_sim_matrix = cat_embeds @ cat_embeds.T
        
        # Extract upper triangle (excluding diagonal)
        intra_sims = []
        for i in range(n_variants):
            for j in range(i+1, n_variants):
                intra_sims.append(cat_sim_matrix[i, j])
        
        intra_sims = np.array(intra_sims)
        
        per_category_stats[cat] = {
            "num_variants": n_variants,
            "mean_intra_similarity": float(intra_sims.mean()) if len(intra_sims) > 0 else 1.0,
            "std_intra_similarity": float(intra_sims.std()) if len(intra_sims) > 0 else 0.0,
            "min_intra_similarity": float(intra_sims.min()) if len(intra_sims) > 0 else 1.0,
            "max_intra_similarity": float(intra_sims.max()) if len(intra_sims) > 0 else 1.0,
        }
        
        offset += n_variants
    
    summary = {
        "categories": categories,
        "total_prompts": len(all_prompts),
        "embedding_shape": list(all_embeds.shape),
        "embedding_dim": int(all_embeds.shape[1]),
        "within_category_similarity": {
            "mean": float(within_cat_sims.mean()),
            "std": float(within_cat_sims.std()),
            "min": float(within_cat_sims.min()),
            "max": float(within_cat_sims.max()),
            "count": int(len(within_cat_sims))
        },
        "cross_category_similarity": {
            "mean": float(cross_cat_sims.mean()),
            "std": float(cross_cat_sims.std()),
            "min": float(cross_cat_sims.min()),
            "max": float(cross_cat_sims.max()),
            "count": int(len(cross_cat_sims))
        },
        "per_category": per_category_stats,
        "interpretation": (
            "High within-category similarity (near 1.0) indicates prompt template variations "
            "produce nearly identical embeddings, making detection robust to wording. "
            "Lower within-category similarity suggests prompt sensitivity. "
            "Cross-category similarity should be lower than within-category for good discrimination."
        )
    }
    
    json_path = os.path.join(output_dir, "text_embedding_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Wrote {json_path}")
    print(f"\nSummary:")
    print(f"  Within-category mean similarity: {summary['within_category_similarity']['mean']:.4f}")
    print(f"  Cross-category mean similarity:  {summary['cross_category_similarity']['mean']:.4f}")
    print(f"  Embedding dimension: {summary['embedding_dim']}")


if __name__ == "__main__":
    main()
