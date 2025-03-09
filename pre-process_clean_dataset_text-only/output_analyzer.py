import os
import re
import json
import argparse
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from collections import Counter

def analyze_processed_file(file_path: str) -> Dict[str, Any]:
    """Analyze the quality of a processed file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic statistics
    stats = {
        "file_path": file_path,
        "total_chars": len(content),
        "total_words": len(content.split()),
        "total_lines": len(content.splitlines()),
        "total_paragraphs": len(re.split(r'\n\n+', content))
    }
    
    # Analyze markdown structure
    headings = {
        "h1": len(re.findall(r'^# ', content, re.MULTILINE)),
        "h2": len(re.findall(r'^## ', content, re.MULTILINE)),
        "h3": len(re.findall(r'^### ', content, re.MULTILINE))
    }
    stats["headings"] = headings
    
    # Check for Sanskrit terms
    sanskrit_terms = re.findall(r'\b\w*[āīūṛṝḷḹṃḥṅñṭḍṇśṣ]\w*\b', content)
    stats["sanskrit_term_count"] = len(sanskrit_terms)
    stats["common_sanskrit_terms"] = dict(Counter(sanskrit_terms).most_common(10))
    
    # Check for tables
    stats["table_count"] = content.count('|---')
    
    # Check for remaining OCR artifacts
    image_refs = re.findall(r'!\[\]\([^)]+\)', content)
    page_refs = re.findall(r'_page_\d+_', content)
    stats["potential_artifacts"] = {
        "image_references": len(image_refs),
        "page_references": len(page_refs)
    }
    
    # Check for name replacements
    nithyananda_mentions = re.findall(r'THE SUPREME PONTIFF OF HINDUISM BHAGAWAN SRI NITHYANANDA PARAMASHIVAM', content)
    stats["name_replacements"] = len(nithyananda_mentions)
    
    return stats

def analyze_dataset_file(file_path: str) -> Dict[str, Any]:
    """Analyze a JSONL dataset file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError:
                continue
    
    # Calculate statistics
    stats = {
        "file_path": file_path,
        "total_examples": len(examples),
        "avg_example_length": sum(len(ex.get("text", "")) for ex in examples) / len(examples) if examples else 0
    }
    
    # Length distribution
    length_distribution = {}
    for ex in examples:
        text = ex.get("text", "")
        words = len(text.split())
        bucket = (words // 100) * 100
        length_distribution[bucket] = length_distribution.get(bucket, 0) + 1
    
    stats["length_distribution"] = length_distribution
    
    return stats

def visualize_stats(processed_stats: Dict[str, Any], dataset_stats: Dict[str, Any], output_dir: str):
    """Create visualizations of the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heading distribution
    if "headings" in processed_stats:
        headings = processed_stats["headings"]
        plt.figure(figsize=(10, 6))
        plt.bar(headings.keys(), headings.values())
        plt.title("Heading Distribution")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "heading_distribution.png"))
        plt.close()
    
    # 2. Common Sanskrit terms
    if "common_sanskrit_terms" in processed_stats:
        terms = processed_stats["common_sanskrit_terms"]
        plt.figure(figsize=(12, 6))
        plt.bar(terms.keys(), terms.values())
        plt.title("Common Sanskrit Terms")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sanskrit_terms.png"))
        plt.close()
    
    # 3. Example length distribution
    if "length_distribution" in dataset_stats:
        dist = dataset_stats["length_distribution"]
        buckets = sorted(dist.keys())
        counts = [dist[bucket] for bucket in buckets]
        labels = [f"{b}-{b+99}" for b in buckets]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, counts)
        plt.title("Example Length Distribution (words)")
        plt.xlabel("Word Count Range")
        plt.ylabel("Number of Examples")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "example_length_distribution.png"))
        plt.close()

def print_report(processed_stats: Dict[str, Any], dataset_stats: Dict[str, Any] = None):
    """Print a human-readable report of the analysis."""
    print("=" * 60)
    print("PROCESSED FILE ANALYSIS")
    print("=" * 60)
    print(f"File: {processed_stats['file_path']}")
    print(f"Total characters: {processed_stats['total_chars']:,}")
    print(f"Total words: {processed_stats['total_words']:,}")
    print(f"Total lines: {processed_stats['total_lines']:,}")
    print(f"Total paragraphs: {processed_stats['total_paragraphs']:,}")
    
    print("\nMARKDOWN STRUCTURE:")
    print(f"H1 headings: {processed_stats['headings']['h1']}")
    print(f"H2 headings: {processed_stats['headings']['h2']}")
    print(f"H3 headings: {processed_stats['headings']['h3']}")
    
    print("\nSANSKRIT CONTENT:")
    print(f"Sanskrit terms found: {processed_stats['sanskrit_term_count']}")
    print("Common Sanskrit terms:")
    for term, count in processed_stats.get('common_sanskrit_terms', {}).items():
        print(f"  - {term}: {count}")
    
    print("\nTABLES AND FORMATTING:")
    print(f"Tables found: {processed_stats['table_count']}")
    
    print("\nPOTENTIAL ARTIFACTS:")
    artifacts = processed_stats.get('potential_artifacts', {})
    print(f"Image references: {artifacts.get('image_references', 0)}")
    print(f"Page references: {artifacts.get('page_references', 0)}")
    
    print(f"\nName replacements: {processed_stats.get('name_replacements', 0)}")
    
    if dataset_stats:
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        print(f"File: {dataset_stats['file_path']}")
        print(f"Total examples: {dataset_stats['total_examples']:,}")
        print(f"Average example length: {dataset_stats['avg_example_length']:.1f} characters")
        
        print("\nEXAMPLE LENGTH DISTRIBUTION:")
        dist = dataset_stats.get('length_distribution', {})
        for bucket in sorted(dist.keys()):
            print(f"  {bucket}-{bucket+99} words: {dist[bucket]} examples")

def main():
    parser = argparse.ArgumentParser(description="Analyze processed files and datasets")
    parser.add_argument("--processed-file", required=True, help="Path to processed markdown file")
    parser.add_argument("--dataset-file", help="Path to JSONL dataset file")
    parser.add_argument("--output-dir", default="analysis", help="Directory to save visualizations")
    parser.add_argument("--save-json", action="store_true", help="Save analysis as JSON")
    
    args = parser.parse_args()
    
    # Analyze processed file
    processed_stats = analyze_processed_file(args.processed_file)
    
    # Analyze dataset file if provided
    dataset_stats = None
    if args.dataset_file:
        dataset_stats = analyze_dataset_file(args.dataset_file)
    
    # Print report
    print_report(processed_stats, dataset_stats)
    
    # Create visualizations
    if dataset_stats:
        visualize_stats(processed_stats, dataset_stats, args.output_dir)
    
    # Save JSON if requested
    if args.save_json:
        result = {
            "processed_file": processed_stats,
            "dataset_file": dataset_stats
        }
        
        output_path = os.path.join(args.output_dir, "analysis_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    main()