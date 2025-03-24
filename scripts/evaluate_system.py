#!/usr/bin/env python3
"""
Script to evaluate the RAG system against a baseline LLM.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import llm
from src.rag import setup_rag_pipeline
from src.utils import extract_champions, calculate_accuracy, get_true_positive_champions
from config import GROUND_TRUTH, EVAL_QUERIES, DATA_DIR


def evaluate_rag_system(qa_chain, queries, ground_truth, output_dir):
    """
    Evaluate the RAG system's performance across different lane queries.
    
    Args:
        qa_chain: The RAG pipeline.
        queries: Dictionary of lane queries.
        ground_truth: Dictionary of ground truth champions.
        output_dir: Directory to save evaluation results.
        
    Returns:
        dict: Evaluation results.
    """
    results = {}
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
    
    print("\n--- League of Legends Champion Recommendation Evaluation (RAG) ---\n")
    
    for lane, query in queries.items():
        print(f"Testing query: {query}")
        
        # Get response from the RAG system
        response = qa_chain.invoke({"query": query})
        result_text = response["result"]
        
        # Extract champions from the response
        extracted_champions = extract_champions(result_text)
        
        # Find true positive champions
        true_positive_champions = get_true_positive_champions(extracted_champions, ground_truth[lane])
        
        # Calculate metrics
        precision, recall, f1 = calculate_accuracy(extracted_champions, ground_truth[lane])
        
        # Store results
        results[lane] = {
            "query": query,
            "extracted_champions": extracted_champions,
            "ground_truth": ground_truth[lane],
            "true_positives": true_positive_champions,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "response": result_text
        }
        
        # Print lane results
        print(f"\nLane: {lane}")
        print(f"Extracted champions: {', '.join(extracted_champions)}")
        print(f"Ground truth: {', '.join(ground_truth[lane])}")
        print(f"Correctly identified: {', '.join(true_positive_champions)}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("-" * 50)
        
        overall_precision += precision
        overall_recall += recall
        overall_f1 += f1
    
    # Calculate overall metrics
    num_lanes = len(queries)
    overall_precision /= num_lanes
    overall_recall /= num_lanes
    overall_f1 /= num_lanes
    
    # Print overall results
    print("\n--- Overall Results (RAG) ---")
    print(f"Average Precision: {overall_precision:.2f}")
    print(f"Average Recall: {overall_recall:.2f}")
    print(f"Average F1 Score: {overall_f1:.2f}")
    
    # Create and save visualization
    visualize_results(results, output_dir, "rag")
    
    # Save detailed results
    save_detailed_results(results, output_dir, "rag")
    
    return results


def evaluate_direct_llm(llm_model, queries, ground_truth, output_dir):
    """
    Evaluate the LLM's performance without RAG across different lane queries.
    
    Args:
        llm_model: The language model.
        queries: Dictionary of lane queries.
        ground_truth: Dictionary of ground truth champions.
        output_dir: Directory to save evaluation results.
        
    Returns:
        dict: Evaluation results.
    """
    results = {}
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
    
    print("\n--- Direct LLM (No RAG) Champion Recommendation Evaluation ---\n")
    
    for lane, query in queries.items():
        print(f"Testing query: {query}")
        
        # Direct LLM call without retrieval
        system_prompt = "You are a game expert and a helpful assistant for League of Legends players. Please recommend the best champions for the lane or role mentioned in the question."
        full_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nPlease provide a detailed answer with specific champion recommendations."
        
        # Get response from the LLM
        result_text = llm_model.invoke(full_prompt)
        
        # Extract champions from the response
        extracted_champions = extract_champions(result_text)
        
        # Find true positive champions
        true_positive_champions = get_true_positive_champions(extracted_champions, ground_truth[lane])
        
        # Calculate metrics
        precision, recall, f1 = calculate_accuracy(extracted_champions, ground_truth[lane])
        
        # Store results
        results[lane] = {
            "query": query,
            "extracted_champions": extracted_champions,
            "ground_truth": ground_truth[lane],
            "true_positives": true_positive_champions,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "response": result_text
        }
        
        # Print lane results
        print(f"\nLane: {lane}")
        print(f"Extracted champions: {', '.join(extracted_champions)}")
        print(f"Ground truth: {', '.join(ground_truth[lane])}")
        print(f"Correctly identified: {', '.join(true_positive_champions)}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("-" * 50)
        
        overall_precision += precision
        overall_recall += recall
        overall_f1 += f1
    
    # Calculate overall metrics
    num_lanes = len(queries)
    overall_precision /= num_lanes
    overall_recall /= num_lanes
    overall_f1 /= num_lanes
    
    # Print overall results
    print("\n--- Overall Direct LLM Results ---")
    print(f"Average Precision: {overall_precision:.2f}")
    print(f"Average Recall: {overall_recall:.2f}")
    print(f"Average F1 Score: {overall_f1:.2f}")
    
    # Create and save visualization
    visualize_results(results, output_dir, "direct_llm")
    
    # Save detailed results
    save_detailed_results(results, output_dir, "direct_llm")
    
    return results


def visualize_results(results, output_dir, prefix):
    """
    Create visualizations of the evaluation results.
    
    Args:
        results: Evaluation results.
        output_dir: Directory to save visualizations.
        prefix: Prefix for output files.
    """
    # Prepare data for plotting
    lanes = list(results.keys())
    precision_values = [results[lane]["precision"] for lane in lanes]
    recall_values = [results[lane]["recall"] for lane in lanes]
    f1_values = [results[lane]["f1"] for lane in lanes]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        "Lane": lanes + lanes + lanes,
        "Metric": ["Precision"] * len(lanes) + ["Recall"] * len(lanes) + ["F1 Score"] * len(lanes),
        "Value": precision_values + recall_values + f1_values
    })
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    sns.barplot(x="Lane", y="Value", hue="Metric", data=df)
    
    # Add labels and title
    title = "RAG System Evaluation Metrics by Lane" if prefix == "rag" else "Direct LLM Evaluation Metrics by Lane"
    plt.title(title, fontsize=16)
    plt.xlabel("Lane", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.ylim(0, 1.0)
    
    # Add score labels on top of bars
    for i, bar in enumerate(plt.gca().patches):
        plt.gca().text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            fontsize=9
        )
    
    # Improve layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_evaluation_results.png"))
    plt.close()
    
    # Create a champion recommendation accuracy plot
    plt.figure(figsize=(14, 10))
    
    for i, lane in enumerate(lanes):
        extracted = results[lane]["extracted_champions"]
        ground_truth = results[lane]["ground_truth"]
        
        plt.subplot(3, 2, i+1)
        
        # Create sets for Venn diagram-like visualization
        extracted_set = set([champ.lower() for champ in extracted])
        truth_set = set([champ.lower() for champ in ground_truth])
        
        # Calculate intersection and differences
        correct = extracted_set.intersection(truth_set)
        missed = truth_set - extracted_set
        incorrect = extracted_set - truth_set
        
        # Create bar chart
        categories = ['Correct', 'Missed', 'Incorrect']
        values = [len(correct), len(missed), len(incorrect)]
        colors = ['green', 'orange', 'red']
        
        bars = plt.bar(categories, values, color=colors)
        
        # Add champion names as text
        correct_list = [champ for champ in extracted if champ.lower() in correct]
        missed_list = [champ for champ in ground_truth if champ.lower() in missed]
        incorrect_list = [champ for champ in extracted if champ.lower() in incorrect]
        
        # Add labels above bars with champion names
        plt.text(0, values[0] + 0.1, ', '.join(correct_list), ha='center', wrap=True)
        if len(missed_list) > 0:
            plt.text(1, values[1] + 0.1, ', '.join(missed_list), ha='center', wrap=True)
        if len(incorrect_list) > 0:
            plt.text(2, values[2] + 0.1, ', '.join(incorrect_list), ha='center', wrap=True)
        
        plt.title(f"{lane} Lane Champion Recommendations")
        plt.ylim(0, max(values) + 2)  # Add space for text
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_champion_accuracy.png"))
    plt.close()


def save_detailed_results(results, output_dir, prefix):
    """
    Save detailed results to a CSV file.
    
    Args:
        results: Evaluation results.
        output_dir: Directory to save results.
        prefix: Prefix for output files.
    """
    rows = []
    
    for lane, data in results.items():
        extracted = ', '.join(data["extracted_champions"])
        ground_truth = ', '.join(data["ground_truth"])
        true_positives = ', '.join(data["true_positives"])
        
        row = {
            "Lane": lane,
            "Query": data["query"],
            "Extracted Champions": extracted,
            "Ground Truth Champions": ground_truth,
            "Correctly Identified": true_positives,
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1 Score": data["f1"]
        }
        
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{prefix}_evaluation_results.csv"), index=False)
    print(f"Detailed results saved to {prefix}_evaluation_results.csv")
    
    # Save full responses for reference
    responses = {lane: data["response"] for lane, data in results.items()}
    with open(os.path.join(output_dir, f"{prefix}_responses.json"), 'w') as f:
        json.dump(responses, f, indent=2)


def save_comparison_results(rag_results, direct_llm_results, output_dir):
    """
    Create and save visualizations comparing RAG vs Direct LLM performance.
    
    Args:
        rag_results: RAG evaluation results.
        direct_llm_results: Direct LLM evaluation results.
        output_dir: Directory to save results.
    """
    lanes = list(rag_results.keys())
    
    # Create comparison metrics dataframe
    comparison_data = []
    
    for lane in lanes:
        comparison_data.extend([
            {"Lane": lane, "Metric": "Precision", "System": "RAG", "Value": rag_results[lane]["precision"]},
            {"Lane": lane, "Metric": "Recall", "System": "RAG", "Value": rag_results[lane]["recall"]},
            {"Lane": lane, "Metric": "F1", "System": "RAG", "Value": rag_results[lane]["f1"]},
            {"Lane": lane, "Metric": "Precision", "System": "Direct LLM", "Value": direct_llm_results[lane]["precision"]},
            {"Lane": lane, "Metric": "Recall", "System": "Direct LLM", "Value": direct_llm_results[lane]["recall"]},
            {"Lane": lane, "Metric": "F1", "System": "Direct LLM", "Value": direct_llm_results[lane]["f1"]}
        ])
    
    df = pd.DataFrame(comparison_data)
    
    # Create comparison visualizations
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(["Precision", "Recall", "F1"]):
        plt.subplot(1, 3, i+1)
        metric_df = df[df["Metric"] == metric]
        
        ax = sns.barplot(x="Lane", y="Value", hue="System", data=metric_df)
        
        plt.title(f"{metric} Comparison: RAG vs Direct LLM", fontsize=14)
        plt.xlabel("Lane", fontsize=12)
        plt.ylabel(f"{metric} Score", fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for i, bar in enumerate(ax.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rag_vs_direct_comparison.png"))
    plt.close()
    
    # Save detailed comparison to CSV
    comparison_rows = []
    
    for lane in lanes:
        rag_correct = set([x.lower() for x in get_true_positive_champions(
            rag_results[lane]["extracted_champions"], 
            rag_results[lane]["ground_truth"]
        )])
        
        direct_correct = set([x.lower() for x in get_true_positive_champions(
            direct_llm_results[lane]["extracted_champions"], 
            direct_llm_results[lane]["ground_truth"]
        )])
        
        # Unique to RAG
        rag_only = rag_correct - direct_correct
        # Unique to Direct LLM
        direct_only = direct_correct - rag_correct
        # Common to both
        common = rag_correct.intersection(direct_correct)
        
        row = {
            "Lane": lane,
            "RAG Precision": rag_results[lane]["precision"],
            "Direct LLM Precision": direct_llm_results[lane]["precision"],
            "RAG Recall": rag_results[lane]["recall"],
            "Direct LLM Recall": direct_llm_results[lane]["recall"],
            "RAG F1": rag_results[lane]["f1"],
            "Direct LLM F1": direct_llm_results[lane]["f1"],
            "RAG Champions": ', '.join(rag_results[lane]["extracted_champions"]),
            "Direct LLM Champions": ', '.join(direct_llm_results[lane]["extracted_champions"]),
            "Correctly Identified by Both": ', '.join([c for c in common]),
            "Correctly Identified Only by RAG": ', '.join([c for c in rag_only]),
            "Correctly Identified Only by Direct LLM": ', '.join([c for c in direct_only])
        }
        
        comparison_rows.append(row)
    
    # Calculate average improvements
    avg_precision_improvement = sum([rag_results[lane]["precision"] - direct_llm_results[lane]["precision"] for lane in lanes]) / len(lanes)
    avg_recall_improvement = sum([rag_results[lane]["recall"] - direct_llm_results[lane]["recall"] for lane in lanes]) / len(lanes)
    avg_f1_improvement = sum([rag_results[lane]["f1"] - direct_llm_results[lane]["f1"] for lane in lanes]) / len(lanes)
    
    # Add summary row
    summary_row = {
        "Lane": "AVERAGE",
        "RAG Precision": sum([rag_results[lane]["precision"] for lane in lanes]) / len(lanes),
        "Direct LLM Precision": sum([direct_llm_results[lane]["precision"] for lane in lanes]) / len(lanes),
        "RAG Recall": sum([rag_results[lane]["recall"] for lane in lanes]) / len(lanes),
        "Direct LLM Recall": sum([direct_llm_results[lane]["recall"] for lane in lanes]) / len(lanes),
        "RAG F1": sum([rag_results[lane]["f1"] for lane in lanes]) / len(lanes),
        "Direct LLM F1": sum([direct_llm_results[lane]["f1"] for lane in lanes]) / len(lanes),
        "RAG Champions": "",
        "Direct LLM Champions": "",
        "Correctly Identified by Both": f"Precision Improvement: {avg_precision_improvement:.2f}",
        "Correctly Identified Only by RAG": f"Recall Improvement: {avg_recall_improvement:.2f}",
        "Correctly Identified Only by Direct LLM": f"F1 Improvement: {avg_f1_improvement:.2f}"
    }
    
    comparison_rows.append(summary_row)
    
    # Create DataFrame and save to CSV
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(output_dir, "rag_vs_direct_comparison.csv"), index=False)
    print("Comparison results saved to rag_vs_direct_comparison.csv")
    
    return comparison_df


def main():
    """Evaluate the LoL RAG Assistant against a baseline LLM."""
    parser = argparse.ArgumentParser(description="Evaluate LoL RAG Assistant")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    output_dir = args.output if args.output else os.path.join(DATA_DIR, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Evaluating LoL RAG Assistant ===")
    
    # Setup RAG pipeline
    print("\nSetting up RAG pipeline...")
    rag_chain = setup_rag_pipeline()
    
    print("\n=== Starting RAG System Evaluation ===")
    # Run RAG evaluation
    rag_results = evaluate_rag_system(rag_chain, EVAL_QUERIES, GROUND_TRUTH, output_dir)
    
    print("\n=== Starting Direct LLM Evaluation (No RAG) ===")
    # Run direct LLM evaluation (without RAG)
    direct_llm_results = evaluate_direct_llm(llm, EVAL_QUERIES, GROUND_TRUTH, output_dir)
    
    # Create and save comparison visualizations and data
    comparison_df = save_comparison_results(rag_results, direct_llm_results, output_dir)
    
    # Print key findings
    print("\n=== Key Performance Comparison: RAG vs Direct LLM ===\n")
    
    # Get the average row from comparison DataFrame
    avg_row = comparison_df[comparison_df["Lane"] == "AVERAGE"].iloc[0]
    
    # Calculate absolute improvements
    precision_improvement = avg_row["RAG Precision"] - avg_row["Direct LLM Precision"]
    recall_improvement = avg_row["RAG Recall"] - avg_row["Direct LLM Recall"]
    f1_improvement = avg_row["RAG F1"] - avg_row["Direct LLM F1"]
    
    # Calculate relative improvements (percentage)
    precision_pct = (precision_improvement / avg_row["Direct LLM Precision"]) * 100 if avg_row["Direct LLM Precision"] > 0 else float('inf')
    recall_pct = (recall_improvement / avg_row["Direct LLM Recall"]) * 100 if avg_row["Direct LLM Recall"] > 0 else float('inf')
    f1_pct = (f1_improvement / avg_row["Direct LLM F1"]) * 100 if avg_row["Direct LLM F1"] > 0 else float('inf')
    
    print(f"Precision: RAG {avg_row['RAG Precision']:.2f} vs Direct LLM {avg_row['Direct LLM Precision']:.2f}")
    print(f"Absolute improvement: {precision_improvement:.2f} ({precision_pct:.1f}%)")
    
    print(f"\nRecall: RAG {avg_row['RAG Recall']:.2f} vs Direct LLM {avg_row['Direct LLM Recall']:.2f}")
    print(f"Absolute improvement: {recall_improvement:.2f} ({recall_pct:.1f}%)")
    
    print(f"\nF1 Score: RAG {avg_row['RAG F1']:.2f} vs Direct LLM {avg_row['Direct LLM F1']:.2f}")
    print(f"Absolute improvement: {f1_improvement:.2f} ({f1_pct:.1f}%)")
    
    print("\nComplete results and visualizations have been saved to CSV files and PNG images.")


if __name__ == "__main__":
    main()