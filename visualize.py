#!/usr/bin/env python3
"""
visualize.py - FET-FGVC Training Visualization

Generate comprehensive visualizations from training results:
- Training curves (loss, accuracy, learning rate)
- Overfitting indicators
- Per-class performance analysis
- Metrics distribution

Usage:
    python visualize.py <experiment_name>
    
Example:
    python visualize.py cub_swin_baseline
"""

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import pandas as pd
import numpy as np

# Configuration
EVALUATION_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints/EvaluationResults'

def plot_training_curves(history_path, save_path=None):
    """
    Plot comprehensive training curves
    """
    with open(history_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Training Progress: {data['experiment_name']}", fontsize=16, fontweight='bold')
    
    # 1. Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    best_epoch = history['best_epoch']
    if best_epoch > 0:
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax1.legend(fontsize=10)
    
    # 2. Accuracy Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axhline(y=history['best_val_acc'], color='g', linestyle=':', alpha=0.5, linewidth=2,
                label=f"Best Val Acc: {history['best_val_acc']:.4f}")
    ax2.legend(fontsize=10)
    
    # 3. Learning Rate Schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'purple', linewidth=2, marker='D', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2)
    
    # 4. Overfitting Indicator
    ax4 = axes[1, 1]
    acc_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    
    ax4.plot(epochs, acc_gap, 'b-', linewidth=2, marker='o', markersize=4, label='Train-Val Gap')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Train-Val Accuracy Gap', fontsize=12)
    ax4.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Warning Threshold (0.1)')
    ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add warning zone
    max_gap = max(acc_gap) if max(acc_gap) > 0.1 else 0.2
    ax4.fill_between(epochs, 0.1, max_gap, alpha=0.15, color='red', label='Overfitting Zone')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Training curves saved to: {save_path}')
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Experiment: {data['experiment_name']}")
    print(f"Started: {data['start_time']}")
    print(f"Last Update: {data['last_update']}")
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nBest Performance:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Val Accuracy: {history['best_val_acc']:.4f}")
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Train-Val Gap: {acc_gap[-1]:.4f}")
    
    if acc_gap[-1] > 0.1:
        print(f"\n⚠️  Warning: Significant overfitting detected!")
    
    print(f"{'='*70}\n")


def plot_per_class_performance(csv_path, top_n=20, bottom_n=20, save_path=None):
    """
    Plot per-class performance metrics
    """
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('F1_Score', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top N classes
    ax1 = axes[0]
    top_df = df_sorted.head(top_n)
    x_pos = range(len(top_df))
    
    width = 0.4
    ax1.barh([p - width/2 for p in x_pos], top_df['F1_Score'], width, 
             color='green', alpha=0.7, label='F1 Score')
    ax1.barh([p + width/2 for p in x_pos], top_df['Accuracy'], width,
             color='blue', alpha=0.7, label='Accuracy')
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(top_df['Class_Name'], fontsize=8)
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_title(f'🏆 Top {top_n} Performing Classes', fontsize=14, fontweight='bold', color='green')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    ax1.set_xlim([0, 1.0])
    
    # 2. Bottom N classes
    ax2 = axes[1]
    bottom_df = df_sorted.tail(bottom_n)
    x_pos = range(len(bottom_df))
    
    ax2.barh([p - width/2 for p in x_pos], bottom_df['F1_Score'], width,
             color='red', alpha=0.7, label='F1 Score')
    ax2.barh([p + width/2 for p in x_pos], bottom_df['Accuracy'], width,
             color='orange', alpha=0.7, label='Accuracy')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(bottom_df['Class_Name'], fontsize=8)
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_title(f'⚠️  Bottom {bottom_n} Performing Classes (Need Improvement)', 
                 fontsize=14, fontweight='bold', color='red')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    ax2.set_xlim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Per-class performance plot saved to: {save_path}')
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"PER-CLASS PERFORMANCE STATISTICS")
    print(f"{'='*70}")
    print(f"Total Classes: {len(df)}")
    print(f"\nOverall Metrics:")
    print(f"  Mean F1-Score: {df['F1_Score'].mean():.4f} (±{df['F1_Score'].std():.4f})")
    print(f"  Mean Accuracy: {df['Accuracy'].mean():.4f} (±{df['Accuracy'].std():.4f})")
    print(f"  Mean Precision: {df['Precision'].mean():.4f} (±{df['Precision'].std():.4f})")
    print(f"  Mean Recall: {df['Recall'].mean():.4f} (±{df['Recall'].std():.4f})")
    
    print(f"\nPerformance Distribution:")
    print(f"  Excellent (F1 > 0.8): {(df['F1_Score'] > 0.8).sum()} classes")
    print(f"  Good (F1 0.6-0.8): {((df['F1_Score'] >= 0.6) & (df['F1_Score'] <= 0.8)).sum()} classes")
    print(f"  Fair (F1 0.4-0.6): {((df['F1_Score'] >= 0.4) & (df['F1_Score'] < 0.6)).sum()} classes")
    print(f"  Poor (F1 < 0.4): {(df['F1_Score'] < 0.4).sum()} classes")
    print(f"{'='*70}\n")


def plot_metrics_distribution(csv_path, save_path=None):
    """Plot distribution of metrics across all classes"""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Metrics Distribution Across All Classes', fontsize=16, fontweight='bold')
    
    metrics = ['Precision', 'Recall', 'F1_Score', 'Accuracy']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # Histogram
        n, bins, patches = ax.hist(df[metric], bins=30, color=color, alpha=0.7, edgecolor='black')
        
        # Mean and median lines
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='darkgreen', linestyle='--', linewidth=2.5,
                   label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Classes', fontsize=11)
        ax.set_title(f'{metric.replace("_", " ")} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'✓ Metrics distribution plot saved to: {save_path}')
    
    plt.show()


def generate_all_visualizations(exp_name, evaluation_dir=None):
    """
    Generate all visualizations for an experiment
    """
    evaluation_dir = evaluation_dir or EVALUATION_DIR
    exp_dir = os.path.join(evaluation_dir, exp_name)
    
    if not os.path.exists(exp_dir):
        print(f"\n{'='*70}")
        print(f"✗ Experiment directory not found!")
        print(f"{'='*70}")
        print(f"Looking for: {exp_dir}")
        print(f"\nMake sure:")
        print(f"  1. Training has been run")
        print(f"  2. Experiment name is correct: '{exp_name}'")
        print(f"  3. Google Drive is mounted")
        print(f"{'='*70}\n")
        return
    
    print(f"\n{'='*70}")
    print(f"GENERATING VISUALIZATIONS: {exp_name}")
    print(f"{'='*70}\n")
    
    # 1. Plot training curves
    history_path = os.path.join(exp_dir, 'training_history.json')
    if os.path.exists(history_path):
        curves_save_path = os.path.join(exp_dir, 'training_curves.png')
        print("📊 Generating training curves...")
        plot_training_curves(history_path, curves_save_path)
    else:
        print(f"⚠️  Training history not found: {history_path}")
    
    # 2. Plot per-class performance
    csv_path = os.path.join(exp_dir, 'per_class_results_latest.csv')
    if os.path.exists(csv_path):
        perf_save_path = os.path.join(exp_dir, 'per_class_performance.png')
        print("\n📊 Generating per-class performance charts...")
        plot_per_class_performance(csv_path, save_path=perf_save_path)
        
        # 3. Plot metrics distribution
        dist_save_path = os.path.join(exp_dir, 'metrics_distribution.png')
        print("\n📊 Generating metrics distribution...")
        plot_metrics_distribution(csv_path, save_path=dist_save_path)
    else:
        print(f"⚠️  Per-class results not found: {csv_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Saved to: {exp_dir}")
    print(f"\n📄 Generated files:")
    
    # List generated files
    if os.path.exists(exp_dir):
        for f in sorted(os.listdir(exp_dir)):
            filepath = os.path.join(exp_dir, f)
            if os.path.isfile(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                icon = "📊" if f.endswith('.png') else "📄" if f.endswith('.csv') else "📋"
                print(f"   {icon} {f:40} ({size_kb:>8.1f} KB)")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        exp_name = 'cub_swin_baseline'
        print(f"⚠️  No experiment name provided. Using default: '{exp_name}'")
        print(f"Usage: python visualize.py <experiment_name>\n")
    
    generate_all_visualizations(exp_name)