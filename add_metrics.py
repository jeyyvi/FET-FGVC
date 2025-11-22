#!/usr/bin/env python3
"""
add_metrics.py - FET-FGVC Training Metrics Enhancement

This script enhances train.py to include:
- Per-class metrics (Precision, Recall, F1-Score, Accuracy, Support)
- Training history tracking (JSON)
- Classification reports (CSV + JSON)
- Overfitting detection
- Automatic saving to Google Drive EvaluationResults folder

Usage:
    python add_metrics.py
"""

import re
import os
import shutil

def add_metrics_to_training():
    """Add comprehensive metrics tracking to train.py"""
    
    train_py_path = 'train.py'
    
    if not os.path.exists(train_py_path):
        print("✗ train.py not found!")
        return
    
    # Read the current train.py
    with open(train_py_path, 'r') as f:
        content = f.read()
    
    # Check if already enhanced
    if 'class TrainingHistory:' in content:
        print("⚠️  train.py already enhanced with metrics!")
        return
    
    print("Enhancing train.py with metrics tracking...\n")
    
    # ========================================================================
    # 1. Add imports
    # ========================================================================
    additional_imports = """
import json
from datetime import datetime
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
"""
    
    if 'from sklearn.metrics' not in content:
        content = content.replace('import torch', additional_imports + '\nimport torch', 1)
        print("✓ Added required imports")
    
    # ========================================================================
    # 2. Add metrics tracking code
    # ========================================================================
    metrics_code = '''

# ============================================================================
# METRICS AND EVALUATION - Added by add_metrics.py
# ============================================================================

EVALUATION_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints/EvaluationResults'

class TrainingHistory:
    """Track and save comprehensive training history"""
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(EVALUATION_DIR, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'epoch_time': [], 'best_epoch': 0, 'best_val_acc': 0.0,
        }
        self.start_time = datetime.now()
        print(f"📁 Evaluation results will be saved to: {self.exp_dir}")
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Update history with new epoch data"""
        self.history['train_loss'].append(float(train_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_loss'].append(float(val_loss))
        self.history['val_acc'].append(float(val_acc))
        self.history['learning_rate'].append(float(lr))
        self.history['epoch_time'].append(float(epoch_time))
        
        if val_acc > self.history['best_val_acc']:
            self.history['best_val_acc'] = float(val_acc)
            self.history['best_epoch'] = epoch
        
        self.save()
    
    def save(self):
        """Save history to JSON file"""
        try:
            history_path = os.path.join(self.exp_dir, 'training_history.json')
            
            save_data = {
                'experiment_name': self.exp_name,
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_epochs': len(self.history['train_loss']),
                'history': self.history
            }
            
            with open(history_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
        except Exception as e:
            print(f'✗ Error saving history: {e}')
    
    def get_overfitting_indicator(self):
        """Calculate overfitting indicators"""
        if len(self.history['train_loss']) < 2:
            return None
        
        train_val_gap = self.history['train_acc'][-1] - self.history['val_acc'][-1]
        loss_gap = self.history['val_loss'][-1] - self.history['train_loss'][-1]
        
        if len(self.history['val_acc']) >= 3:
            val_trend = np.mean(np.diff(self.history['val_acc'][-3:]))
            train_trend = np.mean(np.diff(self.history['train_acc'][-3:]))
            diverging = train_trend > 0 and val_trend < 0
        else:
            diverging = False
        
        return {
            'accuracy_gap': float(train_val_gap),
            'loss_gap': float(loss_gap),
            'diverging': diverging,
            'warning': train_val_gap > 0.1 or diverging
        }


def compute_per_class_metrics(all_preds, all_labels, class_names, num_classes):
    """Compute detailed per-class metrics"""
    try:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
        )
        
        per_class_acc = []
        for i in range(num_classes):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == i).sum() / mask.sum()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        per_class_results = []
        for i in range(num_classes):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            per_class_results.append({
                'class_id': int(i), 'class_name': class_name,
                'precision': float(precision[i]), 'recall': float(recall[i]),
                'f1_score': float(f1[i]), 'support': int(support[i]),
                'accuracy': float(per_class_acc[i])
            })
        
        report = classification_report(
            all_labels, all_preds, labels=list(range(num_classes)),
            target_names=[class_names[i] if i < len(class_names) else f"Class_{i}" for i in range(num_classes)],
            zero_division=0, output_dict=True
        )
        
        return per_class_results, report
    except Exception as e:
        print(f'✗ Error computing metrics: {e}')
        return None, None


def save_classification_report(per_class_results, report, epoch, exp_name):
    """Save classification report and per-class results to EvaluationResults"""
    try:
        exp_dir = os.path.join(EVALUATION_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save per-class results as CSV
        csv_path = os.path.join(exp_dir, f'per_class_results_epoch_{epoch}.csv')
        with open(csv_path, 'w') as f:
            f.write('Class_ID,Class_Name,Precision,Recall,F1_Score,Support,Accuracy\\n')
            for item in per_class_results:
                f.write(f"{item['class_id']},{item['class_name']},"
                       f"{item['precision']:.4f},{item['recall']:.4f},"
                       f"{item['f1_score']:.4f},{item['support']},{item['accuracy']:.4f}\\n")
        
        # Save full report as JSON
        report_path = os.path.join(exp_dir, f'classification_report_epoch_{epoch}.json')
        report_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'per_class_results': per_class_results,
            'overall_metrics': {
                'accuracy': report['accuracy'],
                'macro_avg': report['macro avg'],
                'weighted_avg': report['weighted avg']
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Also save latest versions
        latest_csv = os.path.join(exp_dir, 'per_class_results_latest.csv')
        latest_json = os.path.join(exp_dir, 'classification_report_latest.json')
        shutil.copy(csv_path, latest_csv)
        shutil.copy(report_path, latest_json)
        
    except Exception as e:
        print(f'✗ Error saving reports: {e}')


def print_top_and_bottom_classes(per_class_results, n=5):
    """Print top and bottom performing classes"""
    if not per_class_results:
        return
    
    sorted_results = sorted(per_class_results, key=lambda x: x['f1_score'], reverse=True)
    
    print(f"\\n{'='*70}")
    print(f"TOP {n} PERFORMING CLASSES (by F1-Score):")
    print(f"{'='*70}")
    for i, item in enumerate(sorted_results[:n], 1):
        print(f"{i}. {item['class_name'][:40]:40} F1: {item['f1_score']:.4f} "
              f"Acc: {item['accuracy']:.4f} Support: {item['support']}")
    
    print(f"\\n{'='*70}")
    print(f"BOTTOM {n} PERFORMING CLASSES (by F1-Score):")
    print(f"{'='*70}")
    for i, item in enumerate(sorted_results[-n:][::-1], 1):
        print(f"{i}. {item['class_name'][:40]:40} F1: {item['f1_score']:.4f} "
              f"Acc: {item['accuracy']:.4f} Support: {item['support']}")
    print(f"{'='*70}\\n")

'''
    
    # Insert after checkpoint functions (or before main if no checkpoints)
    insertion_point = content.find('def load_checkpoint_from_gdrive')
    if insertion_point == -1:
        insertion_point = content.find('\ndef main(')
    if insertion_point == -1:
        insertion_point = content.find('if __name__')
    
    if insertion_point != -1:
        content = content[:insertion_point] + '\n' + metrics_code + content[insertion_point:]
        print("✓ Added TrainingHistory and metrics functions")
    
    # ========================================================================
    # 3. Modify validate function
    # ========================================================================
    
    # Add parameters to validate signature
    if 'def validate(' in content:
        content = re.sub(
            r'def validate\((.*?)\):',
            r'def validate(\1, class_names=None, num_classes=200):',
            content, count=1
        )
        print("✓ Updated validate function signature")
    
    # Add prediction collection
    validate_pattern = r'(def validate.*?)(\n    .*?for.*?in.*?loader.*?:)'
    if re.search(validate_pattern, content, re.DOTALL):
        content = re.sub(
            validate_pattern,
            r'\1\n    all_preds = []\n    all_labels = []\2',
            content, count=1, flags=re.DOTALL
        )
        print("✓ Added prediction tracking")
    
    # Collect predictions inside loop
    if '_, pred = torch.max(outputs' in content:
        content = re.sub(
            r'(_, pred = torch\.max\(outputs.*?)\n',
            r'\1\n        all_preds.extend(pred.cpu().numpy())\n        all_labels.extend(targets.cpu().numpy())\n',
            content, count=1
        )
    
    # Compute metrics before return
    if 'def validate' in content:
        content = re.sub(
            r'(def validate.*?)(    return)',
            r'\1    per_class_results, classification_rep = None, None\n    if class_names:\n        try:\n            per_class_results, classification_rep = compute_per_class_metrics(all_preds, all_labels, class_names, num_classes)\n        except Exception as e:\n            print(f"Error: {e}")\n    \n\2',
            content, count=1, flags=re.DOTALL
        )
        
        # Update return statement
        content = re.sub(
            r'return (.*?), (.*?), acc_avg([^,])',
            r'return \1, \2, acc_avg, per_class_results, classification_rep\3',
            content, count=1
        )
        print("✓ Modified validate to compute metrics")
    
    # ========================================================================
    # 4. Modify main function
    # ========================================================================
    
    # Add class names loading
    if 'model = ' in content and 'class_names = []' not in content:
        content = re.sub(
            r'(model = .*?\.to\(device\))',
            r'''\1
    
    # Load class names for metrics
    class_names = []
    classes_file = os.path.join(cfg.dataset.txt_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip().split()[1].replace('_', ' ') for line in f.readlines()]
    else:
        class_names = [f'Class_{i}' for i in range(cfg.dataset.num_classes)]
    ''',
            content, count=1, flags=re.DOTALL
        )
        print("✓ Added class names loading")
    
    # Initialize training history
    if 'class_names = []' in content:
        content = re.sub(
            r"(class_names = \[f'Class_\{i\}' for i in range\(cfg\.dataset\.num_classes\)\])",
            r"\1\n    training_history = TrainingHistory(args.name)",
            content, count=1
        )
        print("✓ Added training history initialization")
    
    # Update validate calls
    content = re.sub(
        r'validate\((.*?loader.*?)\)',
        r'validate(\1, class_names, cfg.dataset.num_classes)',
        content
    )
    
    # Update validation result unpacking
    content = re.sub(
        r'\(loss_val, losses_val, acc_val\) = validate',
        r'(loss_val, losses_val, acc_val, per_class_results, classification_rep) = validate',
        content
    )
    print("✓ Updated validate calls")
    
    # Add epoch timing
    if 'for epoch in range' in content:
        content = re.sub(
            r'(for epoch in range.*?:)',
            r'\1\n        epoch_start_time = time.time()',
            content, count=1
        )
    
    # Add history update after epoch logging
    epoch_log_pattern = r"(logger\.info\(.*?\[Epoch:.*?\].*?\n)"
    if re.search(epoch_log_pattern, content, re.DOTALL):
        history_update = '''
        
        # Update training history and save metrics
        current_lr = optimizers[0].param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        training_history.update(epoch, loss_train, acc_train, loss_val, acc_val, current_lr, epoch_time)
        
        # Save classification report
        if per_class_results:
            save_classification_report(per_class_results, classification_rep, epoch, args.name)
            print_top_and_bottom_classes(per_class_results, n=5)
        
        # Check overfitting
        overfitting_info = training_history.get_overfitting_indicator()
        if overfitting_info and overfitting_info['warning']:
            print(f"⚠️  Overfitting Warning: Train-Val Gap = {overfitting_info['accuracy_gap']:.4f}")
        
'''
        content = re.sub(epoch_log_pattern, r'\1' + history_update, content, count=1, flags=re.DOTALL)
        print("✓ Added training history updates")
    
    # Write back
    with open(train_py_path, 'w') as f:
        f.write(content)
    
    print("\n" + "="*70)
    print("✓ Successfully enhanced train.py!")
    print("="*70)
    print(f"\n📁 All evaluation results will be saved to:")
    print(f"   /content/drive/MyDrive/FET-FGVC-checkpoints/EvaluationResults/")
    print("\nFeatures added:")
    print("  • Per-class metrics (Precision, Recall, F1, Accuracy, Support)")
    print("  • Classification reports (CSV + JSON)")
    print("  • Training history tracking")
    print("  • Overfitting detection")
    print("  • Top/bottom class performance display")
    print("="*70)


if __name__ == '__main__':
    print("="*70)
    print("FET-FGVC Training Metrics Enhancement")
    print("="*70)
    print()
    
    add_metrics_to_training()