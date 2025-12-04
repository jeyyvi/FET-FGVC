# -*- coding: utf-8 -*-
# At the very top of train.py, **before** importing matplotlib
import os
os.environ.pop('MPLBACKEND', None)  # Remove the problematic env variable

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for scripts

from datetime import datetime
import math
import time
import logging
import traceback
import argparse
import json
import shutil
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, precision_recall_fscore_support

from timm.layers import resample_abs_pos_embed

from thop import profile
from network.GlobalBranch import FET_FGVC, create_global_branch, supported_arch
from modules.datasets import BatchDataset, BalancedBatchSampler
from modules import utils, losses


utils.fix_seed()

# # ============================================================================
# # GOOGLE DRIVE CHECKPOINT FUNCTIONS
# # ============================================================================
# GDRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints2'
# EVALUATION_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints2/EvaluationResults'


# def save_checkpoint_to_gdrive(epoch, model, optimizers, best_val, exp_name):
#     """Save checkpoint to Google Drive after each epoch"""
#     try:
#         # Create experiment directory
#         exp_dir = os.path.join(GDRIVE_CHECKPOINT_DIR, exp_name)
#         os.makedirs(exp_dir, exist_ok=True)
        
#         # Prepare checkpoint dictionary
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'best_val': best_val,
#         }
        
#         # Add optimizer states
#         if isinstance(optimizers, list):
#             checkpoint['optimizer_state_dicts'] = [opt.state_dict() for opt in optimizers]
#         else:
#             checkpoint['optimizer_state_dict'] = optimizers.state_dict()
        
#         # Save epoch checkpoint
#         checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
#         local_path = checkpoint_name
#         gdrive_path = os.path.join(exp_dir, checkpoint_name)
        
#         # Save locally first
#         torch.save(checkpoint, local_path)
#         logging.info(f'💾 Saved local checkpoint: {local_path}')
        
#         # Copy to Google Drive
#         shutil.copy(local_path, gdrive_path)
#         logging.info(f'☁️  Copied to Google Drive: {gdrive_path}')
        
#         # Clean up old local checkpoint
#         if os.path.exists(local_path):
#             os.remove(local_path)
        
#         return gdrive_path
        
#     except Exception as e:
#         logging.error(f'❌ Error saving checkpoint: {e}')
#         return None


# def save_best_checkpoint_to_gdrive(checkpoint_path, exp_name):
#     """Save the best checkpoint to Google Drive"""
#     try:
#         if checkpoint_path is None:
#             logging.warning('⚠️  No checkpoint path provided for best model')
#             return
            
#         exp_dir = os.path.join(GDRIVE_CHECKPOINT_DIR, exp_name)
#         best_path = os.path.join(exp_dir, 'best_checkpoint.pth')
        
#         if os.path.exists(checkpoint_path):
#             shutil.copy(checkpoint_path, best_path)
#             logging.info(f'⭐ BEST checkpoint saved: {best_path}')
#         else:
#             logging.warning(f'⚠️  Checkpoint not found: {checkpoint_path}')
        
#     except Exception as e:
#         logging.error(f'❌ Error saving best checkpoint: {e}')



# ============================================================================
# GOOGLE DRIVE CHECKPOINT FUNCTIONS (Keep Only Top 5)
# ============================================================================
import os
import glob
import shutil
import logging
import torch

GDRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints2'
EVALUATION_DIR = '/content/drive/MyDrive/FET-FGVC-checkpoints2/EvaluationResults'
MAX_CHECKPOINTS = 5  # Keep only top 5 best checkpoints


def cleanup_old_checkpoints(exp_dir, current_epoch, best_val_history):
    """
    Keep only the top 5 best checkpoints based on validation accuracy.
    
    Args:
        exp_dir: Experiment directory path
        current_epoch: Current epoch number
        best_val_history: Dict mapping epoch -> validation accuracy
    """
    try:
        # Get all checkpoint files (excluding best_checkpoint.pth)
        checkpoint_files = glob.glob(os.path.join(exp_dir, 'checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) <= MAX_CHECKPOINTS:
            return  # No need to cleanup yet
        
        # Create list of (epoch, val_acc, filepath) tuples
        checkpoint_info = []
        for ckpt_path in checkpoint_files:
            # Extract epoch number from filename
            basename = os.path.basename(ckpt_path)
            try:
                epoch_num = int(basename.replace('checkpoint_epoch_', '').replace('.pth', ''))
                val_acc = best_val_history.get(epoch_num, 0.0)
                checkpoint_info.append((epoch_num, val_acc, ckpt_path))
            except ValueError:
                continue
        
        # Sort by validation accuracy (descending)
        checkpoint_info.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top MAX_CHECKPOINTS
        checkpoints_to_keep = set([info[2] for info in checkpoint_info[:MAX_CHECKPOINTS]])
        
        # Delete checkpoints not in top 5
        deleted_count = 0
        for _, _, ckpt_path in checkpoint_info[MAX_CHECKPOINTS:]:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                deleted_count += 1
                logging.info(f'🗑️  Removed old checkpoint: {os.path.basename(ckpt_path)}')
        
        if deleted_count > 0:
            logging.info(f'✓ Cleanup complete: kept top {MAX_CHECKPOINTS} checkpoints, removed {deleted_count}')
            
            # Log which checkpoints are kept
            kept_epochs = [info[0] for info in checkpoint_info[:MAX_CHECKPOINTS]]
            logging.info(f'📦 Keeping checkpoints from epochs: {sorted(kept_epochs)}')
        
    except Exception as e:
        logging.error(f'❌ Error during checkpoint cleanup: {e}')


def save_checkpoint_to_gdrive(epoch, model, optimizers, best_val, exp_name, best_val_history):
    """
    Save checkpoint to Google Drive and maintain only top 5 best checkpoints.
    
    Args:
        epoch: Current epoch number
        model: Model to save
        optimizers: List of optimizers
        best_val: Current best validation accuracy
        exp_name: Experiment name
        best_val_history: Dict mapping epoch -> validation accuracy
    """
    try:
        # Create experiment directory
        exp_dir = os.path.join(GDRIVE_CHECKPOINT_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val': best_val,
            'val_acc': best_val_history.get(epoch, 0.0),  # Current epoch's val acc
        }
        
        # Add optimizer states
        if isinstance(optimizers, list):
            checkpoint['optimizer_state_dicts'] = [opt.state_dict() for opt in optimizers]
        else:
            checkpoint['optimizer_state_dict'] = optimizers.state_dict()
        
        # Save epoch checkpoint
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        local_path = checkpoint_name
        gdrive_path = os.path.join(exp_dir, checkpoint_name)
        
        # Save locally first
        torch.save(checkpoint, local_path)
        logging.info(f'💾 Saved local checkpoint: {local_path}')
        
        # Copy to Google Drive
        shutil.copy(local_path, gdrive_path)
        logging.info(f'☁️  Copied to Google Drive: {gdrive_path}')
        
        # Clean up old local checkpoint
        if os.path.exists(local_path):
            os.remove(local_path)
        
        # Cleanup old checkpoints (keep only top 5)
        cleanup_old_checkpoints(exp_dir, epoch, best_val_history)
        
        return gdrive_path
        
    except Exception as e:
        logging.error(f'❌ Error saving checkpoint: {e}')
        return None


def save_best_checkpoint_to_gdrive(checkpoint_path, exp_name):
    """Save the best checkpoint to Google Drive"""
    try:
        if checkpoint_path is None:
            logging.warning('⚠️  No checkpoint path provided for best model')
            return
            
        exp_dir = os.path.join(GDRIVE_CHECKPOINT_DIR, exp_name)
        best_path = os.path.join(exp_dir, 'best_checkpoint.pth')
        
        if os.path.exists(checkpoint_path):
            shutil.copy(checkpoint_path, best_path)
            logging.info(f'⭐ BEST checkpoint saved: {best_path}')
        else:
            logging.warning(f'⚠️  Checkpoint not found: {checkpoint_path}')
        
    except Exception as e:
        logging.error(f'❌ Error saving best checkpoint: {e}')

# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

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
        logging.info(f"📁 Evaluation results will be saved to: {self.exp_dir}")
    
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
            logging.error(f'✗ Error saving history: {e}')
    
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
        logging.error(f'✗ Error computing metrics: {e}')
        return None, None


def save_classification_report(per_class_results, report, epoch, exp_name):
    """Save classification report and per-class results to EvaluationResults"""
    try:
        exp_dir = os.path.join(EVALUATION_DIR, exp_name)
        
        # Create separate folders for per-class results and classification reports
        per_class_dir = os.path.join(exp_dir, 'per_class_results')
        classification_dir = os.path.join(exp_dir, 'classification_reports')
        os.makedirs(per_class_dir, exist_ok=True)
        os.makedirs(classification_dir, exist_ok=True)
        
        # Save per-class results as CSV in dedicated folder
        csv_path = os.path.join(per_class_dir, f'epoch_{epoch}.csv')
        with open(csv_path, 'w') as f:
            f.write('Class_ID,Class_Name,Precision,Recall,F1_Score,Support,Accuracy\n')
            for item in per_class_results:
                f.write(f"{item['class_id']},{item['class_name']},"
                       f"{item['precision']:.4f},{item['recall']:.4f},"
                       f"{item['f1_score']:.4f},{item['support']},{item['accuracy']:.4f}\n")
        
        # Save full report as JSON in dedicated folder
        report_path = os.path.join(classification_dir, f'epoch_{epoch}.json')
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
        
        # Also save latest versions in main experiment directory
        latest_csv = os.path.join(exp_dir, 'per_class_results_latest.csv')
        latest_json = os.path.join(exp_dir, 'classification_report_latest.json')
        shutil.copy(csv_path, latest_csv)
        shutil.copy(report_path, latest_json)
        
        logging.info(f'📊 Saved metrics: {per_class_dir}/ and {classification_dir}/')
        
    except Exception as e:
        logging.error(f'✗ Error saving reports: {e}')

def print_top_and_bottom_classes(per_class_results, n=5):
    """Print top and bottom performing classes"""
    if not per_class_results:
        return
    
    sorted_results = sorted(per_class_results, key=lambda x: x['f1_score'], reverse=True)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"TOP {n} PERFORMING CLASSES (by F1-Score):")
    logging.info(f"{'='*70}")
    for i, item in enumerate(sorted_results[:n], 1):
        logging.info(f"{i}. {item['class_name'][:40]:40} F1: {item['f1_score']:.4f} "
              f"Acc: {item['accuracy']:.4f} Support: {item['support']}")
    
    logging.info(f"\n{'='*70}")
    logging.info(f"BOTTOM {n} PERFORMING CLASSES (by F1-Score):")
    logging.info(f"{'='*70}")
    for i, item in enumerate(sorted_results[-n:][::-1], 1):
        logging.info(f"{i}. {item['class_name'][:40]:40} F1: {item['f1_score']:.4f} "
              f"Acc: {item['accuracy']:.4f} Support: {item['support']}")
    logging.info(f"{'='*70}\n")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():

    # prepare model
    load_pretrained = False if (args.from_scratch or args.finetune or args.vis_mode) else True
    if args.img_size == 224:
        size = 224
    elif args.img_size == 448:
        size = 384
    batch_size = (args.sample_classes*args.sample_images)
    model_cfg = {
        "num_classes": cfg.dataset.num_classes, 
        "image_size": args.img_size, 
        "size": size, 
        "load_pretrained": load_pretrained, 
        "drop_rate": cfg.backbone.drop_rate, 
        "attn_drop_rate": cfg.backbone.drop_path_rate, 
        "drop_path_rate": cfg.backbone.attn_drop_rate, 
        "window_size": args.window_size, 
        "dynamic": not args.nodynamic, 
        "base_keep_rate": args.base_keep_rate, 
        "pruning_loc": args.pruning_loc,
    }
    local_cfg = {
        "depth": args.local_depth,
        "num_parts": args.num_parts,
        "batch_size": batch_size, 
        "part_channels": args.part_channels, 
        "gaussian_ksize": args.gaussian_ksize,
    }
    model = FET_FGVC(args.arch, model_cfg, local_cfg, args.nopfi, args.nolocal, cfg.backbone.pretrain_root, args.local_from_stage)
    
    logging.info("Calculate MACs & FLOPs ...")
    inputs = torch.randn((1, 3, args.img_size, args.img_size))
    macs, num_params = profile(model, (inputs,), verbose=False)
    logging.info("\nParams(M):{:.2f}, MACs(G):{:.2f}, FLOPs(G):~{:.2f}".format(num_params/(1000**2), macs/(1000**3), 2*macs/(1000**3)))
    logging.info("")
    
    if args.distill:
        teacher_model = create_global_branch(args.arch, model_cfg, only_teacher_model=True)
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None
    
    logging.info("\nargs: {}".format(args))
    logging.info("\nconfigs: {}".format(cfg))
    if not args.vis_mode:
        logging.info("\nNetwork config: \n{}".format(model))

    # load trained weights
    if args.weights_dir:
        if args.finetune:
            weights_path = os.path.join(args.weights_dir, "best.pth")
        else:
            weights_path = os.path.join(args.weights_dir, "last.pth")
        logging.info("Load weights from {}".format(weights_path))
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""):v for k,v in state_dict.items()}
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith("attn_mask") and args.arch.startswith("swin"):
                continue
            elif k.endswith("edge_index"):
                continue
            elif k.startswith("local_branch.before_gcn_fc"):
                continue
            elif k.endswith("relative_position_index") or k.endswith("relative_position_bias_table"):
                continue
            elif k.endswith("edge_weight"):
                filtered_state_dict[k] = v[:args.num_parts*(args.num_parts-1)*batch_size]
            elif k.endswith("pos_embed"):
                filtered_state_dict[k] = resample_abs_pos_embed(
                    v,
                    new_size=model.backbone.patch_embed.grid_size,
                    num_prefix_tokens=1,
                    interpolation='bilinear',
                    antialias=False,
                    verbose=True,
                )
            else:
                filtered_state_dict[k] = v
        model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)

    # Load class names for metrics
    class_names = []
    classes_file = os.path.join(cfg.dataset.txt_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip().split()[1].replace('_', ' ') for line in f.readlines()]
    else:
        class_names = [f'Class_{i}' for i in range(cfg.dataset.num_classes)]
    
    # Initialize training history tracker
    training_history = TrainingHistory(args.name) if not args.vis_mode else None

    scale_size = int(round(512*args.img_size/448))
    transform1 = transforms.Compose([
                                transforms.Resize([scale_size,scale_size]),
                                transforms.RandomCrop([args.img_size,args.img_size]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)
                                )])
    transform2 = transforms.Compose([
                                transforms.Resize([scale_size,scale_size]),
                                transforms.CenterCrop([args.img_size,args.img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)
                                )])
    
    [backbone_names, other_names], [backbone_params, other_params] = model.get_param_groups()
    logging.info(f"\nbackbone_names:{backbone_names}\nother_names:{other_names}\n")
    
    if not args.vis_mode:
        ### optimizers, loss functions
        if args.finetune:
            optimizers = [
                torch.optim.AdamW(model.parameters(), lr=cfg.train.backbone_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
            ]
        else:
            optimizers = [
                torch.optim.AdamW(backbone_params, lr=cfg.train.backbone_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
                torch.optim.AdamW(other_params, lr=cfg.train.others_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
            ]
        schedulers = [
            utils.WarmupCosineSchedule(optimizer, warmup_steps=cfg.train.warmup_epochs, t_total=int(1.1*args.epochs))
            for optimizer in optimizers
        ]
        ce_criterion = losses.LabelSmoothingCrossEntropy().to(device)
        mse_criterion = nn.MSELoss().to(device)
        rank_criterion = None if args.nopfi else nn.MarginRankingLoss(margin=0.05).to(device)
        if args.nodynamic:
            dynamic_criterion = None
        else:
            if model.has_cls_token:
                if args.ratio_weight is None:
                    args.ratio_weight = 2
                dynamic_criterion = losses.DistillDiffPruningLoss_dynamic(teacher_model, ratio_weight=args.ratio_weight, distill_weight=0.5, pruning_loc=model.pruning_loc, keep_ratio=model.keep_rate, mse_token=True)
            else:
                if args.ratio_weight is None:
                    args.ratio_weight = 10
                dynamic_criterion = losses.ConvNextDistillDiffPruningLoss(teacher_model, ratio_weight=args.ratio_weight, distill_weight=0.5, keep_ratio=model.keep_rate, swin_token=True)
            
        ### Resume training (FIXED to handle optimizer mismatch)
        start_epoch = 0
        best_val = None
        
        # Check for params.pth (contains epoch and optimizer states)
        params_path = os.path.join(args.weights_dir, "params.pth") if args.weights_dir else None
        
        if params_path and os.path.exists(params_path):
            logging.info(f"🔄 Found params.pth, attempting to resume training...")
            try:
                state_dict = torch.load(params_path, map_location="cpu", weights_only=False)
                start_epoch = state_dict["epoch"]
                best_val = state_dict.get("best_val", None)
                
                # Load optimizer states (handle mismatches gracefully)
                saved_optimizer_states = state_dict.get('optimizer_state_dicts', [])
                
                # Check if number of optimizers matches
                if isinstance(optimizers, list):
                    num_current_optimizers = len(optimizers)
                else:
                    num_current_optimizers = 1
                    optimizers = [optimizers]  # Convert to list for uniform handling
                
                num_saved_optimizers = len(saved_optimizer_states)
                
                if num_current_optimizers == num_saved_optimizers:
                    # Perfect match - load all optimizer states
                    for idx, opt_state in enumerate(saved_optimizer_states):
                        optimizers[idx].load_state_dict(opt_state)
                        logging.info(f"  ✓ Loaded optimizer {idx} state")
                    logging.info(f"✓ Successfully resumed from epoch {start_epoch}")
                    
                elif num_current_optimizers == 1 and num_saved_optimizers == 2:
                    # Resuming from 2-optimizer checkpoint to 1-optimizer mode (fine-tuning)
                    # We can only load the first optimizer's state as a starting point
                    logging.warning(f"⚠️  Checkpoint has {num_saved_optimizers} optimizers, but model uses {num_current_optimizers}")
                    logging.warning(f"    Loading only the first optimizer state (backbone)")
                    optimizers[0].load_state_dict(saved_optimizer_states[0])
                    logging.info(f"  ✓ Loaded backbone optimizer state")
                    logging.info(f"✓ Resumed from epoch {start_epoch} (optimizer partially loaded)")
                    
                elif num_current_optimizers == 2 and num_saved_optimizers == 1:
                    # Resuming from 1-optimizer checkpoint to 2-optimizer mode
                    # This is your current situation!
                    logging.warning(f"⚠️  Checkpoint has {num_saved_optimizers} optimizer, but model uses {num_current_optimizers}")
                    logging.warning(f"    Skipping optimizer state loading - optimizers will start fresh")
                    logging.warning(f"    Only epoch number and best_val will be restored")
                    logging.info(f"✓ Resumed from epoch {start_epoch} (without optimizer states)")
                    
                else:
                    logging.warning(f"⚠️  Optimizer count mismatch: checkpoint has {num_saved_optimizers}, model has {num_current_optimizers}")
                    logging.warning(f"    Skipping optimizer state loading")
                    logging.info(f"✓ Resumed from epoch {start_epoch} (without optimizer states)")
                
                if best_val is not None:
                    logging.info(f"✓ Previous best validation accuracy: {best_val:.4f}")
                    
            except Exception as e:
                logging.warning(f"⚠️  Failed to load resume state: {e}")
                logging.info("Starting fresh training instead...")
                start_epoch = 0
                best_val = None
        else:
            logging.info("No params.pth found, starting from epoch 0")

        # Data loading code
        train_dataset = BatchDataset(cfg.dataset.root_dir, cfg.train.stage, cfg.dataset.txt_dir, transform=transform1)
        if args.nopfi:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
        else:
            train_sampler = BalancedBatchSampler(train_dataset, args.sample_classes, args.sample_images)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfg.train.num_workers, pin_memory=True)

    val_dataset = BatchDataset(cfg.dataset.root_dir, cfg.val.stage, cfg.dataset.txt_dir, transform=transform2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.val.num_workers, pin_memory=True)

    if not args.vis_mode:
        logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
        start_time = datetime.now().replace(microsecond=0)
        loss_list = []
        acc_list = []
        val_loss_list = []
        val_acc_list = []
        scaler = None

        best_val_history = {}  # Track validation accuracy for each epoch
        
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # Training
            loss, acc = train(scaler, train_loader, model, [ce_criterion, rank_criterion, dynamic_criterion, mse_criterion], optimizers, epoch)
            [scheduler.step() for scheduler in schedulers]
            loss_list.append(loss)
            acc_list.append(acc)
            
            # Validation
            val_loss, val_acc, per_class_results, classification_rep = validate(
                val_loader, model, ce_criterion, epoch, class_names, cfg.dataset.num_classes
            )
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
            eta = utils.cal_eta(start_time, epoch+1, args.epochs)
            logging.info(
                "[Epoch:{}/{}] eta:{} lr:{:.6f} loss:{:.6f} acc:{:.6f} val_loss:{:.6f} val_acc:{:.6f}".format(
                    epoch+1, args.epochs, eta, optimizers[0].param_groups[0]['lr'], loss, acc, val_loss, val_acc
                )
            )
            
            utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
            
            # Update training history and save metrics
            current_lr = optimizers[0].param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            training_history.update(epoch+1, loss, acc, val_loss, val_acc, current_lr, epoch_time)
            
            # Save classification report
            if per_class_results:
                save_classification_report(per_class_results, classification_rep, epoch+1, args.name)
                print_top_and_bottom_classes(per_class_results, n=5)
            
            # Check overfitting
            overfitting_info = training_history.get_overfitting_indicator()
            if overfitting_info and overfitting_info['warning']:
                logging.info(f"⚠️  Overfitting Warning: Train-Val Gap = {overfitting_info['accuracy_gap']:.4f}")
            
            # save model
            torch.save({
                'epoch': epoch+1,
                'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers], 
                'best_val': best_val,
                }, params_save_path)
            torch.save(model.state_dict(), os.path.join(model_last_path))

            best_val_history[epoch + 1] = val_acc
            
            # Save checkpoint to Google Drive
            checkpoint_path = save_checkpoint_to_gdrive(
                epoch=epoch+1, 
                model=model, 
                optimizers=optimizers, 
                best_val=best_val, 
                exp_name=args.name,
                best_val_history=best_val_history
            )
            
            if best_val is None or val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), model_best_path)
                logging.info("Saved best model.")
                save_best_checkpoint_to_gdrive(checkpoint_path, args.name)

        utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
        
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info('Training time: {:.2f} hours'.format(round((datetime.now().replace(microsecond=0)-start_time).seconds/3600, 2)))


def cal_loss(model, inputs, targets, criterions, epoch, with_acc=False):
    ce_criterion, rank_criterion, dynamic_criterion, mse_criterion = criterions
    acc = 0
    if args.nopfi:
        if model.has_cls_token:
            logits, global_feature, decision_mask_list, _part_masks, global_patch_features, decision_mask = model(inputs)
        else:
            logits, global_feature, decision_mask_list, _part_masks = model(inputs)
    else:
        if model.has_cls_token:
            logits, pfi_logits, pfi_targets, self_scores, other_scores, global_feature, decision_mask_list, _part_masks, global_patch_features, decision_mask = model(inputs, targets)
        else:
            logits, pfi_logits, pfi_targets, self_scores, other_scores, global_feature, decision_mask_list, _part_masks = model(inputs, targets)
    loss = ce_criterion(logits, targets)
    loss_str = f"ce_loss:{loss.item()}, "
    if not args.nopfi:
        pfi_loss = 2 * ce_criterion(pfi_logits, pfi_targets)
        loss += pfi_loss
        loss_str += f"pfi_loss:{pfi_loss.item()}, "
        flags = torch.ones([self_scores.size(0),]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flags)
        loss += rank_loss
        loss_str += f"rank_loss:{rank_loss.item()}, "
    if dynamic_criterion:
        if model.has_cls_token:
            dynamic_loss = dynamic_criterion(inputs, [global_feature, global_patch_features, decision_mask, decision_mask_list])
        else:
            dynamic_loss = dynamic_criterion(inputs, [global_feature, decision_mask_list])
        loss += dynamic_loss
        loss_str += f"dynamic_loss:{dynamic_loss.item()}"

    if with_acc:
        acc = utils.cal_accuracy(logits, targets)
    loss.backward()

    if torch.isnan(loss):
        logging.error("Nan is detected in total loss!")
        exit(-1)

    if with_acc:
        return loss, loss_str, acc
    else:
        return loss, loss_str


def train(scaler, train_loader, model, criterions, optimizers, epoch):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    total = len(train_loader)
    for i, (inputs, targets, filenames) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        [optimizer.zero_grad() for optimizer in optimizers]
        loss, loss_str, acc = cal_loss(model, inputs, targets, criterions, epoch, with_acc=True)
        [optimizer.step() for optimizer in optimizers]

        if i % cfg.train.log_step == 0:
            logging.info("Trainning epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f} loss_detail: {}".format(epoch+1, args.epochs, i+1, total, loss.item(), acc, loss_str))
        batch_loss_list.append(loss.item())
        batch_acc_list.append(acc)

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def validate(val_loader, model, ce_criterion, epoch, class_names=None, num_classes=200):
    model.eval()
    batch_loss_list = []
    batch_acc_list = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, targets, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            if len(outputs.size()) == 1:
                outputs = torch.unsqueeze(outputs, dim=0)

            loss = ce_criterion(outputs, targets)
            batch_loss_list.append(loss.item())

            acc = utils.cal_accuracy(outputs, targets)
            batch_acc_list.append(acc)
            
            # Collect predictions for metrics
            _, pred = torch.max(outputs.data, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            if i % cfg.train.log_step == 0:
                logging.info("Validating epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f}".format(epoch+1, args.epochs, i+1, total, loss.item(), acc))
    
    # Compute per-class metrics
    per_class_results, classification_rep = None, None
    if class_names:
        try:
            per_class_results, classification_rep = compute_per_class_metrics(all_preds, all_labels, class_names, num_classes)
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
    
    return np.mean(batch_loss_list), np.mean(batch_acc_list), per_class_results, classification_rep


def visualize(val_loader, model, part_attn_save_dir):
    model.eval()
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, targets, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            patch_features, decision_mask, parts_masks = model(inputs, flag="visual")
            B, L, D = parts_masks.shape
            size = int(math.sqrt(L))
            parts_masks = parts_masks[0].reshape(size, size, D).detach().cpu().numpy()
            rows = 3
            columns = 4
            fig, axs = plt.subplots(rows,columns)
            for row in range(rows):
                for col in range(columns):
                    idx = row*columns+col
                    if idx < D:
                        axs[row,col].imshow(parts_masks[:,:,idx], cmap="jet", vmin=0, vmax=1)
            plt.savefig(os.path.join(part_attn_save_dir, "{}.png".format(i)))
            plt.close()
            if i % cfg.train.log_step == 0:
                logging.info("Visualize: batch:{}/{}".format(i+1, total))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--arch", type=str, required=True, choices=supported_arch, help="model architecture")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--sample_classes", type=int, default=2, help="sample n classes from all classes each time")
    parser.add_argument("--sample_images", type=int, default=10, help="sample n images from each classes each time")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--window_size", type=int, default=7, help="image_size:224,window_size:7, image_size:384,window_size:12;")
    parser.add_argument("--weights_dir", type=str, default=None, help=".pth weights directory")
    parser.add_argument("--lr", type=float, default=None, help="backbone learning rate")
    parser.add_argument("--distill", action="store_true", help="use teacher model")
    parser.add_argument("--from_scratch", action="store_true", help="without pretrain weights")
    parser.add_argument("--finetune", action="store_true", help="train in finetuning mode")
    parser.add_argument("--nopfi", action="store_true", help="without pfi module")
    parser.add_argument("--nolocal", action="store_true", help="without local branch")
    parser.add_argument("--nodynamic", action="store_true", help="without dynamic design in global branch")
    parser.add_argument("--vis_mode", action="store_true", help="only visualize")

    parser.add_argument("--ratio_weight", type=int, default=None, help="if None, set 2 for vit, set 10 for swin.")

    parser.add_argument("--pruning_loc", type=int, default=6, help="pruning_loc: 2, 4, 6, 8, 10, 12, 14, 16")
    parser.add_argument("--base_keep_rate", type=float, default=0.5, help="base keep rate in DynamicSwin backbone")

    parser.add_argument("--gaussian_ksize", type=int, default=15, help="gaussian_ksize: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21")
    parser.add_argument("--num_parts", type=int, default=8, help="number of parts in LocalBranch")
    parser.add_argument("--part_channels", type=int, default=16, help="number of channels for each part in LocalBranch")
    parser.add_argument("--local_from_stage", type=int, default=-1, help="[0,1,2,3] for SwinT or -1 for all architecture")
    parser.add_argument("--local_depth", type=int, default=3, help="number of blocks in LocalBranch")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)

    if args.gaussian_ksize == 0:
        args.gaussian_ksize = None
    if args.lr:
        cfg.train.backbone_lr = args.lr
        cfg.train.others_lr = args.lr * 5
    if args.vis_mode:
        print("[in VISUALIZE mode]")
        assert args.weights_dir is not None, "args.weights_dir shouldn't be None."

    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    if not args.vis_mode:
        model_best_path  = "./middle/models/{}-{}/best.pth".format(args.name, timestamp)
        model_last_path  = "./middle/models/{}-{}/last.pth".format(args.name, timestamp)
        params_save_path = "./middle/models/{}-{}/params.pth".format(args.name, timestamp)
        log_path = "./middle/logs/{}-{}.log".format(args.name, timestamp)
        history_save_path = "./middle/history/{}-{}.png".format(args.name, timestamp)
        os.makedirs("./middle/logs/", exist_ok=True)
        os.makedirs("./middle/models/{}-{}/".format(args.name, timestamp), exist_ok=True)
        os.makedirs("./middle/history/", exist_ok=True)
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
        )
    else:
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()]
        )
    part_attn_save_dir = "./middle/parts-attn/"
    os.makedirs("./middle/parts-attn/", exist_ok=True)
    logging.info(f"timestamp:{timestamp}")

    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device("cpu")

    try:
        main()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)