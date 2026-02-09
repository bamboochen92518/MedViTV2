import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import requests
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import csv
import pandas as pd

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from datasets import build_dataset
from distutils.util import strtobool
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large
#from MedViTV1 import MedViT_small, MedViT_base, MedViT_large


model_classes = {
    'MedViT_tiny': MedViT_tiny,
    'MedViT_small': MedViT_small,
    'MedViT_base': MedViT_base,
    'MedViT_large': MedViT_large
}

model_urls = {
    "MedViT_tiny": "https://dl.dropbox.com/scl/fi/496jbihqp360jacpji554/MedViT_tiny.pth?rlkey=6hb9froxugvtg8l639jmspxfv&st=p9ef06j8&dl=0",
    "MedViT_small": "https://dl.dropbox.com/scl/fi/6nnec8hxcn5da6vov7h2a/MedViT_small.pth?rlkey=yf5twra1cv6ep2oqr79tbzyg5&st=rwx5hy8z&dl=0",
    "MedViT_base": "https://dl.dropbox.com/scl/fi/q5c0u515dd4oc8j55bhi9/MedViT_base.pth?rlkey=5duw3uomnsyjr80wykvedjhas&st=incconx4&dl=0",
    "MedViT_large": "https://dl.dropbox.com/scl/fi/owujijpsl6vwd481hiydd/MedViT_large.pth?rlkey=cx9lqb4a1288nv4xlmux13zoe&st=kcehwbrb&dl=0"
}

def download_checkpoint(url, path):
    print(f"Downloading checkpoint from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    print(f"Checkpoint downloaded and saved to {path}")

# Define the MNIST training routine with detailed metrics logging
def train_mnist(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path, data_flag, task, num_classes_in_group=None, group_labels=None):
    best_acc = 0.0
    
    # Prepare CSV file for logging metrics
    csv_path = save_path.replace('.pth', '_metrics.csv')
    
    # Determine number of classes
    if num_classes_in_group is not None:
        # Group training mode
        num_classes = num_classes_in_group
        use_medmnist_evaluator = False
    else:
        # Original training mode
        info = INFO[data_flag]
        num_classes = len(info['label'])
        use_medmnist_evaluator = True
    
    # Create CSV header with dynamic per-class AP and AUC columns
    header = ['epoch', 'train_loss', 'val_accuracy', 'f1_score', 'auc', 'mAP']
    
    # Use original label names if group_labels is provided
    if group_labels is not None:
        # Add per-class AUC columns
        for label in group_labels:
            header.append(f'AUC_class_{label}')
        # Add per-class AP columns
        for label in group_labels:
            header.append(f'AP_class_{label}')
    else:
        # Add per-class AUC columns
        for i in range(num_classes):
            header.append(f'AUC_class_{i}')
        # Add per-class AP columns
        for i in range(num_classes):
            header.append(f'AP_class_{i}')
    
    # Initialize CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    print(f"📊 Metrics will be saved to: {csv_path}")
    
    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            
            if task == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
                loss = loss_function(outputs, labels)
            else:
                labels = labels.squeeze().long()
                loss = loss_function(outputs.squeeze(0), labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        # Validation phase
        net.eval()
        y_true_list = []
        y_score_list = []
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                inputs, targets = val_data
                outputs = net(inputs.to(device))
                
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = torch.sigmoid(outputs)  # Use sigmoid for multi-label
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                
                y_true_list.append(targets.cpu())
                y_score_list.append(outputs.cpu())
        
        # Concatenate all predictions and targets
        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_score = torch.cat(y_score_list, dim=0).numpy()
        
        # Calculate per-class AUC first (needed for both evaluation modes)
        auc_per_class = []
        
        if task == 'multi-label, binary-class':
            # Multi-label: calculate AUC for each class
            for i in range(num_classes):
                try:
                    if len(np.unique(y_true[:, i])) > 1:  # Check if class has both 0 and 1
                        auc = roc_auc_score(y_true[:, i], y_score[:, i])
                        auc_per_class.append(auc)
                    else:
                        auc_per_class.append(0.0)
                except:
                    auc_per_class.append(0.0)
        else:
            # Multi-class: calculate AUC using one-vs-rest
            y_true_labels = y_true.flatten().astype(int)
            y_true_onehot = label_binarize(y_true_labels, classes=list(range(num_classes)))
            
            # Handle binary classification case (only 2 classes)
            if num_classes == 2 and y_true_onehot.shape[1] == 1:
                y_true_onehot = np.hstack([1 - y_true_onehot, y_true_onehot])
            
            for i in range(num_classes):
                try:
                    if len(np.unique(y_true_onehot[:, i])) > 1:
                        auc = roc_auc_score(y_true_onehot[:, i], y_score[:, i])
                        auc_per_class.append(auc)
                    else:
                        auc_per_class.append(0.0)
                except:
                    auc_per_class.append(0.0)
        
        # Calculate mean AUC
        auc_score = np.mean(auc_per_class)
        
        # Calculate metrics
        if use_medmnist_evaluator:
            # Use MedMNIST Evaluator for original full dataset
            evaluator = Evaluator(data_flag, 'test', size=224, root='./data')
            metrics = evaluator.evaluate(y_score)
            # Note: We override the evaluator's AUC with our per-class calculated one for consistency
            # auc_score_evaluator = metrics[0]  # This is MedMNIST's AUC
            val_accuracy = metrics[1]
        else:
            # Calculate metrics manually for group training
            if task == 'multi-label, binary-class':
                # For multi-label: use threshold 0.5
                y_pred = (y_score > 0.5).astype(int)
                # Accuracy: exact match ratio
                val_accuracy = np.mean(np.all(y_pred == y_true, axis=1))
            else:
                # For multi-class
                y_pred = np.argmax(y_score, axis=1)
                y_true_labels = y_true.flatten().astype(int)
                val_accuracy = np.mean(y_pred == y_true_labels)
        
        # Calculate F1 score
        if task == 'multi-label, binary-class':
            # For multi-label classification
            y_pred = (y_score > 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            # For multi-class classification
            y_pred = np.argmax(y_score, axis=1)
            if len(y_true.shape) > 1:
                y_true_labels = y_true.flatten()
            else:
                y_true_labels = y_true
            f1 = f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        
        # Calculate per-class AP and mAP
        ap_per_class = []
        
        if task == 'multi-label, binary-class':
            # Multi-label case
            for i in range(num_classes):
                try:
                    if len(np.unique(y_true[:, i])) > 1:  # Check if class has both 0 and 1
                        ap = average_precision_score(y_true[:, i], y_score[:, i])
                        ap_per_class.append(ap)
                    else:
                        ap_per_class.append(0.0)
                except:
                    ap_per_class.append(0.0)
        else:
            # Multi-class case: convert to one-hot encoding
            y_true_labels = y_true.flatten().astype(int)
            y_true_onehot = label_binarize(y_true_labels, classes=list(range(num_classes)))
            
            # Handle binary classification case (only 2 classes)
            if num_classes == 2 and y_true_onehot.shape[1] == 1:
                y_true_onehot = np.hstack([1 - y_true_onehot, y_true_onehot])
            
            for i in range(num_classes):
                try:
                    ap = average_precision_score(y_true_onehot[:, i], y_score[:, i])
                    ap_per_class.append(ap)
                except:
                    ap_per_class.append(0.0)
        
        # Calculate mAP (mean Average Precision)
        mAP = np.mean(ap_per_class)
        
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        
        # Print metrics
        print(f'[epoch {epoch + 1}/{epochs}] train_loss: {avg_train_loss:.4f}  '
              f'val_acc: {val_accuracy:.4f}  f1: {f1:.4f}  auc: {auc_score:.4f}  mAP: {mAP:.4f}')
        
        # Print per-class AUC and AP
        auc_str = "  Per-class AUC: " + ", ".join([f"class{i}={auc:.4f}" for i, auc in enumerate(auc_per_class)])
        print(auc_str)
        ap_str = "  Per-class AP: " + ", ".join([f"class{i}={ap:.4f}" for i, ap in enumerate(ap_per_class)])
        print(ap_str)
        
        # Save metrics to CSV: [epoch, train_loss, val_acc, f1, auc, mAP, AUC_class_0, ..., AP_class_0, ...]
        row = [epoch + 1, avg_train_loss, val_accuracy, f1, auc_score, mAP] + auc_per_class + ap_per_class
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Save best model
        if val_accuracy > best_acc:
            print('\n💾 Saving checkpoint...')
            best_acc = val_accuracy
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'f1': f1,
                'auc': auc_score,
                'mAP': mAP,
            }
            torch.save(state, save_path)

    print('\n✅ Finished Training')
    print(f'📊 Metrics saved to: {csv_path}')
    print(f'🏆 Best validation accuracy: {best_acc:.4f}')

# Define the non-MNIST training routine
def specificity_per_class(conf_matrix):
    """Calculates specificity for each class."""
    specificity = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity.append(tn / (tn + fp))
    return specificity

def overall_accuracy(conf_matrix):
    """Calculates overall accuracy for multi-class."""
    tp_tn_sum = conf_matrix.trace()  # Sum of all diagonal elements (TP for all classes)
    total_sum = conf_matrix.sum()  # Sum of all elements in the matrix
    return tp_tn_sum / total_sum

def train_other(epochs, net, train_loader, test_loader, optimizer, scheduler, loss_function, device, save_path):
    best_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # Training Loop
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"
        
        # Validation Loop
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store raw probabilities/logits for AUC
        acc = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # Raw outputs (logits)
                probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
                
                predict_y = torch.max(probs, dim=1)[1]  # Predicted class

                # Collect predictions, labels, and probabilities
                all_preds.extend(predict_y.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
        # Calculate metrics
        val_accurate = acc / len(test_loader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')  # Sensitivity
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Confusion Matrix for multi-class
        conf_matrix = confusion_matrix(all_labels, all_preds)
        specificity = specificity_per_class(conf_matrix)  # List of specificities per class
        avg_specificity = sum(specificity) / len(specificity)  # Average specificity

        # Overall Accuracy calculation
        overall_acc = overall_accuracy(conf_matrix)

        # One-hot encode the labels for AUC calculation
        n_classes = len(conf_matrix)
        all_labels_one_hot = label_binarize(all_labels, classes=list(range(n_classes)))

        try:
            # Compute AUC for multi-class
            auc = roc_auc_score(all_labels_one_hot, all_probs, multi_class='ovr')
        except ValueError:
            auc = float('nan')  # Handle edge case where AUC can't be computed

        # Print metrics
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} '
              f'val_accuracy: {val_accurate:.4f} precision: {precision:.4f} '
              f'recall: {recall:.4f} specificity: {avg_specificity:.4f} '
              f'f1_score: {f1:.4f} auc: {auc:.4f} overall_accuracy: {overall_acc:.4f}')
        
        #print(f'lr: {scheduler.get_last_lr()[-1]:.8f}')
        
        # Save best model
        if val_accurate > best_acc:
            print('\nSaving checkpoint...')
            best_acc = val_accurate
            state = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
            }
            torch.save(state, save_path)

    print('Finished Training')

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    model_name = args.model_name
    dataset_name = args.dataset
    pretrained = args.pretrained
    
    # ✨ Early check: Determine save path and check if training already completed
    if hasattr(args, 'label_head') and args.label_head is not None:
        # Label range mode: use head-tail in filename
        save_path = f'./results/{model_name}_{dataset_name}_class{args.label_head}to{args.label_tail}.pth'
    elif hasattr(args, 'group') and args.group is not None:
        # Group training mode
        save_path = f'./results/{model_name}_{dataset_name}_group{args.group}.pth'
    else:
        # Original full dataset mode
        save_path = f'./results/{model_name}_{dataset_name}.pth'
    
    # Check if CSV file already exists
    csv_path = save_path.replace('.pth', '_metrics.csv')
    if os.path.exists(csv_path):
        print("\n" + "="*60)
        print("⚠️  Training already completed!")
        print("="*60)
        print(f"  CSV file found: {csv_path}")
        print(f"  Skipping training to avoid duplication.")
        print(f"  To retrain, delete the CSV file first.")
        print("="*60 + "\n")
        return
    
    # Determine loss function based on dataset
    if args.dataset.endswith('mnist'):
        info = INFO[args.dataset]
        task = info['task']
        
        # Use appropriate loss function based on task type
        if task == "multi-label, binary-class":
            loss_function = nn.BCEWithLogitsLoss()
        else:
            loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()
    
    model_class = model_classes.get(model_name)

    batch_size = args.batch_size
    lr = args.lr
    
    train_dataset, test_dataset, nb_classes = build_dataset(args=args)
    val_num = len(test_dataset)
    train_num = len(train_dataset)
    
    # scheduler max iteration
    eta = args.epochs * train_num // args.batch_size

    # Select model
    if model_name in model_classes:
        model_class = model_classes[model_name]
        net = model_class(num_classes=nb_classes).cuda()
        if pretrained:
            checkpoint_path = args.checkpoint_path
            if not os.path.exists(checkpoint_path):
                checkpoint_url = model_urls.get(model_name)
                if not checkpoint_url:
                    raise ValueError(f"Checkpoint URL for model {model_name} not found.")
                download_checkpoint(checkpoint_url, f'./{model_name}.pth')
                checkpoint_path = f'./{model_name}.pth'

            checkpoint = torch.load(checkpoint_path)
            state_dict = net.state_dict()
            for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            net.load_state_dict(checkpoint, strict=False)
    else:
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=nb_classes).cuda()

    
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eta, eta_min=5e-6)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = args.epochs
    best_acc = 0.0
    
    # Ensure results directory exists
    os.makedirs('./results', exist_ok=True)
    
    train_steps = len(train_loader)

    if dataset_name.endswith('mnist'):
        # Pass nb_classes and group_labels when using group training or label range, None otherwise
        if hasattr(args, 'label_head') and args.label_head is not None:
            # Label range mode
            num_classes_param = nb_classes
            from datasets import get_label_range
            group_labels_param = get_label_range(dataset_name, args.label_head, args.label_tail)
        elif hasattr(args, 'group') and args.group is not None:
            # Group training mode
            num_classes_param = nb_classes
            from datasets import get_label_groups
            label_groups = get_label_groups(dataset_name)
            group_labels_param = label_groups[args.group] if label_groups else None
        else:
            # Original full dataset mode
            num_classes_param = None
            group_labels_param = None
        
        train_mnist(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path, dataset_name, task, 
        num_classes_in_group=num_classes_param, group_labels=group_labels_param)
    else:
        train_other(epochs, net, train_loader, test_loader,
        optimizer, scheduler, loss_function, device, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for MedViT models.')
    parser.add_argument('--model_name', type=str, default='MedViT_tiny', help='Model name to use.')
    #tissuemnist, pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist, retinamnist, breastmnist, bloodmnist,
    #organamnist, organcmnist, organsmnist'
    parser.add_argument('--dataset', type=str, default='PAD', help='Dataset to use.')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--pretrained', type=lambda x: bool(strtobool(x)), default=False, help="Whether to use pretrained weights (True/False).")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/MedViT_tiny.pth', help='Path to the checkpoint file.')
    
    # ✨ Add group parameter
    parser.add_argument('--group', type=int, default=None, 
                       help='Train on a specific label group (e.g., 0, 1, or 2). '
                            'If None, trains on all classes. '
                            'Available for datasets with group definitions (e.g., chestmnist).')
    
    # ✨ Add label range parameters
    parser.add_argument('--label_head', type=int, default=None,
                       help='Starting class for label range (inclusive). '
                            'Must be used together with --label_tail. '
                            'Uses sorted order: [3, 2, 0, 5, 4, 7, 8, 12, 1, 10, 9, 11, 6, 13] for chestmnist.')
    parser.add_argument('--label_tail', type=int, default=None,
                       help='Ending class for label range (inclusive). '
                            'Must be used together with --label_head.')
    
    # ✨ Add sample parameter
    parser.add_argument('--sample', type=float, default=None,
                       help='Fraction of dataset to use for training (0 < sample <= 1.0). '
                            'E.g., --sample 0.1 uses 10%% of the data. '
                            'If None, uses full dataset.')

    args = parser.parse_args()
    
    # Validate label range parameters
    if (args.label_head is not None and args.label_tail is None) or \
       (args.label_head is None and args.label_tail is not None):
        parser.error("--label_head and --label_tail must be used together")
    
    if args.label_head is not None and args.group is not None:
        parser.error("Cannot use both --group and --label_head/--label_tail at the same time")
    
    # Validate sample parameter
    if args.sample is not None and (args.sample <= 0 or args.sample > 1.0):
        parser.error("--sample must be between 0 and 1.0 (exclusive of 0, inclusive of 1.0)")
    
    main(args)

# python main.py --model_name 'convnext_tiny' --dataset 'PAD'
