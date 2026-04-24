import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor


SEED = 42
MODEL_NAME = "google/vit-base-patch16-224-in21k"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT classifier for pizza vs not_pizza.")
    parser.add_argument("--data-dir", default="/opt/project/dataset/pizza_not_pizza")
    parser.add_argument("--output-dir", default="/opt/project/tmp/class11_vit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-layers", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--balance-threshold", type=float, default=1.5)
    parser.add_argument("--disable-balanced-sampling", action="store_true")
    parser.add_argument("--disable-threshold-search", action="store_true")
    parser.add_argument(
        "--train-exclude-file",
        default=None,
        help="Optional text file listing train image paths to exclude, one per line.",
    )
    return parser.parse_args()


def get_processor_config():
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    size = processor.size
    if hasattr(size, "get"):
        image_size = size.get("height", size.get("shortest_edge", 224))
    else:
        image_size = size
    return image_size, processor.image_mean, processor.image_std


def build_transforms(image_size, image_mean, image_std):
    resize_size = int(image_size * 256 / 224)
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]),
        "val": transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ]),
    }


def build_datasets(data_dir, data_transforms):
    required_splits = ["train", "val"]
    for split in required_splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(f"Missing required split folder: {split_path}")

    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in required_splits
    }

    train_classes = image_datasets["train"].classes
    val_classes = image_datasets["val"].classes
    if train_classes != val_classes:
        raise ValueError(
            f"Train/val class mismatch: train={train_classes}, val={val_classes}"
        )

    return image_datasets


def normalize_rel_path(path):
    return path.replace("\\", "/").lstrip("./")


def load_exclude_paths(exclude_file):
    with open(exclude_file, "r", encoding="utf-8") as f:
        return {
            normalize_rel_path(line.strip())
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def apply_train_exclusions(train_dataset, data_dir, exclude_file):
    if not exclude_file:
        return 0
    if not os.path.isfile(exclude_file):
        raise FileNotFoundError(f"Train exclude file not found: {exclude_file}")

    excluded_rel_paths = load_exclude_paths(exclude_file)
    if not excluded_rel_paths:
        return 0

    kept_samples = []
    excluded_count = 0
    data_dir = os.path.abspath(data_dir)
    train_root = os.path.abspath(os.path.join(data_dir, "train"))

    for path, label in train_dataset.samples:
        rel_from_train = normalize_rel_path(os.path.relpath(path, train_root))
        rel_from_data_dir = normalize_rel_path(os.path.relpath(path, data_dir))
        rel_from_cwd = normalize_rel_path(os.path.relpath(path))
        if (
            rel_from_train in excluded_rel_paths
            or rel_from_data_dir in excluded_rel_paths
            or rel_from_cwd in excluded_rel_paths
        ):
            excluded_count += 1
            continue
        kept_samples.append((path, label))

    if not kept_samples:
        raise ValueError("All training samples were excluded. Check the exclude file.")

    train_dataset.samples = kept_samples
    train_dataset.imgs = kept_samples
    train_dataset.targets = [label for _, label in kept_samples]
    return excluded_count


def build_train_sampler(train_dataset):
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_dataset.targets]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(image_datasets, batch_size, num_workers, use_balanced_sampling):
    train_sampler = build_train_sampler(image_datasets["train"]) if use_balanced_sampling else None
    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    return dataloaders, dataset_sizes


def get_split_class_counts(dataset):
    counts = np.bincount(dataset.targets)
    return counts.tolist()


def get_imbalance_ratio(class_counts):
    non_zero_counts = [count for count in class_counts if count > 0]
    if not non_zero_counts:
        return 1.0
    return max(non_zero_counts) / min(non_zero_counts)


def build_model(class_names, device):
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in enumerate(class_names)}

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def freeze_backbone(model):
    for param in model.vit.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_top_layers(model, num_layers):
    freeze_backbone(model)
    if num_layers <= 0:
        return

    encoder_layers = model.vit.encoder.layer
    num_layers = min(num_layers, len(encoder_layers))

    for layer in encoder_layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    for param in model.vit.layernorm.parameters():
        param.requires_grad = True


def build_optimizer(model, lr):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)


def compute_classification_metrics(all_labels, all_preds):
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return {
        "accuracy": epoch_acc,
        "precision": epoch_precision,
        "recall": epoch_recall,
        "f1": epoch_f1,
    }


def find_best_binary_threshold(all_labels, positive_probs):
    labels = np.asarray(all_labels)
    probs = np.asarray(positive_probs)

    best_threshold = 0.5
    best_metrics = None
    for threshold in np.linspace(0.2, 0.8, 61):
        preds = (probs >= threshold).astype(int)
        metrics = compute_classification_metrics(labels, preds)
        if best_metrics is None or (
            metrics["accuracy"] > best_metrics["accuracy"]
            or (
                metrics["accuracy"] == best_metrics["accuracy"]
                and metrics["f1"] > best_metrics["f1"]
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def run_one_epoch(model, dataloader, criterion, optimizer, device, phase, threshold_search_enabled):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_positive_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        if outputs.shape[1] == 2:
            positive_probs = torch.softmax(outputs, dim=1)[:, 1]
            all_positive_probs.extend(positive_probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds)
    best_threshold = 0.5

    if (
        threshold_search_enabled
        and phase == "val"
        and all_positive_probs
    ):
        best_threshold, threshold_metrics = find_best_binary_threshold(all_labels, all_positive_probs)
        metrics = threshold_metrics

    return {
        "loss": epoch_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "threshold": best_threshold,
    }


def save_checkpoint(model, class_names, save_path, history, best_val_metrics, args, image_size, image_mean, image_std):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": MODEL_NAME,
        "best_val_metrics": best_val_metrics,
        "history": history,
        "train_args": vars(args),
        "image_size": image_size,
        "image_mean": image_mean,
        "image_std": image_std,
    }
    torch.save(checkpoint, save_path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.output_dir, "vit_model_best.pth")
    history_path = os.path.join(args.output_dir, "vit_training_history.json")

    image_size, image_mean, image_std = get_processor_config()

    print("=" * 60)
    print("ViT Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Head-only epochs: {args.head_epochs}")
    print(f"Head learning rate: {args.head_lr}")
    print(f"Fine-tune learning rate: {args.finetune_lr}")
    print(f"Fine-tune top layers: {args.finetune_layers}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Num workers: {args.num_workers}")
    print(f"Seed: {args.seed}")

    data_transforms = build_transforms(image_size, image_mean, image_std)
    image_datasets = build_datasets(args.data_dir, data_transforms)
    excluded_train_count = apply_train_exclusions(
        image_datasets["train"],
        data_dir=args.data_dir,
        exclude_file=args.train_exclude_file,
    )
    class_names = image_datasets["train"].classes
    train_class_counts = get_split_class_counts(image_datasets["train"])
    val_class_counts = get_split_class_counts(image_datasets["val"])
    train_imbalance_ratio = get_imbalance_ratio(train_class_counts)
    use_balanced_sampling = (
        not args.disable_balanced_sampling
        and train_imbalance_ratio >= args.balance_threshold
    )
    threshold_search_enabled = not args.disable_threshold_search
    dataloaders, dataset_sizes = build_dataloaders(
        image_datasets=image_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_balanced_sampling=use_balanced_sampling,
    )
    print(f"Class names: {class_names}")
    print(f"Class to index: {image_datasets['train'].class_to_idx}")
    print(f"Train size: {dataset_sizes['train']}")
    print(f"Val size: {dataset_sizes['val']}")
    print(f"Train class counts: {train_class_counts}")
    print(f"Val class counts: {val_class_counts}")
    if excluded_train_count:
        print(f"Excluded train images: {excluded_train_count}")
    print(f"Train imbalance ratio: {train_imbalance_ratio:.3f}")
    print(f"Validation threshold search: {'enabled' if threshold_search_enabled else 'disabled'}")
    if use_balanced_sampling:
        print("Training uses balanced sampling to reduce class bias.")
    else:
        print("Training uses normal shuffled batches because the class balance is already reasonable.")

    model = build_model(class_names, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_val_f1 = -1.0
    best_val_metrics = None
    early_stop_counter = 0
    history = []

    current_stage = None
    optimizer = None
    scheduler = None
    best_head_model_wts = None

    for epoch in range(args.epochs):
        stage = "head_only" if epoch < args.head_epochs else "fine_tune"

        if stage != current_stage:
            current_stage = stage
            if stage == "head_only":
                freeze_backbone(model)
                optimizer = build_optimizer(model, args.head_lr)
            else:
                if best_head_model_wts is not None:
                    model.load_state_dict(best_head_model_wts)
                unfreeze_top_layers(model, args.finetune_layers)
                optimizer = build_optimizer(model, args.finetune_lr)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
            )
            print("\n" + "=" * 60)
            print(f"Switching to stage: {stage}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8f}")

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()

        train_metrics = run_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            threshold_search_enabled=False,
        )
        val_metrics = run_one_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            phase="val",
            threshold_search_enabled=threshold_search_enabled,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time

        print(
            f"Stage: {stage} | "
            f"LR: {current_lr:.8f} | "
            f"Epoch time: {epoch_time:.2f} sec"
        )
        print(
            "Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            "Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Threshold: {val_metrics['threshold']:.3f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "stage": stage,
            "train": train_metrics,
            "val": val_metrics,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }
        history.append(epoch_record)

        is_better = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                val_metrics["accuracy"] == best_val_accuracy
                and val_metrics["f1"] > best_val_f1
            )
        )
        if is_better:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = {
                "epoch": epoch + 1,
                "stage": stage,
                **val_metrics,
            }
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model=model,
                class_names=class_names,
                save_path=checkpoint_path,
                history=history,
                best_val_metrics=best_val_metrics,
                args=args,
                image_size=image_size,
                image_mean=image_mean,
                image_std=image_std,
            )
            print(f"New best model saved to {checkpoint_path}")
            early_stop_counter = 0
            if stage == "head_only":
                best_head_model_wts = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            print(f"No validation F1 improvement. Early-stop counter: {early_stop_counter}/{args.patience}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if early_stop_counter >= args.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    if best_val_metrics is not None:
        print(
            f"Best validation result came from epoch {best_val_metrics['epoch']} "
            f"({best_val_metrics['stage']}) | "
            f"Loss: {best_val_metrics['loss']:.4f} | "
            f"Acc: {best_val_metrics['accuracy']:.4f} | "
            f"Precision: {best_val_metrics['precision']:.4f} | "
            f"Recall: {best_val_metrics['recall']:.4f} | "
            f"F1: {best_val_metrics['f1']:.4f}"
        )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history JSON: {history_path}")


if __name__ == "__main__":
    main()
