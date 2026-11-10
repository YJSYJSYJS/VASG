"""Shared argument, data, logging and checkpoint utilities for pretraining."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset.acav import ACAV

def build_parser(stage):
    p = argparse.ArgumentParser(description=f'VASG {stage} pretraining')
    p.add_argument('--mel-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=100 if stage == 'reconstruction' else 50)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-decay', type=float, default=0.99)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-size', type=int, default=100)
    p.add_argument('--embedding-dim', type=int, default=64)
    p.add_argument('--num-embeddings', type=int, default=512)
    p.add_argument('--commitment-weight', type=float, default=0.25)
    p.add_argument('--reconstruction-weight', type=float, default=1e6 if stage == 'reconstruction' else 1.0)
    p.add_argument('--embedding-weight', type=float, default=1.0)
    return p


def prepare_training(args, parser, stage):
    for name in ('epochs', 'batch_size', 'embedding_dim', 'num_embeddings'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.workers < 0 or args.val_size < 0 or args.lr <= 0 or args.lr_decay <= 0:
        parser.error('workers/val-size must be nonnegative and lr/lr-decay must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output = args.output_dir
    # Do not silently overwrite an earlier experiment.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f'Output directory must be empty: {output}')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['stage'] = stage
    config['torch_version'] = str(torch.__version__)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    image_transform = transforms.Compose([
        transforms.Resize(286), transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    val_transform = transforms.Compose([
        transforms.Resize(286), transforms.CenterCrop(256), transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    image_dir = getattr(args, 'image_dir', None)
    dataset = ACAV(args.mel_dir, image_dir, image_transform)
    if args.val_size >= len(dataset):
        parser.error('val-size must be smaller than the dataset')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_indices, train_indices = indices[:args.val_size], indices[args.val_size:]
    val_dataset = ACAV(args.mel_dir, image_dir, val_transform)
    (output / 'split.json').write_text(json.dumps({
        'train': [str(dataset.files[i]) for i in train_indices],
        'validation': [str(dataset.files[i]) for i in val_indices]}, indent=2) + '\n')
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=args.batch_size,
        num_workers=args.workers)
    return device, output, config, train_loader, val_loader


def save_epoch(output, index, train_metrics, val_metrics, generator, projection,
               optimizer, scheduler, config, best):
    row = {'epoch': index + 1, 'lr': optimizer.param_groups[0]['lr'],
           'train': train_metrics, 'validation': val_metrics}
    print(json.dumps(row), flush=True)
    with (output / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    scheduler.step()
    score = (val_metrics or train_metrics)['total']
    state = {'epoch': index + 1, 'generator': generator.state_dict(),
             'projection': projection.state_dict() if projection is not None else None,
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'config': config}
    torch.save(state, output / 'last.pt')
    if score < best:
        torch.save(state, output / 'best.pt')
    return min(best, score)
