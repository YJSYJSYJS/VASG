"""Convert precomputed Mel arrays to vMel using a trained VASG generator."""
import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from models.vq_vae import VQVAE


def load_generator(checkpoint_path, device):
    """Accept an exported anet.pt or the generator in a best.pt/last.pt state."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a generator state dictionary or a training checkpoint')
    state = checkpoint.get('generator', checkpoint)
    if not isinstance(state, Mapping) or 'vq_layer.embedding.weight' not in state:
        raise ValueError('Checkpoint does not contain VASG generator weights')
    codebook = state['vq_layer.embedding.weight']
    if not isinstance(codebook, torch.Tensor) or codebook.ndim != 2:
        raise ValueError('Invalid generator codebook weights')
    num_embeddings, embedding_dim = codebook.shape
    generator = VQVAE(in_channels=1, embedding_dim=embedding_dim,
                      num_embeddings=num_embeddings)
    generator.load_state_dict(state, strict=True)
    return generator.to(device).eval().requires_grad_(False)


def load_mel(path, input_scale):
    mel = np.load(path, allow_pickle=False)
    # A single example, with an optional singleton channel dimension.
    if mel.ndim not in (2, 3) or (mel.ndim == 3 and mel.shape[0] != 1):
        raise ValueError(f'{path}: expected (128, T) or (1, 128, T), got {mel.shape}')
    if mel.shape[-2] != 128 or mel.shape[-1] < 8:
        raise ValueError(f'{path}: expected 128 Mel bins and at least 8 time frames')
    if not np.issubdtype(mel.dtype, np.number) or np.iscomplexobj(mel):
        raise ValueError(f'{path}: expected real numeric Mel values')
    mel = mel.astype(np.float32)
    if not np.isfinite(mel).all():
        raise ValueError(f'{path}: nonfinite Mel values')
    if input_scale == 'acav-db':
        mel = (mel + 80.0) / (80.0 + 1e-10)
    if mel.ndim == 2:
        mel = mel[None, ...]
    return torch.from_numpy(mel).unsqueeze(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='anet.pt, best.pt or last.pt from VASG pretraining')
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--input-scale', choices=['acav-db', 'normalized'], default='acav-db',
                        help='acav-db applies original ACAV normalization; normalized passes values through')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error('Input and output directories must be separate, non-nested directories')
    if not source.is_dir():
        parser.error(f'Input directory does not exist: {source}')
    paths = sorted(source.rglob('*.npy'))
    if not paths:
        parser.error(f'No .npy files found under {source}')
    targets = [destination / path.relative_to(source) for path in paths]
    for target in targets:
        if target.exists():
            parser.error(f'Output already exists: {target}; use a new output directory')
    device = torch.device(args.device)
    generator = load_generator(args.checkpoint, device)
    with torch.inference_mode():
        for path, target in zip(paths, targets):
            audio = load_mel(path, args.input_scale).to(device)
            vmel = generator(audio)[0]
            result = vmel[0].cpu().numpy().astype(np.float32)
            if not np.isfinite(result).all():
                raise FloatingPointError(f'{path}: generator produced nonfinite values')
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also protects against a concurrent overwrite.
            with target.open('xb') as stream:
                np.save(stream, result, allow_pickle=False)
            print(f'{path.relative_to(source)} -> {target}', flush=True)
    print(f'Converted {len(paths)} Mel arrays to vMel.', flush=True)


if __name__ == '__main__':
    main()
