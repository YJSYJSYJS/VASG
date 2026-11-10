"""Load paired, precomputed ACAV Mel arrays and image frames."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ACAV(Dataset):
    def __init__(self, mel_dir, image_dir=None, image_transform=None):
        self.files = sorted(Path(mel_dir).glob('*.npy'))
        if not self.files:
            raise ValueError(f'No .npy Mel arrays found in {mel_dir}')
        self.image_transform = image_transform
        self.images = None
        if image_dir is not None:
            self.images = []
            for path in self.files:
                candidates = [Path(image_dir) / (path.stem + ext)
                              for ext in ('.jpg', '.jpeg', '.png')]
                matches = [p for p in candidates if p.is_file()]
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one paired image for {path.name}: {matches}')
                self.images.append(matches[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        mel = np.load(path, allow_pickle=False)
        if mel.ndim == 2:
            mel = mel[None, ...]
        if mel.shape != (1, 128, 216):
            raise ValueError(f'{path}: expected (128, 216) or (1, 128, 216), got {mel.shape}')
        if not np.isfinite(mel).all():
            raise ValueError(f'{path}: nonfinite Mel values')
        # Preserve the original ACAV dB normalization; do not clip.
        item = {'audio': torch.from_numpy(((mel.astype(np.float32) + 80.0) / (80.0 + 1e-10))),
                'path': str(path)}
        if self.images is not None:
            with Image.open(self.images[index]) as image:
                image = image.convert('RGB')
                item['image'] = self.image_transform(image) if self.image_transform else image.copy()
        return item
