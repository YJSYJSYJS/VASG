# VASG

Code for **Generating visual-adaptive audio representation for audio recognition** (Pattern Recognition Letters, 2025). [Paper](https://doi.org/10.1016/j.patrec.2025.03.020)

## Installation

Create and activate the environment.

```bash
conda env create -f environment.yml
conda activate vasg-pretrain
```

## VQ-VAE pretraining

Train the VQ-VAE to reconstruct Mel-spectrograms.

```bash
python train_vqvae.py \
  --mel-dir data/acav/mel \
  --output-dir outputs/vqvae
```

## Audiovisual correlation training

Load the pretrained VQ-VAE and train it with paired Mel-spectrograms and images using InfoNCE and BCT losses.

```bash
python train_avc.py \
  --mel-dir data/acav/mel \
  --image-dir data/acav/img \
  --generator-checkpoint outputs/vqvae/anet.pt \
  --output-dir outputs/avc
```

## Mel to vMel

Convert Mel-spectrograms to vMel using the trained generator.

```bash
python convert_vmel.py \
  --checkpoint outputs/avc/anet.pt \
  --input-dir data/acav/mel \
  --output-dir outputs/vmel
```
