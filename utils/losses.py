"""Audiovisual objectives using normalized audio and image features."""
import torch
from torch.nn import functional as F

def audiovisual_losses(audio, image, temperature=1.0):
    audio = F.normalize(audio, dim=-1)
    image = F.normalize(image, dim=-1)
    bct = F.mse_loss(audio @ audio.T, image @ image.T)
    logits = (audio @ image.T) / temperature
    labels = torch.arange(audio.shape[0], device=audio.device)
    nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return bct, nce
