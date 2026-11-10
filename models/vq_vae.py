"""Core architecture extracted from the original VASG source; parameter names preserved."""
import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)
        self.permute_order_in = [0, 2, 3, 1]
        self.permute_order_out = [0, 3, 1, 2]

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
        quantized_latents = self.embedding(encoding_inds)
        quantized_latents = self.embedding(encoding_inds).view(latents_shape)
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        quantized_latents = latents + (quantized_latents - latents).detach()
        quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()
        return (quantized_latents, vq_loss, encoding_inds)

class VQVAE(nn.Module):

    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int, beta: float=0.25, **kwargs) -> None:
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.prevq_conv = nn.Conv2d(256, embedding_dim, kernel_size=1)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
        self.prevq_upconv = nn.ConvTranspose2d(embedding_dim, 256, kernel_size=1)
        self.upconv5 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)
        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv10 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, input, **kwargs):
        x1 = nn.functional.relu(self.conv1(input))
        x2 = nn.functional.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = nn.functional.relu(self.conv2(x2))
        x3 = nn.functional.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = nn.functional.relu(self.conv3(x3))
        x4 = nn.functional.max_pool2d(x3, kernel_size=2, stride=2)
        encoding = nn.functional.relu(self.conv4(x4))
        prevq = self.prevq_conv(encoding)
        quantized_inputs, vq_loss, vq_inds = self.vq_layer(prevq)
        prevq_up = self.prevq_upconv(quantized_inputs, output_size=encoding.size())
        x5 = self.upconv5(prevq_up, output_size=x3.size())
        x5 = torch.cat([x5, x3], dim=1)
        x6 = nn.functional.relu(self.conv6(x5))
        x7 = self.upconv7(x6, output_size=x2.size())
        x7 = torch.cat([x7, x2], dim=1)
        x8 = nn.functional.relu(self.conv8(x7))
        x9 = self.upconv9(x8, output_size=x1.size())
        x9 = torch.cat([x9, x1], dim=1)
        x10 = nn.functional.relu(self.conv10(x9))
        x11 = nn.functional.sigmoid(self.conv11(x10))
        return [x11, vq_loss, vq_inds, encoding]
