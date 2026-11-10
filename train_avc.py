"""Stage 2: train the generator and projection with audiovisual correlation."""
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from models.proj import Proj
from utils.losses import audiovisual_losses
import torch
from models.vq_vae import VQVAE
from utils.training import build_parser, prepare_training, save_epoch


def run_epoch(generator, projection, visual, loader, optimizer, device, args, training):
    generator.train(training)
    projection.train(training)
    visual.eval()
    totals = dict(total=0.0, reconstruction=0.0, embedding=0.0, bct=0.0, nce=0.0)
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            audio = batch['audio'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstructed, vq_loss, _, encoded = generator(audio)
            rec = torch.nn.functional.mse_loss(reconstructed, audio)
            bct = nce = audio.new_zeros(())
            with torch.no_grad():
                image_features = visual(batch['image'].to(device))
            bct, nce = audiovisual_losses(projection(encoded), image_features, args.temperature)
            loss = args.embedding_weight * vq_loss + args.reconstruction_weight * rec
            loss = loss + args.bct_weight * bct + args.nce_weight * nce
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training/validation loss')
            if training:
                loss.backward()
                optimizer.step()
            n = audio.shape[0]
            count += n
            for key, value in zip(totals, (loss, rec, vq_loss, bct, nce)):
                totals[key] += value.detach().item() * n
    return {key: value / count for key, value in totals.items()} if count else None



def main(argv=None):
    parser = build_parser('avc')
    parser.add_argument('--image-dir', required=True, type=Path)
    parser.add_argument('--generator-checkpoint', required=True, type=Path)
    parser.add_argument('--bct-weight', type=float, default=1.0)
    parser.add_argument('--nce-weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--visual-weights', choices=['imagenet', 'none'], default='imagenet',
                        help='none is for smoke tests only, not paper experiments')
    args = parser.parse_args(argv)
    if args.temperature <= 0:
        parser.error('temperature must be positive')
    if args.batch_size < 2:
        parser.error('AVC requires batch-size >= 2')
    device, output, config, train_loader, val_loader = prepare_training(args, parser, 'avc')
    if len(train_loader.dataset) < 2:
        parser.error('AVC requires at least two training samples')

    generator = VQVAE(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        beta=args.commitment_weight,
    ).to(device)
    generator.load_state_dict(torch.load(
        args.generator_checkpoint, map_location=device, weights_only=True))
    projection = Proj().to(device)
    weights = ResNet50_Weights.DEFAULT if args.visual_weights == 'imagenet' else None
    visual = resnet50(weights=weights)
    visual.fc = torch.nn.Identity()
    visual = visual.to(device).requires_grad_(False).eval()

    parameters = list(generator.parameters()) + list(projection.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best = float('inf')
    for index in range(args.epochs):
        train_metrics = run_epoch(generator, projection, visual, train_loader,
                                  optimizer, device, args, training=True)
        val_metrics = run_epoch(generator, projection, visual, val_loader,
                                optimizer, device, args, training=False)
        best = save_epoch(output, index, train_metrics, val_metrics, generator, projection,
                          optimizer, scheduler, config, best)
    torch.save(generator.state_dict(), output / 'anet.pt')
    torch.save(projection.state_dict(), output / 'proj.pt')


if __name__ == '__main__':
    main()
