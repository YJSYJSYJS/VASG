"""Stage 1: train VQ-VAE with reconstruction and embedding losses."""
import torch
from models.vq_vae import VQVAE
from utils.training import build_parser, prepare_training, save_epoch


def run_epoch(generator, loader, optimizer, device, args, training):
    generator.train(training)
    totals = dict(total=0.0, reconstruction=0.0, embedding=0.0, bct=0.0, nce=0.0)
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            audio = batch['audio'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstructed, vq_loss, _, _ = generator(audio)
            rec = torch.nn.functional.mse_loss(reconstructed, audio)
            bct = nce = audio.new_zeros(())
            loss = args.embedding_weight * vq_loss + args.reconstruction_weight * rec
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
    parser = build_parser('reconstruction')
    args = parser.parse_args(argv)
    device, output, config, train_loader, val_loader = prepare_training(
        args, parser, 'reconstruction')

    generator = VQVAE(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        beta=args.commitment_weight,
    ).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best = float('inf')
    for index in range(args.epochs):
        train_metrics = run_epoch(generator, train_loader, optimizer, device, args, training=True)
        val_metrics = run_epoch(generator, val_loader, optimizer, device, args, training=False)
        best = save_epoch(output, index, train_metrics, val_metrics, generator, None,
                          optimizer, scheduler, config, best)
    torch.save(generator.state_dict(), output / 'anet.pt')


if __name__ == '__main__':
    main()
