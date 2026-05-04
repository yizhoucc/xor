"""Checkpoint utilities — save model weights every epoch to user_data."""
import os
import torch

CHECKPOINT_BASE = '/user_data/yizhouc3/xor_checkpoints'


def save_checkpoint(model, save_dir, epoch, prefix='model', extra=None):
    """Save model state_dict to user_data checkpoint dir.

    Args:
        model: nn.Module
        save_dir: experiment-specific subdir name (e.g., 'warmstart_d128')
        epoch: epoch number
        prefix: filename prefix
        extra: dict of extra info to save alongside state_dict
    """
    ckpt_dir = os.path.join(CHECKPOINT_BASE, save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    if extra:
        state.update(extra)

    path = os.path.join(ckpt_dir, f'{prefix}_ep{epoch:03d}.pth')
    torch.save(state, path)
    return path


def save_inner_weights(inner_net, save_dir, epoch, name='inner'):
    """Save just InnerNet weights (small, for visualization)."""
    ckpt_dir = os.path.join(CHECKPOINT_BASE, save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    if hasattr(inner_net, 'net'):
        state = inner_net.net.state_dict()
    else:
        state = inner_net.state_dict()

    path = os.path.join(ckpt_dir, f'{name}_ep{epoch:03d}.pth')
    torch.save(state, path)
    return path
