"""Checkpoint utilities — save model weights every epoch to user_data."""
import os
import torch

CHECKPOINT_BASE = '/user_data/yizhouc3/xor_checkpoints'


def save_checkpoint(model, save_dir, epoch, prefix='model', optimizer=None, metrics=None):
    """Save full checkpoint to user_data.

    Saves: model state_dict, optimizer state, epoch, metrics.
    """
    ckpt_dir = os.path.join(CHECKPOINT_BASE, save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if metrics is not None:
        state['metrics'] = metrics

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
