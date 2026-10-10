"""Opt-in image-only training mode and exact consumed-mode accounting.

The default remains the historical per-window random sampling of all three
input modes. This module does not admit any dataset or authorize a GPU job.
"""
import torch


def validate_train_input_mode(mode, failure_enabled):
    if type(mode) is not str or mode not in ('sampled', 'image'):
        raise ValueError('unknown training input mode')
    if mode == 'image' and failure_enabled is not True:
        raise ValueError('image-only training requires the fixed failure-state recipe')
    return mode


def select_training_modes(batch_size, device, mode):
    if type(mode) is not str or mode not in ('sampled', 'image'):
        raise ValueError('unknown training input mode')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive integer batch size required')
    # Draw even for image-only training to keep the subsequent Torch RNG stream
    # aligned with the sampled-mode run as far as possible.
    sampled = torch.randint(0, 3, (batch_size,), device=device)
    if mode == 'image':
        sampled.zero_()
    return sampled


def audit_image_mode_counts(counts, actual_total):
    if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.int64 or
            counts.shape != (3,) or counts.device.type != 'cpu' or
            type(actual_total) is not int or actual_total < 1):
        raise ValueError('invalid image-only mode exposure accounting')
    values = counts.tolist()
    if values != [actual_total, 0, 0]:
        raise ValueError('image-only mode exposures differ from actual consumed rows')
    return dict(image=values[0], point=values[1], mixed=values[2],
                total=actual_total)
