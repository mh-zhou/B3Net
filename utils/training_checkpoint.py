from collections.abc import Mapping
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA RNG state requires a CUDA device.')
        torch.cuda.set_rng_state(state['cuda'].cpu())


def replay_training_iterator(loader, resume_state=None):
    if resume_state is None:
        epoch_start_rng = capture_rng_state()
        return iter(loader), 0, epoch_start_rng
    epoch_start_rng = resume_state['epoch_start_rng']
    restore_rng_state(epoch_start_rng)
    iterator = iter(loader)
    next_batch = resume_state['next_batch']
    for _ in range(next_batch):
        next(iterator)
    restore_rng_state(resume_state['checkpoint_rng'])
    return iterator, next_batch, epoch_start_rng


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError('Unsupported checkpoint metadata type: ' + type(value).__name__)


def atomic_save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                            dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_checkpoint(path, *, model, optimizer, scheduler, config,
                             global_step, epoch, next_batch, batches_per_epoch,
                             metrics, rng_by_rank):
    state = {
        'format_version': 2,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'iter_count': global_step,
        'epoch': epoch,
        'next_batch': next_batch,
        'batches_per_epoch': batches_per_epoch,
        'world_size': len(rng_by_rank),
        'config': _plain(config),
        'metrics': _plain(metrics),
        'rng_by_rank': rng_by_rank,
        'resume_state': rng_by_rank[0],
    }
    atomic_save_checkpoint(state, path)


def load_training_checkpoint(path, *, config, world_size=1, batches_per_epoch):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get('format_version') != 2:
        raise ValueError('Training resume requires a version-2 checkpoint with cursor and RNG states.')
    if world_size != 1 or checkpoint['world_size'] != 1:
        raise ValueError('Exact training resume is currently validated for one GPU only.')
    protocol_keys = (
        'train_db_name', 'val_db_name', 'trBatch', 'valBatch', 'nworkers', 'ignore_index',
        'intermediate_supervision', 'max_iter', 'optimizer', 'optimizer_kwargs',
        'scheduler', 'scheduler_kwargs', 'model', 'backbone', 'head', 'embed_dim',
        'mtt_resolution_downsample_rate', 'PRED_OUT_NUM_CONSTANT', 'REFINE_DEPTH',
        'DILATED_RATIO', 'SQUEEZE_RATIO', 'task_dictionary', 'loss_kwargs',
        'grad_clip_param', 'TRAIN', 'TEST', 'seed',
    )
    current = _plain(config)
    changed = [key for key in protocol_keys
               if current.get(key) != checkpoint['config'].get(key)]
    if changed:
        raise ValueError('Training protocol differs from checkpoint: ' + ', '.join(changed))
    step = checkpoint['global_step']
    if step != checkpoint['iter_count'] or step != checkpoint['scheduler']['last_epoch']:
        raise ValueError('Checkpoint update count and scheduler position disagree.')
    if not 0 <= step <= current['max_iter']:
        raise ValueError('Checkpoint update count is outside the training budget.')
    if checkpoint['batches_per_epoch'] != batches_per_epoch:
        raise ValueError('Training loader length differs from checkpoint.')
    if not 0 <= checkpoint['next_batch'] <= batches_per_epoch:
        raise ValueError('Checkpoint batch cursor is out of range.')
    resume_state = checkpoint['rng_by_rank'][0]
    if resume_state['next_batch'] != checkpoint['next_batch']:
        raise ValueError('Checkpoint RNG cursor disagrees with batch cursor.')
    for name in ('epoch_start_rng', 'checkpoint_rng'):
        if not all(key in resume_state[name] for key in ('python', 'numpy', 'torch', 'cuda')):
            raise ValueError('Checkpoint RNG state is incomplete.')
    checkpoint['resume_state'] = resume_state
    return checkpoint
