import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Check the training configuration")
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import torch.distributed as dist
    from utils.common_config import (get_criterion, get_model, get_optimizer,
                                     get_train_dataloader, get_train_dataset,
                                     get_transformations)
    from utils.config import create_config
    from utils.utils import to_cuda

    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    dist.init_process_group('nccl')
    try:
        if dist.get_world_size() != 1:
            raise ValueError('This acceptance check requires one GPU.')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        cv2.setNumThreads(0)
        p = create_config(args.config, {'run_mode': 'infer'})
        transforms, _ = get_transformations(p)
        dataset = get_train_dataset(p, transforms)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
        sampler.set_epoch(0)
        loader = get_train_dataloader(p, dataset, sampler)
        model = get_model(p, pretrained=True).cuda()
        parameter_count = sum(value.numel() for value in model.parameters())
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        criterion = get_criterion(p).cuda()
        scheduler, optimizer = get_optimizer(p, model)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for step, cpu_batch in enumerate(loader):
            batch = to_cuda(cpu_batch)
            assert tuple(batch['image'].shape) == (4, 3, 448, 576)
            output = model(batch['image'])
            losses = criterion(output, batch, tasks=p.TASKS.NAMES)
            if not all(torch.isfinite(value).all().item() for value in losses.values()):
                raise FloatingPointError('Non-finite training loss.')
            optimizer.zero_grad(set_to_none=False)
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), error_if_nonfinite=True, **p.grad_clip_param)
            optimizer.step()
            scheduler.step()
            record = {'completed_step': step + 1,
                      'losses': {key: value.item() for key, value in losses.items()},
                      'gradient_norm_before_clip': norm.item(),
                      'learning_rate_next_step': optimizer.param_groups[0]['lr']}
            records.append(record)
            print(json.dumps(record), flush=True)
            if step == 1:
                break
        for name, value in model.named_parameters():
            if not torch.isfinite(value).all().item():
                raise FloatingPointError('Non-finite parameter: ' + name)
        result = {'status': 'passed', 'config': args.config,
                  'parameter_count': parameter_count, 'seed': 42,
                  'input_shape': [4, 3, 448, 576], 'dtype': 'float32',
                  'steps': records,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'smoke_weights_saved': False,
                  'note': 'Runtime acceptance only; no accuracy claim or formal training progress.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                               encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
