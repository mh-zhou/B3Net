import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint structure")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Used to construct the model, not evidence of its past training settings.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from utils.checkpoint import load_checkpoint_strict
    from utils.common_config import get_model
    from utils.config import create_config

    before = args.checkpoint.stat()
    p = create_config(str(args.config), {'run_mode': 'infer'})
    model = get_model(p, pretrained=False).cpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loading = load_checkpoint_strict(
        model, args.checkpoint, args.output_dir / 'checkpoint_keys.csv')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    optimizer_fields = ('lr', 'initial_lr', 'betas', 'eps', 'weight_decay', 'amsgrad')
    scheduler_fields = ('max_iterations', 'gamma', 'min_lr', 'base_lrs',
                        'last_epoch', '_step_count', '_last_lr')
    optimizer = checkpoint.get('optimizer', {})
    scheduler = checkpoint.get('scheduler', {})
    parameter_groups = [
        {key: group[key] for key in optimizer_fields if key in group}
        for group in optimizer.get('param_groups', [])
    ]
    step_counts = Counter()
    for state in optimizer.get('state', {}).values():
        if 'step' in state:
            value = state['step']
            step_counts[str(int(value.item() if torch.is_tensor(value) else value))] += 1
    after = args.checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source checkpoint changed during the audit; discard this result.')
    result = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_bytes': before.st_size,
        'checkpoint_mtime_utc': datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
        'model_construction_config': str(args.config.resolve()),
        'config_is_historical_training_evidence': False,
        'strict_loading': loading,
        'parameters': sum(value.numel() for value in model.parameters()),
        'parameter_tensors': len(list(model.named_parameters())),
        'buffer_tensors': len(list(model.named_buffers())),
        'backbone': p.backbone,
        'refinement_depths': list(p.REFINE_DEPTH),
        'tasks': list(p.TASKS.NAMES),
        'task_output_channels': dict(p.TASKS.NUM_OUTPUT),
        'pfe_parameters': sum(value.numel() for name, value in model.named_parameters()
                              if '.pfe.' in name),
        'metadata': {key: checkpoint[key] for key in ('epoch', 'iter_count', 'global_step',
                                                     'format_version') if key in checkpoint},
        'optimizer_parameter_groups': parameter_groups,
        'optimizer_step_counts': dict(step_counts),
        'scheduler': {key: scheduler[key] for key in scheduler_fields if key in scheduler},
        'embedded_config_fields': [key for key in ('config', 'args', 'cfg') if key in checkpoint],
        'not_established_by_state_dict': [
            'training batch size', 'validation batch size', 'seed',
            'input geometry', 'data augmentation', 'gradient clipping',
            'loss weights', 'checkpoint selection', 'warmup or freezing history'],
        'device': 'cpu',
        'forward_or_accuracy_evaluation_performed': False,
        'source_unchanged_during_audit': True,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False)
    (args.output_dir / 'checkpoint_summary.json').write_text(serialized + '\n', encoding='utf-8')
    print(serialized)

if __name__ == '__main__':
    main()
