#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple, Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoint import convert_state_dict as convert_public_state_dict
from utils.checkpoint import public_key

ROOT_RULES = [
    ('multi_task_decoder.trfl.', 'multi_task_decoder.feature_propagator.'),
]

CITY_RULES = []


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith('module.') else key


def _apply_rules(key: str, rules) -> Tuple[str, bool]:
    new_key = key
    changed = False
    for src, dst in rules:
        if new_key.startswith(src):
            new_key = dst + new_key[len(src):]
            changed = True
    return new_key, changed


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(ckpt, dict):
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            return ckpt['model'], 'model'
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            return ckpt['state_dict'], 'state_dict'
    if isinstance(ckpt, dict):
        return ckpt, 'raw'
    raise TypeError('Unsupported checkpoint format')


def convert_state_dict(state_dict: Dict[str, torch.Tensor], branch: str) -> Tuple[OrderedDict, Dict[str, int]]:
    if branch == 'root':
        converted = convert_public_state_dict(state_dict)
        stripped = sum(key.startswith('module.') for key in state_dict)
        renamed = sum(public_key(key) != _strip_module_prefix(key) for key in state_dict)
        return converted, {
            'total_keys': len(state_dict),
            'stripped_module_prefix': stripped,
            'renamed_keys': renamed,
            'unchanged_keys': len(state_dict) - renamed,
        }
    rules = ROOT_RULES if branch == 'root' else CITY_RULES
    converted = OrderedDict()
    stats = {
        'total_keys': 0,
        'stripped_module_prefix': 0,
        'renamed_keys': 0,
        'unchanged_keys': 0,
    }
    for old_key, value in state_dict.items():
        stats['total_keys'] += 1
        key = old_key
        stripped = _strip_module_prefix(key)
        if stripped != key:
            stats['stripped_module_prefix'] += 1
            key = stripped
        new_key, changed = _apply_rules(key, rules)
        if changed:
            stats['renamed_keys'] += 1
        else:
            stats['unchanged_keys'] += 1
        if new_key in converted:
            raise KeyError(f'Duplicate key after conversion: {new_key}')
        converted[new_key] = value
    return converted, stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert legacy checkpoints to pure B3 naming.')
    parser.add_argument('--input', required=True, type=Path, help='Path to the legacy checkpoint file.')
    parser.add_argument('--output', required=True, type=Path, help='Path to the converted checkpoint file.')
    parser.add_argument('--branch', choices=['root', 'city'], default='root', help='Checkpoint target branch.')
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint: {}'.format(args.output))
    ckpt = torch.load(args.input, map_location='cpu', weights_only=True)
    state_dict, container_type = _extract_state_dict(ckpt)
    converted_state_dict, stats = convert_state_dict(state_dict, branch=args.branch)

    if container_type == 'model':
        ckpt['model'] = converted_state_dict
        out_obj = ckpt
    elif container_type == 'state_dict':
        ckpt['state_dict'] = converted_state_dict
        out_obj = ckpt
    else:
        out_obj = converted_state_dict

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, args.output)

    print('Conversion finished.')
    print(f'Input : {args.input}')
    print(f'Output: {args.output}')
    print(f'Branch: {args.branch}')
    print(f'Container type: {container_type}')
    for k, v in stats.items():
        print(f'{k}: {v}')
    if args.branch == 'root':
        print('Applied mapping: multi_task_decoder.trfl.* -> multi_task_decoder.feature_propagator.*')
    else:
        print('No branch-specific rename rule was needed; only DataParallel prefixes were normalized if present.')

if __name__ == '__main__':
    main()
