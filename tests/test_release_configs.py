from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def check_variants():
    import torch

    sys.path.insert(0, str(ROOT))
    from utils.config import create_config
    from utils.common_config import get_optimizer

    cases = (
        ('nyud/internimage_b.yaml', 'intern_b', 4, 20000, 'adamW', 2e-5, 0.05, 18, 1, 0.95, 5, 15),
        ('nyud/vit_b.yaml', 'vitB', 3, 26000, 'adam', 2e-5, 1e-6, 18, 2, 0.95, 10, 50),
        ('nyud/vit_l.yaml', 'vitL', 3, 25000, 'adam', 2e-5, 5e-6, 18, 2, 0.95, 10, 70),
        ('nyud/variants/internimage_l_20k.yaml', 'intern_l', 4, 20000, 'adamW', 5e-5, 0.03, 12, 1, 0.95, 5, 15),
        ('pascal/internimage_b.yaml', 'intern_b', 4, 45000, 'adamW', 3e-5, 0.03, 12, 1, 0.95, 5, 20),
        ('pascal/vit_b.yaml', 'vitB', 4, 45000, 'adamW', 1.5e-5, 0.03, 12, 2, 0.98, 5, 20),
        ('pascal/vit_l.yaml', 'vitL', 4, 45000, 'adamW', 1.5e-5, 0.03, 12, 2, 0.98, 5, 20),
    )
    keys = set('version_name out_dir train_db_name val_db_name trBatch valBatch '
               'nworkers ignore_index intermediate_supervision val_interval epochs '
               'max_iter optimizer optimizer_kwargs scheduler grad_clip_param model '
               'backbone head embed_dim mtt_resolution_downsample_rate '
               'PRED_OUT_NUM_CONSTANT REFINE_DEPTH DILATED_RATIO SQUEEZE_RATIO '
               'task_dictionary loss_kwargs'.split())
    for name, backbone, batch, steps, opt, lr, wd, depth, down, balance, normal, edge in cases:
        path = ROOT / 'configs' / name
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert set(config) == keys, name
        is_nyud = name.startswith('nyud/')
        database = 'NYUD' if is_nyud else 'PASCALContext'
        assert config['train_db_name'] == config['val_db_name'] == database
        assert config['backbone'] == backbone
        assert config['trBatch'] == batch
        assert config['max_iter'] == steps
        assert config['optimizer'] == opt
        expected_optimizer = {'lr': lr, 'weight_decay': wd}
        if opt == 'adamW':
            expected_optimizer['betas'] = {'a': 0.9, 'b': 0.999}
        assert config['optimizer_kwargs'] == expected_optimizer, name
        assert config['REFINE_DEPTH'] == [depth] * 3
        assert config['DILATED_RATIO'] == [[1, 2, 5] * (depth // 3)] * 3
        assert config['SQUEEZE_RATIO'] == [[2] * depth] * 3
        assert config['mtt_resolution_downsample_rate'] == down
        assert config['grad_clip_param'] == {'max_norm': 10 if is_nyud else 5, 'norm_type': 2}
        expected_weights = {'semseg': 1.0, 'normals': normal, 'edge': edge}
        expected_weights.update({'depth': 1.0} if is_nyud else {'human_parts': 2.0, 'sal': 3.0})
        assert config['loss_kwargs'] == {'loss_weights': expected_weights}
        expected_tasks = {'include_' + task: True for task in expected_weights}
        expected_tasks['edge_w'] = balance
        assert config['task_dictionary'] == expected_tasks
        for key, value in {'valBatch': 6, 'nworkers': 2, 'ignore_index': 255,
                           'intermediate_supervision': True, 'val_interval': 1000,
                           'epochs': 999999, 'scheduler': 'poly', 'model': 'B3Net',
                           'head': 'mlp', 'embed_dim': 512, 'PRED_OUT_NUM_CONSTANT': 64}.items():
            assert config[key] == value, (name, key)
        parsed = create_config(str(path), {'run_mode': 'infer'})
        assert parsed.TRAIN.SCALE == parsed.TEST.SCALE == ((448, 576) if is_nyud else (512, 512))
        assert set(parsed.TASKS.NAMES) == set(expected_weights)
        expected_channels = ({'semseg': 40, 'depth': 1, 'normals': 3, 'edge': 1} if is_nyud
                             else {'semseg': 21, 'human_parts': 7, 'sal': 2, 'normals': 3, 'edge': 1})
        assert parsed.TASKS.NUM_OUTPUT == expected_channels
        assert parsed.edge_w == balance
        scheduler, optimizer = get_optimizer(parsed, torch.nn.Linear(1, 1))
        assert type(optimizer) is (torch.optim.AdamW if opt == 'adamW' else torch.optim.Adam)
        assert optimizer.defaults['lr'] == lr and optimizer.defaults['weight_decay'] == wd
        assert optimizer.defaults['betas'] == (0.9, 0.999) and optimizer.defaults['eps'] == 1e-8
        assert scheduler.max_iterations == steps and scheduler.gamma == 0.9 and scheduler.min_lr == 0
    print('Seven non-primary variant configs, derived fields and optimizers passed.')


def main():
    assert {p.name for p in (ROOT / 'configs/nyud/variants').iterdir()} == {'internimage_l_20k.yaml'}
    names = set()
    for path in (ROOT / 'configs').rglob('*.yaml'):
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        dataset = path.relative_to(ROOT / 'configs').parts[0]
        assert config['version_name'] == f'b3_{dataset}_{path.stem}'
        assert config['version_name'] not in names
        names.add(config['version_name'])
        assert config['out_dir'] == './outputs/'
        assert not {'warmup_iters', 'warmup_start_lr', 'freeze_backbone_iters', 'poly_power'} & config.keys()
    cases = (
        ('nyud', 'internimage_l.yaml',
         'variants/internimage_l_20k.yaml', 26000, 20000, 5e-5, 3),
        ('pascal', 'internimage_l.yaml',
         'variants/internimage_l_40k.yaml', 45000, 40000, 3e-5, 4),
    )
    for dataset, default_name, short_name, default_steps, short_steps, lr, batch in cases:
        folder = ROOT / 'configs' / dataset
        default = yaml.safe_load((folder / default_name).read_text(encoding='utf-8'))
        short = yaml.safe_load((folder / short_name).read_text(encoding='utf-8'))
        assert default['max_iter'] == default_steps
        assert short['max_iter'] == short_steps
        assert default['version_name'] != short['version_name']
        assert default['trBatch'] == batch
        assert short['trBatch'] == 4
        assert default['REFINE_DEPTH'] == [12, 12, 12]
        assert default['optimizer_kwargs']['lr'] == lr
        assert default['optimizer_kwargs']['weight_decay'] == 0.03
        assert default['scheduler'] == 'poly'
        assert default['val_interval'] == 1000
        assert default_steps % default['val_interval'] == 0
        assert short_steps % short['val_interval'] == 0
        for config in (default, short):
            config.pop('version_name')
            config.pop('max_iter')
            config.pop('trBatch')
        assert default == short, dataset
    print('Release defaults and optional 20K/40K budgets passed.')
    check_variants()

if __name__ == '__main__':
    main()
