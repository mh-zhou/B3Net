import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='b3_launchers_') as folder:
        root = Path(folder) / 'repository with spaces'
        root.mkdir()
        (root / 'scripts').mkdir()
        scripts = ('run_PASCALContext.sh', 'run_nyud.sh', 'scripts/train_pascal.sh',
                   'scripts/train_nyud.sh', 'scripts/train_nyud_reference.sh')
        for script in scripts:
            shutil.copy2(ROOT / script, root / script)
            subprocess.run(['bash', '-n', str(root / script)], check=True)
        launcher = root / 'torchrun'
        launcher.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$PWD" "$PYTHONPATH" "$@"\n')
        launcher.chmod(0o755)
        env = os.environ.copy()
        env.update(PATH=f'{root}{os.pathsep}{env["PATH"]}', B3_DATA_ROOT='/data with spaces',
                   PYTHONPATH='/existing modules')
        prefix = ['--standalone', '--nnodes=1', '--nproc_per_node=1', 'train.py']

        def run(script, *args, **updates):
            return subprocess.run(['bash', str(root / script), *args],
                                  env=dict(env, **updates), capture_output=True, text=True)

        def check(result, config, dataset):
            assert result.returncode == 0, result.stderr
            expected = [str(root), f'{root}/models/ops_dcnv3:/existing modules', *prefix,
                        '--config_exp', config, '--dataset_log', dataset]
            assert result.stdout.split('\0') == expected + [''], result.stdout

        for dataset, script, wrapper in (('nyud', 'run_nyud.sh', 'scripts/train_nyud.sh'),
                                         ('pascal', 'run_PASCALContext.sh', 'scripts/train_pascal.sh')):
            config = f'configs/{dataset}/internimage_l.yaml'
            check(run(script), config, dataset.upper())
            check(run(wrapper), config, dataset.upper())
            config = 'configs/custom model.yaml'
            check(run(script, config), config, dataset.upper())
            check(run(wrapper, config), config, dataset.upper())
            result = run(script, B3_DATA_ROOT='')
            assert result.returncode != 0 and not result.stdout and 'Set B3_DATA_ROOT' in result.stderr
        check(run('scripts/train_nyud_reference.sh'),
              'configs/nyud/variants/internimage_l_20k.yaml', 'NYUD')
    print('Training launcher argument checks passed.')


if __name__ == '__main__':
    main()
