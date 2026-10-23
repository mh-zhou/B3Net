from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.training_checkpoint import (atomic_save_checkpoint, capture_rng_state,
                                       load_training_checkpoint, replay_training_iterator,
                                       save_training_checkpoint)

with patch.dict(sys.modules, {
    'evaluation.evaluate_utils': SimpleNamespace(PerformanceMeter=object),
    'utils.test_utils': SimpleNamespace(test_phase=None),
}):
    from utils.train_utils import PolynomialLR, train_phase


class RandomDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        image = torch.tensor([index / 16, random.random(), np.random.rand(),
                              torch.rand(()).item()])
        return image, image.sum().reshape(1)


def make_training_state():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.4),
                                torch.nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = PolynomialLR(optimizer, max_iterations=6)
    return model, optimizer, scheduler


def update(state, batch):
    model, optimizer, scheduler = state
    optimizer.zero_grad(set_to_none=False)
    torch.nn.functional.mse_loss(model(batch[0]), batch[1]).backward()
    optimizer.step()
    scheduler.step()


class TrainingCheckpointTest(unittest.TestCase):
    def test_exact_next_batch_and_dropout_update(self):
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                random.seed(19)
                np.random.seed(19)
                torch.manual_seed(19)
                state = make_training_state()
                loader = DataLoader(RandomDataset(), batch_size=2, shuffle=True,
                                    num_workers=workers)
                iterator, cursor, epoch_rng = replay_training_iterator(loader)
                self.assertEqual(cursor, 0)
                for _ in range(3):
                    update(state, next(iterator))
                config = {'max_iter': 6, 'nworkers': workers, 'trBatch': 2, 'seed': 19}
                path = Path(directory) / 'checkpoint.pth'
                save_training_checkpoint(
                    path, model=state[0], optimizer=state[1], scheduler=state[2],
                    config=config, global_step=3, epoch=0, next_batch=3,
                    batches_per_epoch=len(loader), metrics={'loss': np.float64(0.1)},
                    rng_by_rank=[{'epoch_start_rng': epoch_rng,
                                  'checkpoint_rng': capture_rng_state(), 'next_batch': 3}])
                expected_batch = next(iterator)
                update(state, expected_batch)
                expected = {key: value.clone() for key, value in state[0].state_dict().items()}
                del iterator

                restored = make_training_state()
                checkpoint = load_training_checkpoint(
                    path, config=config, batches_per_epoch=len(loader))
                self.assertEqual(checkpoint['global_step'], 3)
                restored[0].load_state_dict(checkpoint['model'], strict=True)
                restored[1].load_state_dict(checkpoint['optimizer'])
                restored[2].load_state_dict(checkpoint['scheduler'])
                iterator, cursor, _ = replay_training_iterator(loader, checkpoint['resume_state'])
                self.assertEqual(cursor, 3)
                actual_batch = next(iterator)
                for actual, expected_value in zip(actual_batch, expected_batch):
                    self.assertTrue(torch.equal(actual, expected_value))
                update(restored, actual_batch)
                for key, actual in restored[0].state_dict().items():
                    self.assertTrue(torch.equal(actual, expected[key]), key)
                del iterator

                with self.assertRaisesRegex(ValueError, 'protocol differs'):
                    load_training_checkpoint(path, config={**config, 'trBatch': 4},
                                             batches_per_epoch=len(loader))
                with self.assertRaisesRegex(ValueError, 'one GPU'):
                    load_training_checkpoint(path, config=config, world_size=2,
                                             batches_per_epoch=len(loader))

    def test_completed_budget_does_not_touch_training_state(self):
        class Untouchable:
            def train(self):
                raise AssertionError('Model must not be touched at the completed budget')

            def __iter__(self):
                raise AssertionError('Loader must not be touched at the completed budget')

        self.assertEqual(train_phase(SimpleNamespace(max_iter=3), None, Untouchable(),
                                     None, Untouchable(), None, None, None, 0,
                                     None, None, 3), (True, 3))

    def test_atomic_failure_preserves_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            atomic_save_checkpoint({'value': torch.tensor(7)}, path)
            with patch('utils.training_checkpoint.torch.save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    atomic_save_checkpoint({'value': torch.tensor(9)}, path)
            self.assertEqual(torch.load(path, weights_only=True)['value'].item(), 7)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_polynomial_budget_boundary(self):
        _, optimizer, scheduler = make_training_state()
        for _ in range(6):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.0)
        with self.assertRaisesRegex(ValueError, 'update budget'):
            scheduler.step()

    def test_legacy_checkpoint_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pth'
            torch.save({'model': {}, 'iter_count': 2}, path)
            with self.assertRaisesRegex(ValueError, 'version-2'):
                load_training_checkpoint(path, config={'max_iter': 6}, batches_per_epoch=8)

if __name__ == '__main__':
    unittest.main()
