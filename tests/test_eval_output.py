from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.utils import get_output
from evaluation.eval_edge import EdgeMeter
from utils.custom_collate import collate_mil


class EdgeOutputTest(unittest.TestCase):
    def test_validation_batch_and_metadata(self):
        for batch_size in (1, 6):
            sample = {'image': torch.zeros(3, 4, 5),
                      'meta': {'img_name': 'sample', 'img_size': (4, 5)}}
            batch = collate_mil([sample] * batch_size)
            self.assertEqual(batch['image'].shape, (batch_size, 3, 4, 5))
            self.assertEqual(batch['meta']['img_name'], ['sample'] * batch_size)
            self.assertEqual(len(batch['meta']['img_size']), batch_size)
            for size in batch['meta']['img_size']:
                self.assertTrue(torch.equal(size, torch.tensor([4, 5])))

    def test_batch_axis_and_values(self):
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                logits = torch.linspace(-3, 3, batch_size * 12).reshape(
                    batch_size, 1, 3, 4
                )
                result = get_output(logits, 'edge')
                expected = 255 * 1 / (1 + torch.exp(-logits[:, 0]))
                self.assertEqual(result.shape, (batch_size, 3, 4))
                self.assertTrue(torch.equal(result, expected))

    def test_edge_loss_matches_legacy_values(self):
        for batch_size in (1, 2, 6):
            with self.subTest(batch_size=batch_size):
                logits = torch.linspace(-3, 3, batch_size * 12).reshape(
                    batch_size, 1, 3, 4
                )
                gt = (logits > 0).float()
                gt[:, :, 0, 0] = 255
                legacy_pred = torch.squeeze(
                    255 * 1 / (1 + torch.exp(-logits.permute(0, 2, 3, 1)))
                )
                legacy_gt = gt.squeeze()
                mask = legacy_gt != 255
                meter = EdgeMeter(pos_weight=0.95, ignore_index=255)
                expected = meter.loss_function(
                    legacy_pred[mask].float().squeeze() / 255,
                    legacy_gt[mask],
                ).item()
                meter.update(get_output(logits, 'edge'), gt)
                self.assertEqual(meter.get_score(verbose=False)['loss'], expected)

if __name__ == '__main__':
    unittest.main()
