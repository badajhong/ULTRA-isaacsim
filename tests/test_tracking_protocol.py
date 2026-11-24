import copy
import unittest
from pathlib import Path

import yaml

from learning.tracking_protocol import configure_tracking_randomization


class TrackingProtocolTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.cfg = yaml.safe_load((root / 'ultra/data/cfg/g1_student_vae.yaml').read_text())

    def test_nominal_disables_independent_object_randomization(self):
        original = copy.deepcopy(self.cfg)
        configure_tracking_randomization(self.cfg, 'nominal')
        dr = self.cfg['domain_rand']
        self.assertFalse(dr['domain_rand_general'])
        self.assertEqual(dr['obj_mass_range'], [1.0, 1.0])
        self.assertEqual(dr['obj_mass_rare_every'], 0)
        self.assertEqual(dr['obj_com_range'], [0.0, 0.0])
        self.assertEqual(dr['obj_inertia_range'], [1.0, 1.0])
        self.assertFalse(self.cfg['noise']['add_noise'])
        self.assertEqual(self.cfg['env']['pointScaleMin'], 1.0)
        self.assertEqual(self.cfg['env']['pointNoiseStd'], 0.0)
        # Nominal evaluation must not weaken the success/termination or controller settings.
        for name in ('control', 'sim'):
            self.assertEqual(self.cfg[name], original[name])
        for name in ('enableEarlyTermination', 'terminationHeight'):
            self.assertEqual(self.cfg['env'][name], original['env'][name])

    def test_config_mode_preserves_all_training_settings(self):
        original = copy.deepcopy(self.cfg)
        configure_tracking_randomization(self.cfg, 'config')
        self.assertEqual(self.cfg, original)

    def test_unknown_mode_rejected_without_mutation(self):
        original = copy.deepcopy(self.cfg)
        with self.assertRaises(ValueError):
            configure_tracking_randomization(self.cfg, 'unknown')
        self.assertEqual(self.cfg, original)


if __name__ == '__main__':
    unittest.main()
