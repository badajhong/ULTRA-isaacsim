"""CPU checks for tracking evaluation of distilled and V3 fine-tuned policies.

Run: PYTHONPATH=ultra python -m unittest discover -s tests -p 'test_tracking_checkpoint.py'
"""

import contextlib
import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from learning.tracking_checkpoint import TRAIN_CONFIG_DIR, prepare_tracking_evaluation
from learning import ultra_network_builder_obj_v2, ultra_network_builder_obj_v3
from learning.ultra_models import ModelUltraContinuous
from learning.tracking_evaluator import TrackingSuccessPlayer


class TrackingCheckpointTest(unittest.TestCase):
    def args(self, **overrides):
        values = dict(eval_episodes=1, test=True, task="UltraDistillObjV2Point",
                      eval_full_clips=True, eval_training_conditions=True, task_mode=None,
                      eval_model="auto", checkpoint="unused.pth",
                      cfg_train=str(TRAIN_CONFIG_DIR / "g1_student_vae.yaml"))
        return SimpleNamespace(**{**values, **overrides})

    def prepare(self, args, state=None, epoch=6000):
        checkpoint = {"model": state or {"a2c_network.mu.weight": torch.ones(1)}}
        if epoch is not None:
            checkpoint["epoch"] = epoch
        with patch("learning.tracking_checkpoint.torch.load", return_value=checkpoint):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_tracking_evaluation(args)

    def test_auto_selects_complete_finetune_network_and_preserves_tracking_task(self):
        args = self.args()
        self.prepare(args, {"a2c_network.rl_finetune_film.weight": torch.ones(1)})
        self.assertEqual(args.eval_model, "finetune")
        self.assertEqual(args.eval_checkpoint_epoch, 6000)
        self.assertEqual(args.task, "UltraDistillObjV2Point")
        self.assertEqual(Path(args.cfg_train), TRAIN_CONFIG_DIR / "g1_student_finetune.yaml")
        params = yaml.safe_load(Path(args.cfg_train).read_text())["params"]
        self.assertTrue(params["network"]["modal_config"]["freeze_encoder_decoder"])
        self.assertTrue(params["network"]["modal_config"]["critic_separate_backbone"])
        self.assertEqual(params["config"]["horizon_length"], 16)

    def test_distillation_config_and_custom_configs_are_preserved(self):
        args = self.args()
        original = args.cfg_train
        self.prepare(args)
        self.assertEqual(args.eval_model, "distillation")
        self.assertEqual(args.cfg_train, original)
        custom = self.args(cfg_train="custom_finetune.yaml")
        self.prepare(custom, {"a2c_network.rl_finetune_film.weight": torch.ones(1)})
        self.assertEqual(custom.cfg_train, "custom_finetune.yaml")

    def test_wrong_explicit_architecture_is_rejected_before_partial_weight_loading(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.prepare(self.args(eval_model="distillation"),
                         {"a2c_network.rl_finetune_film.weight": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.prepare(self.args(eval_model="finetune"))

    def test_training_never_inspects_or_changes_its_checkpoint(self):
        args = self.args(eval_episodes=0, eval_full_clips=False, eval_training_conditions=False,
                         test=False, task="UltraDistillObjV3RL")
        before = copy.deepcopy(vars(args))
        with patch("learning.tracking_checkpoint.torch.load", side_effect=AssertionError("training checkpoint read")):
            prepare_tracking_evaluation(args)
        self.assertEqual(vars(args), before)

    def test_tracking_rejects_goal_offset_task_and_invalid_protocols(self):
        for overrides in (dict(task="UltraDistillObjV3RL"), dict(test=False),
                          dict(eval_episodes=-1), dict(task_mode="full_track"),
                          dict(eval_full_clips=False)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                prepare_tracking_evaluation(self.args(**overrides))

    def test_checkpoint_without_epoch_is_available_for_full_clip_evaluation(self):
        args = self.args(eval_training_conditions=False)
        self.prepare(args, epoch=None)
        self.assertIsNone(args.eval_checkpoint_epoch)

    def test_teacher_checkpoint_is_not_loaded_into_student_network(self):
        args = self.args(eval_policy='teacher', eval_training_conditions=False,
                         checkpoint='teacher.pth')
        with patch('learning.tracking_checkpoint.torch.load', side_effect=AssertionError('student load')):
            prepare_tracking_evaluation(args)
        self.assertEqual(args.eval_model, 'teacher')
        self.assertEqual(args.checkpoint, 'teacher.pth')

    def test_teacher_actions_use_unmasked_privileged_observations(self):
        player = object.__new__(TrackingSuccessPlayer)
        teacher_obs = torch.tensor([[0.5, -0.25]])
        class Teacher(torch.nn.Module):
            def forward(self, inputs):
                torch.testing.assert_close(inputs['obs'], teacher_obs)
                return {'mus': torch.tensor([[1.5, -0.5]]),
                        'actions': torch.tensor([[0.0, 0.0]])}
        player.env = SimpleNamespace(task=SimpleNamespace(model=Teacher(), device=torch.device('cpu')))
        player.device = torch.device('cpu')
        player.eval_policy = 'teacher'
        player._last_expert = {'teacher_obs': teacher_obs}
        action = player.get_action({'obs': torch.zeros(1, 3)}, is_determenistic=True)
        torch.testing.assert_close(action, torch.tensor([[1.0, -0.5]]))

    def test_real_network_reload_keeps_prior_encoder_and_finetune_adapter_actions(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for kind, module in (("distillation", ultra_network_builder_obj_v2),
                                 ("finetune", ultra_network_builder_obj_v3)):
                with self.subTest(kind=kind):
                    cfg_path = TRAIN_CONFIG_DIR / ("g1_student_finetune.yaml" if kind == "finetune"
                                                   else "g1_student_vae.yaml")
                    def build(config_path):
                        params = yaml.safe_load(Path(config_path).read_text())["params"]
                        builder = module.UltraBuilder()
                        builder.load(params["network"])
                        with contextlib.redirect_stdout(io.StringIO()):
                            model = ModelUltraContinuous(builder).build({
                                "actions_num": 29, "input_shape": (1496,), "num_seqs": 1, "value_size": 1})
                        return model.eval()

                    original = build(cfg_path)
                    if kind == "finetune":
                        # A nonzero learned adapter must survive loading and affect actions.
                        with torch.no_grad():
                            original.a2c_network.rl_finetune_film.bias.fill_(0.3)
                    args = self.args()
                    self.prepare(args, original.state_dict())
                    self.assertEqual(args.eval_model, kind)
                    restored = build(args.cfg_train)
                    restored.load_state_dict(original.state_dict(), strict=True)
                    obs = torch.randn(2, 1496)
                    teacher_obs = torch.randn(2, 4052)
                    noise = torch.randn(2, 64)
                    for encoder in (False, True):
                        inputs = dict(obs=obs, teacher_obs=teacher_obs, vae_noise=noise, is_train=False,
                                      prev_actions=None, rnn_states=None, with_encoder=encoder,
                                      with_vae=False, skip_critic=True)
                        with torch.no_grad():
                            expected = original(dict(inputs))["mus"]
                            actual = restored(dict(inputs))["mus"]
                            if kind == "finetune":
                                trained_bias = restored.a2c_network.rl_finetune_film.bias.clone()
                                restored.a2c_network.rl_finetune_film.bias.zero_()
                                without_adapter = restored(dict(inputs))["mus"]
                                restored.a2c_network.rl_finetune_film.bias.copy_(trained_bias)
                                self.assertGreater((actual - without_adapter).abs().max().item(), 1e-6)
                        self.assertEqual(actual.shape, (2, 29))
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if kind == "finetune":
                        # Exercise the player's actual prior inference entry point,
                        # then compare with the RL agent's with_vae=True prior branch.
                        player = object.__new__(TrackingSuccessPlayer)
                        player.model = restored
                        player._preproc_obs = lambda value: value
                        player._vae_noise = noise.clone()
                        player._last_expert = {"teacher_obs": teacher_obs}
                        player.states = None
                        player.device = torch.device("cpu")
                        player.eval_policy = "prior"
                        player.config = {"vae_dim": 64}
                        with torch.no_grad(), patch.object(restored.a2c_network, "_encoder",
                                side_effect=AssertionError("prior evaluation used privileged encoder")):
                            evaluated = player.get_action({"obs": obs}, is_determenistic=True)
                            trained = restored(dict(obs=obs, teacher_obs=teacher_obs, vae_noise=noise,
                                is_train=False, prev_actions=None, rnn_states=None,
                                with_encoder=False, with_vae=True, skip_critic=True))["mus"].clamp(-1, 1)
                        torch.testing.assert_close(evaluated, trained, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
