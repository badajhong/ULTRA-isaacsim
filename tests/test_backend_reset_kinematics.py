"""Reset must synchronize link poses before the next observation is read."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from env.sim.backend import IsaacSimBackend


class BackendResetKinematicsTest(unittest.TestCase):
    def test_writing_reset_joint_state_updates_articulation_without_stepping(self):
        backend = object.__new__(IsaacSimBackend)
        backend.device = 'cpu'
        backend.num_envs = 2
        backend.num_actors_per_env = 2
        backend.num_dofs = 2
        backend._dof_perm = torch.tensor([1, 0])
        backend._dof_pos = torch.zeros(2, 2)
        backend._dof_vel = torch.zeros(2, 2)
        backend.robot_view = SimpleNamespace(set_dof_positions=Mock(), set_dof_velocities=Mock())
        physics_view = SimpleNamespace(update_articulations_kinematic=Mock())
        backend.sim = SimpleNamespace(physics_sim_view=physics_view, step=Mock())
        state = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])

        backend.set_dof_state_tensor_indexed(state, torch.tensor([2]))

        torch.testing.assert_close(backend._dof_pos[1], torch.tensor([7., 5.]))
        torch.testing.assert_close(backend._dof_vel[1], torch.tensor([8., 6.]))
        physics_view.update_articulations_kinematic.assert_called_once_with()
        backend.sim.step.assert_not_called()


if __name__ == '__main__':
    unittest.main()
