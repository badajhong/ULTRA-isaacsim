# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys

import numpy as np
import torch


# Base class for RL tasks.  The simulator is created by ``create_sim`` (see env.tasks.humanoid) and is an
# ``env.sim.backend.IsaacSimBackend``; ``self.viewer`` is its GUI window, or None when running headless.
class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False):
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.headless = cfg["headless"]

        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.start_times = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.data_id = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.contact_reset = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float)
        self.extras = {}

        # optional observation / action noise (``noise_lambda`` entries), empty by default
        self.dr_randomizations = {}

        # create envs, sim and viewer
        self.create_sim()

        self.enable_viewer_sync = True
        self.viewer = self.sim.viewer

        # if running with a viewer, set the camera (Z-up)
        if self.viewer is not None:
            self.viewer.look_at([20.0, 25.0, 3.0], [10.0, 15.0, 0.0])

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = 'z'
            sim_params.gravity = (0.0, 0.0, -9.81)
            return 2
        return 1

    def create_sim(self):
        raise NotImplementedError

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
        if self.viewer and self.enable_viewer_sync:
            # exits the process when the window has been closed
            self.viewer.render()

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError
