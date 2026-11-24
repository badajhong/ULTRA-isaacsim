# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import random

import numpy as np
import yaml

from env.sim.sim_params import SimParams, parse_sim_config

SIM_TIMESTEP = 1.0 / 1000.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    import torch

    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["params"]["torch_deterministic"] = True

    exp_name = cfg_train["params"]["config"]['name']

    if args.experiment != 'Base':
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, args.physics_engine)

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True

    if args.checkpoint != "Base":
        cfg_train["params"]["load_path"] = args.checkpoint
        
    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed

    cfg["args"] = args

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.num_subscenes = args.subscenes
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def _parse_device_str(device_str):
    if device_str in ("cpu", "cuda"):
        return device_str, 0
    device, _, device_id = device_str.partition(":")
    if device != "cuda" or not device_id.isdigit():
        raise ValueError(f'Invalid device string "{device_str}"')
    return device, int(device_id)


def parse_arguments(description="ULTRA", custom_parameters=()):
    """Command line of the former Isaac Gym scripts (``gymutil.parse_arguments``) plus Isaac Sim options."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sim_device", type=str, default="cuda:0", help="Physics device in PyTorch-like syntax")
    parser.add_argument("--pipeline", type=str, default="gpu", help="Tensor pipeline (cpu/gpu)")
    parser.add_argument("--graphics_device_id", type=int, default=0, help="Unused; kept for compatibility")
    parser.add_argument("--physx", action="store_true", help="Unused; PhysX is the only engine")
    parser.add_argument("--num_threads", type=int, default=0, help="Unused by Isaac Sim; kept for compatibility")
    parser.add_argument("--subscenes", type=int, default=0, help="Unused; kept for compatibility")
    parser.add_argument("--slices", type=int, help="Unused; kept for compatibility")
    parser.add_argument("--enable_cameras", action="store_true",
                        help="Enable Isaac Sim rendering in headless mode (headless image capture)")
    for argument in custom_parameters:
        kwargs = {k: v for k, v in argument.items() if k != "name"}
        parser.add_argument(argument["name"], **kwargs)
    args = parser.parse_args()

    args.sim_device_type, args.compute_device_id = _parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()
    assert pipeline in ("cpu", "gpu", "cuda"), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = pipeline in ("gpu", "cuda")
    if args.sim_device_type != "cuda" and args.use_gpu_pipeline:
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = "CPU"
        args.use_gpu_pipeline = False
    args.physics_engine = "physx"
    args.use_gpu = args.sim_device_type == "cuda"
    if args.slices is None:
        args.slices = args.subscenes
    return args


def get_args(benchmark=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--eval_episodes", "type": int, "default": 0,
            "help": "Evaluate tracking success over this many completed student episodes (requires --test)"},
        {"name": "--eval_full_clips", "action": "store_true", "default": False,
            "help": "Evaluate every supplied motion once, from its first frame"},
        {"name": "--eval_training_conditions", "action": "store_true", "default": False,
            "help": "With --eval_full_clips, use training-style random starts, masks and rollout settings"},
        {"name": "--eval_randomization", "choices": ["config", "nominal"], "default": "config",
            "help": "Tracking evaluation physics/sensing: preserve config or disable training randomization"},
        {"name": "--eval_policy", "choices": ["prior", "encoder", "teacher"], "default": "prior",
            "help": "Evaluation action source: student prior, privileged student encoder, or teacher"},
        {"name": "--eval_model", "choices": ["auto", "distillation", "finetune"], "default": "auto",
            "help": "Tracking evaluation network; auto detects distilled/V3 finetune checkpoints"},
        {"name": "--eval_output", "type": str, "default": "",
            "help": "Write full-clip tracking metrics and per-motion results to this JSON file"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--resume_from", "type": str, "default": "",
            "help": "Initialize training from a student checkpoint"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--multi_gpu", "action": "store_true", "default": False,
            "help": "Enable torch.distributed (DDP) multi-GPU training; launch with torchrun (see scripts/train_student_multigpu.sh)"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--projtype", "type": str, "default": "None",
            "help": "Can be None, Auto, Mouse"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_env", "type": str, "default": "Base", "help": "Environment configuration file (.yaml)"},
        {"name": "--cfg_train", "type": str, "default": "Base", "help": "Training configuration file (.yaml)"},
        {"name": "--motion_file", "type": str,
            "default": "", "help": "Specify reference motion file"},
        {"name": "--play_dataset", "action": "store_true", "default": False,
            "help": "Display the dataset"},
        {"name": "--init_vel", "action": "store_true", "default": False,
            "help": "Init the object velocity at the first frame"},
        {"name": "--save_images", "action": "store_true", "default": False,
            "help": "save images for viewer"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--task_mode", "type": str, "default": None,
            "help": "Student observation preset for playback: full_track, sparse_track, object_obs"},
        {"name": "--obj_obs", "type": str, "default": "points",
            "help": "Object observation used when task_mode=object_obs: points, pos, none"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--frames_scale", "type": float, "default": 0.,
            "help": "Set the fps scale for the reference HOI dataset"},
        {"name": "--ball_size", "type": float, "default": 0.,
            "help": "Set the ball size scale"},
        {"name": "--cg1", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 1"},
        {"name": "--cg2", "type": float, "default": -1.,
            "help": "Set the contact graph reward weight 2"},
        {"name": "--ig", "type": float, "default": -1.,
            "help": "Set the interaction graph reward weight"},
        {"name": "--op", "type": float, "default": -1.,
            "help": "Set the object position reward weight"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--output_path", "type": str, "default": "output/", "help": "Specify output directory"},
        {"name": "--llc_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."}]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
