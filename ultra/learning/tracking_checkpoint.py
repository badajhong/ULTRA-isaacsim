"""Select a checkpoint's policy architecture independently of the tracking task."""

from pathlib import Path

import torch


TRAIN_CONFIG_DIR = Path(__file__).resolve().parents[1] / "data/cfg/train/rlg"


def prepare_tracking_evaluation(args):
    """Resolve evaluation-only model selection before loading the runner config.

    Finetune policies need the V3 network and its saved FiLM/critic weights. They
    still run in the V2 tracking task so reference tracking scores can be compared
    with distilled policies. Custom --cfg_train files remain explicit overrides.
    """
    if args.eval_episodes and (
            args.eval_episodes < 0 or not args.test or args.task != "UltraDistillObjV2Point"):
        raise ValueError("--eval_episodes requires a positive count, --test and task UltraDistillObjV2Point")
    if args.eval_full_clips and not args.eval_episodes:
        raise ValueError("--eval_full_clips requires --eval_episodes and --test")
    if args.eval_training_conditions and (not args.eval_full_clips or args.task_mode):
        raise ValueError("--eval_training_conditions requires --eval_full_clips and no fixed --task_mode")
    if not args.eval_episodes:
        if getattr(args, "eval_randomization", "config") != "config":
            raise ValueError("--eval_randomization requires --eval_episodes")
        if args.eval_model != "auto":
            raise ValueError("--eval_model requires --eval_episodes")
        return

    if getattr(args, 'eval_policy', 'prior') == 'teacher':
        if args.eval_training_conditions:
            raise ValueError('Teacher evaluation does not support student training conditions')
        if args.eval_model != 'auto':
            raise ValueError('Teacher evaluation requires --eval_model auto')
        args.eval_model = 'teacher'
        args.eval_checkpoint_epoch = None
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    finetune = any(key.startswith(("a2c_network.rl_finetune_film.",
                                   "a2c_network.critic_value.")) for key in state)
    detected = "finetune" if finetune else "distillation"
    if args.eval_model != "auto" and args.eval_model != detected:
        raise ValueError(f"--eval_model={args.eval_model} conflicts with the {detected} checkpoint")
    args.eval_model = detected
    args.eval_checkpoint_epoch = int(checkpoint["epoch"]) if "epoch" in checkpoint else None

    # The shipped evaluation wrappers pass the distilled config. Replace it only
    # when a finetune checkpoint is detected; preserve caller-supplied custom configs.
    if finetune and Path(args.cfg_train).resolve() == TRAIN_CONFIG_DIR / "g1_student_vae.yaml":
        args.cfg_train = str(TRAIN_CONFIG_DIR / "g1_student_finetune.yaml")
    print(f"[tracking evaluation] Model={detected}; task={args.task}; "
          f"network config={args.cfg_train}. Finetune goal offsets are not used.", flush=True)
