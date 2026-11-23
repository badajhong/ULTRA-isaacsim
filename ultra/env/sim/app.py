"""Start the Isaac Sim application (must happen before importing ``isaaclab`` / ``omni`` modules)."""

import os
import sys

_SIMULATION_APP = None


def launch_app(headless=True, device="cuda:0", distributed=False, enable_cameras=False):
    """Launch (once) and return the Isaac Sim ``SimulationApp``.

    Args:
        headless: run without the GUI window.
        device: simulation device, ``"cuda:<id>"`` or ``"cpu"``.
        distributed: multi-GPU run launched with ``torchrun``; the device is taken from ``LOCAL_RANK``.
        enable_cameras: enable the rendering pipeline in headless mode (needed only for headless image capture).
    """
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP

    from isaaclab.app import AppLauncher

    # Kit parses sys.argv; hide ULTRA's own command line from it while the app starts.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        launcher = AppLauncher(
            headless=headless,
            device=device,
            distributed=distributed,
            enable_cameras=enable_cameras,
        )
    finally:
        sys.argv = saved_argv
    _SIMULATION_APP = launcher.app
    return _SIMULATION_APP


def launch_app_from_args(args):
    """Launch the app from ULTRA's parsed command line (see ``utils.config.get_args``)."""
    distributed = bool(getattr(args, "multi_gpu", False))
    if distributed:
        device = "cuda:" + os.environ.get("LOCAL_RANK", "0")
    else:
        device = args.sim_device if args.use_gpu_pipeline else "cpu"
    return launch_app(
        headless=args.headless,
        device=device,
        distributed=distributed,
        enable_cameras=getattr(args, "enable_cameras", False),
    )


def get_app():
    return _SIMULATION_APP


def close_app(exit_code=0):
    """Terminate the process with ``exit_code``.

    Kit's graceful shutdown can hang after GPU physics runs, so the process exits directly (after finishing an
    active wandb run).  Checkpoints and rollouts are already written at this point.
    """
    global _SIMULATION_APP
    if _SIMULATION_APP is None:
        return
    _SIMULATION_APP = None
    wandb = sys.modules.get("wandb")
    if wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            wandb.finish(exit_code=exit_code)
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def run_and_close(main_fn):
    """Run ``main_fn`` and exit with its status, printing any exception before Isaac Sim is torn down."""
    import traceback

    exit_code = 0
    try:
        main_fn()
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        traceback.print_exc()
        exit_code = 130
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    close_app(exit_code)
