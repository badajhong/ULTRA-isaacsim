"""Simulation parameters (replacement for ``isaacgym.gymapi.SimParams``).

The ``sim`` section of ULTRA's YAML files keeps Isaac Gym's names (``physx.num_position_iterations``, ...); this
module parses it and the backend maps every field to its Isaac Sim / PhysX 5 counterpart.
"""

from types import SimpleNamespace


def _physx_defaults():
    # Isaac Gym's PhysX defaults, so that YAML files that omit a field behave as before.
    return SimpleNamespace(
        solver_type=1,
        num_position_iterations=4,
        num_velocity_iterations=1,
        contact_offset=0.02,
        rest_offset=0.001,
        bounce_threshold_velocity=0.2,
        max_depenetration_velocity=100.0,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
        max_gpu_contact_pairs=1024 * 1024,
        default_buffer_size_multiplier=2.0,
        num_threads=0,
        num_subscenes=0,
        use_gpu=True,
    )


class SimParams:
    def __init__(self):
        self.dt = 1.0 / 60.0
        self.substeps = 1
        self.up_axis = "z"
        self.gravity = (0.0, 0.0, -9.81)
        self.use_gpu_pipeline = True
        self.num_client_threads = 0
        self.physx = _physx_defaults()

    def __repr__(self):
        return f"SimParams(dt={self.dt}, substeps={self.substeps}, gravity={self.gravity}, physx={vars(self.physx)})"


def parse_sim_config(cfg_sim, sim_params):
    """Update ``sim_params`` from the ``sim`` section of an ULTRA YAML (same keys as ``gymutil.parse_sim_config``)."""
    for key, value in cfg_sim.items():
        if key == "physx":
            for physx_key, physx_value in value.items():
                setattr(sim_params.physx, physx_key, physx_value)
        elif key == "flex":
            continue  # Flex is an Isaac Gym-only engine
        elif key == "gravity":
            sim_params.gravity = tuple(float(v) for v in value)
        elif key == "up_axis":
            sim_params.up_axis = str(value).lower()
        else:
            setattr(sim_params, key, value)
    return sim_params
