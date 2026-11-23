"""Isaac Sim / Isaac Lab simulation backend with Isaac Gym-style state tensors.

One ``IsaacSimBackend`` owns the physics scene: a ground plane and, in every environment, one robot articulation
(actor 0) and optionally one rigid object (actor 1).  It exposes the tensors ULTRA's tasks were written for:

====================================  ===============================  =======================================
attribute                             shape                            content
====================================  ===============================  =======================================
``root_state_tensor``                 ``[num_envs * actors, 13]``      pos(3), quat xyzw(4), lin vel(3), ang vel(3)
``dof_state_tensor``                  ``[num_envs * dofs, 2]``         position, velocity
``rigid_body_state_tensor``           ``[num_envs * bodies, 13]``      robot links then object body
``net_contact_force_tensor``          ``[num_envs * bodies, 3]``       net contact force per body (world frame)
``dof_force_tensor``                  ``[num_envs * dofs]``            joint force along each DOF
====================================  ===============================  =======================================

Positions are environment-local and bodies / DOFs use Isaac Gym's order, exactly like Isaac Gym's tensor API.
Velocities are world-frame centre-of-mass velocities, as in Isaac Gym.  ``refresh_*`` copies the simulator
state into these tensors in place, and ``set_*`` writes rows back to the simulator.
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from env.sim.assets import DEFAULT_CACHE_DIR, prepare_object_usd, prepare_robot_usd, rad_to_deg
from env.sim.urdf_info import parse_urdf

ROBOT_ACTOR = 0
OBJECT_ACTOR = 1


@dataclass
class RobotSpec:
    """Robot articulation options.  Per-DOF lists use Isaac Gym's DOF order."""

    urdf_path: str
    init_height: float = 0.89
    position_control: bool = False          # PhysX PD drives (``set_dof_position_target_tensor``)
    stiffness: Optional[Sequence[float]] = None
    damping: Optional[Sequence[float]] = None
    effort_limit: Optional[Sequence[float]] = None   # None: URDF limits
    armature: Optional[Sequence[float]] = None
    # Isaac Gym AssetOptions defaults
    angular_damping: float = 0.5
    linear_damping: float = 0.0
    max_angular_velocity: float = 64.0      # rad/s
    max_linear_velocity: float = 1000.0
    self_collisions: bool = True
    collision_filter: Dict[str, int] = field(default_factory=dict)  # link name -> Isaac Gym shape filter bits
    # V-HACD settings of the collision meshes
    max_convex_hulls: int = 5
    hull_vertex_limit: int = 16
    voxel_resolution: int = 60000


@dataclass
class ObjectSpec:
    """One object type.  Environment ``i`` gets object type ``i % len(objects)``."""

    name: str
    mesh_path: str
    density: float
    max_convex_hulls: int = 10
    hull_vertex_limit: int = 64
    voxel_resolution: int = 300000
    angular_damping: float = 0.5
    linear_damping: float = 0.0
    max_angular_velocity: float = 64.0
    max_linear_velocity: float = 1000.0


@dataclass
class PlaneSpec:
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


class IsaacSimBackend:
    def __init__(
        self,
        sim_params,
        device: str,
        headless: bool,
        num_envs: int,
        env_spacing: float,
        robot: RobotSpec,
        objects: Optional[List[ObjectSpec]] = None,
        plane: Optional[PlaneSpec] = None,
        object_scale: float = 1.0,
        usd_cache_dir: str = DEFAULT_CACHE_DIR,
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.headless = headless
        self.sim_params = sim_params
        self.dt = float(sim_params.dt)
        self.robot_spec = robot
        self.object_specs = list(objects or [])
        self.plane_spec = plane or PlaneSpec()
        self.object_scale = float(object_scale)
        self.usd_cache_dir = usd_cache_dir

        self.urdf_info = parse_urdf(robot.urdf_path)
        self.body_names = list(self.urdf_info.body_names)
        self.dof_names = list(self.urdf_info.dof_names)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        self.has_object = len(self.object_specs) > 0
        self.num_actors_per_env = 2 if self.has_object else 1
        self.bodies_per_env = self.num_bodies + (1 if self.has_object else 0)
        self._frame_count = 0
        self.track_dof_forces = robot.position_control

        if getattr(sim_params, "substeps", 1) != 1:
            print(f"[ULTRA][Isaac Sim] sim.substeps={sim_params.substeps} is not supported; using 1 substep")

        self._create_sim()
        self._create_scene(env_spacing)
        self.sim.reset()
        self._init_views()
        self._alloc_buffers()
        self.refresh_all()
        self.viewer = None if headless else IsaacSimViewer(self)

    # ----------------------------------------------------------------------------------------------------------
    # construction
    # ----------------------------------------------------------------------------------------------------------
    def _create_sim(self):
        import isaaclab.sim as sim_utils

        p = self.sim_params.physx
        physx_kwargs = dict(
            solver_type=int(p.solver_type),
            bounce_threshold_velocity=float(p.bounce_threshold_velocity),
            friction_offset_threshold=float(p.friction_offset_threshold),
            friction_correlation_distance=float(p.friction_correlation_distance),
            gpu_max_rigid_contact_count=int(max(p.max_gpu_contact_pairs, 2**23)),
            gpu_max_rigid_patch_count=2**20,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**23,
        )
        # optional explicit Isaac Sim buffer sizes, e.g. ``sim.physx.gpu_max_rigid_patch_count`` in the YAML
        for key in list(vars(p)):
            if key.startswith("gpu_"):
                physx_kwargs[key] = int(getattr(p, key))

        cfg = sim_utils.SimulationCfg(
            dt=self.dt,
            device=self.device,
            gravity=tuple(self.sim_params.gravity),
            physx=sim_utils.PhysxCfg(**physx_kwargs),
            # Isaac Gym's default shape material: friction 1, restitution 0
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        )
        self.sim = sim_utils.SimulationContext(cfg)

    def _rigid_props(self, spec):
        import isaaclab.sim as sim_utils

        p = self.sim_params.physx
        return sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=spec.linear_damping,
            angular_damping=spec.angular_damping,
            max_linear_velocity=spec.max_linear_velocity,
            max_angular_velocity=rad_to_deg(spec.max_angular_velocity),
            max_depenetration_velocity=float(p.max_depenetration_velocity),
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=int(p.num_position_iterations),
            solver_velocity_iteration_count=int(p.num_velocity_iterations),
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        )

    def _collision_props(self):
        import isaaclab.sim as sim_utils

        p = self.sim_params.physx
        return sim_utils.CollisionPropertiesCfg(contact_offset=float(p.contact_offset), rest_offset=float(p.rest_offset))

    def _create_scene(self, env_spacing):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg

        robot = self.robot_spec
        p = self.sim_params.physx
        robot_usd = prepare_robot_usd(robot, self.urdf_info, self.usd_cache_dir,
                                      contact_offset=float(p.contact_offset), rest_offset=float(p.rest_offset))

        def per_dof(values, default):
            if values is None:
                return default
            return {name: float(v) for name, v in zip(self.dof_names, values)}

        zeros = {name: 0.0 for name in self.dof_names}
        effort = robot.effort_limit if robot.effort_limit is not None else self.urdf_info.dof_effort
        actuator = ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=per_dof(robot.stiffness, zeros) if robot.position_control else zeros,
            damping=per_dof(robot.damping, zeros) if robot.position_control else zeros,
            effort_limit_sim=per_dof(effort, None),
            armature=per_dof(robot.armature, zeros),
            friction=0.0,
        )
        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd,
                activate_contact_sensors=True,
                rigid_props=self._rigid_props(robot),
                collision_props=self._collision_props(),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=robot.self_collisions,
                    solver_position_iteration_count=int(p.num_position_iterations),
                    solver_velocity_iteration_count=int(p.num_velocity_iterations),
                    sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, robot.init_height), joint_pos={".*": 0.0}),
            actuators={"joints": actuator},
        )

        @_configclass
        class _SceneCfg(InteractiveSceneCfg):
            pass

        heterogeneous = len(self.object_specs) > 1
        scene_cfg = _SceneCfg(
            num_envs=self.num_envs,
            env_spacing=2.0 * env_spacing,  # Isaac Gym's spacing is the half-size of an environment
            replicate_physics=not heterogeneous,
            filter_collisions=True,
        )
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=_GroundPlaneSpawnerCfg(
                static_friction=self.plane_spec.static_friction,
                dynamic_friction=self.plane_spec.dynamic_friction,
                restitution=self.plane_spec.restitution,
                contact_offset=float(p.contact_offset),
                rest_offset=float(p.rest_offset),
            ),
        )
        scene_cfg.robot = robot_cfg
        scene_cfg.robot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0)

        if self.has_object:
            spec0 = self.object_specs[0]
            object_usds = [prepare_object_usd(s, i, self.usd_cache_dir) for i, s in enumerate(self.object_specs)]
            scale = (self.object_scale,) * 3
            common = dict(
                activate_contact_sensors=True,
                rigid_props=self._rigid_props(spec0),
                collision_props=self._collision_props(),
            )
            if heterogeneous:
                spawn = sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=[sim_utils.UsdFileCfg(usd_path=path, scale=scale) for path in object_usds],
                    random_choice=False,
                    **common,
                )
            else:
                spawn = sim_utils.UsdFileCfg(usd_path=object_usds[0], scale=scale, **common)
            scene_cfg.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
            scene_cfg.object_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object", update_period=0.0)

        self.scene = InteractiveScene(scene_cfg)
        if self.has_object:
            self.object_type_ids = self._read_object_types()

    def _read_object_types(self):
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        types = []
        for i, env_path in enumerate(self.scene.env_prim_paths):
            prim = stage.GetPrimAtPath(f"{env_path}/Object")
            attr = prim.GetAttribute("ultra:objectType") if prim.IsValid() else None
            types.append(attr.Get() if attr is not None and attr.IsValid() else -1)
        types = torch.tensor(types, dtype=torch.long)
        expected = torch.arange(self.num_envs) % len(self.object_specs)
        if not torch.equal(types, expected):
            raise RuntimeError(
                "object assets were not assigned round-robin (env i -> object i % num_objects); "
                f"got {types[:16].tolist()}..."
            )
        return types

    def _init_views(self):
        self.robot = self.scene["robot"]
        self.robot_view = self.robot.root_physx_view
        self.robot_contact = self.scene["robot_contact"]
        self.env_origins = self.scene.env_origins.to(self.device).clone()
        self._all_env_idx = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._all_env_idx_cpu = self._all_env_idx.cpu()

        physx_links = list(self.robot_view.shared_metatype.link_names)
        physx_dofs = list(self.robot_view.shared_metatype.dof_names)
        missing = [n for n in self.body_names if n not in physx_links] + [n for n in self.dof_names if n not in physx_dofs]
        if missing or len(physx_dofs) != self.num_dofs:
            raise RuntimeError(f"robot bodies/DOFs differ between the URDF and Isaac Sim: missing={missing}, dofs={physx_dofs}")
        self.num_physx_links = len(physx_links)
        # body_perm[i] = Isaac Sim link index of Isaac Gym body i (same for DOFs)
        self._body_perm = torch.tensor([physx_links.index(n) for n in self.body_names], device=self.device)
        self._dof_perm = torch.tensor([physx_dofs.index(n) for n in self.dof_names], device=self.device)
        self._body_perm_cpu = self._body_perm.cpu()

        sensor_bodies = list(self.robot_contact.body_names)
        pairs = [(self.body_names.index(n), j) for j, n in enumerate(sensor_bodies) if n in self.body_names]
        self._contact_gym_ids = torch.tensor([g for g, _ in pairs], device=self.device, dtype=torch.long)
        self._contact_sensor_ids = torch.tensor([s for _, s in pairs], device=self.device, dtype=torch.long)
        self._num_sensor_bodies = len(sensor_bodies)

        if self.has_object:
            self.object = self.scene["object"]
            self.object_view = self.object.root_physx_view
            self.object_contact = self.scene["object_contact"]

        # Isaac Gym-ordered joint limits as simulated
        limits = self.robot_view.get_dof_limits().to(self.device).view(self.num_envs, -1, 2)[0]
        self.dof_limits_lower = limits[self._dof_perm, 0].clone()
        self.dof_limits_upper = limits[self._dof_perm, 1].clone()

    def _alloc_buffers(self):
        n, d, f = self.num_envs, self.num_dofs, torch.float32
        dev = self.device
        self.root_state_tensor = torch.zeros(n * self.num_actors_per_env, 13, device=dev, dtype=f)
        self.dof_state_tensor = torch.zeros(n * d, 2, device=dev, dtype=f)
        self.rigid_body_state_tensor = torch.zeros(n * self.bodies_per_env, 13, device=dev, dtype=f)
        self.net_contact_force_tensor = torch.zeros(n * self.bodies_per_env, 3, device=dev, dtype=f)
        self.dof_force_tensor = torch.zeros(n * d, device=dev, dtype=f)
        # scratch buffers in Isaac Sim's (full-size) layout for indexed writes
        self._robot_pose = torch.zeros(n, 7, device=dev, dtype=f)
        self._robot_vel = torch.zeros(n, 6, device=dev, dtype=f)
        self._dof_pos = torch.zeros(n, d, device=dev, dtype=f)
        self._dof_vel = torch.zeros(n, d, device=dev, dtype=f)
        self._dof_cmd = torch.zeros(n, d, device=dev, dtype=f)
        self._last_actuation = torch.zeros(n, d, device=dev, dtype=f)
        if self.has_object:
            self._object_pose = torch.zeros(n, 7, device=dev, dtype=f)
            self._object_vel = torch.zeros(n, 6, device=dev, dtype=f)

    # ----------------------------------------------------------------------------------------------------------
    # simulation
    # ----------------------------------------------------------------------------------------------------------
    def simulate(self):
        """Advance physics by one ``dt``."""
        self.sim.step(render=False)
        self._frame_count += 1

    def get_frame_count(self):
        return self._frame_count

    def set_gravity(self, gravity):
        import carb

        g = [float(v) for v in gravity]
        self.sim.physics_sim_view.set_gravity(carb.Float3(g[0], g[1], g[2]))

    # ----------------------------------------------------------------------------------------------------------
    # state reads (Isaac Gym layout, in place)
    # ----------------------------------------------------------------------------------------------------------
    def refresh_all(self):
        self.refresh_actor_root_state_tensor()
        self.refresh_dof_state_tensor()
        self.refresh_rigid_body_state_tensor()
        self.refresh_net_contact_force_tensor()
        self.refresh_dof_force_tensor()

    def refresh_actor_root_state_tensor(self):
        n = self.num_envs
        rs = self.root_state_tensor.view(n, self.num_actors_per_env, 13)
        pose = self.robot_view.get_root_transforms().view(n, 7)
        rs[:, ROBOT_ACTOR, 0:3] = pose[:, 0:3] - self.env_origins
        rs[:, ROBOT_ACTOR, 3:7] = pose[:, 3:7]
        rs[:, ROBOT_ACTOR, 7:13] = self.robot_view.get_root_velocities().view(n, 6)
        if self.has_object:
            pose = self.object_view.get_transforms().view(n, 7)
            rs[:, OBJECT_ACTOR, 0:3] = pose[:, 0:3] - self.env_origins
            rs[:, OBJECT_ACTOR, 3:7] = pose[:, 3:7]
            rs[:, OBJECT_ACTOR, 7:13] = self.object_view.get_velocities().view(n, 6)

    def refresh_dof_state_tensor(self):
        n, d = self.num_envs, self.num_dofs
        ds = self.dof_state_tensor.view(n, d, 2)
        ds[..., 0] = self.robot_view.get_dof_positions().view(n, d)[:, self._dof_perm]
        ds[..., 1] = self.robot_view.get_dof_velocities().view(n, d)[:, self._dof_perm]

    def refresh_rigid_body_state_tensor(self):
        n, b = self.num_envs, self.num_bodies
        rb = self.rigid_body_state_tensor.view(n, self.bodies_per_env, 13)
        pose = self.robot_view.get_link_transforms().view(n, self.num_physx_links, 7)[:, self._body_perm]
        rb[:, :b, 0:3] = pose[..., 0:3] - self.env_origins.unsqueeze(1)
        rb[:, :b, 3:7] = pose[..., 3:7]
        rb[:, :b, 7:13] = self.robot_view.get_link_velocities().view(n, self.num_physx_links, 6)[:, self._body_perm]
        if self.has_object:
            pose = self.object_view.get_transforms().view(n, 7)
            rb[:, b, 0:3] = pose[:, 0:3] - self.env_origins
            rb[:, b, 3:7] = pose[:, 3:7]
            rb[:, b, 7:13] = self.object_view.get_velocities().view(n, 6)

    def refresh_net_contact_force_tensor(self):
        n = self.num_envs
        cf = self.net_contact_force_tensor.view(n, self.bodies_per_env, 3)
        forces = self.robot_contact.contact_physx_view.get_net_contact_forces(dt=self.dt).view(n, self._num_sensor_bodies, 3)
        cf[:, self._contact_gym_ids] = forces[:, self._contact_sensor_ids]
        if self.has_object:
            cf[:, self.num_bodies] = self.object_contact.contact_physx_view.get_net_contact_forces(dt=self.dt).view(n, 3)

    def refresh_dof_force_tensor(self):
        n, d = self.num_envs, self.num_dofs
        out = self.dof_force_tensor.view(n, d)
        if self.track_dof_forces:
            out[:] = self.robot_view.get_dof_projected_joint_forces().view(n, d)[:, self._dof_perm]
        else:
            out[:] = self._last_actuation

    # ----------------------------------------------------------------------------------------------------------
    # state writes
    # ----------------------------------------------------------------------------------------------------------
    def set_actor_root_state_tensor(self, root_states):
        n, a = self.num_envs, self.num_actors_per_env
        ids = torch.arange(n * a, device=self.device, dtype=torch.int32)
        self.set_actor_root_state_tensor_indexed(root_states, ids)

    def set_actor_root_state_tensor_indexed(self, root_states, actor_ids):
        """Write rows ``actor_ids`` (Isaac Gym actor indices ``env * actors_per_env + actor``) of ``root_states``."""
        actor_ids = actor_ids.to(self.device).long().flatten()
        if actor_ids.numel() == 0:
            return
        a = self.num_actors_per_env
        states = root_states.view(self.num_envs, a, 13)
        env_ids = actor_ids // a
        actor = actor_ids % a
        robot_envs = env_ids[actor == ROBOT_ACTOR]
        if robot_envs.numel() > 0:
            self._robot_pose[robot_envs, 0:3] = states[robot_envs, ROBOT_ACTOR, 0:3] + self.env_origins[robot_envs]
            self._robot_pose[robot_envs, 3:7] = _unit_quat(states[robot_envs, ROBOT_ACTOR, 3:7])
            self._robot_vel[robot_envs] = states[robot_envs, ROBOT_ACTOR, 7:13]
            idx = robot_envs.to(torch.int32)
            self.robot_view.set_root_transforms(self._robot_pose, idx)
            self.robot_view.set_root_velocities(self._robot_vel, idx)
        if self.has_object:
            object_envs = env_ids[actor == OBJECT_ACTOR]
            if object_envs.numel() > 0:
                self._object_pose[object_envs, 0:3] = states[object_envs, OBJECT_ACTOR, 0:3] + self.env_origins[object_envs]
                self._object_pose[object_envs, 3:7] = _unit_quat(states[object_envs, OBJECT_ACTOR, 3:7])
                self._object_vel[object_envs] = states[object_envs, OBJECT_ACTOR, 7:13]
                idx = object_envs.to(torch.int32)
                self.object_view.set_transforms(self._object_pose, idx)
                self.object_view.set_velocities(self._object_vel, idx)

    def set_dof_state_tensor_indexed(self, dof_state, actor_ids):
        """Write the DOF states of the robot actors in ``actor_ids`` (object actors have no DOFs)."""
        actor_ids = actor_ids.to(self.device).long().flatten()
        a = self.num_actors_per_env
        env_ids = actor_ids[actor_ids % a == ROBOT_ACTOR] // a
        if env_ids.numel() == 0:
            return
        ds = dof_state.view(self.num_envs, self.num_dofs, 2)
        rows = env_ids.unsqueeze(1)
        cols = self._dof_perm.unsqueeze(0)
        self._dof_pos[rows, cols] = ds[env_ids, :, 0]
        self._dof_vel[rows, cols] = ds[env_ids, :, 1]
        idx = env_ids.to(torch.int32)
        self.robot_view.set_dof_positions(self._dof_pos, idx)
        self.robot_view.set_dof_velocities(self._dof_vel, idx)

    def set_dof_actuation_force_tensor(self, forces):
        forces = forces.view(self.num_envs, self.num_dofs)
        self._last_actuation.copy_(forces)
        self._dof_cmd[:, self._dof_perm] = forces
        self.robot_view.set_dof_actuation_forces(self._dof_cmd, self._all_env_idx)

    def set_dof_position_target_tensor(self, targets):
        targets = targets.view(self.num_envs, self.num_dofs)
        self._dof_cmd[:, self._dof_perm] = targets
        self.robot_view.set_dof_position_targets(self._dof_cmd, self._all_env_idx)

    # ----------------------------------------------------------------------------------------------------------
    # physical properties (CPU tensors, Isaac Gym body order)
    # ----------------------------------------------------------------------------------------------------------
    def find_body_index(self, name):
        """Isaac Gym rigid-body index of a robot link, or -1."""
        return self.body_names.index(name) if name in self.body_names else -1

    def get_robot_masses(self):
        return self.robot_view.get_masses().view(self.num_envs, -1)[:, self._body_perm_cpu].clone()

    def get_robot_coms(self):
        return self.robot_view.get_coms().view(self.num_envs, -1, 7)[:, self._body_perm_cpu, 0:3].clone()

    def get_robot_inertias(self):
        return self.robot_view.get_inertias().view(self.num_envs, -1, 9)[:, self._body_perm_cpu].clone()

    def set_robot_masses(self, masses):
        data = self.robot_view.get_masses().view(self.num_envs, -1).clone()
        data[:, self._body_perm_cpu] = masses.cpu().to(data.dtype)
        self.robot_view.set_masses(data, self._all_env_idx_cpu)

    def set_robot_coms(self, coms):
        data = self.robot_view.get_coms().view(self.num_envs, -1, 7).clone()
        data[:, self._body_perm_cpu, 0:3] = coms.cpu().to(data.dtype)
        self.robot_view.set_coms(data, self._all_env_idx_cpu)

    def set_robot_inertias(self, inertias):
        data = self.robot_view.get_inertias().view(self.num_envs, -1, 9).clone()
        data[:, self._body_perm_cpu] = inertias.cpu().to(data.dtype)
        self.robot_view.set_inertias(data, self._all_env_idx_cpu)

    def set_robot_friction(self, friction):
        """Set static and dynamic friction of every robot shape, one value per environment."""
        mats = self.robot_view.get_material_properties().view(self.num_envs, -1, 3).clone()
        f = torch.as_tensor(friction, dtype=mats.dtype).cpu().view(self.num_envs, 1)
        mats[:, :, 0] = f
        mats[:, :, 1] = f
        self.robot_view.set_material_properties(mats, self._all_env_idx_cpu)

    def get_object_masses(self):
        return self.object_view.get_masses().view(self.num_envs).clone()

    def get_object_coms(self):
        return self.object_view.get_coms().view(self.num_envs, 7)[:, 0:3].clone()

    def get_object_inertias(self):
        return self.object_view.get_inertias().view(self.num_envs, 9).clone()

    def set_object_masses(self, masses):
        data = torch.as_tensor(masses, dtype=torch.float32).cpu().view(self.object_view.get_masses().shape)
        self.object_view.set_masses(data, self._all_env_idx_cpu)

    def set_object_coms(self, coms):
        data = self.object_view.get_coms().clone()
        data.view(self.num_envs, 7)[:, 0:3] = torch.as_tensor(coms, dtype=data.dtype).cpu()
        self.object_view.set_coms(data, self._all_env_idx_cpu)

    def set_object_inertias(self, inertias):
        data = torch.as_tensor(inertias, dtype=torch.float32).cpu().view(self.object_view.get_inertias().shape)
        self.object_view.set_inertias(data, self._all_env_idx_cpu)

    def set_object_material(self, friction, restitution):
        """Set static/dynamic friction and restitution of the object shapes, one value per environment."""
        mats = self.object_view.get_material_properties().view(self.num_envs, -1, 3).clone()
        f = torch.as_tensor(friction, dtype=mats.dtype).cpu().view(self.num_envs, 1)
        r = torch.as_tensor(restitution, dtype=mats.dtype).cpu().view(self.num_envs, 1)
        mats[:, :, 0] = f
        mats[:, :, 1] = f
        mats[:, :, 2] = r
        self.object_view.set_material_properties(mats, self._all_env_idx_cpu)

    # ----------------------------------------------------------------------------------------------------------
    # visuals
    # ----------------------------------------------------------------------------------------------------------
    def set_object_color(self, env_id, rgb):
        """Recolour the object of one environment (used by dataset playback to show contact)."""
        if self.viewer is None or not self.has_object:
            return
        self.viewer.set_object_color(env_id, rgb)


def _unit_quat(q):
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)


# ----------------------------------------------------------------------------------------------------------------
# ground plane spawner (avoids downloading Isaac Lab's default ground USD from Nucleus)
# ----------------------------------------------------------------------------------------------------------------
def _spawn_ground_plane(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    sim_utils.standardize_xform_ops(root.GetPrim(), translation=translation, orientation=orientation)
    plane = UsdGeom.Plane.Define(stage, f"{prim_path}/collisionPlane")
    plane.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    plane.CreateWidthAttr().Set(cfg.size)
    plane.CreateLengthAttr().Set(cfg.size)
    plane.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    collision = PhysxSchema.PhysxCollisionAPI.Apply(plane.GetPrim())
    collision.CreateContactOffsetAttr().Set(cfg.contact_offset)
    collision.CreateRestOffsetAttr().Set(cfg.rest_offset)

    material_path = f"{prim_path}/physicsMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=cfg.static_friction, dynamic_friction=cfg.dynamic_friction, restitution=cfg.restitution
    )
    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_physics_material(plane.GetPrim().GetPath().pathString, material_path)

    half = cfg.size / 2.0
    visual = UsdGeom.Mesh.Define(stage, f"{prim_path}/visual")
    visual.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0), Gf.Vec3f(half, -half, 0),
                                                 Gf.Vec3f(half, half, 0), Gf.Vec3f(-half, half, 0)]))
    visual.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    visual.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    visual.CreateNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 4))
    visual.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.35, 0.37, 0.4)]))
    return root.GetPrim()


def _configclass(cls):
    from isaaclab.utils import configclass

    return configclass(cls)


def _make_ground_cfg_class():
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.utils import configclass

    @configclass
    class GroundPlaneSpawnerCfg(SpawnerCfg):
        func: object = _spawn_ground_plane
        size: float = 1000.0
        static_friction: float = 1.0
        dynamic_friction: float = 1.0
        restitution: float = 0.0
        contact_offset: float = 0.02
        rest_offset: float = 0.0

    return GroundPlaneSpawnerCfg


class _LazyGroundCfg:
    """Defers the Isaac Lab import until the app is running."""

    _cls = None

    def __call__(self, **kwargs):
        if _LazyGroundCfg._cls is None:
            _LazyGroundCfg._cls = _make_ground_cfg_class()
        return _LazyGroundCfg._cls(**kwargs)


_GroundPlaneSpawnerCfg = _LazyGroundCfg()


# ----------------------------------------------------------------------------------------------------------------
# viewer
# ----------------------------------------------------------------------------------------------------------------
class IsaacSimViewer:
    """Isaac Sim GUI window used where ULTRA used the Isaac Gym viewer.

    Coordinates passed to the drawing / camera helpers are environment-local, like Isaac Gym's per-env drawing.
    """

    width, height = 1600, 900

    def __init__(self, backend):
        from isaaclab.sim.utils import get_current_stage

        self.backend = backend
        self._stage = get_current_stage()
        self._draw = None
        self._annotator = None
        self._object_meshes = {}
        try:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # debug drawing is optional
            print(f"[ULTRA][Isaac Sim] debug drawing disabled: {exc}")

    def _origin(self, env_id):
        return self.backend.env_origins[int(env_id)].detach().cpu().numpy()

    def is_running(self):
        from env.sim.app import get_app

        app = get_app()
        return app is None or app.is_running()

    def render(self):
        if not self.is_running():
            sys.exit()
        self.backend.sim.render()

    def look_at(self, eye, target, env_id=0):
        origin = self._origin(env_id)
        self.backend.sim.set_camera_view(eye=list(np.asarray(eye) + origin), target=list(np.asarray(target) + origin))

    def clear_lines(self):
        if self._draw is not None:
            self._draw.clear_lines()
            self._draw.clear_points()

    def draw_lines(self, env_id, vertices, colors, width=1.0):
        """Draw line segments; ``vertices`` is ``[2 * k, 3]`` and ``colors`` ``[k, 3]`` (or ``[2 * k, 3]``)."""
        if self._draw is None:
            return
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3) + self._origin(env_id)
        cols = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if len(cols) == len(verts):
            cols = cols[0::2]
        starts = [tuple(v) for v in verts[0::2]]
        ends = [tuple(v) for v in verts[1::2]]
        rgba = [tuple(c) + (1.0,) for c in cols]
        self._draw.draw_lines(starts, ends, rgba, [float(width)] * len(starts))

    def draw_wire_sphere(self, env_id, position, radius, color, segments=12):
        """Three orthogonal circles, like ``gymutil.WireframeSphereGeometry``."""
        c = np.asarray(position, dtype=np.float32)
        angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
        verts = []
        for axes in ((0, 1), (0, 2), (1, 2)):
            pts = np.zeros((segments + 1, 3), dtype=np.float32)
            pts[:, axes[0]] = circle[:, 0]
            pts[:, axes[1]] = circle[:, 1]
            pts += c
            for i in range(segments):
                verts.extend([pts[i], pts[i + 1]])
        self.draw_lines(env_id, np.asarray(verts), np.asarray([color] * (len(verts) // 2)))

    def set_object_color(self, env_id, rgb):
        from pxr import Gf, UsdGeom, Vt

        meshes = self._object_meshes.get(env_id)
        if meshes is None:
            root = self._stage.GetPrimAtPath(f"{self.backend.scene.env_prim_paths[env_id]}/Object")
            meshes = [UsdGeom.Mesh(p) for p in _iter_prims(root) if p.IsA(UsdGeom.Mesh)]
            self._object_meshes[env_id] = meshes
        for mesh in meshes:
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in rgb])]))

    def capture_rgb(self):
        """Return the current viewport image as an ``[H, W, 3]`` uint8 array (or None)."""
        try:
            if self._annotator is None:
                import omni.replicator.core as rep

                product = rep.create.render_product("/OmniverseKit_Persp", (self.width, self.height))
                self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                self._annotator.attach([product])
                self.backend.sim.render()
            data = self._annotator.get_data()
            if data is None or getattr(data, "size", 0) == 0:
                return None
            return np.asarray(data)[..., :3].copy()
        except Exception as exc:
            print(f"[ULTRA][Isaac Sim] frame capture failed: {exc}")
            return None

    def write_image(self, path):
        frame = self.capture_rgb()
        if frame is None:
            return False
        from PIL import Image

        Image.fromarray(frame).save(path)
        return True


def _iter_prims(root):
    from pxr import Usd

    if root is None or not root.IsValid():
        return []
    return list(Usd.PrimRange(root))
