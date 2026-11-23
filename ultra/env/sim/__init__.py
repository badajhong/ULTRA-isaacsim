"""Isaac Sim (Isaac Lab) simulation backend for ULTRA.

ULTRA was written against Isaac Gym's tensor API. This package provides the same data layout on top of
Isaac Sim 5.x / Isaac Lab 2.x so that the task code keeps working unchanged:

* state tensors use Isaac Gym's layout: root states ``[num_envs * actors, 13]`` (pos, quat xyzw, lin vel,
  ang vel), DOF states ``[num_envs * dofs, 2]``, rigid-body states ``[num_envs * bodies, 13]`` and net contact
  forces ``[num_envs * bodies, 3]``;
* bodies and DOFs follow Isaac Gym's URDF ordering (depth first, children sorted by name), which the released
  motion data and checkpoints depend on;
* positions are expressed in each environment's local frame, as in Isaac Gym.

``app.launch_app`` must run before anything imports ``isaaclab`` / ``omni`` modules.
"""
