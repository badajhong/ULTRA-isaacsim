"""URDF metadata needed to reproduce Isaac Gym's body / DOF ordering without Isaac Gym."""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_MOVABLE_JOINTS = ("revolute", "continuous", "prismatic")


@dataclass
class UrdfInfo:
    body_names: list          # rigid bodies in Isaac Gym order
    dof_names: list           # movable joints in Isaac Gym order
    dof_lower: list
    dof_upper: list
    dof_effort: list
    dof_velocity: list
    massless_bodies: list = field(default_factory=list)  # links without an <inertial> element


def parse_urdf(path):
    """Parse a URDF and return its bodies and DOFs in Isaac Gym's order.

    Isaac Gym (with ``collapse_fixed_joints=False``) orders rigid bodies depth first from the root link and visits
    the children of every link sorted by link name; DOFs follow the order of their child bodies.  ULTRA's motion
    data, hard-coded body indices and released checkpoints all use this order.
    """
    root = ET.parse(path).getroot()
    links = [link.get("name") for link in root.findall("link")]
    massless = [link.get("name") for link in root.findall("link") if link.find("inertial") is None]
    children = {}
    parent_of = {}
    joint_of_child = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        children.setdefault(parent, []).append(child)
        parent_of[child] = parent
        joint_of_child[child] = joint
    roots = [name for name in links if name not in parent_of]
    if len(roots) != 1:
        raise ValueError(f"{path}: expected exactly one root link, found {roots}")

    body_names, dof_names, lower, upper, effort, velocity = [], [], [], [], [], []

    def visit(link):
        body_names.append(link)
        joint = joint_of_child.get(link)
        if joint is not None and joint.get("type") in _MOVABLE_JOINTS:
            limit = joint.find("limit")
            dof_names.append(joint.get("name"))
            if joint.get("type") == "continuous" or limit is None:
                lower.append(-float("inf"))
                upper.append(float("inf"))
            else:
                lower.append(float(limit.get("lower", 0.0)))
                upper.append(float(limit.get("upper", 0.0)))
            effort.append(float(limit.get("effort", 0.0)) if limit is not None else 0.0)
            velocity.append(float(limit.get("velocity", 0.0)) if limit is not None else 0.0)
        for child in sorted(children.get(link, [])):
            visit(child)

    visit(roots[0])
    return UrdfInfo(body_names, dof_names, lower, upper, effort, velocity, massless)


def urdf_collision_meshes(path):
    """Return the mesh files referenced by <collision> elements (absolute paths)."""
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    meshes = []
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is not None:
                meshes.append(os.path.join(base, mesh.get("filename")))
    return meshes
