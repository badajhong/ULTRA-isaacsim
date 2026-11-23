"""Build (and cache) the USD assets ULTRA simulates in Isaac Sim.

* The robot URDF is converted with Isaac Lab's ``UrdfConverter`` (fixed links kept, convex decomposition
  colliders), then patched so that it matches the Isaac Gym asset ULTRA was trained with: V-HACD parameters,
  Gym's per-shape collision filter bits, and massless sensor frames.
* Object meshes (``.obj``) are written directly as single rigid-body USDs with density-based mass and convex
  decomposition colliders, which is what Isaac Gym built from the object URDFs.

Converted files are cached under ``ultra/data/assets/usd/`` keyed by a hash of the source file and the options.
"""

import hashlib
import json
import math
import os

_CACHE_VERSION = "ultra-usd-v1"
DEFAULT_CACHE_DIR = os.path.join("ultra", "data", "assets", "usd")

# Isaac Gym gives links without an <inertial> element zero mass; PhysX needs a positive value.
MASSLESS_LINK_MASS = 1.0e-3
MASSLESS_LINK_INERTIA = 1.0e-7


def _file_digest(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _cache_key(*parts):
    h = hashlib.sha1(_CACHE_VERSION.encode())
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def filtered_link_pairs(filter_bits):
    """Link pairs that Isaac Gym would not collide: shapes collide unless ``filter_a & filter_b != 0``."""
    names = sorted(filter_bits)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i:]:
            if a != b and filter_bits[a] & filter_bits[b]:
                pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------------------------------------------
# Robot
# --------------------------------------------------------------------------------------------------------------
def prepare_robot_usd(spec, urdf_info, cache_dir=DEFAULT_CACHE_DIR, contact_offset=0.02, rest_offset=0.0):
    """Convert ``spec.urdf_path`` to USD (cached) and return the USD path."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf_path = os.path.abspath(spec.urdf_path)
    name = os.path.splitext(os.path.basename(urdf_path))[0]
    options = {
        "self_collisions": spec.self_collisions,
        "max_convex_hulls": spec.max_convex_hulls,
        "hull_vertex_limit": spec.hull_vertex_limit,
        "voxel_resolution": spec.voxel_resolution,
        "collision_filter": spec.collision_filter,
        "contact_offset": contact_offset,
        "rest_offset": rest_offset,
    }
    key = _cache_key(_file_digest(urdf_path), options)
    out_dir = os.path.abspath(os.path.join(cache_dir, f"{name}_{key}"))
    usd_path = os.path.join(out_dir, f"{name}.usd")
    done_marker = os.path.join(out_dir, ".ultra_ready")
    if os.path.isfile(usd_path) and os.path.isfile(done_marker):
        return usd_path

    print(f"[ULTRA][Isaac Sim] converting {urdf_path} -> {usd_path}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    drive = UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="none",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=out_dir,
        usd_file_name=f"{name}.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=spec.self_collisions,
        collider_type="convex_decomposition",
        joint_drive=drive,
    )
    UrdfConverter(cfg)
    _patch_robot_usd(usd_path, spec, urdf_info, contact_offset, rest_offset)
    with open(done_marker, "w") as f:
        json.dump(options, f, indent=1)
    return usd_path


def _patch_robot_usd(usd_path, spec, urdf_info, contact_offset, rest_offset):
    import glob

    from pxr import PhysxSchema, Usd, UsdPhysics

    # 1. Colliders. The converter instances the collision meshes, so they cannot be edited through the composed
    #    asset (instance proxies are read-only): patch every layer of the asset that defines collision prims.
    layers = [usd_path] + sorted(glob.glob(os.path.join(os.path.dirname(usd_path), "configuration", "*.usd*")))
    colliders = 0
    for layer_path in layers:
        stage = Usd.Stage.Open(layer_path)
        stage.SetEditTarget(stage.GetRootLayer())
        changed = False
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexDecomposition")
                api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                api.CreateMaxConvexHullsAttr().Set(int(spec.max_convex_hulls))
                api.CreateHullVertexLimitAttr().Set(int(spec.hull_vertex_limit))
                api.CreateVoxelResolutionAttr().Set(int(spec.voxel_resolution))
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(float(contact_offset))
            collision.CreateRestOffsetAttr().Set(float(rest_offset))
            colliders += 1
            changed = True
        if changed:
            stage.GetRootLayer().Save()

    # 2. Links (not instanced): massless frames and Isaac Gym's collision filter bits, authored in the root layer.
    stage = Usd.Stage.Open(usd_path)
    stage.SetEditTarget(stage.GetRootLayer())
    links = {prim.GetName(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    missing = [n for n in urdf_info.body_names if n not in links]
    if missing:
        raise RuntimeError(f"URDF links missing from the converted USD {usd_path}: {missing}")
    if colliders == 0:
        raise RuntimeError(f"no collision prims found in {usd_path}")

    for name in urdf_info.massless_bodies:
        mass_api = UsdPhysics.MassAPI.Apply(links[name])
        mass_api.CreateMassAttr().Set(MASSLESS_LINK_MASS)
        mass_api.CreateDiagonalInertiaAttr().Set((MASSLESS_LINK_INERTIA,) * 3)

    pairs = 0
    for a, b in filtered_link_pairs(spec.collision_filter):
        if a in links and b in links:
            UsdPhysics.FilteredPairsAPI.Apply(links[a]).CreateFilteredPairsRel().AddTarget(links[b].GetPath())
            pairs += 1

    stage.GetRootLayer().Save()
    print(f"[ULTRA][Isaac Sim] patched {colliders} colliders, {len(links)} links, {pairs} filtered link pairs "
          f"in {usd_path}", flush=True)


# --------------------------------------------------------------------------------------------------------------
# Objects
# --------------------------------------------------------------------------------------------------------------
def prepare_object_usd(spec, type_index, cache_dir=DEFAULT_CACHE_DIR):
    """Write a single-rigid-body USD for ``spec.mesh_path`` (cached) and return its path."""
    mesh_path = os.path.abspath(spec.mesh_path)
    options = {
        "density": spec.density,
        "max_convex_hulls": spec.max_convex_hulls,
        "hull_vertex_limit": spec.hull_vertex_limit,
        "voxel_resolution": spec.voxel_resolution,
        "type_index": type_index,
    }
    key = _cache_key(_file_digest(mesh_path), options)
    out_dir = os.path.abspath(os.path.join(cache_dir, "objects"))
    usd_path = os.path.join(out_dir, f"{spec.name}_{key}.usda")
    if os.path.isfile(usd_path):
        return usd_path
    os.makedirs(out_dir, exist_ok=True)
    _write_object_usd(usd_path, mesh_path, spec, type_index)
    return usd_path


def _write_object_usd(usd_path, mesh_path, spec, type_index):
    import numpy as np
    import trimesh
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    root_prim = root.GetPrim()
    root_prim.CreateAttribute("ultra:objectType", Sdf.ValueTypeNames.Int).Set(int(type_index))
    root_prim.CreateAttribute("ultra:objectName", Sdf.ValueTypeNames.String).Set(spec.name)

    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateDensityAttr().Set(float(spec.density))

    geom = UsdGeom.Mesh.Define(stage, "/Object/geometry")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    geom.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(verts))
    geom.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    geom.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    geom.CreateExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
    geom.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.2, 0.45, 0.85)]))

    geom_prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(geom_prim)
    UsdPhysics.MeshCollisionAPI.Apply(geom_prim).CreateApproximationAttr().Set("convexDecomposition")
    decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(geom_prim)
    decomposition.CreateMaxConvexHullsAttr().Set(int(spec.max_convex_hulls))
    decomposition.CreateHullVertexLimitAttr().Set(int(spec.hull_vertex_limit))
    decomposition.CreateVoxelResolutionAttr().Set(int(spec.voxel_resolution))
    UsdPhysics.MassAPI.Apply(geom_prim).CreateDensityAttr().Set(float(spec.density))
    stage.GetRootLayer().Save()


def rad_to_deg(value):
    return value * 180.0 / math.pi
