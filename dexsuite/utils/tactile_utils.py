"""Robot-level tactile helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import genesis.utils.geom as gu
import numpy as np
import torch
import trimesh

from dexsuite.utils.orientation_utils import quat_to_rotmat_wxyz_torch


@dataclass(frozen=True)
class _PadSpec:
    name: str
    link: object
    link_idx_local: int
    points_local: torch.Tensor
    center_local: torch.Tensor
    radius: float
    normal_axis_local: torch.Tensor
    rows: int
    cols: int


@dataclass(frozen=True)
class _ExternalGeomSpec:
    geom_idx: int
    link: object
    center_local: torch.Tensor
    radius: float


@dataclass(frozen=True)
class _GeomSDFCache:
    T_mesh_to_sdf: torch.Tensor
    sdf_res: torch.Tensor
    sdf_val: torch.Tensor
    sdf_grad: torch.Tensor
    sdf_max: float
    sdf_cell_size: float
    init_pos: torch.Tensor
    init_quat: torch.Tensor


@dataclass(frozen=True)
class _PadKinematics:
    pos: torch.Tensor
    quat: torch.Tensor
    linvel: torch.Tensor
    angvel: torch.Tensor
    net_force: torch.Tensor


@dataclass
class _ClosestSurfaceState:
    signed_distance: torch.Tensor
    normal_world: torch.Tensor
    indenter_linvel: torch.Tensor
    indenter_angvel: torch.Tensor
    closest_link_local: torch.Tensor
    indenter_quat: torch.Tensor


class RigidPadTactile:
    """Read EasySim-style tactile observations for rigid fingertip pads.

    Pad samples are returned in the pad-local grid convention with Genesis contact forces mapped onto configured tactile links. Internal caches are keyed by geometry and reused across reads.
    """

    _TACTILE_KN = 1.0
    _TACTILE_KT = 0.1
    _TACTILE_MU = 2.0
    _NORMALIZE_SWITCH = 2e-4
    _NORMALIZE_FALLBACK = 8e-4
    _SOFT_CONTACT_MARGIN = 1.5e-3
    _PROXIMITY_MARGIN = 5e-3

    def __init__(
        self,
        entity,
        *,
        link_names: tuple[str, ...],
        grid_size: tuple[int, int, int] | None = None,
        point_distance: float = 0.002,
    ) -> None:
        """Initialize the object.

        Args:
        entity: Genesis rigid entity that owns tactile links.
        link_names: Link names treated as tactile pads.
        grid_size: Tactile grid definition `(rows, depth, cols)`. Defaults to None.
        point_distance: Distance in meters between adjacent tactile points. Defaults to None.
        """
        self.entity = entity
        self.link_names = tuple(link_names)
        self.grid_size = tuple(int(v) for v in grid_size) if grid_size else (12, 1, 32)
        self.point_distance = float(point_distance)
        self._single_env = bool(getattr(self.entity._solver, "n_envs", 0) == 0)
        self._device = self.entity.get_pos().device
        self._solver = self.entity._solver
        self._own_geom_start = int(self.entity.geom_start)
        self._own_geom_end = int(self.entity.geom_end)
        self._geom_cache: dict[int, _GeomSDFCache] = {}

        self._pads = tuple(self._build_pad(name) for name in self.link_names)
        self._external_geoms = tuple(
            self._build_external_geom(geom_idx)
            for geom_idx in range(len(self._solver.geoms))
            if geom_idx < self._own_geom_start or geom_idx >= self._own_geom_end
        )

    def _build_pad(self, name: str) -> _PadSpec:
        link = self.entity.get_link(name)
        if len(link.geoms) == 0:
            raise ValueError(f"Tactile link '{name}' has no collision geoms.")
        geom = link.geoms[0]

        verts_local = gu.transform_by_trans_quat(
            np.asarray(geom.init_verts, dtype=np.float32),
            np.asarray(geom.init_pos, dtype=np.float32),
            np.asarray(geom.init_quat, dtype=np.float32),
        )
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts_local, dtype=np.float32),
            faces=np.asarray(geom.init_faces, dtype=np.int32),
            process=False,
        )

        points_local, normal_axis_local, rows, cols = self._generate_tactile_points(mesh)
        points_center_local = points_local.mean(axis=0).astype(np.float32)
        points_radius = float(
            np.linalg.norm(points_local - points_center_local[None, :], axis=-1).max(),
        )
        return _PadSpec(
            name=name,
            link=link,
            link_idx_local=int(link.idx_local),
            points_local=torch.as_tensor(
                points_local,
                dtype=torch.float32,
                device=self._device,
            ),
            center_local=torch.as_tensor(
                points_center_local,
                dtype=torch.float32,
                device=self._device,
            ),
            radius=points_radius,
            normal_axis_local=torch.as_tensor(
                normal_axis_local,
                dtype=torch.float32,
                device=self._device,
            ),
            rows=rows,
            cols=cols,
        )

    def _build_external_geom(self, geom_idx: int) -> _ExternalGeomSpec:
        geom = self._solver.geoms[int(geom_idx)]
        verts_local = gu.transform_by_trans_quat(
            np.asarray(geom.init_verts, dtype=np.float32),
            np.asarray(geom.init_pos, dtype=np.float32),
            np.asarray(geom.init_quat, dtype=np.float32),
        )
        center_local = verts_local.mean(axis=0).astype(np.float32)
        radius = float(
            np.linalg.norm(verts_local - center_local[None, :], axis=-1).max(),
        )
        return _ExternalGeomSpec(
            geom_idx=int(geom_idx),
            link=geom.link,
            center_local=torch.as_tensor(
                center_local,
                dtype=torch.float32,
                device=self._device,
            ),
            radius=radius,
        )

    def _generate_tactile_points(
        self,
        mesh: trimesh.Trimesh,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        def _raise_incomplete_tactile_hits() -> None:
            raise RuntimeError("Incomplete tactile ray hits.")

        rows = int(self.grid_size[0])
        cols = int(self.grid_size[-1])
        num_divs = (rows, cols)

        elastomer_dims = np.diff(mesh.bounds, axis=0).squeeze()
        slim_axis = int(elastomer_dims.argmin())
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0

        grid_axes = []
        idx = 0
        for axis_i in range(3):
            if axis_i == slim_axis:
                grid_axes.append([-1.0])
                continue
            axis_grid = np.linspace(
                center[axis_i]
                - self.point_distance * (num_divs[idx] + 1) / 2.0,
                center[axis_i]
                + self.point_distance * (num_divs[idx] + 1) / 2.0,
                num_divs[idx] + 2,
            )
            grid_axes.append(axis_grid[1:-1])
            idx += 1

        ray_origins = np.array(list(product(grid_axes[0], grid_axes[1], grid_axes[2])), dtype=np.float32)
        ray_directions = np.zeros_like(ray_origins)
        ray_directions[:, slim_axis] = 1.0

        try:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            _, index_ray, locations = intersector.intersects_id(
                ray_origins,
                ray_directions,
                return_locations=True,
                multiple_hits=False,
            )
            points_local = locations[index_ray.argsort()]
            if points_local.shape[0] != rows * cols:
                _raise_incomplete_tactile_hits()
        except (AttributeError, ImportError, RuntimeError, ValueError):
            # The pad mesh is flat and rectangular. If trimesh ray queries are not
            # available, project onto the mesh bounds face just like the URDF mesh.
            points_local = ray_origins.copy()
            points_local[:, slim_axis] = mesh.bounds[0, slim_axis]

        normal_axis_local = np.eye(3, dtype=np.float32)[slim_axis]
        return points_local.astype(np.float32), normal_axis_local, rows, cols

    def _geom_sdf(self, geom_idx: int) -> _GeomSDFCache:
        geom_idx = int(geom_idx)
        cached = self._geom_cache.get(geom_idx)
        if cached is not None:
            return cached

        geom = self._solver.geoms[geom_idx]
        cache = _GeomSDFCache(
            T_mesh_to_sdf=torch.as_tensor(
                geom.T_mesh_to_sdf,
                dtype=torch.float32,
                device=self._device,
            ),
            sdf_res=torch.as_tensor(
                geom.sdf_res,
                dtype=torch.float32,
                device=self._device,
            ),
            sdf_val=torch.as_tensor(
                geom.sdf_val,
                dtype=torch.float32,
                device=self._device,
            ),
            sdf_grad=torch.as_tensor(
                geom.sdf_grad,
                dtype=torch.float32,
                device=self._device,
            ),
            sdf_max=float(geom.sdf_max),
            sdf_cell_size=float(geom.sdf_cell_size),
            init_pos=torch.as_tensor(geom.init_pos, dtype=torch.float32, device=self._device),
            init_quat=torch.as_tensor(geom.init_quat, dtype=torch.float32, device=self._device),
        )
        self._geom_cache[geom_idx] = cache
        return cache

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        rotmat = quat_to_rotmat_wxyz_torch(quat)
        return torch.matmul(rotmat, vec.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
        out = quat.clone()
        out[..., 1:] *= -1.0
        return out

    def _point_velocities(
        self,
        link_linvel: torch.Tensor,
        link_angvel: torch.Tensor,
        link_quat: torch.Tensor,
        points_link_local: torch.Tensor,
    ) -> torch.Tensor:
        world_offsets = self._quat_apply(
            link_quat.unsqueeze(0).expand(points_link_local.shape[0], -1),
            points_link_local,
        )
        return torch.linalg.cross(
            link_angvel.unsqueeze(0).expand_as(world_offsets),
            world_offsets,
        ) + link_linvel.unsqueeze(0).expand_as(world_offsets)

    def _query_sdf_world(
        self,
        geom_idx: int,
        points_world: torch.Tensor,
        *,
        env_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geom = self._solver.geoms[int(geom_idx)]
        cache = self._geom_sdf(int(geom_idx))

        geom_pos_all = geom.get_pos()
        geom_quat_all = geom.get_quat()
        if self._single_env:
            geom_pos = geom_pos_all
            geom_quat = geom_quat_all
        else:
            geom_pos = geom_pos_all[env_index]
            geom_quat = geom_quat_all[env_index]

        points_mesh = gu.inv_transform_by_trans_quat(points_world, geom_pos, geom_quat)
        points_sdf = gu.transform_by_T(points_mesh, cache.T_mesh_to_sdf)

        sdf_res = cache.sdf_res
        outside = ((points_sdf >= (sdf_res - 1.0)) | (points_sdf <= 0.0)).any(dim=-1)

        signed_distance = torch.empty(points_sdf.shape[0], dtype=torch.float32, device=self._device)
        grad_sdf = torch.empty(points_sdf.shape[0], 3, dtype=torch.float32, device=self._device)

        if outside.any():
            center = (sdf_res - 1.0) * 0.5
            delta = points_sdf[outside] - center
            signed_distance[outside] = (
                torch.linalg.norm(delta, dim=-1) / float(cache.sdf_cell_size)
            ) + float(cache.sdf_max)
            grad_sdf[outside] = torch.nn.functional.normalize(delta, dim=-1)

        if (~outside).any():
            inside_points = points_sdf[~outside]
            base = torch.floor(inside_points).to(dtype=torch.long)
            max_base = (sdf_res.to(dtype=torch.long) - 2).view(1, 3)
            base = torch.minimum(torch.maximum(base, torch.zeros_like(base)), max_base)

            inside_sd = torch.zeros(inside_points.shape[0], dtype=torch.float32, device=self._device)
            inside_grad = torch.zeros(inside_points.shape[0], 3, dtype=torch.float32, device=self._device)

            for ox, oy, oz in product((0, 1), repeat=3):
                pos_cell = base + torch.tensor(
                    [ox, oy, oz],
                    dtype=torch.long,
                    device=self._device,
                )
                pos_cell_f = pos_cell.to(dtype=torch.float32)
                w_xyz = 1.0 - torch.abs(inside_points - pos_cell_f)
                w = w_xyz[:, 0] * w_xyz[:, 1] * w_xyz[:, 2]
                inside_sd += w * cache.sdf_val[
                    pos_cell[:, 0],
                    pos_cell[:, 1],
                    pos_cell[:, 2],
                ]
                inside_grad += w.unsqueeze(-1) * cache.sdf_grad[
                    pos_cell[:, 0],
                    pos_cell[:, 1],
                    pos_cell[:, 2],
                ]

            signed_distance[~outside] = inside_sd
            grad_sdf[~outside] = inside_grad

        normal_mesh = torch.nn.functional.normalize(grad_sdf, dim=-1)
        normal_world = self._quat_apply(
            geom_quat.unsqueeze(0).expand(points_world.shape[0], -1),
            normal_mesh,
        )
        closest_mesh = points_mesh - signed_distance.unsqueeze(-1) * normal_mesh
        closest_link_local = gu.transform_by_trans_quat(
            closest_mesh,
            cache.init_pos,
            cache.init_quat,
        )

        return signed_distance, normal_mesh, normal_world, closest_mesh, closest_link_local

    def _external_contact_geom_indices(
        self,
        contacts: dict[str, torch.Tensor],
        pad_link_idx_global: int,
        *,
        env_index: int,
    ) -> list[int]:
        if self._single_env:
            link_a = contacts["link_a"]
            link_b = contacts["link_b"]
            geom_a = contacts["geom_a"]
            geom_b = contacts["geom_b"]
            valid_mask = torch.ones_like(link_a, dtype=torch.bool)
        else:
            link_a = contacts["link_a"][env_index]
            link_b = contacts["link_b"][env_index]
            geom_a = contacts["geom_a"][env_index]
            geom_b = contacts["geom_b"][env_index]
            valid_mask = contacts["valid_mask"][env_index]

        pad_is_a = valid_mask & (link_a == int(pad_link_idx_global))
        pad_is_b = valid_mask & (link_b == int(pad_link_idx_global))
        other_geom = torch.cat([geom_b[pad_is_a], geom_a[pad_is_b]], dim=0)

        if other_geom.numel() == 0:
            return []

        external = (other_geom < self._own_geom_start) | (other_geom >= self._own_geom_end)
        other_geom = other_geom[external]
        if other_geom.numel() == 0:
            return []

        return [int(v) for v in torch.unique(other_geom).tolist()]

    def _nearby_external_geom_indices(
        self,
        pad: _PadSpec,
        *,
        env_index: int,
    ) -> list[int]:
        link_pos_all = pad.link.get_pos()
        link_quat_all = pad.link.get_quat()
        if self._single_env:
            pad_link_pos = link_pos_all
            pad_link_quat = link_quat_all
        else:
            pad_link_pos = link_pos_all[env_index]
            pad_link_quat = link_quat_all[env_index]

        pad_center_world = gu.transform_by_trans_quat(
            pad.center_local.unsqueeze(0),
            pad_link_pos,
            pad_link_quat,
        ).squeeze(0)

        nearby: list[int] = []
        max_dist_pad = pad.radius + self._PROXIMITY_MARGIN
        for geom_spec in self._external_geoms:
            geom_link_pos_all = geom_spec.link.get_pos()
            geom_link_quat_all = geom_spec.link.get_quat()
            if self._single_env:
                geom_link_pos = geom_link_pos_all
                geom_link_quat = geom_link_quat_all
            else:
                geom_link_pos = geom_link_pos_all[env_index]
                geom_link_quat = geom_link_quat_all[env_index]

            geom_center_world = gu.transform_by_trans_quat(
                geom_spec.center_local.unsqueeze(0),
                geom_link_pos,
                geom_link_quat,
            ).squeeze(0)
            center_dist = torch.linalg.norm(geom_center_world - pad_center_world)
            if float(center_dist.item()) <= (max_dist_pad + geom_spec.radius):
                nearby.append(geom_spec.geom_idx)

        return nearby

    def _normalize_tactile_points(self, points_raw: torch.Tensor) -> torch.Tensor:
        points = points_raw.clone()
        normal_force = points[:, 3].clamp(min=0.0)
        max_value = normal_force.max()
        if float(max_value.item()) < self._NORMALIZE_SWITCH:
            normal_force = normal_force / self._NORMALIZE_FALLBACK
        else:
            normal_force = normal_force / max_value.clamp(min=1e-9)
        points[:, 3] = normal_force
        return points

    def _select_env_tensor(
        self,
        value: torch.Tensor,
        env_index: int,
    ) -> torch.Tensor:
        return value if self._single_env else value[env_index]

    def _pad_kinematics(self, pad: _PadSpec, *, env_index: int) -> _PadKinematics:
        link = pad.link
        net_force_all = self.entity.get_links_net_contact_force()

        if self._single_env:
            pad_force = net_force_all[pad.link_idx_local]
        else:
            pad_force = net_force_all[env_index, pad.link_idx_local]

        return _PadKinematics(
            pos=self._select_env_tensor(link.get_pos(), env_index),
            quat=self._select_env_tensor(link.get_quat(), env_index),
            linvel=self._select_env_tensor(link.get_vel(), env_index),
            angvel=self._select_env_tensor(link.get_ang(), env_index),
            net_force=pad_force,
        )

    def _empty_closest_surface_state(self, num_points: int) -> _ClosestSurfaceState:
        normal_world = torch.zeros(
            num_points,
            3,
            dtype=torch.float32,
            device=self._device,
        )
        return _ClosestSurfaceState(
            signed_distance=torch.full(
                (num_points,),
                float("inf"),
                dtype=torch.float32,
                device=self._device,
            ),
            normal_world=normal_world,
            indenter_linvel=torch.zeros_like(normal_world),
            indenter_angvel=torch.zeros_like(normal_world),
            closest_link_local=torch.zeros_like(normal_world),
            indenter_quat=torch.zeros(
                num_points,
                4,
                dtype=torch.float32,
                device=self._device,
            ),
        )

    def _update_closest_surface_state(
        self,
        state: _ClosestSurfaceState,
        *,
        geom_idx: int,
        points_world: torch.Tensor,
        env_index: int,
    ) -> None:
        geom = self._solver.geoms[int(geom_idx)]
        sd, _, normal_world, _, closest_link_local = self._query_sdf_world(
            geom_idx,
            points_world,
            env_index=env_index,
        )
        update = sd < state.signed_distance
        if not bool(update.any().item()):
            return

        indenter_link = geom.link
        indenter_linvel = self._select_env_tensor(indenter_link.get_vel(), env_index)
        indenter_angvel = self._select_env_tensor(indenter_link.get_ang(), env_index)
        indenter_quat = self._select_env_tensor(indenter_link.get_quat(), env_index)

        n_update = int(update.sum().item())
        state.signed_distance[update] = sd[update]
        state.normal_world[update] = normal_world[update]
        state.closest_link_local[update] = closest_link_local[update]
        state.indenter_linvel[update] = indenter_linvel.unsqueeze(0).expand(
            n_update,
            -1,
        )
        state.indenter_angvel[update] = indenter_angvel.unsqueeze(0).expand(
            n_update,
            -1,
        )
        state.indenter_quat[update] = indenter_quat.unsqueeze(0).expand(n_update, -1)

    def _normal_force_from_contact(
        self,
        pad: _PadSpec,
        kin: _PadKinematics,
        state: _ClosestSurfaceState,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        tactile_points_linvel = self._point_velocities(
            kin.linvel,
            kin.angvel,
            kin.quat,
            pad.points_local,
        )
        indenter_offsets_world = self._quat_apply(
            state.indenter_quat,
            state.closest_link_local,
        )
        closest_points_linvel = torch.linalg.cross(
            state.indenter_angvel,
            indenter_offsets_world,
        ) + state.indenter_linvel

        relative_linvel = tactile_points_linvel - closest_points_linvel
        relative_vt = relative_linvel - state.normal_world * (
            state.normal_world * relative_linvel
        ).sum(dim=-1, keepdim=True)

        normal_force_norm = self._TACTILE_KN * depth
        normal_force_world = normal_force_norm.unsqueeze(-1) * state.normal_world

        relative_vt_norm = relative_vt.norm(dim=-1)
        friction_static_norm = self._TACTILE_KT * relative_vt_norm
        friction_dynamic_norm = self._TACTILE_MU * normal_force_norm
        friction_norm = torch.minimum(friction_static_norm, friction_dynamic_norm)
        friction_world = -friction_norm.unsqueeze(-1) * (
            relative_vt / relative_vt_norm.clamp(min=1e-9).unsqueeze(-1)
        )

        tactile_force = normal_force_world + friction_world
        tactile_force_local = self._quat_apply(
            self._quat_conjugate(
                kin.quat.unsqueeze(0).expand(pad.points_local.shape[0], -1),
            ),
            tactile_force,
        )
        return (tactile_force_local * pad.normal_axis_local.unsqueeze(0)).sum(
            dim=-1,
            keepdim=True,
        )

    def _read_pad(
        self,
        pad: _PadSpec,
        contacts: dict[str, torch.Tensor],
        *,
        env_index: int,
    ) -> dict[str, torch.Tensor]:
        kin = self._pad_kinematics(pad, env_index=env_index)
        points_world = gu.transform_by_trans_quat(pad.points_local, kin.pos, kin.quat)
        candidate_geoms = self._external_contact_geom_indices(
            contacts,
            int(pad.link.idx),
            env_index=env_index,
        )
        if not candidate_geoms:
            candidate_geoms = self._nearby_external_geom_indices(
                pad,
                env_index=env_index,
            )

        normal_force = torch.zeros(
            pad.points_local.shape[0],
            1,
            dtype=torch.float32,
            device=self._device,
        )
        state = self._empty_closest_surface_state(pad.points_local.shape[0])

        for geom_idx in candidate_geoms:
            self._update_closest_surface_state(
                state,
                geom_idx=geom_idx,
                points_world=points_world,
                env_index=env_index,
            )

        depth = (self._SOFT_CONTACT_MARGIN - state.signed_distance).clamp(min=0.0)
        if bool((depth > 0.0).any().item()):
            normal_force = self._normal_force_from_contact(pad, kin, state, depth)

        points_raw = torch.cat((points_world, normal_force.clamp(min=0.0)), dim=-1)
        points_norm = self._normalize_tactile_points(points_raw)
        grid_raw = points_raw[:, 3].reshape(pad.rows, pad.cols)
        grid_norm = points_norm[:, 3].reshape(pad.rows, pad.cols)

        return {
            "contact": (grid_raw.max().view(1) > 0.0).to(dtype=torch.float32),
            "force": kin.net_force,
            "force_norm": torch.linalg.norm(kin.net_force, dim=-1, keepdim=True),
            "pos": kin.pos,
            "quat": kin.quat,
            "points_raw": points_raw,
            "points": points_norm,
            "force_grid": grid_raw,
            "force_grid_norm": grid_norm,
        }

    def _stack_env_outputs(
        self,
        per_env: list[dict[str, dict[str, torch.Tensor]]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        out: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
        pad_keys = self.link_names
        field_keys = per_env[0][pad_keys[0]].keys()

        tactile_points = []
        tactile_points_raw = []
        for pad_name in pad_keys:
            out[pad_name] = {}
            for field in field_keys:
                stacked = torch.stack([env_out[pad_name][field] for env_out in per_env], dim=0)
                out[pad_name][field] = stacked
            tactile_points.append(out[pad_name]["points"])
            tactile_points_raw.append(out[pad_name]["points_raw"])

        out["tactile_points"] = torch.cat(tactile_points, dim=1)
        out["tactile_points_raw"] = torch.cat(tactile_points_raw, dim=1)
        return out

    def read(self) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Return EasySim-style tactile observations plus per-pad summaries."""
        contacts = self.entity.get_contacts(exclude_self_contact=True)

        if self._single_env:
            out: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {}
            tactile_points = []
            tactile_points_raw = []
            for pad in self._pads:
                pad_obs = self._read_pad(pad, contacts, env_index=0)
                out[pad.name] = pad_obs
                tactile_points.append(pad_obs["points"])
                tactile_points_raw.append(pad_obs["points_raw"])
            out["tactile_points"] = torch.cat(tactile_points, dim=0)
            out["tactile_points_raw"] = torch.cat(tactile_points_raw, dim=0)
            return out

        per_env = []
        for env_index in range(int(self._solver.n_envs)):
            env_out = {
                pad.name: self._read_pad(pad, contacts, env_index=env_index)
                for pad in self._pads
            }
            per_env.append(env_out)
        return self._stack_env_outputs(per_env)

    def read_summary(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return compact per-pad contact and force summaries.

        This avoids the full tactile grid/SDF path and is intended for
        low-dimensional RL policies that only consume contact flags and net
        contact-force magnitudes.
        """
        contacts = self.entity.get_contacts(exclude_self_contact=True)
        net_force_all = self.entity.get_links_net_contact_force()
        out: dict[str, dict[str, torch.Tensor]] = {}

        for pad in self._pads:
            if self._single_env:
                link_a = contacts["link_a"]
                link_b = contacts["link_b"]
                valid_mask = torch.ones_like(link_a, dtype=torch.bool)
                contact = (
                    valid_mask
                    & ((link_a == int(pad.link.idx)) | (link_b == int(pad.link.idx)))
                ).any()
                force = net_force_all[pad.link_idx_local]
                contact = contact.reshape(1).to(dtype=torch.float32)
            else:
                link_a = contacts["link_a"]
                link_b = contacts["link_b"]
                valid_mask = contacts["valid_mask"]
                contact = (
                    valid_mask
                    & ((link_a == int(pad.link.idx)) | (link_b == int(pad.link.idx)))
                ).any(dim=-1, keepdim=True)
                force = net_force_all[:, pad.link_idx_local]
                contact = contact.to(dtype=torch.float32)

            out[pad.name] = {
                "contact": contact,
                "force": force,
                "force_norm": torch.linalg.norm(force, dim=-1, keepdim=True),
                "pos": pad.link.get_pos(),
                "quat": pad.link.get_quat(),
            }

        return out
