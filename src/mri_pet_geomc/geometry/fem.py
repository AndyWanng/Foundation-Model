"""Deterministic coarse volumetric FEM and Laplace--Beltrami eigenbasis."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from torch import Tensor


_CUBE_TETRAHEDRA = np.asarray(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int64,
)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _geometry_hash(*arrays: np.ndarray, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_brain_mask(
    mask: str | Path | np.ndarray | Tensor,
    affine: np.ndarray | Tensor | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if isinstance(mask, (str, Path)):
        import nibabel as nib

        source = Path(mask).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        image = nib.load(str(source), mmap=True)
        value = np.asarray(image.dataobj) > 0
        transform = np.asarray(image.affine, dtype=np.float64)
        mask_source = str(source)
    else:
        value = np.asarray(torch.as_tensor(mask).cpu() if isinstance(mask, Tensor) else mask) > 0
        if affine is None:
            raise ValueError("affine is required when mask is an array/tensor.")
        transform = np.asarray(affine, dtype=np.float64)
        mask_source = "in_memory"
    if value.ndim != 3 or min(value.shape) < 3 or not value.any():
        raise ValueError("FEM reference mask must be a non-empty 3-D array.")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("FEM affine must be finite 4x4.")
    if abs(float(np.linalg.det(transform[:3, :3]))) <= 1.0e-12:
        raise ValueError("FEM affine spatial block is singular.")
    return value.astype(bool), transform, mask_source, _array_hash(value.astype(np.uint8))


def _axis(size: int, stride: int) -> np.ndarray:
    values = list(range(0, int(size), int(stride)))
    if values[-1] != size - 1:
        values.append(size - 1)
    return np.asarray(values, dtype=np.int64)


def _occupied_cells(
    mask: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
) -> list[tuple[int, int, int]]:
    occupied: list[tuple[int, int, int]] = []
    ax, ay, az = axes
    for i in range(len(ax) - 1):
        for j in range(len(ay) - 1):
            for k in range(len(az) - 1):
                block = mask[
                    ax[i] : ax[i + 1] + 1,
                    ay[j] : ay[j + 1] + 1,
                    az[k] : az[k + 1] + 1,
                ]
                if float(block.mean()) >= threshold:
                    occupied.append((i, j, k))
    return occupied


def _referenced_vertex_count(cells: list[tuple[int, int, int]]) -> int:
    vertices: set[tuple[int, int, int]] = set()
    for i, j, k in cells:
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    vertices.add((i + dx, j + dy, k + dz))
    return len(vertices)


def _choose_lattice(
    mask: np.ndarray,
    *,
    target_nodes: int,
    minimum_nodes: int,
    occupancy_threshold: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[tuple[int, int, int]], int]:
    estimate = max(1, int(round((float(mask.sum()) / max(target_nodes, 1)) ** (1.0 / 3.0))))
    maximum = max(1, min(mask.shape) - 1)
    candidates = sorted(
        set(range(max(1, estimate - 5), min(maximum, estimate + 6) + 1)) | {1, estimate}
    )
    best: tuple[float, int, tuple[np.ndarray, np.ndarray, np.ndarray], list[tuple[int, int, int]], int] | None = None
    for stride in candidates:
        axes = tuple(_axis(size, stride) for size in mask.shape)
        cells = _occupied_cells(mask, axes, occupancy_threshold)
        count = _referenced_vertex_count(cells)
        if not cells or count < minimum_nodes:
            continue
        score = abs(np.log(max(count, 1) / float(target_nodes)))
        candidate = (float(score), stride, axes, cells, count)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    if best is None:
        raise RuntimeError(
            f"Could not construct a connected-size candidate with at least {minimum_nodes} nodes."
        )
    return best[2], best[3], best[1]


def _world_coordinates(voxel: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (voxel.astype(np.float64), np.ones((len(voxel), 1), dtype=np.float64)), axis=1
    )
    return (homogeneous @ affine.T)[:, :3]


def _build_tetrahedral_mesh(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    cells: list[tuple[int, int, int]],
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_map: dict[tuple[int, int, int], int] = {}
    voxel_vertices: list[tuple[int, int, int]] = []
    tetrahedra: list[list[int]] = []
    ax, ay, az = axes
    for i, j, k in cells:
        cube_keys = [
            (i, j, k),
            (i + 1, j, k),
            (i, j + 1, k),
            (i + 1, j + 1, k),
            (i, j, k + 1),
            (i + 1, j, k + 1),
            (i, j + 1, k + 1),
            (i + 1, j + 1, k + 1),
        ]
        cube: list[int] = []
        for key in cube_keys:
            index = vertex_map.get(key)
            if index is None:
                index = len(voxel_vertices)
                vertex_map[key] = index
                voxel_vertices.append((int(ax[key[0]]), int(ay[key[1]]), int(az[key[2]])))
            cube.append(index)
        cube_array = np.asarray(cube, dtype=np.int64)
        tetrahedra.extend(cube_array[_CUBE_TETRAHEDRA].tolist())
    if not tetrahedra:
        raise RuntimeError("The occupied lattice produced no tetrahedra.")
    world = _world_coordinates(np.asarray(voxel_vertices, dtype=np.float64), affine)
    return world, np.asarray(tetrahedra, dtype=np.int64)


def _keep_largest_mesh_component(
    nodes: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    parent = np.arange(len(nodes), dtype=np.int64)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for tetra in tetrahedra:
        for vertex in tetra[1:]:
            union(int(tetra[0]), int(vertex))
    roots = np.asarray([find(index) for index in range(len(nodes))])
    unique, counts = np.unique(roots, return_counts=True)
    keep_root = int(unique[int(np.argmax(counts))])
    keep = roots == keep_root
    removed = int((~keep).sum())
    mapping = np.full(len(nodes), -1, dtype=np.int64)
    mapping[keep] = np.arange(int(keep.sum()))
    keep_tetra = np.all(keep[tetrahedra], axis=1)
    return nodes[keep], mapping[tetrahedra[keep_tetra]], removed


def _assemble_fem_matrices(
    nodes: np.ndarray, tetrahedra: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    row: list[int] = []
    column: list[int] = []
    data: list[float] = []
    mass = np.zeros(len(nodes), dtype=np.float64)
    volumes = np.zeros(len(tetrahedra), dtype=np.float64)
    for element_index, tetra in enumerate(tetrahedra):
        xyz = nodes[tetra]
        system = np.concatenate((np.ones((4, 1)), xyz), axis=1)
        determinant = float(np.linalg.det(system))
        volume = abs(determinant) / 6.0
        if not np.isfinite(volume) or volume <= 1.0e-10:
            raise RuntimeError(f"Degenerate FEM tetrahedron {element_index}: volume={volume}.")
        coefficients = np.linalg.inv(system)
        gradients = coefficients[1:, :].T
        local = volume * (gradients @ gradients.T)
        for local_i, global_i in enumerate(tetra):
            mass[global_i] += volume / 4.0
            for local_j, global_j in enumerate(tetra):
                row.append(int(global_i))
                column.append(int(global_j))
                data.append(float(local[local_i, local_j]))
        volumes[element_index] = volume
    stiffness = sparse.coo_matrix((data, (row, column)), shape=(len(nodes), len(nodes))).tocsr()
    stiffness.sum_duplicates()
    stiffness = 0.5 * (stiffness + stiffness.T)
    stiffness.eliminate_zeros()
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise RuntimeError("FEM assembly produced non-positive mass.")
    return stiffness.tocsr(), mass, volumes


def _solve_generalized_eigenproblem(
    stiffness: sparse.csr_matrix,
    mass: np.ndarray,
    modes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = len(mass)
    if not 1 <= int(modes) < n:
        raise ValueError(f"modes must satisfy 1 <= modes < nodes ({n}).")
    inv_sqrt = sparse.diags(1.0 / np.sqrt(mass))
    normalized = (inv_sqrt @ stiffness @ inv_sqrt).tocsr()
    if n <= 256:
        eigenvalues, transformed = np.linalg.eigh(normalized.toarray())
        eigenvalues, transformed = eigenvalues[:modes], transformed[:, :modes]
    else:
        # ARPACK otherwise chooses a random starting vector, which can rotate
        # retained near-degenerate modes and change the geometry hash across
        # identical launches.  A fixed, non-constant v0 makes the sparse path
        # reproducible before the existing sign canonicalization below.
        v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
        v0 /= np.linalg.norm(v0)
        try:
            eigenvalues, transformed = sparse_linalg.eigsh(
                normalized,
                k=modes,
                which="SM",
                v0=v0,
                tol=1.0e-9,
                maxiter=max(10000, 50 * n),
            )
        except sparse_linalg.ArpackNoConvergence:
            eigenvalues, transformed = sparse_linalg.eigsh(
                normalized,
                k=modes,
                sigma=-1.0e-9,
                which="LM",
                v0=v0,
                tol=1.0e-9,
                maxiter=max(10000, 50 * n),
            )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    transformed = np.asarray(transformed[:, order], dtype=np.float64)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1.0e-10
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0.0
    if np.min(eigenvalues) < -tolerance:
        raise RuntimeError(f"FEM spectrum contains a negative eigenvalue {np.min(eigenvalues)}.")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    basis = transformed / np.sqrt(mass)[:, None]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    gram = basis.T @ (mass[:, None] * basis)
    residuals = []
    for index, value in enumerate(eigenvalues):
        left = stiffness @ basis[:, index]
        right = value * mass * basis[:, index]
        scale = max(float(np.linalg.norm(left) + np.linalg.norm(right)), 1.0e-10)
        residuals.append(float(np.linalg.norm(left - right) / scale))
    checks = {
        "mass_orthogonality_max_abs": float(np.max(np.abs(gram - np.eye(modes)))),
        "generalized_residual_max_nonconstant": float(max(residuals[1:] or residuals)),
        "constant_mode_absolute_residual": float(
            np.linalg.norm(stiffness @ basis[:, 0] - eigenvalues[0] * mass * basis[:, 0])
        ),
    }
    return eigenvalues, basis, checks


@dataclass(frozen=True)
class FEMGeometry:
    node_xyz_mm: Tensor
    tetrahedra: Tensor
    mass: Tensor
    stiffness_indices: Tensor
    stiffness_values: Tensor
    eigenvalues: Tensor
    eigenvectors: Tensor
    metadata: dict[str, Any]

    @property
    def node_count(self) -> int:
        return int(self.mass.numel())

    @property
    def mode_count(self) -> int:
        return int(self.eigenvalues.numel())

    def stiffness_sparse(self) -> Tensor:
        return torch.sparse_coo_tensor(
            self.stiffness_indices,
            self.stiffness_values,
            (self.node_count, self.node_count),
        ).coalesce()

    def validate(self, *, tolerance: float = 2.0e-4) -> None:
        n, k = self.node_count, self.mode_count
        if self.node_xyz_mm.shape != (n, 3) or self.tetrahedra.ndim != 2 or self.tetrahedra.shape[1] != 4:
            raise ValueError("Invalid FEM node/tetrahedron shape.")
        if self.eigenvectors.shape != (n, k) or self.stiffness_indices.shape[0] != 2:
            raise ValueError("Invalid FEM spectral/stiffness shape.")
        if self.stiffness_values.shape != (self.stiffness_indices.shape[1],):
            raise ValueError("stiffness values do not match indices.")
        if torch.any(self.mass <= 0) or not torch.isfinite(self.mass).all():
            raise ValueError("FEM mass must be finite and positive.")
        if torch.any(self.tetrahedra < 0) or torch.any(self.tetrahedra >= n):
            raise ValueError("FEM tetrahedron references an invalid node.")
        if torch.any(torch.diff(self.eigenvalues) < -1.0e-7) or self.eigenvalues[0] < -1.0e-7:
            raise ValueError("FEM eigenvalues must be sorted and non-negative.")
        gram = self.eigenvectors.double().T @ (
            self.mass.double()[:, None] * self.eigenvectors.double()
        )
        orthogonality_error = float(
            (gram - torch.eye(k, dtype=torch.float64)).abs().max()
        )
        if orthogonality_error > tolerance:
            raise ValueError("FEM basis is not M-orthonormal.")
        basis = self.eigenvectors.double()
        stiffness = self.stiffness_sparse().double()
        left = torch.sparse.mm(stiffness, basis)
        right = self.mass.double()[:, None] * basis * self.eigenvalues.double()[None, :]
        denominator = (left.norm(dim=0) + right.norm(dim=0)).clamp_min(1.0e-10)
        residuals = (left - right).norm(dim=0) / denominator
        nonconstant = residuals[1:] if k > 1 else residuals
        residual_error = float(nonconstant.max())
        if not torch.isfinite(residuals).all() or residual_error > tolerance:
            raise ValueError(
                "FEM basis does not satisfy the generalized eigenproblem: "
                f"maximum nonconstant residual={residual_error:.6g}"
            )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "node_xyz_mm": self.node_xyz_mm.cpu(),
            "tetrahedra": self.tetrahedra.cpu(),
            "mass": self.mass.cpu(),
            "stiffness_indices": self.stiffness_indices.cpu(),
            "stiffness_values": self.stiffness_values.cpu(),
            "eigenvalues": self.eigenvalues.cpu(),
            "eigenvectors": self.eigenvectors.cpu(),
            "metadata": dict(self.metadata),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FEMGeometry":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise RuntimeError("Unsupported geometry bundle.")
        bundle = cls(
            payload["node_xyz_mm"],
            payload["tetrahedra"],
            payload["mass"],
            payload["stiffness_indices"],
            payload["stiffness_values"],
            payload["eigenvalues"],
            payload["eigenvectors"],
            dict(payload["metadata"]),
        )
        bundle.validate()
        return bundle



def build_fem_geometry(
    mask: str | Path | np.ndarray | Tensor,
    *,
    nodes: int = 1024,
    modes: int = 128,
    affine: np.ndarray | Tensor | None = None,
    occupancy_threshold: float = 0.20,
    output_path: str | Path | None = None,
) -> FEMGeometry:
    """Build and optionally save a deterministic approximate-size volumetric FEM.

    ``nodes`` is a target: a regular lattice stride is chosen to approach it;
    the exact connected node count is recorded in metadata.
    """

    if int(nodes) < 8 or int(modes) < 2 or int(modes) >= int(nodes):
        raise ValueError("Require nodes>=8 and 2<=modes<nodes.")
    if not 0.0 < float(occupancy_threshold) <= 1.0:
        raise ValueError("occupancy_threshold must lie in (0,1].")
    brain, transform, mask_source, mask_hash = _load_brain_mask(mask, affine)
    axes, cells, stride = _choose_lattice(
        brain,
        target_nodes=int(nodes),
        minimum_nodes=int(modes) + 1,
        occupancy_threshold=float(occupancy_threshold),
    )
    coordinates, tetrahedra = _build_tetrahedral_mesh(axes, cells, transform)
    coordinates, tetrahedra, removed = _keep_largest_mesh_component(coordinates, tetrahedra)
    if len(coordinates) <= modes:
        raise RuntimeError(
            f"Largest FEM component has {len(coordinates)} nodes, insufficient for {modes} modes."
        )
    stiffness, mass, volumes = _assemble_fem_matrices(coordinates, tetrahedra)
    eigenvalues, eigenvectors, checks = _solve_generalized_eigenproblem(stiffness, mass, int(modes))
    coo = stiffness.tocoo()
    indices = np.stack((coo.row, coo.col), axis=0).astype(np.int64)
    base_metadata: dict[str, Any] = {
        "schema_version": 1,
        "kind": "structured_tetrahedral_linear_fem",
        "mask_source": mask_source,
        "mask_sha256": mask_hash,
        "mask_shape": list(brain.shape),
        "affine": transform.tolist(),
        "brain_voxels": int(brain.sum()),
        "target_nodes": int(nodes),
        "actual_nodes": int(len(coordinates)),
        "tetrahedra": int(len(tetrahedra)),
        "modes": int(modes),
        "lattice_stride_voxels": int(stride),
        "occupancy_threshold": float(occupancy_threshold),
        "removed_disconnected_nodes": int(removed),
        "total_volume_mm3": float(volumes.sum()),
        "minimum_tetrahedron_volume_mm3": float(volumes.min()),
        **checks,
    }
    base_metadata["geometry_hash"] = _geometry_hash(
        coordinates,
        tetrahedra,
        mass,
        indices,
        coo.data,
        eigenvalues,
        eigenvectors,
        metadata=base_metadata,
    )
    bundle = FEMGeometry(
        torch.from_numpy(coordinates).double(),
        torch.from_numpy(tetrahedra).long(),
        torch.from_numpy(mass).double(),
        torch.from_numpy(indices).long(),
        torch.from_numpy(coo.data.astype(np.float64)).double(),
        torch.from_numpy(eigenvalues).double(),
        torch.from_numpy(eigenvectors).double(),
        base_metadata,
    )
    bundle.validate()
    if output_path is not None:
        bundle.save(output_path)
    return bundle
