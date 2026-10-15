"""Mesh Analyzer Module for Additive Manufacturing (3D/5D FDM).

Performs deep geometric analysis of 3D meshes (STL/OBJ/PLY) using trimesh:
- Bounding box dimensions (X, Y, Z in mm)
- Volume (cm3), surface area (cm2), watertight / manifold status
- Overhang detection (angle > 45 deg relative to Z- build vector)
- First layer adhesion & aspect ratio / topple risk
- Automated printing technology verdict (Standard 3D + supports vs 5D multi-axis)
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import trimesh


@dataclass
class OverhangStats:
    threshold_deg: float
    overhang_area_cm2: float
    total_area_cm2: float
    overhang_percent: float
    face_count: int
    overhang_face_count: int


@dataclass
class AdhesionStats:
    contact_area_cm2: float
    height_mm: float
    aspect_ratio: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    recommendation: str


@dataclass
class MeshAnalysisResult:
    filename: str
    is_watertight: bool
    dimensions_mm: Dict[str, float]  # x, y, z
    volume_cm3: float
    surface_area_cm2: float
    overhangs: OverhangStats
    adhesion: AdhesionStats
    recommended_technology: str  # "3D_STANDARD", "3D_WITH_SUPPORTS", "5D_CONTINUOUS"
    verdict_reason: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MeshAnalyzer:
    """Industrial geometry analyzer for 3D/5D FDM printing."""

    def __init__(self, overhang_threshold_deg: float = 45.0, first_layer_tolerance_mm: float = 0.4):
        self.overhang_threshold_deg = overhang_threshold_deg
        self.first_layer_tolerance_mm = first_layer_tolerance_mm

    def load_mesh(self, mesh_path: str | Path) -> trimesh.Trimesh:
        path = Path(mesh_path)
        if not path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {path}")

        loaded = trimesh.load(str(path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise ValueError("Loaded scene contains no geometry.")
            # Merge scene geometries into a single mesh
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise TypeError(f"Unsupported geometry type: {type(loaded)}")

        # Clean up degenerate faces and duplicate vertices
        mesh.remove_infinite_values()
        mesh.remove_degenerate_faces()
        return mesh

    def calculate_overhangs(self, mesh: trimesh.Trimesh) -> OverhangStats:
        """Calculate faces that exceed overhang angle threshold.

        In FDM, 0 deg normal is horizontal, pointing straight down is -Z (180 deg).
        An overhang relative to build plate (-Z vector [0, 0, -1]):
        Angle between face normal and downward vector [0, 0, -1].
        If face points down (Z normal < 0):
        Angle with horizontal = 90 - arccos(|nz|).
        Overhang > 45 deg occurs when normal angle from vertical Z downward is < 45 deg
        (i.e., normal.z < -cos(radians(45)) = -0.7071).
        More precisely, angle against build plate vector [0, 0, -1]:
        cos_theta = -face_normals[:, 2]
        When cos_theta > cos(radians(threshold_deg)), the face points downward steeper
        than (90 - threshold) from horizontal.
        """
        normals = mesh.face_normals
        areas = mesh.area_faces
        total_area = float(np.sum(areas))

        if total_area <= 1e-9:
            return OverhangStats(
                threshold_deg=self.overhang_threshold_deg,
                overhang_area_cm2=0.0,
                total_area_cm2=0.0,
                overhang_percent=0.0,
                face_count=len(normals),
                overhang_face_count=0,
            )

        # Vector pointing straight down into the print bed
        down_vector = np.array([0.0, 0.0, -1.0])
        # Dot product with downward normal gives cos of angle with downward vector
        cos_down = np.dot(normals, down_vector)
        cos_down = np.clip(cos_down, -1.0, 1.0)
        angles_from_down_deg = np.degrees(np.arccos(cos_down))

        # Overhang threshold: overhang > 45 deg means angle from downward vector <= (90 - 45) = 45 deg
        # Exclude perfectly flat bottom faces if they lie directly on Z=0 (first layer contact)
        z_min = float(mesh.bounds[0][2])
        face_centers_z = mesh.triangles_center[:, 2]
        is_first_layer = face_centers_z <= (z_min + self.first_layer_tolerance_mm)
        is_flat_bottom = is_first_layer & (angles_from_down_deg < 5.0)

        # Overhang condition: pointing downwards and steeper than threshold, excluding first-layer base
        overhang_mask = (angles_from_down_deg <= (90.0 - self.overhang_threshold_deg)) & (~is_flat_bottom)

        overhang_area_mm2 = float(np.sum(areas[overhang_mask]))
        overhang_faces = int(np.sum(overhang_mask))

        overhang_area_cm2 = overhang_area_mm2 / 100.0
        total_area_cm2 = total_area / 100.0
        overhang_percent = (overhang_area_cm2 / total_area_cm2) * 100.0 if total_area_cm2 > 0 else 0.0

        return OverhangStats(
            threshold_deg=self.overhang_threshold_deg,
            overhang_area_cm2=round(overhang_area_cm2, 3),
            total_area_cm2=round(total_area_cm2, 3),
            overhang_percent=round(overhang_percent, 2),
            face_count=len(normals),
            overhang_face_count=overhang_faces,
        )

    def analyze_adhesion(self, mesh: trimesh.Trimesh) -> AdhesionStats:
        """Estimate contact area with print bed (Z=z_min) and topple risk.

        Aspect ratio = Height_Z / sqrt(Contact_Area)
        """
        bounds = mesh.bounds
        dx = float(bounds[1][0] - bounds[0][0])
        dy = float(bounds[1][1] - bounds[0][1])
        dz = float(bounds[1][2] - bounds[0][2])

        z_min = float(bounds[0][2])
        # Find triangles whose center is near z_min and normal points downward
        face_centers_z = mesh.triangles_center[:, 2]
        normals_z = mesh.face_normals[:, 2]
        areas = mesh.area_faces

        bed_mask = (face_centers_z <= (z_min + self.first_layer_tolerance_mm)) & (normals_z < -0.8)
        contact_area_mm2 = float(np.sum(areas[bed_mask]))

        # Fallback if bottom has complex chamfers/fillets: slice at first layer
        if contact_area_mm2 < 1.0:
            try:
                slice_2d = mesh.section(plane_origin=[0, 0, z_min + 0.1], plane_normal=[0, 0, 1])
                if slice_2d is not None:
                    contact_area_mm2 = float(slice_2d.area)
            except Exception:
                contact_area_mm2 = max(1.0, min(dx * dy * 0.05, 50.0))

        contact_area_cm2 = max(0.01, contact_area_mm2 / 100.0)
        characteristic_width = math.sqrt(contact_area_cm2 * 100.0)  # in mm
        aspect_ratio = round(dz / characteristic_width if characteristic_width > 0 else 99.0, 2)

        if aspect_ratio > 6.0 or contact_area_cm2 < 0.5:
            risk = "CRITICAL"
            rec = "Критический риск отрыва детали: используйте широкую кайму (Brim >= 8 мм) или переориентируйте деталь."
        elif aspect_ratio > 3.5:
            risk = "HIGH"
            rec = "Высокий риск опрокидывания: требуется Brim 5 мм и снижение скорости внешних контуров."
        elif aspect_ratio > 2.0:
            risk = "MEDIUM"
            rec = "Умеренный риск: рекомендуется кайма Brim 3 мм для предотвращения отлипания углов."
        else:
            risk = "LOW"
            rec = "Отличная адгезия к платформе: стандартный skirt достаточен."

        return AdhesionStats(
            contact_area_cm2=round(contact_area_cm2, 3),
            height_mm=round(dz, 2),
            aspect_ratio=aspect_ratio,
            risk_level=risk,
            recommendation=rec,
        )

    def analyze(self, mesh_path: str | Path) -> MeshAnalysisResult:
        """Run complete geometry analysis and produce Rosatom 3D/5D verdict."""
        path = Path(mesh_path)
        mesh = self.load_mesh(path)

        bounds = mesh.bounds
        dx = float(bounds[1][0] - bounds[0][0])
        dy = float(bounds[1][1] - bounds[0][1])
        dz = float(bounds[1][2] - bounds[0][2])

        is_watertight = bool(mesh.is_watertight)

        # Volume calculation (watertight mesh has accurate volume; non-watertight uses convex hull or approximate)
        if is_watertight:
            vol_cm3 = float(abs(mesh.volume) / 1000.0)
        else:
            vol_cm3 = float(abs(mesh.convex_hull.volume) * 0.75 / 1000.0)

        area_cm2 = float(mesh.area / 100.0)

        overhangs = self.calculate_overhangs(mesh)
        adhesion = self.analyze_adhesion(mesh)

        notes = []
        if not is_watertight:
            notes.append("Внимание: Сетка не является герметичной (not watertight). Возможны артефакты при слайсинге.")

        # Verdict logic for Rosatom 3D vs 5D printing:
        # 5D printing excels at:
        # 1. Overhangs > 10% where support removal damages surface finish or internal cavities
        # 2. Critical aspect ratios / complex geometry requiring rotation
        # 3. High-performance filaments (PEEK, PA12-CF) where supports waste expensive material
        if overhangs.overhang_percent > 12.0:
            recommended_tech = "5D_CONTINUOUS"
            verdict = (
                f"Рекомендуется 5D-печать (Stereotech 5-Axis): доля нависаний составляет {overhangs.overhang_percent}%. "
                "5-осевая кинематика позволяет печатать без поддержек за счет поворота/наклона стола (оси A/C), "
                "сохраняя анизотропную прочность вдоль траектории волокон."
            )
        elif overhangs.overhang_percent > 2.0:
            recommended_tech = "3D_WITH_SUPPORTS"
            verdict = (
                f"Допустима классическая 3D-печать с поддержками (нависания {overhangs.overhang_percent}%). "
                "При наличии 5-осевого станка рекомендуется 5D-режим для экономии материала и исключения постобработки."
            )
        else:
            recommended_tech = "3D_STANDARD"
            verdict = (
                f"Рекомендуется стандартная 3D-печать: деталь технологична, доля нависаний всего {overhangs.overhang_percent}%. "
                "Поддержки не требуются."
            )

        return MeshAnalysisResult(
            filename=path.name,
            is_watertight=is_watertight,
            dimensions_mm={"x": round(dx, 2), "y": round(dy, 2), "z": round(dz, 2)},
            volume_cm3=round(vol_cm3, 3),
            surface_area_cm2=round(area_cm2, 3),
            overhangs=overhangs,
            adhesion=adhesion,
            recommended_technology=recommended_tech,
            verdict_reason=verdict,
            notes=notes,
        )


def analyze_stl_file(file_path: str | Path) -> Dict[str, Any]:
    """Helper entry point for external modules and MCP tools."""
    analyzer = MeshAnalyzer()
    return analyzer.analyze(file_path).to_dict()
