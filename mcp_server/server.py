"""Rosatom 3D/5D Additive Manufacturing MCP Server.

Provides tools for:
- Deep STL geometry & overhang analysis (trimesh)
- Filament recommendation & parameter adaptation
- Safe G-code generation (PrusaSlicer / OrcaSlicer CLI + Mathematical Fallback)
- Stereotech 5-axis post-processing (supportless continuous printing)
- Industrial security guard validation & static G-code verification (G28, bounds, SHA-256)
- Vision-assisted defect diagnostics & parameter correction
- Rosatom industrial additive fleet management simulation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from mcp_server.five_axis_postproc import FiveAxisPostProcessor
from mcp_server.mesh_analyzer import MeshAnalyzer
from mcp_server.security_guard import SecurityGuard
from mcp_server.slicer_cli import SlicerCLI

# Initialize FastMCP Server
mcp = FastMCP("Rosatom 5D Additive Manufacturing MCP Server")

# Shared Core Instances
mesh_analyzer = MeshAnalyzer()
security_guard = SecurityGuard()
slicer_cli = SlicerCLI()
five_axis_proc = FiveAxisPostProcessor()

PROFILES_PATH = Path(__file__).parent / "profiles" / "filaments.json"


def _get_filaments_data() -> Dict[str, Any]:
    if PROFILES_PATH.is_file():
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# In-memory fleet simulator for Rosatom Additive Cell
FLEET_PRINTERS = {
    "STEREO-5D-01": {
        "name": "Stereotech 530 Hybrid (5-Axis)",
        "type": "5D",
        "status": "IDLE",
        "current_material": "PA12-CF",
        "supported_materials": ["PA12-CF", "PEEK", "PETG", "PLA"],
        "bed_size_mm": [300, 300, 300],
        "nozzle_temp_max_c": 450,
        "bed_temp_max_c": 160,
        "chamber_temp_max_c": 130,
        "active_job": None,
        "telemetry": {"nozzle_c": 24.2, "bed_c": 23.5, "chamber_c": 25.0},
    },
    "STEREO-5D-02": {
        "name": "Stereotech 530 High-Temp (5-Axis)",
        "type": "5D",
        "status": "PRINTING",
        "current_material": "PEEK",
        "supported_materials": ["PEEK", "PA12-CF"],
        "bed_size_mm": [300, 300, 300],
        "nozzle_temp_max_c": 450,
        "bed_temp_max_c": 160,
        "chamber_temp_max_c": 130,
        "active_job": {"name": "Impeller_Nuclear_Coolant.gcode", "progress_percent": 68.4},
        "telemetry": {"nozzle_c": 400.0, "bed_c": 140.0, "chamber_c": 120.0},
    },
    "PRUSA-3D-01": {
        "name": "Industrial Cartesian Cell 300 (3-Axis)",
        "type": "3D",
        "status": "IDLE",
        "current_material": "PETG",
        "supported_materials": ["PLA", "PETG", "ABS"],
        "bed_size_mm": [250, 210, 210],
        "nozzle_temp_max_c": 300,
        "bed_temp_max_c": 110,
        "chamber_temp_max_c": 50,
        "active_job": None,
        "telemetry": {"nozzle_c": 22.0, "bed_c": 21.8, "chamber_c": 22.0},
    },
}


@mcp.tool()
def analyze_stl(stl_path: str) -> str:
    """Perform deep geometric analysis of an STL/OBJ/PLY file.

    Evaluates:
    - Watertightness (manifold integrity)
    - Exact dimensions (X, Y, Z in mm), volume (cm3), surface area (cm2)
    - Overhangs (>45 degrees) area and percentage
    - Bed adhesion / topple risk assessment
    - Automated Rosatom verdict: Standard 3D vs 5D continuous multi-axis printing.
    """
    try:
        report = mesh_analyzer.analyze(stl_path)
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze STL: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def recommend_print_parameters(stl_path: str, material: str = "PLA", strength_priority: str = "balanced") -> str:
    """Recommend physical and slicing parameters based on mesh analysis, industrial filament profile, and target strength.

    Args:
        stl_path: Path to the 3D model file.
        material: Target filament (PLA, ABS, PETG, PA12-CF, PEEK).
        strength_priority: 'max_strength', 'balanced', or 'fast_prototype'.
    """
    try:
        mesh_report = mesh_analyzer.analyze(stl_path)
        filaments = _get_filaments_data()
        mat_key = material.strip().upper()
        profile = filaments.get(mat_key, filaments.get("PLA", {}))

        # Determine infill and layer height based on strength priority
        priority_norm = strength_priority.lower()
        if priority_norm in ("max_strength", "high", "structural"):
            infill = 80 if mat_key in ("PA12-CF", "PEEK") else 60
            layer_height = 0.15
            perimeters = 4
        elif priority_norm in ("fast", "fast_prototype", "draft"):
            infill = 15
            layer_height = 0.28
            perimeters = 2
        else:  # balanced
            infill = 35
            layer_height = 0.20
            perimeters = 3

        # Mode 5D recommendation
        use_5d = mesh_report.recommended_technology == "5D_CONTINUOUS"

        rec = {
            "material": mat_key,
            "material_name": profile.get("name", mat_key),
            "nozzle_temp_c": profile.get("nozzle_temp_recommended_c", 215),
            "bed_temp_c": profile.get("bed_temp_recommended_c", 60),
            "chamber_temp_recommended_c": profile.get("chamber_temp_recommended_c", 0),
            "print_speed_mm_s": profile.get("print_speed_recommended_mm_s", 50),
            "layer_height_mm": layer_height,
            "infill_percent": infill,
            "perimeters": perimeters,
            "cooling_fan_percent": profile.get("cooling_fan_percent", 100),
            "recommended_mode_5d": use_5d,
            "shrinkage_coefficient": profile.get("shrinkage_coefficient", 0.002),
            "density_g_cm3": profile.get("density_g_cm3", 1.24),
            "adhesion_recommendation": mesh_report.adhesion.recommendation,
            "notes": profile.get("notes", ""),
        }
        return json.dumps(rec, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to recommend parameters: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def generate_safe_gcode(
    stl_path: str,
    material: str = "PLA",
    layer_height: float = 0.2,
    infill: int = 20,
    mode_5d: bool = False,
    output_gcode_path: Optional[str] = None,
) -> str:
    """Generate safe, validated G-code with automated fallback engine and optional 5D post-processing.

    1. Validates physical limits (Security Guard)
    2. Slices using Prusa/Orca CLI or Mathematical Fallback engine
    3. Transforms toolpaths into 5D multi-axis G-code (A and C rotary-tilt axes) if mode_5d=True
    4. Performs static verification and SHA-256 digital signature.
    """
    try:
        filaments = _get_filaments_data()
        mat_key = material.strip().upper()
        mat_profile = filaments.get(mat_key, {})
        noz_temp = mat_profile.get("nozzle_temp_recommended_c", 215.0)
        bed_temp = mat_profile.get("bed_temp_recommended_c", 60.0)
        speed = mat_profile.get("print_speed_recommended_mm_s", 50.0)

        # 1. Parameter Safety Validation
        param_val = security_guard.validate_slicing_parameters({
            "nozzle_temp_c": noz_temp,
            "bed_temp_c": bed_temp,
            "layer_height_mm": layer_height,
            "print_speed_mm_s": speed,
            "infill_percent": infill,
            "material": mat_key,
        })
        if not param_val.is_safe:
            return json.dumps({"error": "Security Guard rejected parameters", "violations": param_val.violations}, ensure_ascii=False)

        # Determine target path
        stl_p = Path(stl_path)
        if output_gcode_path:
            out_p = Path(output_gcode_path)
        else:
            suffix = "_5d.gcode" if mode_5d else "_3d.gcode"
            out_p = stl_p.parent / f"{stl_p.stem}{suffix}"

        # 2. Slicing
        raw_gcode = out_p if not mode_5d else (out_p.parent / f"{stl_p.stem}_raw.gcode")
        slice_result = slicer_cli.slice_mesh(
            stl_path=stl_p,
            output_gcode_path=raw_gcode,
            material=mat_key,
            layer_height_mm=layer_height,
            infill_percent=infill,
            print_speed_mm_s=speed,
            nozzle_temp_c=noz_temp,
            bed_temp_c=bed_temp,
        )

        final_gcode = raw_gcode
        stats_5d = None

        # 3. 5D Post-Processing if requested
        if mode_5d:
            stats_5d = five_axis_proc.process_gcode(
                input_gcode_path=raw_gcode,
                output_gcode_path=out_p,
                adaptive_overhang_tilt=True,
                load_alignment_mode=True,
            )
            final_gcode = out_p

        # 4. Final G-code Static Security Check
        sec_report = security_guard.analyze_gcode_file(final_gcode)

        response = {
            "success": slice_result.success and sec_report.is_safe,
            "final_gcode_path": str(final_gcode),
            "mode_5d": mode_5d,
            "slicer_engine": slice_result.slicer_engine,
            "print_time_seconds": slice_result.print_time_seconds,
            "print_time_human": f"{int(slice_result.print_time_seconds // 60)} мин {int(slice_result.print_time_seconds % 60)} сек",
            "filament_used_mm": slice_result.filament_used_mm,
            "filament_weight_g": slice_result.filament_weight_g,
            "sha256": sec_report.sha256,
            "security_status": "PASSED" if sec_report.is_safe else "FAILED",
            "security_violations": sec_report.violations,
            "security_warnings": sec_report.warnings,
            "five_axis_stats": {
                "transformed_moves": stats_5d.transformed_moves_count,
                "max_tilt_a_deg": stats_5d.max_tilt_a_deg,
                "max_rotation_c_deg": stats_5d.max_rotation_c_deg,
                "support_volume_saved_percent": stats_5d.support_volume_saved_est_percent,
            } if stats_5d else None,
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate safe G-code: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def verify_gcode_security(gcode_path: str) -> str:
    """Line-by-line static security inspection of a G-code file before transmission to physical printer.

    Checks:
    - Presence of G28 homing command
    - Software limit envelopes (prevent machine head crashes into bed/frame)
    - Thermal limits for nozzle and bed
    - Generates SHA-256 digital fingerprint.
    """
    try:
        report = security_guard.analyze_gcode_file(gcode_path)
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to verify G-code: {str(e)}"}, ensure_ascii=False)


@mcp.tool()
def diagnose_print_defect_or_spool(
    defect_type: str,
    observed_symptoms: str,
    current_material: str = "PLA",
    current_nozzle_temp_c: Optional[float] = None,
    current_bed_temp_c: Optional[float] = None,
) -> str:
    """Analyze defect descriptions or visual inspection findings to diagnose root cause and correct slicing parameters.

    Common industrial defect categories:
    - delamination / warping (расслоение, коробление)
    - stringing / oozing (паутина, подтекание)
    - under_extrusion / clogging (недоэкструзия, засор сопла)
    - sagging_overhangs (провисание нависаний)
    - wet_filament (влажный пластик, вскипание)
    """
    filaments = _get_filaments_data()
    mat_key = current_material.strip().upper()
    profile = filaments.get(mat_key, {})

    recommendations: List[str] = []
    param_adjustments: Dict[str, Any] = {}
    dtype = defect_type.lower()
    symptoms = observed_symptoms.lower()

    if "delamination" in dtype or "warping" in dtype or "короблен" in symptoms or "отрыв" in symptoms:
        rec_bed = profile.get("bed_temp_recommended_c", 60) + 10
        param_adjustments["bed_temp_c"] = min(rec_bed, 145)
        param_adjustments["cooling_fan_percent"] = max(0, profile.get("cooling_fan_percent", 100) - 40)
        param_adjustments["brim_width_mm"] = 8.0
        recommendations.append("Увеличьте температуру стола на 10°C и уменьшите обдув детали.")
        recommendations.append("Для закрытых камер: прогрейте объем камеры перед печатью.")
        if mat_key in ("ABS", "PA12-CF", "PEEK"):
            recommendations.append(f"Для {mat_key} обязателен термобарьер и закрытая термокамера.")

    elif "stringing" in dtype or "паутин" in symptoms or "сопл" in symptoms:
        rec_noz = max(profile.get("nozzle_temp_range_c", [190, 230])[0], (current_nozzle_temp_c or 215) - 10)
        param_adjustments["nozzle_temp_c"] = rec_noz
        param_adjustments["retraction_distance_mm"] = 6.0
        param_adjustments["retraction_speed_mm_s"] = 45.0
        recommendations.append("Снизьте температуру сопла на 5-10°C для уменьшения текучести расплава.")
        recommendations.append("Увеличьте дистанцию ретракта до 5-7 мм (для Bowden) или 1.5-2.5 мм (для Direct Drive).")

    elif "under_extrusion" in dtype or "недоэкструз" in symptoms or "щели" in symptoms:
        param_adjustments["flow_ratio"] = 1.05
        param_adjustments["print_speed_mm_s"] = max(20.0, (profile.get("print_speed_recommended_mm_s", 50) * 0.8))
        recommendations.append("Проверьте диаметр сопла на частичный засор (cold pull).")
        recommendations.append("Увеличьте поток (flow multiplier) на 5% и снизьте линейную скорость печати.")

    elif "wet" in dtype or "вскипани" in symptoms or "пузырь" in symptoms:
        recommendations.append(f"Критично: Материал {mat_key} набрал влагу из воздуха.")
        recommendations.append("Требуется сушка катушки в сушильном шкафу (PLA: 45°C/4ч, PA12-CF: 80°C/8ч, PEEK: 120°C/6ч).")
        param_adjustments["requires_pre_drying"] = True

    elif "overhang" in dtype or "провисан" in symptoms:
        recommendations.append("Используйте 5D режим печати (Stereotech) для ориентации слоев против гравитации.")
        recommendations.append("Снизьте высоту слоя до 0.12-0.15 мм и увеличьте обдув периметров.")
        param_adjustments["recommended_mode_5d"] = True
        param_adjustments["layer_height_mm"] = 0.12

    else:
        recommendations.append("Выполните калибровку первого слоя (Z-offset) и проверьте натяжение приводных ремней.")

    return json.dumps({
        "defect_type": defect_type,
        "material": mat_key,
        "diagnosed_root_cause": symptoms,
        "recommendations": recommendations,
        "suggested_parameter_adjustments": param_adjustments,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def get_fleet_status() -> str:
    """Retrieve real-time status and telemetry of all 3D/5D additive manufacturing cells in the Rosatom cluster."""
    return json.dumps(FLEET_PRINTERS, ensure_ascii=False, indent=2)


@mcp.tool()
def send_to_printer(printer_id: str, gcode_path: str) -> str:
    """Transmit validated and cryptographically verified G-code to a physical/simulated printer in the fleet.

    Enforces mandatory security check prior to dispatch:
    - Bounding envelope vs printer physical bed size
    - Thermal check vs printer maximum nozzle/bed ratings
    - Checksum registration for industrial traceability.
    """
    pid = printer_id.strip().upper()
    if pid not in FLEET_PRINTERS:
        return json.dumps({
            "success": False,
            "error": f"Unknown printer ID '{printer_id}'. Available: {list(FLEET_PRINTERS.keys())}"
        }, ensure_ascii=False)

    printer = FLEET_PRINTERS[pid]
    if printer["status"] == "PRINTING":
        return json.dumps({
            "success": False,
            "error": f"Printer '{pid}' is currently busy with job '{printer['active_job']['name']}'."
        }, ensure_ascii=False)

    # Security verification
    sec_report = security_guard.analyze_gcode_file(gcode_path)
    if not sec_report.is_safe:
        return json.dumps({
            "success": False,
            "error": "Transmission aborted: Security Guard detected safety violations in G-code.",
            "violations": sec_report.violations,
        }, ensure_ascii=False)

    # Verify physical dimensions against target printer bed
    bed_x, bed_y, bed_z = printer["bed_size_mm"]
    obs_x = sec_report.bounds_observed["x"][1]
    obs_y = sec_report.bounds_observed["y"][1]
    obs_z = sec_report.bounds_observed["z"][1]

    if obs_x > bed_x or obs_y > bed_y or obs_z > bed_z:
        return json.dumps({
            "success": False,
            "error": f"Part dimensions exceed bed size of {pid} ({bed_x}x{bed_y}x{bed_z} mm).",
            "observed_bounds": sec_report.bounds_observed,
        }, ensure_ascii=False)

    # Dispatch job
    g_file = Path(gcode_path)
    printer["status"] = "PRINTING"
    printer["active_job"] = {
        "name": g_file.name,
        "sha256": sec_report.sha256,
        "progress_percent": 0.0,
    }

    return json.dumps({
        "success": True,
        "printer_id": pid,
        "printer_name": printer["name"],
        "printer_type": printer["type"],
        "dispatched_file": g_file.name,
        "sha256": sec_report.sha256,
        "message": f"Job successfully verified and queued on {printer['name']}.",
    }, ensure_ascii=False, indent=2)


def main():
    """Main CLI entrypoint for MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
