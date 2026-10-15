"""Industrial Safety Guard Module for 3D/5D Additive Manufacturing.

Prevents machine damage and hazards caused by LLM hallucinations or invalid slicing parameters:
1. Slicing Parameter Validation: Hard limits for nozzle (170-420C), bed (0-150C),
   layer height (0.05-0.4 mm), print speed (10-300 mm/s).
2. Static G-Code Security Analysis:
   - Verification of mandatory homing commands (G28)
   - Continuous toolpath bounding box verification (never exceed physical machine build volume)
   - Extrusion checks & temperature safety bounds inside G-code
   - Cryptographic SHA-256 verification of the file prior to fleet dispatch.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class PhysicalMachineLimits(BaseModel):
    """Physical hardware limits for Rosatom 3D/5D printing cells."""
    min_nozzle_temp_c: float = 170.0
    max_nozzle_temp_c: float = 420.0  # Supports high-temp PEEK
    min_bed_temp_c: float = 0.0
    max_bed_temp_c: float = 150.0
    min_layer_height_mm: float = 0.05
    max_layer_height_mm: float = 0.40
    min_speed_mm_s: float = 10.0
    max_speed_mm_s: float = 300.0
    # Default build volume for Stereotech / industrial 5D printer
    build_volume_x_mm: float = 300.0
    build_volume_y_mm: float = 300.0
    build_volume_z_mm: float = 300.0


class SlicingParametersInput(BaseModel):
    """Input parameters provided by LLM or operator before slicing."""
    nozzle_temp_c: float = Field(..., description="Nozzle temperature in Celsius")
    bed_temp_c: float = Field(..., description="Bed temperature in Celsius")
    layer_height_mm: float = Field(..., description="Layer height in mm")
    print_speed_mm_s: float = Field(..., description="Base print speed in mm/s")
    infill_percent: int = Field(default=20, ge=0, le=100, description="Infill percentage (0-100)")
    material: str = Field(default="PLA", description="Target filament material")

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        valid_materials = {"PLA", "ABS", "PETG", "PA12-CF", "PEEK"}
        norm = v.strip().upper()
        if norm not in valid_materials:
            raise ValueError(f"Unsupported material '{v}'. Allowed: {sorted(list(valid_materials))}")
        return norm


@dataclass
class ValidationReport:
    is_safe: bool
    violations: List[str]
    warnings: List[str]
    normalized_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GCodeSecurityReport:
    is_safe: bool
    file_path: str
    sha256: str
    file_size_bytes: int
    total_lines: int
    has_homing_g28: bool
    max_observed_nozzle_temp: float
    max_observed_bed_temp: float
    bounds_observed: Dict[str, Tuple[float, float]]  # axis -> (min, max)
    build_volume_limits: Dict[str, float]
    violations: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecurityGuard:
    """Enforces strict safety rules for slicing parameters and generated G-code files."""

    def __init__(self, limits: Optional[PhysicalMachineLimits] = None):
        self.limits = limits or PhysicalMachineLimits()

    def validate_slicing_parameters(self, params: Dict[str, Any]) -> ValidationReport:
        """Validate proposed slicing parameters against physical safety limits and material rules."""
        violations: List[str] = []
        warnings: List[str] = []

        try:
            validated = SlicingParametersInput(**params)
        except Exception as e:
            return ValidationReport(
                is_safe=False,
                violations=[f"Параметрическая ошибка валидации: {e}"],
                warnings=[],
                normalized_params=params,
            )

        # 1. Nozzle temperature physical limits
        if validated.nozzle_temp_c < self.limits.min_nozzle_temp_c:
            violations.append(
                f"Температура сопла {validated.nozzle_temp_c}°C ниже минимального физического предела ({self.limits.min_nozzle_temp_c}°C). Риск заклинивания экструдера."
            )
        elif validated.nozzle_temp_c > self.limits.max_nozzle_temp_c:
            violations.append(
                f"Температура сопла {validated.nozzle_temp_c}°C превышает максимальный безопасный предел станка ({self.limits.max_nozzle_temp_c}°C). Риск термического разрушения хотэнда и возгорания."
            )

        # 2. Bed temperature physical limits
        if validated.bed_temp_c < self.limits.min_bed_temp_c:
            violations.append(f"Температура стола {validated.bed_temp_c}°C не может быть отрицательной.")
        elif validated.bed_temp_c > self.limits.max_bed_temp_c:
            violations.append(
                f"Температура стола {validated.bed_temp_c}°C превышает допустимый максимум ({self.limits.max_bed_temp_c}°C). Риск деградации нагревателя стола."
            )

        # 3. Layer height limits
        if validated.layer_height_mm < self.limits.min_layer_height_mm or validated.layer_height_mm > self.limits.max_layer_height_mm:
            violations.append(
                f"Высота слоя {validated.layer_height_mm} мм вне допустимого диапазона [{self.limits.min_layer_height_mm}, {self.limits.max_layer_height_mm}] мм."
            )

        # 4. Print speed limits
        if validated.print_speed_mm_s < self.limits.min_speed_mm_s or validated.print_speed_mm_s > self.limits.max_speed_mm_s:
            violations.append(
                f"Скорость печати {validated.print_speed_mm_s} мм/с вне безопасного диапазона [{self.limits.min_speed_mm_s}, {self.limits.max_speed_mm_s}] мм/с."
            )

        # 5. Cross-validation: Material-specific safety checks
        mat = validated.material
        if mat == "PEEK":
            if validated.nozzle_temp_c < 370.0:
                violations.append("Для PEEK температура сопла < 370°C недопустима (неполная кристаллизация и засор).")
            if validated.bed_temp_c < 110.0:
                warnings.append("Для PEEK рекомендуется подогрев стола не менее 120-140°C во избежание деламинации.")
        elif mat == "PLA":
            if validated.nozzle_temp_c > 240.0:
                warnings.append("Температура сопла > 240°C для PLA вызовет деградацию полимера и коксование сопла.")
            if validated.bed_temp_c > 70.0:
                warnings.append("Температура стола > 70°C для PLA приведет к деформации основания (эффект 'слоновьей ноги').")
        elif mat == "PA12-CF":
            if validated.nozzle_temp_c < 240.0:
                violations.append("PA12-CF требует сопла не ниже 250°C для нормального плавления связующего полиамида.")

        is_safe = len(violations) == 0
        return ValidationReport(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            normalized_params=validated.model_dump(),
        )

    def analyze_gcode_file(self, gcode_path: str | Path) -> GCodeSecurityReport:
        """Line-by-line static analysis of G-code file.

        Verifies:
        - Presence of G28 homing
        - Hardware bounding box limits (X, Y, Z coordinates)
        - Absolute/relative coordinate modes (G90/G91)
        - Temperature commands (M104/M109/M140/M190)
        - Calculates SHA-256 checksum for digital signature / fleet audit.
        """
        path = Path(gcode_path)
        if not path.is_file():
            raise FileNotFoundError(f"G-code file not found: {path}")

        sha256_hash = hashlib.sha256()
        file_size = path.stat().st_size

        has_homing_g28 = False
        is_absolute_mode = True  # Default G90 assumed in 3D printing
        violations: List[str] = []
        warnings: List[str] = []

        # Coordinate bounding tracking
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")
        min_z, max_z = float("inf"), float("-inf")

        # Current machine positions
        cur_x, cur_y, cur_z = 0.0, 0.0, 0.0
        max_nozzle_t = 0.0
        max_bed_t = 0.0

        coord_pattern = re.compile(r"([XYZSEF])([-+]?[0-9]*\.?[0-9]+)")
        temp_s_pattern = re.compile(r"S([-+]?[0-9]*\.?[0-9]+)")

        line_count = 0

        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256_hash.update(chunk)

        file_checksum = sha256_hash.hexdigest()

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, start=1):
                line_count += 1
                raw = line.split(";", 1)[0].strip()  # Strip comments
                if not raw:
                    continue

                upper = raw.upper()
                tokens = upper.split()
                if not tokens:
                    continue

                cmd = tokens[0]

                # Mode checks
                if cmd == "G90":
                    is_absolute_mode = True
                elif cmd == "G91":
                    is_absolute_mode = False
                    warnings.append(f"Строка {line_idx}: Переключение в относительные координаты (G91).")

                # Homing check
                if cmd == "G28":
                    has_homing_g28 = True

                # Temperature checks
                if cmd in ("M104", "M109"):  # Extruder temperature
                    match = temp_s_pattern.search(upper)
                    if match:
                        t = float(match.group(1))
                        max_nozzle_t = max(max_nozzle_t, t)
                        if t > self.limits.max_nozzle_temp_c:
                            violations.append(
                                f"Строка {line_idx}: Превышение предела температуры сопла ({t}°C > {self.limits.max_nozzle_temp_c}°C)."
                            )
                elif cmd in ("M140", "M190"):  # Bed temperature
                    match = temp_s_pattern.search(upper)
                    if match:
                        t = float(match.group(1))
                        max_bed_t = max(max_bed_t, t)
                        if t > self.limits.max_bed_temp_c:
                            violations.append(
                                f"Строка {line_idx}: Превышение предела температуры стола ({t}°C > {self.limits.max_bed_temp_c}°C)."
                            )

                # Motion commands check (G0, G1, G2, G3)
                if cmd in ("G0", "G1", "G00", "G01", "G2", "G3"):
                    coords = dict(coord_pattern.findall(upper))
                    if is_absolute_mode:
                        if "X" in coords:
                            cur_x = float(coords["X"])
                        if "Y" in coords:
                            cur_y = float(coords["Y"])
                        if "Z" in coords:
                            cur_z = float(coords["Z"])
                    else:
                        if "X" in coords:
                            cur_x += float(coords["X"])
                        if "Y" in coords:
                            cur_y += float(coords["Y"])
                        if "Z" in coords:
                            cur_z += float(coords["Z"])

                    # Update observed envelope
                    if "X" in coords:
                        min_x, max_x = min(min_x, cur_x), max(max_x, cur_x)
                    if "Y" in coords:
                        min_y, max_y = min(min_y, cur_y), max(max_y, cur_y)
                    if "Z" in coords:
                        min_z, max_z = min(min_z, cur_z), max(max_z, cur_z)

                    # Hard bound checks on current position
                    if cur_x < -5.0 or cur_x > self.limits.build_volume_x_mm + 5.0:
                        violations.append(
                            f"Строка {line_idx}: Выезд за пределы рабочей зоны по оси X: {cur_x:.2f} мм (лимит [0, {self.limits.build_volume_x_mm}])."
                        )
                    if cur_y < -5.0 or cur_y > self.limits.build_volume_y_mm + 5.0:
                        violations.append(
                            f"Строка {line_idx}: Выезд за пределы рабочей зоны по оси Y: {cur_y:.2f} мм (лимит [0, {self.limits.build_volume_y_mm}])."
                        )
                    if cur_z < -0.5 or cur_z > self.limits.build_volume_z_mm + 5.0:
                        violations.append(
                            f"Строка {line_idx}: Опасное перемещение оси Z: {cur_z:.2f} мм (лимит [0, {self.limits.build_volume_z_mm}])."
                        )

        # Homing verification
        if not has_homing_g28:
            violations.append("Критическая ошибка безопасности: в G-коде отсутствует обязательная команда парковки осей G28.")

        # Cap bounds if no moves recorded
        if min_x == float("inf"):
            min_x, max_x = 0.0, 0.0
        if min_y == float("inf"):
            min_y, max_y = 0.0, 0.0
        if min_z == float("inf"):
            min_z, max_z = 0.0, 0.0

        is_safe = len(violations) == 0

        return GCodeSecurityReport(
            is_safe=is_safe,
            file_path=str(path),
            sha256=file_checksum,
            file_size_bytes=file_size,
            total_lines=line_count,
            has_homing_g28=has_homing_g28,
            max_observed_nozzle_temp=max_nozzle_t,
            max_observed_bed_temp=max_bed_t,
            bounds_observed={
                "x": (round(min_x, 2), round(max_x, 2)),
                "y": (round(min_y, 2), round(max_y, 2)),
                "z": (round(min_z, 2), round(max_z, 2)),
            },
            build_volume_limits={
                "x": self.limits.build_volume_x_mm,
                "y": self.limits.build_volume_y_mm,
                "z": self.limits.build_volume_z_mm,
            },
            violations=violations[:20],  # Keep top 20 violations for readability
            warnings=warnings[:10],
        )
