#!/usr/bin/env python3
"""Shared IQI inference service for delivery and debug wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from gauge.config import (
    CorrectionConfig,
    EnhanceConfig,
    FClipConfig,
    GaugeConfig,
    OCRConfig,
    PipelineConfig,
)
from gauge.services.fclip_stage import (
    FClipInferencer,
    invert_perspective_matrix,
    perspective_transform_points,
    undo_ccw90_points,
)
from gauge.iqi_rules import (
    build_result_status,
    choose_primary_result_code,
    compute_iqi_grade,
    extract_general_fields_from_ocr_items,
    format_allowed_numbers_spec,
    infer_plate_from_ocr_items,
    infer_plate_from_texts,
    normalize_text,
    parse_allowed_numbers_spec,
)
from gauge.record_builders import build_delivery_record, build_iqi_statistics
from gauge.services.ocr_stage import (
    PaddleOCRSubprocessClient,
    build_ocr_item_debug_images,
    draw_ocr_on_roi,
    infer_roi_ocr,
)
from gauge.pipeline_utils import (
    collect_images,
    crop_rotated_polygon,
    enhance_windowing_gray,
    ensure_dir,
    load_image,
    resize_long_side,
    rotate_if_wide,
    to_gray,
)
from gauge.services.roi_stage import build_roi_vis_image, extract_best_obb


class IQIInferencer:
    """Shared service for complete IQI grade inference."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        **kwargs: Any,
    ):
        # ---- Build config from kwargs if not provided (legacy path) ----
        if config is not None:
            self.config = config
        else:
            gauge_weights = kwargs.pop("gauge_weights", "")
            fclip_ckpt = kwargs.pop("fclip_ckpt", None)
            gauge_conf = float(kwargs.pop("gauge_conf", 0.25))
            gauge_iou = float(kwargs.pop("gauge_iou", 0.45))
            gauge_imgsz = int(kwargs.pop("gauge_imgsz", 640))
            gauge_device = kwargs.pop("gauge_device", None)
            gauge_select = str(kwargs.pop("gauge_select", "conf"))
            gauge_class = kwargs.pop("gauge_class", None)
            enhance_mode = str(kwargs.pop("enhance_mode", "windowing"))
            rotate_roi = bool(kwargs.pop("rotate_roi", True))
            enable_correction = bool(kwargs.pop("enable_correction", False))
            correction_model = kwargs.pop("correction_model", None)
            correction_device = kwargs.pop("correction_device", None)
            correction_verbose = bool(kwargs.pop("correction_verbose", False))
            ocr_device = str(kwargs.pop("ocr_device", "gpu"))
            ocr_det_model_name = str(kwargs.pop("ocr_det_model_name", "PP-OCRv5_server_det"))
            ocr_det_model_dir = kwargs.pop("ocr_det_model_dir", None)
            ocr_rec_model_name = str(kwargs.pop("ocr_rec_model_name", "en_PP-OCRv5_mobile_rec"))
            ocr_rec_model_dir = kwargs.pop("ocr_rec_model_dir", None)
            ocr_det_limit_side_len = int(kwargs.pop("ocr_det_limit_side_len", 960))
            ocr_det_limit_type = str(kwargs.pop("ocr_det_limit_type", "max"))
            ocr_min_score = float(kwargs.pop("ocr_min_score", 0.0))
            enable_ocr_orientation = bool(kwargs.pop("enable_ocr_orientation", False))
            ocr_orientation_model = kwargs.pop("ocr_orientation_model", None)
            ocr_orientation_device = kwargs.pop("ocr_orientation_device", None)
            ocr_orientation_verbose = bool(kwargs.pop("ocr_orientation_verbose", False))
            ocr_number_range = kwargs.pop("ocr_number_range", None)
            fclip_device = kwargs.pop("fclip_device", None)
            fclip_model_config = kwargs.pop("fclip_model_config", "config/model.yaml")
            fclip_params = kwargs.pop("fclip_params", "params.yaml")
            fclip_threshold = kwargs.pop("fclip_threshold", None)

            self.config = PipelineConfig(
                gauge=GaugeConfig(
                    weights=gauge_weights,
                    conf=gauge_conf,
                    iou=gauge_iou,
                    imgsz=gauge_imgsz,
                    device=gauge_device,
                    select=gauge_select,
                    gauge_class=gauge_class,
                ),
                fclip=FClipConfig(
                    ckpt=fclip_ckpt,
                    device=fclip_device,
                    fclip_model_config=str(Path(fclip_model_config).resolve()),
                    params=str(Path(fclip_params).resolve()),
                    threshold=fclip_threshold,
                ),
                ocr=OCRConfig(
                    device=ocr_device,
                    det_model_name=ocr_det_model_name,
                    det_model_dir=ocr_det_model_dir,
                    rec_model_name=ocr_rec_model_name,
                    rec_model_dir=ocr_rec_model_dir,
                    det_limit_side_len=ocr_det_limit_side_len,
                    det_limit_type=ocr_det_limit_type,
                    min_score=ocr_min_score,
                    enable_orientation=enable_ocr_orientation,
                    orientation_model=ocr_orientation_model,
                    orientation_device=ocr_orientation_device,
                    orientation_verbose=ocr_orientation_verbose,
                    number_range=format_allowed_numbers_spec(
                        parse_allowed_numbers_spec(
                            ocr_number_range
                            if not isinstance(ocr_number_range, (list, tuple))
                            else format_allowed_numbers_spec(ocr_number_range)
                        )
                    ),
                ),
                correction=CorrectionConfig(
                    enabled=enable_correction,
                    model=correction_model,
                    device=correction_device,
                    verbose=correction_verbose,
                ),
                enhance=EnhanceConfig(
                    mode=enhance_mode,
                    rotate_roi=rotate_roi,
                ),
            )

        # ---- Store legacy attr aliases for backward compatibility ----
        self.gauge_weights = str(self.config.gauge.weights)
        self.fclip_ckpt = self.config.fclip.ckpt
        self.gauge_conf = float(self.config.gauge.conf)
        self.gauge_iou = float(self.config.gauge.iou)
        self.gauge_imgsz = int(self.config.gauge.imgsz)
        self.gauge_device = self.config.gauge.device
        self.gauge_select = str(self.config.gauge.select)
        self.gauge_class = self.config.gauge.class_filter
        self.enhance_mode = str(self.config.enhance.mode)
        self.rotate_roi = bool(self.config.enhance.rotate_roi)
        self.correction_verbose = bool(self.config.correction.verbose)
        self.ocr_det_limit_side_len = int(self.config.ocr.det_limit_side_len)
        self.ocr_det_limit_type = str(self.config.ocr.det_limit_type)
        self.ocr_min_score = float(self.config.ocr.min_score)
        self.ocr_orientation_verbose = bool(self.config.ocr.orientation_verbose)
        self.ocr_allowed_numbers = parse_allowed_numbers_spec(self.config.ocr.number_range)
        self.ocr_number_range = format_allowed_numbers_spec(self.ocr_allowed_numbers)
        self.fclip_model_config = str(self.config.fclip.fclip_model_config)
        self.fclip_params = str(self.config.fclip.params)
        self.fclip_threshold = self.config.fclip.threshold

        # ---- Model initialization (kept from original) ----
        self.corrector = None
        if self.config.correction.enabled:
            if not self.config.correction.model:
                raise ValueError("--correction-model is required when enable_correction=True")
            from gauge.services.weld_correction import WeldOrientationCorrector

            self.corrector = WeldOrientationCorrector(
                model_path=self.config.correction.model,
                model_type="resnet50",
                device=self.config.correction.device,
            )

        self.ocr_text_corrector = None
        if self.config.ocr.enable_orientation:
            if not self.config.ocr.orientation_model:
                raise ValueError("--ocr-orientation-model is required when enable_ocr_orientation=True")
            from gauge.services.ocr_orientation import OCRTextOrientationCorrector

            model_path = Path(self.config.ocr.orientation_model)
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()
            self.ocr_text_corrector = OCRTextOrientationCorrector(
                model_path=model_path,
                model_type="resnet34",
                device=self.config.ocr.orientation_device,
            )

        from ultralytics import YOLO

        self.gauge_model = YOLO(self.gauge_weights)
        self.ocr_backend = PaddleOCRSubprocessClient(
            device=self.config.ocr.device,
            det_model_name=self.config.ocr.det_model_name,
            det_model_dir=self.config.ocr.det_model_dir,
            rec_model_name=self.config.ocr.rec_model_name,
            rec_model_dir=self.config.ocr.rec_model_dir,
            det_limit_side_len=self.ocr_det_limit_side_len,
            det_limit_type=self.ocr_det_limit_type,
        )
        self.fclip_inferencer = None
        if self.fclip_ckpt:
            self.fclip_inferencer = FClipInferencer(
                ckpt_path=self.fclip_ckpt,
                device=self.config.fclip.device,
                model_config=self.fclip_model_config,
                params_file=self.fclip_params,
                threshold=self.fclip_threshold,
            )

        # ---- Build PipelineRunner ----
        from gauge.pipeline import PipelineRunner

        services: Dict[str, Any] = {
            "gauge_model": self.gauge_model,
            "ocr_backend": self.ocr_backend,
            "fclip_inferencer": self.fclip_inferencer,
            "corrector": self.corrector,
            "ocr_text_corrector": self.ocr_text_corrector,
        }
        self.runner = PipelineRunner.from_config(self.config, services=services)

    def close(self) -> None:
        if self.ocr_backend is not None:
            self.ocr_backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_runtime_meta(self) -> Dict[str, Any]:
        return {
            "gauge_weights": self.gauge_weights,
            "fclip_ckpt": self.fclip_ckpt,
            "fclip_model_config": self.fclip_model_config,
            "fclip_params": self.fclip_params,
            "fclip_threshold": self.fclip_threshold,
            "gauge_conf": self.gauge_conf,
            "gauge_iou": self.gauge_iou,
            "gauge_imgsz": self.gauge_imgsz,
            "gauge_select": self.gauge_select,
            "gauge_class": self.gauge_class,
            "enhance_mode": self.enhance_mode,
            "rotation_rule": "ccw90_if_width_gt_height" if self.rotate_roi else "disabled",
            "ocr_min_score": self.ocr_min_score,
            "ocr_number_range": self.ocr_number_range,
            "ocr_det_limit_side_len": self.ocr_det_limit_side_len,
            "ocr_det_limit_type": self.ocr_det_limit_type,
            "full_image_resize_long_side": self.ocr_det_limit_side_len,
            "full_image_ocr_enhance_mode": "windowing",
            "ocr_runtime": "subprocess_det_rec",
        }

    # ------------------------------------------------------------------
    # Legacy static/class helper methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_skipped_wire(status: str, error: Optional[str] = None) -> Dict[str, Any]:
        from gauge.pipeline_utils import build_skipped_wire

        return build_skipped_wire(status, error)

    @staticmethod
    def _scale_box_points(box: Any, scale: float) -> Any:
        from gauge.pipeline_utils import scale_box_points

        return scale_box_points(box, scale)

    @classmethod
    def _scale_ocr_items_to_original(cls, items: Sequence[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
        from gauge.pipeline_utils import scale_ocr_items_to_original

        return scale_ocr_items_to_original(items, scale)

    @classmethod
    def _scale_roi_info_to_original(cls, roi_info: Dict[str, Any], scale: float) -> Dict[str, Any]:
        from gauge.pipeline_utils import scale_roi_info_to_original

        return scale_roi_info_to_original(roi_info, scale)

    @staticmethod
    def _is_usable_ocr_item(item: Dict[str, Any]) -> bool:
        from gauge.pipeline_utils import is_usable_ocr_item

        return is_usable_ocr_item(item)

    @staticmethod
    def _box_points_to_bbox(box: Any) -> Optional[List[float]]:
        from gauge.pipeline_utils import box_points_to_bbox

        return box_points_to_bbox(box)

    @staticmethod
    def _project_roi_box_to_image(
        box: Any,
        crop_inverse_matrix: Optional[np.ndarray],
        pre_rotate_size: Optional[Sequence[int]],
        rotated: bool,
    ) -> Tuple[Any, Any]:
        if box is None:
            return None, None
        try:
            roi_points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None, None
        if roi_points.size == 0:
            return None, None

        roi_unrotated = roi_points
        if rotated:
            if pre_rotate_size is None:
                return roi_points.tolist(), None
            roi_unrotated = undo_ccw90_points(roi_points, pre_rotate_size=pre_rotate_size)

        if crop_inverse_matrix is None:
            return roi_points.tolist(), roi_unrotated.tolist()

        image_points = perspective_transform_points(roi_unrotated, crop_inverse_matrix)
        return image_points.tolist(), roi_unrotated.tolist()

    @classmethod
    def _project_ocr_items_to_image(
        cls,
        items: Sequence[Dict[str, Any]],
        crop_inverse_matrix: Optional[np.ndarray],
        pre_rotate_size: Optional[Sequence[int]],
        rotated: bool,
    ) -> List[Dict[str, Any]]:
        projected_items: List[Dict[str, Any]] = []
        for item in items:
            projected = dict(item)
            box_image, box_unrotated = cls._project_roi_box_to_image(
                item.get("box"),
                crop_inverse_matrix=crop_inverse_matrix,
                pre_rotate_size=pre_rotate_size,
                rotated=rotated,
            )
            projected["box_image"] = box_image
            projected["box_roi_unrotated"] = box_unrotated
            projected_items.append(projected)
        return projected_items

    @classmethod
    def _build_plate_visualization_items(
        cls,
        items: Sequence[Dict[str, Any]],
        source: str,
    ) -> List[Dict[str, Any]]:
        from gauge.pipeline_utils import build_plate_visualization_items

        return build_plate_visualization_items(items, source)

    @staticmethod
    def _select_plate_visualization_items(
        items: Sequence[Dict[str, Any]],
        plate_code: Optional[str],
        allowed_numbers: Optional[Sequence[int]],
    ) -> List[Dict[str, Any]]:
        target_code = normalize_text(plate_code)
        if not target_code:
            return []

        selected: List[Dict[str, Any]] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            parsed = infer_plate_from_texts(
                [text],
                require_jb=True,
                allowed_numbers=allowed_numbers,
            )
            candidates = [normalize_text(code) for code in (parsed.get("candidate_codes") or [])]
            if target_code in candidates:
                selected.append(item)
        return selected

    def _attach_visualization_payload(
        self,
        record: Dict[str, Any],
        *,
        roi_plate_vis_items: Optional[Sequence[Dict[str, Any]]] = None,
        full_plate_vis_items: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        roi = record.get("roi") or {}
        plate = record.get("plate") or {}
        wire = record.get("wire") or {}
        plate_source = str(record.get("plate_source") or "")

        if plate_source == "roi":
            plate_items = list(roi_plate_vis_items or [])
        elif plate_source == "full_image":
            plate_items = list(full_plate_vis_items or [])
        else:
            plate_items = []

        plate_items_selected = self._select_plate_visualization_items(
            plate_items,
            plate_code=plate.get("plate_code"),
            allowed_numbers=self.ocr_allowed_numbers,
        )

        wire_lines = []
        for line in wire.get("lines") or []:
            image_xy = line.get("image_xy")
            if not image_xy:
                continue
            wire_lines.append(
                {
                    "index": line.get("index"),
                    "score": line.get("score"),
                    "image_xy": image_xy,
                }
            )

        record["visualization"] = {
            "roi_polygon_xy": roi.get("polygon"),
            "roi_bbox": roi.get("bbox"),
            "plate_source": plate_source,
            "plate_code": plate.get("plate_code"),
            "candidate_codes": plate.get("candidate_codes") or [],
            "raw_texts": plate.get("raw_texts") or [],
            "plate_text_items": plate_items,
            "plate_text_items_selected": plate_items_selected,
            "wire_lines": wire_lines,
        }

    @staticmethod
    def _merge_prefixed_ocr_timings(
        step_timings: Dict[str, float],
        prefix: str,
        ocr_result: Optional[Dict[str, Any]],
    ) -> None:
        from gauge.pipeline_utils import merge_prefixed_ocr_timings

        merge_prefixed_ocr_timings(step_timings, prefix, ocr_result)

    @staticmethod
    def _finalize_record(
        record: Dict[str, Any],
        error_entries: Sequence[Dict[str, Any]],
        warnings: Sequence[str],
        grade: Optional[int] = None,
    ) -> Dict[str, Any]:
        primary_code = choose_primary_result_code([entry["result_code"] for entry in error_entries])
        status = build_result_status(primary_code)
        record.update(status)
        record["status"] = "ok" if primary_code == 0 else "error"
        record["errors"] = list(error_entries)
        record["warnings"] = list(warnings)

        plate = record.get("plate") or {}
        wire = record.get("wire") or {}
        record["iqi_type"] = plate.get("iqi_type")
        record["plate_code"] = plate.get("plate_code")
        record["plate_number"] = plate.get("number")
        record["wire_count"] = wire.get("wire_count")
        record["iqi_marker_found"] = bool(plate.get("ok"))
        record["general_fields_found"] = bool((record.get("field_statistics") or {}).get("general_fields_found", False))
        record["grade"] = int(grade) if primary_code == 0 and grade is not None else None
        return record

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def infer_image_path(
        self,
        image_path: Path,
        return_debug_artifacts: bool = False,
        debug_timer: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
        try:
            record = self.runner.run(
                image_path,
                return_debug_artifacts=return_debug_artifacts,
            )
            return record.model_dump(), record._debug_artifacts
        except Exception as exc:
            from gauge.iqi_rules import build_result_status

            record: Dict[str, Any] = {
                "image_path": str(image_path),
                "status": "error",
                "ok": False,
                "result_code": 9001,
                "result_name": "internal_error",
                "result_message": str(exc),
                "grade": None,
                "iqi_type": None,
                "plate_code": None,
                "plate_number": None,
                "plate_source": None,
                "wire_count": None,
                "fields": {
                    "component_codes": [],
                    "weld_film_pairs": [],
                    "weld_numbers": [],
                    "film_numbers": [],
                    "pipe_specs": [],
                },
                "field_statistics": {
                    "component_code_count": 0,
                    "weld_film_pair_count": 0,
                    "weld_number_count": 0,
                    "film_number_count": 0,
                    "pipe_spec_count": 0,
                    "general_fields_found": False,
                    "full_image_marker_found": False,
                    "roi_marker_found": False,
                    "iqi_marker_found": False,
                },
                "warnings": [],
                "errors": [{"stage": "pipeline", **build_result_status(9001, str(exc))}],
            }
            return record, None


# ---------------------------------------------------------------------------
# Module-level helper functions (unchanged)
# ---------------------------------------------------------------------------


def build_wire_vis_image(roi_image: np.ndarray, wire_result: Dict[str, Any]) -> np.ndarray:
    vis = roi_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for line in wire_result.get("lines") or []:
        points = line.get("roi_xy") or []
        if len(points) != 2:
            continue
        p0 = (int(round(points[0][0])), int(round(points[0][1])))
        p1 = (int(round(points[1][0])), int(round(points[1][1])))
        cv2.line(vis, p0, p1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    text = f"wire_count={wire_result.get('wire_count')} parsed={wire_result.get('parsed_line_count')}"
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def build_final_result_vis_image(
    image: np.ndarray,
    visualization: Optional[Dict[str, Any]] = None,
    plate_code: Optional[str] = None,
    grade: Optional[int] = None,
    wire_count: Optional[int] = None,
) -> np.ndarray:
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    visualization = dict(visualization or {})

    roi_polygon = visualization.get("roi_polygon_xy") or []
    try:
        roi_pts = np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
    except Exception:
        roi_pts = np.zeros((0, 2), dtype=np.float32)
    if roi_pts.shape[0] >= 3:
        cv2.polylines(
            vis,
            [roi_pts.astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )

    plate_text_items = (
        visualization.get("plate_text_items_selected")
        or visualization.get("plate_text_items")
        or []
    )
    for item in plate_text_items:
        box = item.get("box_image_xy") or []
        try:
            pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if pts.shape[0] < 3:
            continue
        cv2.polylines(
            vis,
            [pts.astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 215, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        label = str(item.get("text", "")).strip() or "[empty]"
        score = item.get("score")
        if score is not None:
            label = f"{label} ({float(score):.2f})"
        x = int(np.min(pts[:, 0]))
        y = max(20, int(np.min(pts[:, 1])) - 8)
        cv2.putText(
            vis,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 215, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for line in visualization.get("wire_lines") or []:
        points = line.get("image_xy") or []
        if len(points) != 2:
            continue
        p0 = (int(round(points[0][0])), int(round(points[0][1])))
        p1 = (int(round(points[1][0])), int(round(points[1][1])))
        cv2.line(vis, p0, p1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    summary_lines = []
    if plate_code:
        summary_lines.append(f"plate={plate_code}")
    if grade is not None:
        summary_lines.append(f"grade={grade}")
    if wire_count is not None:
        summary_lines.append(f"wire_count={wire_count}")
    if summary_lines:
        cv2.putText(
            vis,
            "  ".join(summary_lines),
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return vis


def save_debug_visualizations(
    output_dir: Path,
    sample_dir: Path,
    image: np.ndarray,
    full_ocr_result: Dict[str, Any],
    full_ocr_image: Optional[np.ndarray] = None,
    roi_info: Optional[Dict[str, Any]] = None,
    roi_image: Optional[np.ndarray] = None,
    roi_gray: Optional[np.ndarray] = None,
    roi_ocr_result: Optional[Dict[str, Any]] = None,
    wire_result: Optional[Dict[str, Any]] = None,
    visualization: Optional[Dict[str, Any]] = None,
    plate_code: Optional[str] = None,
    grade: Optional[int] = None,
    wire_count: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_dir(sample_dir)

    input_path = sample_dir / "input.png"
    cv2.imwrite(str(input_path), image)

    payload: Dict[str, Any] = {
        "status_vis_dir": str(sample_dir.relative_to(output_dir)),
        "input_vis_path": str(input_path.relative_to(output_dir)),
    }

    full_base = full_ocr_image if full_ocr_image is not None else image
    full_input_path = sample_dir / "full_ocr_input.png"
    cv2.imwrite(str(full_input_path), full_base)
    full_ocr_vis = draw_ocr_on_roi(full_base, full_ocr_result)
    full_ocr_vis_path = sample_dir / "full_ocr_result.png"
    cv2.imwrite(str(full_ocr_vis_path), full_ocr_vis)

    full_item_vis_rows: List[Dict[str, Any]] = []
    full_debug_rows = build_ocr_item_debug_images(full_base, full_ocr_result)
    if full_debug_rows:
        item_dir = sample_dir / "full_ocr_items"
        ensure_dir(item_dir)
        for row in full_debug_rows:
            crop_index = int(row.get("crop_index", len(full_item_vis_rows)))
            crop_path = item_dir / f"crop_{crop_index:03d}.png"
            rec_input_path = item_dir / f"rec_input_{crop_index:03d}.png"
            rec_result_path = item_dir / f"rec_result_{crop_index:03d}.png"
            cv2.imwrite(str(crop_path), row["crop_image"])
            cv2.imwrite(str(rec_input_path), row["rec_input_image"])
            cv2.imwrite(str(rec_result_path), row["rec_result_image"])
            full_item_vis_rows.append(
                {
                    "crop_index": crop_index,
                    "crop_path": str(crop_path.relative_to(output_dir)),
                    "rec_input_path": str(rec_input_path.relative_to(output_dir)),
                    "rec_result_path": str(rec_result_path.relative_to(output_dir)),
                    "text": row.get("text", ""),
                    "score": row.get("score"),
                }
            )

    payload.update(
        {
            "full_ocr_input_path": str(full_input_path.relative_to(output_dir)),
            "full_ocr_vis_path": str(full_ocr_vis_path.relative_to(output_dir)),
            "full_ocr_item_vis": full_item_vis_rows,
            "ocr_vis_path": str(full_ocr_vis_path.relative_to(output_dir)),
            "ocr_item_vis": full_item_vis_rows,
        }
    )

    if roi_info:
        roi_vis = build_roi_vis_image(image, roi_info)
        roi_vis_path = sample_dir / "ROI.png"
        cv2.imwrite(str(roi_vis_path), roi_vis)
        payload["roi_vis_path"] = str(roi_vis_path.relative_to(output_dir))

    if roi_image is not None:
        roi_image_path = sample_dir / "roi_image.png"
        cv2.imwrite(str(roi_image_path), roi_image)
        payload["roi_image_path"] = str(roi_image_path.relative_to(output_dir))

    if roi_gray is not None:
        roi_gray_path = sample_dir / "roi_gray.png"
        cv2.imwrite(str(roi_gray_path), roi_gray)
        payload["roi_gray_path"] = str(roi_gray_path.relative_to(output_dir))

    if roi_ocr_result is not None and roi_gray is not None:
        roi_ocr_vis = draw_ocr_on_roi(roi_gray, roi_ocr_result)
        roi_ocr_vis_path = sample_dir / "roi_ocr_result.png"
        cv2.imwrite(str(roi_ocr_vis_path), roi_ocr_vis)
        payload["roi_ocr_vis_path"] = str(roi_ocr_vis_path.relative_to(output_dir))

        roi_item_vis_rows: List[Dict[str, Any]] = []
        roi_debug_rows = build_ocr_item_debug_images(roi_gray, roi_ocr_result)
        if roi_debug_rows:
            item_dir = sample_dir / "roi_ocr_items"
            ensure_dir(item_dir)
            for row in roi_debug_rows:
                crop_index = int(row.get("crop_index", len(roi_item_vis_rows)))
                crop_path = item_dir / f"crop_{crop_index:03d}.png"
                rec_input_path = item_dir / f"rec_input_{crop_index:03d}.png"
                rec_result_path = item_dir / f"rec_result_{crop_index:03d}.png"
                cv2.imwrite(str(crop_path), row["crop_image"])
                cv2.imwrite(str(rec_input_path), row["rec_input_image"])
                cv2.imwrite(str(rec_result_path), row["rec_result_image"])
                roi_item_vis_rows.append(
                    {
                        "crop_index": crop_index,
                        "crop_path": str(crop_path.relative_to(output_dir)),
                        "rec_input_path": str(rec_input_path.relative_to(output_dir)),
                        "rec_result_path": str(rec_result_path.relative_to(output_dir)),
                        "text": row.get("text", ""),
                        "score": row.get("score"),
                    }
                )
            payload["roi_ocr_item_vis"] = roi_item_vis_rows

    if wire_result is not None and roi_image is not None:
        wire_vis = build_wire_vis_image(roi_image, wire_result)
        wire_vis_path = sample_dir / "wire_result.png"
        cv2.imwrite(str(wire_vis_path), wire_vis)
        payload["wire_vis_path"] = str(wire_vis_path.relative_to(output_dir))

    if visualization:
        final_result_vis = build_final_result_vis_image(
            image=image,
            visualization=visualization,
            plate_code=plate_code,
            grade=grade,
            wire_count=wire_count,
        )
        final_result_path = sample_dir / "finalresult.png"
        cv2.imwrite(str(final_result_path), final_result_vis)
        payload["final_result_vis_path"] = str(final_result_path.relative_to(output_dir))

    return payload


def collect_input_images(
    image_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_list: Optional[str] = None,
    max_images: Optional[int] = None,
) -> List[Path]:
    if image_path:
        paths = [Path(image_path).resolve()]
    else:
        paths = collect_images(
            Path(image_dir).resolve() if image_dir else None,
            Path(image_list).resolve() if image_list else None,
            max_images=max_images,
        )
    if max_images is not None:
        paths = paths[:max_images]
    return paths
