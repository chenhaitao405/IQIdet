# Graph Report - IQIdet  (2026-05-30)

## Corpus Check
- 1642 files · ~39,718,716 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 886 nodes · 1860 edges · 54 communities (43 shown, 11 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 213 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6bbd1a03`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_IQI Inferencer Core|IQI Inferencer Core]]
- [[_COMMUNITY_OCR Subprocess Pipeline|OCR Subprocess Pipeline]]
- [[_COMMUNITY_FClip Trainer Engine|FClip Trainer Engine]]
- [[_COMMUNITY_Loss Functions|Loss Functions]]
- [[_COMMUNITY_Adaptive Image Processing|Adaptive Image Processing]]
- [[_COMMUNITY_IQI Rules Engine|IQI Rules Engine]]
- [[_COMMUNITY_Region SNR API|Region SNR API]]
- [[_COMMUNITY_FClip Model Heads|FClip Model Heads]]
- [[_COMMUNITY_FClip Inference Utils|FClip Inference Utils]]
- [[_COMMUNITY_Region OCR API|Region OCR API]]
- [[_COMMUNITY_Dataset Crop Augmentation|Dataset Crop Augmentation]]
- [[_COMMUNITY_Pipeline Image Utilities|Pipeline Image Utilities]]
- [[_COMMUNITY_Weld Dataset Pipeline|Weld Dataset Pipeline]]
- [[_COMMUNITY_FClip Line Parsing|FClip Line Parsing]]
- [[_COMMUNITY_FClip Box Core|FClip Box Core]]
- [[_COMMUNITY_Gauge Augmentation Training|Gauge Augmentation Training]]
- [[_COMMUNITY_Wireframe Dataset Pipeline|Wireframe Dataset Pipeline]]
- [[_COMMUNITY_FClip Box Collections|FClip Box Collections]]
- [[_COMMUNITY_FClip Config Box|FClip Config Box]]
- [[_COMMUNITY_FClip Box Internals|FClip Box Internals]]
- [[_COMMUNITY_FClip Box Serialization|FClip Box Serialization]]
- [[_COMMUNITY_LR Schedulers|LR Schedulers]]
- [[_COMMUNITY_YOLO OBB Conversion|YOLO OBB Conversion]]
- [[_COMMUNITY_FClip Box Accessors|FClip Box Accessors]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_FClip Metrics|FClip Metrics]]
- [[_COMMUNITY_FClip Config System|FClip Config System]]
- [[_COMMUNITY_Paddle OCR Wrapper|Paddle OCR Wrapper]]
- [[_COMMUNITY_Gauge Detector Planning|Gauge Detector Planning]]
- [[_COMMUNITY_Box Rationale Notes|Box Rationale Notes]]
- [[_COMMUNITY_Box Rationale Notes|Box Rationale Notes]]
- [[_COMMUNITY_Box Rationale Notes|Box Rationale Notes]]
- [[_COMMUNITY_Box Rationale Notes|Box Rationale Notes]]
- [[_COMMUNITY_Marker Pattern System|Marker Pattern System]]
- [[_COMMUNITY_Grade Fusion Logic|Grade Fusion Logic]]
- [[_COMMUNITY_LCNN Head Module|LCNN Head Module]]
- [[_COMMUNITY_Quadrantal Rotation Config|Quadrantal Rotation Config]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `Box` - 49 edges
2. `IQIInferencer` - 38 edges
3. `OCRTextOrientationCorrector` - 36 edges
4. `PaddleOCRSubprocessClient` - 32 edges
5. `str` - 30 edges
6. `AdaptiveImageProcessor` - 28 edges
7. `str` - 28 edges
8. `FClip` - 25 edges
9. `Any` - 24 edges
10. `str` - 23 edges

## Surprising Connections (you probably didn't know these)
- `bool` --uses--> `IQIInferencer`  [INFERRED]
  scripts/debug/dump_iqi_fclip_trace.py → src/gauge/iqi_inferencer.py
- `Tensor` --uses--> `IQIInferencer`  [INFERRED]
  scripts/debug/dump_iqi_fclip_trace.py → src/gauge/iqi_inferencer.py
- `Subprocess OCR Architecture` --conceptually_related_to--> `PaddleOCRSubprocessClient`  [INFERRED]
  gauge/ocr_stage.py → src/gauge/ocr_stage.py
- `Any` --uses--> `IQIInferencer`  [INFERRED]
  scripts/debug/dump_iqi_fclip_trace.py → src/gauge/iqi_inferencer.py
- `Path` --uses--> `IQIInferencer`  [INFERRED]
  scripts/debug/dump_iqi_fclip_trace.py → src/gauge/iqi_inferencer.py

## Hyperedges (group relationships)
- **IQI Grade Inference Pipeline** — gauge_iqi_inferencer_iqiinferencer, gauge_roi_stage_extract_best_obb, gauge_fclip_stage_fclipinferencer, gauge_ocr_stage_paddleocrsubprocessclient, gauge_ocr_stage_infer_roi_ocr, gauge_pipeline_utils_enhance_windowing_gray, gauge_pipeline_utils_crop_rotated_polygon, gauge_iqi_rules_infer_plate_from_ocr_items, gauge_iqi_rules_compute_iqi_grade, gauge_weld_correction_weldorientationcorrector [INFERRED 0.95]
- **Subprocess OCR Architecture** — gauge_ocr_stage_paddleocrsubprocessclient, gauge_ocr_paddle_worker_main, gauge_ocr_stage_subprocess_ocr_architecture [INFERRED 0.95]
- **Two-tier Orientation Correction** — gauge_weld_correction_weldorientationcorrector, gauge_ocr_orientation_ocrtextorientationcorrector, gauge_adaptive_image_processor_adaptiveimageprocessor [INFERRED 0.85]
- **Heatmap Data Format Producers** — dataset_wireframe_save_heatmap, dataset_wireframe_line_save_heatmap, dataset_york_save_heatmap, dataset_york_line_save_heatmap, dataset_weld_save_heatmap, dataset_crop_croptaugmentation, dataset_resolution_resizeresolution [INFERRED 0.95]
- **FClip Inference Pipeline** — fclip_infer_utils_infer_heatmaps, fclip_nms_non_maximum_suppression, fclip_line_parsing_pointparsing, fclip_line_parsing_onestagelineparsing, fclip_postprocess_postprocess [INFERRED 0.95]
- **IQI Dataset Processing Pipeline** — dataset_weld_order_points, dataset_weld_crop_rotated_rect, dataset_weld_point_in_polygon, dataset_weld_map_lines_perspective, dataset_weld_angle_from_vertical, dataset_weld_rotate_cw90, dataset_weld_enhance_windowing_gray, dataset_weld_augment_and_save, dataset_weld_save_heatmap [INFERRED 0.95]
- **FClip Detection Head Implementations** — models___init___multitaskhead, models___init___linehead, models___init___lcnnhead [INFERRED 0.85]
- **Gauge Detector Training Augmentation** — config_augment_augmentation_pipeline, config_augment_quadrantal_rotation [EXTRACTED 1.00]

## Communities (54 total, 11 thin omitted)

### Community 0 - "IQI Inferencer Core"
Cohesion: 0.12
Nodes (28): FClipInferencer, Torch-based FClip inferencer for wire count and line endpoints., build_delivery_record(), build_final_result_vis_image(), build_iqi_statistics(), build_wire_vis_image(), collect_input_images(), IQIInferencer (+20 more)

### Community 1 - "OCR Subprocess Pipeline"
Cohesion: 0.10
Nodes (43): decode_image(), main(), parse_args(), write_response(), build_ocr_item_debug_images(), build_ocr_statistics(), _configure_paddle_runtime(), _contains_jb() (+35 more)

### Community 2 - "FClip Trainer Engine"
Cohesion: 0.23
Nodes (7): init_lr_scheduler(), WarmUpCosine, WarmUpSingle, build_model(), get_outdir(), main(), _LRScheduler

### Community 3 - "Loss Functions"
Cohesion: 0.06
Nodes (24): ce_loss, focal_loss, gaussian_soft_ce_loss, l12loss, sigmoid_l1_loss, anchor_loss(), balanced_positive_negative_sampler(), ce_loss() (+16 more)

### Community 4 - "Adaptive Image Processing"
Cohesion: 0.08
Nodes (23): AdaptiveImageProcessor, Apply windowing and optional negative transform to weld film images., Resize while preserving aspect ratio, then pad to a square canvas., SquarePadResize, Image, bool, int, ndarray (+15 more)

### Community 5 - "IQI Rules Engine"
Cohesion: 0.18
Nodes (40): IQIInferencer.infer_image_path, IQI Grade Inference Pipeline Architecture, _base_field_record(), _box_center(), _build_candidates_from_marker_sequence(), _build_candidates_from_sequence(), build_result_status(), _build_text_sequences() (+32 more)

### Community 6 - "Region SNR API"
Cohesion: 0.11
Nodes (24): BaseModel, compute_region_snr(), _decode_base64_image(), Field(), get_region_snr_service(), HTTPException, init_region_snr_api(), 计算单个区域的归一化信噪比（base64 输入）。 (+16 more)

### Community 7 - "FClip Model Heads"
Cohesion: 0.08
Nodes (14): build_infer_model(), LCNNHead, LineHead, MultitaskHead, BasicBlock, Bottleneck, conv3x3(), get_hr_config() (+6 more)

### Community 8 - "FClip Inference Utils"
Cohesion: 0.14
Nodes (23): device, draw_count_pair(), get_count_pred(), infer_heatmaps(), parse_lines_1d(), preprocess_gray_image(), scale_lines(), evaluate_count_precision() (+15 more)

### Community 9 - "Region OCR API"
Cohesion: 0.14
Nodes (19): BaseModel, _decode_base64_image(), Field(), get_region_ocr_service(), HTTPException, init_region_ocr_api(), 同步识别单张图片区域（base64 输入），用于前端实时 OCR 框选功能。, recognize_region() (+11 more)

### Community 10 - "Dataset Crop Augmentation"
Cohesion: 0.08
Nodes (10): Dataset, CropAugmentation, CropAugmentation: random crop augmentation for training, randomCrop(), resolution should be the size of the label in npz file,         'NOT' the augmen, offset_wrapper(), translate the offset to gaussian mode, WireframeHuangKun (+2 more)

### Community 11 - "Pipeline Image Utilities"
Cohesion: 0.14
Nodes (26): apply_clahe(), apply_window_level(), auto_window_level(), collect_images(), crop_rotated_polygon(), enhance_windowing_gray(), ensure_dir(), format_polygon() (+18 more)

### Community 12 - "Weld Dataset Pipeline"
Cohesion: 0.14
Nodes (24): angle_from_vertical(), apply_clahe(), apply_window_level(), augment_and_save(), auto_window_level(), crop_rotated_rect(), draw_lines(), enhance_windowing_gray() (+16 more)

### Community 13 - "FClip Line Parsing"
Cohesion: 0.12
Nodes (12): line_parsing_from_npz(), OneStageLineParsing, PointParsing, :param xy: (K, 2)         :param xy_idx: (K,)         :param length_regress: (H,, non_maximum_suppression(), structure_nms(), structure_nms_torch(), plambda() (+4 more)

### Community 14 - "FClip Box Core"
Cohesion: 0.17
Nodes (4): BoxList, Drop in replacement of list, that converts added objects to Box or BoxList     o, _to_yaml(), list

### Community 15 - "Gauge Augmentation Training"
Cohesion: 0.22
Nodes (18): _normalize_quadrantal_rotation(), patch_random_perspective_rotation(), Custom augmentation helpers for gauge training., Split Ultralytics kwargs from project-local custom augmentation config., Patch Ultralytics RandomPerspective to sample around fixed quadrantal angles., split_augment_config(), build_train_args(), drop_none() (+10 more)

### Community 16 - "Wireframe Dataset Pipeline"
Cohesion: 0.33
Nodes (5): main(), main(), save_heatmap(), to_int(), parmap()

### Community 17 - "FClip Box Collections"
Cohesion: 0.18
Nodes (41): _args_snapshot(), _array_diff_stats(), _array_sha256(), _array_stats(), _asset_hashes(), _bbox_from_polygon(), _code_hashes(), _command_output() (+33 more)

### Community 18 - "FClip Config Box"
Cohesion: 0.16
Nodes (7): ConfigBox, Return value of key as an int          :param item: key of value to transform, Return value of key as a float          :param item: key of value to transform, Return value of key as a list          :param item: key of value to transform, Modified box object to add object transforms.      Allows for build in transform, Config file keys are stored in lower case, be a little more         loosey goose, Return value of key as a boolean          :param item: key of value to transform

### Community 19 - "FClip Box Internals"
Cohesion: 0.05
Nodes (39): 7 Stages, Backward Compatibility, Base Class Extraction, BaseOrientationCorrector (`services/correction.py`), BaseRegionService (`services/base.py`), CLI Integration, Configuration Management, Data Models (Pydantic) (+31 more)

### Community 20 - "FClip Box Serialization"
Cohesion: 0.05
Nodes (41): Final Verification Checklist, IQIdet Refactor Implementation Plan, Phase 0: Git Worktree + Branch, Phase 1: Infrastructure (pyproject.toml, models, config, logging, exceptions, services base), Phase 2: Data Types + Error Unification, Phase 3: Pipeline Refactoring, Phase 4: Orientation Merge + Final Polish, Task 0: Create isolated branch (+33 more)

### Community 21 - "LR Schedulers"
Cohesion: 0.23
Nodes (3): dict, Box, Improved dictionary access through dot notation with additional tools.      :par

### Community 22 - "YOLO OBB Conversion"
Cohesion: 0.38
Nodes (10): ensure_dir(), main(), order_points(), parse_args(), polygon_to_obb(), resolve_image_path(), write_data_yaml(), ndarray (+2 more)

### Community 23 - "FClip Box Accessors"
Cohesion: 0.18
Nodes (4): benchmark, np_softmax(), Compute softmax values for each sets of scores in x., object

### Community 24 - "Community 24"
Cohesion: 0.22
Nodes (9): Exception, BoxError, _from_json(), _from_yaml(), _get_box_config(), Due to the way pickling works in python 3, we need to make sure         the box, Non standard dictionary exceptions, Transform a json object string into a Box object. If the incoming         json i (+1 more)

### Community 25 - "FClip Metrics"
Cohesion: 0.48
Nodes (5): ap(), APJ(), mAPJ(), msAP(), msTPFP()

### Community 26 - "FClip Config System"
Cohesion: 0.32
Nodes (7): Namespace, Path, str, main(), parse_args(), Resolve a path against cwd, returning None if input is None., _resolve_path()

### Community 27 - "Paddle OCR Wrapper"
Cohesion: 0.83
Nodes (3): _ensure_bgr(), run_paddle_ocr(), Any

### Community 42 - "Community 42"
Cohesion: 0.15
Nodes (6): ShorthandBox (SBox) allows for     property access of `dict` `json` and `yaml`, Turn the Box and sub Boxes back into a native         python dictionary., Transform the Box object into a JSON string.          :param filename: If provid, Transform the BoxList object into a JSON string.          :param filename: If pr, SBox, _to_json()

### Community 43 - "Community 43"
Cohesion: 0.24
Nodes (4): AttributeError, BoxKeyError, _recursive_tuples(), KeyError

### Community 44 - "Community 44"
Cohesion: 0.21
Nodes (3): ModelPrinter, Temporarily prints things on the screen, Permanently prints things on the screen

### Community 45 - "Community 45"
Cohesion: 0.29
Nodes (3): get_global(), :param fn:         :param i:         :param result:  {, VisualizeResults

### Community 46 - "Community 46"
Cohesion: 0.31
Nodes (7): _camel_killer(), _conversion_checks(), Convert a key into something that is accessible as an attribute, CamelKiller, qu'est-ce que c'est?      Taken from http://stackoverflow.com/a/117, Internal use for checking if a duplicate safe attribute already exists      :par, _safe_attr(), _safe_key()

### Community 47 - "Community 47"
Cohesion: 0.22
Nodes (7): 代码边界, 修改约束, 像质计规则关键点, 开发环境, 推理主流程, 训练与数据, 项目概述

### Community 48 - "Community 48"
Cohesion: 0.24
Nodes (3): Logger, mkdir_if_missing(), Write console output to external text file.     Code imported from https://githu

### Community 50 - "Community 50"
Cohesion: 0.40
Nodes (3): coor_rot90(), save_heatmap(), to_int()

### Community 51 - "Community 51"
Cohesion: 0.40
Nodes (5): C: global configuration container (Box instance), _ensure_box(), load_configs(), M: model configuration shortcut (Box instance), load_config_from_yaml()

### Community 52 - "Community 52"
Cohesion: 0.50
Nodes (3): main(), save_heatmap(), to_int()

## Knowledge Gaps
- **98 isolated node(s):** `Task 0: Create isolated branch`, `Task 1.1: Create pyproject.toml`, `Task 1.2: Remove sys.path hacks from entry points`, `Task 1.3: Create Pydantic data models`, `Task 1.4: Create exception hierarchy` (+93 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Box` connect `LR Schedulers` to `FClip Model Heads`, `Community 42`, `Community 43`, `Community 46`, `FClip Box Core`, `FClip Config Box`, `Community 51`, `Community 53`, `Community 24`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **Why does `FClipInferencer` connect `IQI Inferencer Core` to `FClip Inference Utils`, `IQI Rules Engine`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `OCRTextOrientationCorrector` connect `IQI Inferencer Core` to `OCR Subprocess Pipeline`, `Adaptive Image Processing`, `Region OCR API`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `Box` (e.g. with `_ensure_box()` and `load_configs()`) actually correct?**
  _`Box` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `IQIInferencer` (e.g. with `FClipInferencer` and `OCRTextOrientationCorrector`) actually correct?**
  _`IQIInferencer` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `OCRTextOrientationCorrector` (e.g. with `IQIInferencer` and `.__init__()`) actually correct?**
  _`OCRTextOrientationCorrector` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `PaddleOCRSubprocessClient` (e.g. with `IQIInferencer` and `.__init__()`) actually correct?**
  _`PaddleOCRSubprocessClient` has 18 INFERRED edges - model-reasoned connections that need verification._