# data/
The Kaggle notebook writes `manifest.csv` (common sample manifest: dataset, sample_id, sequence_id, left_image, right_image, mask_path, disparity_path, depth_path, calibration_path, split, annotation_status). Copy it here. Raw datasets are NOT stored in this repo.

Annotation guidance: RSRD pothole masks are model-proposed and manually reviewed (`annotation_status = model_proposed_manually_reviewed`). Unannotated defects must not be treated as reliable background.
