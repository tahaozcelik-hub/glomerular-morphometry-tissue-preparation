#!/usr/bin/env bash
# Full pipeline in the order used for the paper. Training and embedding need a CUDA GPU (about 10 h in total).
set -e
cd "$(dirname "$0")"
MODELS_FM="phikon phikon2 lunit midnight dinov2 imagenet_vitb"
TEST="2ec3f1bb9 3589adb90 57512b7f1 aa05346ff d488c759a"
W=${WORK_DIR:-work}

# metadata and manual outlines
python hubmap_provenance.py
python morphometry.py

# images at 1 µm/pixel, scale checks
python seg/prep_1um.py
python seg/prep_extra.py
python nuclear_ruler.py

# glomerulus segmentation: donor-grouped folds, mixed and single-preparation models
for k in 0 1 2 3; do
  python seg/train.py --fold $k
  python seg/train.py --fold $k --prep FFPE --epochs 20
  python seg/train.py --fold $k --prep fresh_frozen --epochs 20
done
python seg/infer.py
python seg/infer.py --suffix _FFPE --train_only
python seg/infer.py --suffix _fresh_frozen --train_only
python seg/validate.py
python seg/xprep_eval.py
for k in 0 1 2 3; do python seg/infer.py $TEST --force_fold $k; done
python seg/instrument_check.py

# cortex segmentation and extension sections
python seg/cortex.py prep
for k in 0 1 2 3 -1; do python seg/cortex.py train --fold $k; done
python seg/cortex.py eval
python seg/cortex.py predict
python seg/extra_measure.py

# foundation-model embeddings and the sclerosis classifier
python fm/download_models.py
python fm/embed.py hubmap results/pred_glomerulus_morphometry.csv $W/emb/pred $MODELS_FM
python fm/embed.py hubmap_norm results/pred_glomerulus_morphometry.csv $W/emb/prednorm $MODELS_FM
python fm/embed.py hubmap_jpeg results/pred_glomerulus_morphometry.csv $W/emb/predjpeg phikon lunit
python fm/embed.py aidpath $W/emb/aidpath $MODELS_FM
python fm/sclerosis.py cv $MODELS_FM
python fm/sclerosis.py apply phikon results/pred_glomerulus_morphometry.csv $W/emb/pred

# section metrics and statistics
python clinical_metrics.py results/pred_glomerulus_morphometry_scl.csv results/clinical_metrics_pred.csv
python section_tables.py
python paired_analysis.py --input pred_glomerulus_morphometry.csv --tag _pred7 --cortex_only --no_vertex_cov \
  --paired 66999,67026,67112,67347,67548,68304,68555
python paired_analysis.py --input glomerulus_morphometry.csv --tag _manual4_cortex --cortex_only
python cohort_analysis.py
python fm/probe.py results/pred_glomerulus_morphometry.csv $W/emb/pred _raw $MODELS_FM
python fm/probe.py results/pred_glomerulus_morphometry.csv $W/emb/prednorm _reinhard $MODELS_FM
python fm/probe.py results/pred_glomerulus_morphometry.csv $W/emb/predjpeg _jpeg phikon lunit
python fm/mitigation.py results/pred_glomerulus_morphometry.csv $W/emb/pred $MODELS_FM
python review/review_stats.py
