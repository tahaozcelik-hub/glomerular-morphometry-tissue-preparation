# Tissue preparation and AI-based glomerular morphometry

Code, trained models and per-glomerulus measurements for:

> Çalı Özçelik KN, Özçelik STA, Timurkaan S. *Effect of tissue preparation on artificial intelligence–based glomerular morphometry: paired frozen and paraffin sections of human kidneys.* (submitted)

The study measures every glomerulus in PAS-stained whole-slide images of human kidneys from the Human BioMolecular Atlas Program (HuBMAP). Seven kidneys contributed both a fresh-frozen and a formalin-fixed paraffin-embedded (FFPE) section, and 35 donors formed a frozen-section cohort with clinical data.

## Contents

| Path | Purpose |
|---|---|
| `config.py` | Input and output locations (environment variables) |
| `wsi_io.py`, `morphometry.py` | TIFF reader and glomerulus shape features in µm |
| `seg/` | Resampling to 1 µm/pixel, glomerulus U-Net (training, inference, validation), cross-preparation models, cortex U-Net, extension sections |
| `fm/` | Encoder embeddings, sclerosis classifier, preparation and donor probes, embedding alignment |
| `clinical_metrics.py`, `section_tables.py` | Section metrics (MGV, density, cortex area), donor and paired tables |
| `paired_analysis.py` | Mixed models and exact paired tests of the preparation effect |
| `cohort_analysis.py` | Clinical associations in the frozen cohort |
| `nuclear_ruler.py`, `seg/scale_check.py` | Nuclear-size check of the stated pixel size |
| `hubmap_provenance.py` | HuBMAP metadata: tissue blocks, scanners and donor links |
| `review/` | Blinded review tool and its statistics |
| `metadata/extension_sections.csv` | The 27 HuBMAP extension files (one excluded as a duplicate) |
| `results/` | Measurements and tables reported in the paper |
| `run_all.sh` | Full pipeline in order |

## Data

| Dataset | Source | Set `...` to |
|---|---|---|
| HuBMAP "Hacking the Kidney" (20 WSIs, `train/`, `test/`, `HuBMAP-20-dataset_information.csv`) | [Kaggle](https://www.kaggle.com/competitions/hubmap-kidney-segmentation) | `KAGGLE_DIR` |
| HuBMAP PAS image pyramids listed in `metadata/extension_sections.csv` | [HuBMAP Data Portal](https://portal.hubmapconsortium.org) | `EXTRA_DIR` |
| AIDPATH glomeruli (`normal/`, `sclerosed/`) | [Mendeley Data, doi:10.17632/k7nvtgn2x6.3](https://doi.org/10.17632/k7nvtgn2x6.3) | `AIDPATH_DIR` |

Intermediate files (resampled images, predictions, models, embeddings) go to `WORK_DIR` (default `./work`). Defaults for all paths are under `./data`.

## Installation

```bash
conda create -n glom python=3.9
conda activate glom
pip install -r requirements.txt   # install the CUDA build of torch for your system
```

## Usage

`run_all.sh` lists every step. The main ones:

```bash
python seg/prep_1um.py                  # resample to 1 µm/pixel
python seg/train.py --fold 0            # one of four donor-grouped folds
python seg/infer.py                     # out-of-fold and ensemble segmentation
python seg/validate.py                  # agreement with manual outlines, per-glomerulus morphometry
python paired_analysis.py --input pred_glomerulus_morphometry.csv --tag _pred7 --cortex_only --no_vertex_cov \
  --paired 66999,67026,67112,67347,67548,68304,68555
python cohort_analysis.py
```

The statistics run on the files in `results/` together with the cached images; training and embedding need a CUDA GPU.

## Trained models

The weights are attached to the [v1.0 release](../../releases/tag/v1.0): glomerulus U-Nets `fold0`–`fold3` (both preparations), `fold*_FFPE` and `fold*_fresh_frozen` (one preparation), and cortex U-Nets `cortex_fold0`–`cortex_fold3` and `cortex_fold-1` (all challenge WSIs). Place them in `WORK_DIR/models`.

## Results

Main files in `results/`:

- `glomerulus_morphometry.csv`: manual outlines (3,785 glomeruli)
- `pred_glomerulus_morphometry_scl.csv`: automated outlines of the challenge WSIs with sclerosis probability (5,275)
- `extra_glomerulus_morphometry.csv`, `extra_section_metrics.csv`: HuBMAP extension sections
- `clinical_metrics_pred.csv`, `cohort_donors.csv`: section-level and donor-level metrics
- `seg_validation_*.csv`, `xprep_*.csv`, `cortex_oof.csv`: segmentation accuracy
- `paired_*.csv`, `mixed_model_results_*.csv`, `cohort_models.csv`, `table*.csv`: statistics
- `fm_*.csv`, `sclerosis_cv_models.csv`: foundation-model analyses
- `histology_review.csv`, `histology_review_summary.csv`: blinded review of 128 glomeruli

Areas are in µm², volumes in µm³ and densities in glomeruli per mm² of cortex. `prep` is `fresh_frozen` or `FFPE`.

## License

Code: MIT. The images and annotations remain under the terms of their original datasets.

## Contact

Salih Taha Alperen Özçelik, Bingöl University, sozcelik@bingol.edu.tr
