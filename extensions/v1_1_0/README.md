# DynaRO v1.1.0 additions

This version adds the matched contact-mechanism comparison and independent-core
evaluation to the [unchanged v1.0.0 release](https://github.com/Hiddenmn/DynaRO/releases/tag/v1.0.0).
The original DynaRO remains the main model. Explicit-contact and learned-contact
are two equally sized comparison models, not replacements for its checkpoints.

## Materials

- `sameinfo/`: six final 400-update model states, three shared initial states,
  configurations, exact panel identities, and all six DEV and six TEST records.
- `external/`: 3,750 expert-reviewed eight-class PNGs from two additional cores,
  12 fixed input volumes, 480 compact displacement trajectories, and the saved
  five-model, three-seed evaluation records.
- `code/`: model definitions, fixed training update, displacement and evaluation
  kernels, weight loader, inference and evaluation entries.
- `reporting/`: source-core and external fixed 3D cases, figure inputs and
  scripts for the four revised figures. No new model call is needed to plot them.
- `manifest.json`: checksums and sizes. `source_provenance.json` identifies
  byte-preserved components and the path-only portability changes.

The original five-model weights and original study materials remain in
`DynaRO-v1.0.0.zip`. Download that asset only for full model inference or the
original study; saved-result reproduction of this addition is self-contained.

## Reproduce saved numbers and figures

Use Python 3.12 and install `requirements.txt`. PyTorch 2.11.0 was used for the
new experiments; install the build appropriate to your CPU/CUDA environment.

```sh
python -B reproduce_saved.py --output rebuilt_numbers
python -B code/sameinfo_runtime.py --verify-only
python -B reporting/plot_external_figures.py --out rebuilt_figures
python -B -c "from pathlib import Path; import sys; sys.path.insert(0,'reporting'); import plot_source_spatial as p; p.render(Path('rebuilt_figures'))"
python -B -c "from pathlib import Path; import sys; sys.path.insert(0,'reporting'); import plot_learning_compact as p; p.render(Path('rebuilt_figures'))"
```

The first command reaggregates all 27 saved model/seed/split panels and checks
their recorded values. It does not repeat inference. The second loads all six
new final states on CPU and checks their tensor hashes, without a forward pass.
The plotting commands use saved fields and metrics. Arial is preferred; a
DejaVu Sans fallback is available. Raster surface shading is deterministic,
uses the native voxel geometry and does not alter the underlying values.

## Model use and experiments

`python code/sameinfo_runtime.py --help` describes inference from an eight-class
128-cubed `.npy` volume and an explicit S1-S5 action sequence. Predictions are
probabilities on the initial oil-site support, with indices and group inventories.

`python code/evaluate_external.py --model dynaro --seed 20260811` prints a plan.
To run the fixed external panel, provide `--base-release` pointing to the
extracted v1.0.0 root, a new `--output` directory, and `--execute`. `--pair-id`
selects an exact identity for a smaller reproduction check. Existing published
results are never overwritten.

`code/train_sameinfo.py` reconstructs the fixed TRAIN pairing and order in
plan mode. Full training requires the original trajectories regenerated with
the v1.0.0 tools and an explicit `--execute`. Both arms use the same supplied
initial parameter buffers, 400 updates, AdamW, and final-update checkpoints.

`python code/evaluate_sameinfo.py --arm explicit_contact --seed 20260811 --split dev`
prints the fixed contact-control evaluation plan. To run it, provide the
original trajectory tree as `--data-root`, a new `--output`, and `--execute`.
DEV50 uses the separate equal-dose 50-step panel; TEST50 is the prefix of the
100-step TEST panel, matching the original experiment. The supplied final
states are fixed independently of the evaluation. Do not tune on TEST.

The original experiment ran on Linux/RTX 4090. The new portable wrappers have
not been used for an additional full training or evaluation run; the published
states and records are those of the completed original experiments.

## Data interpretation and reuse

The external PNGs are author-generated, expert-reviewed annotations in the
original CT slice numbering and pixel coordinates. Classes 0-7 are background,
rock, water, film oil, corner oil, droplet oil, columnar oil and cluster oil.
These are initial morphology labels; categories stay attached to the initial
oil sites when later inventories are computed. The model receives labels,
not raw grayscale CT. The targets are contact-driven stochastic displacement
trajectories, not measured future flow fields.

The external cores are Ketton (WF1; three axial segments from one core) and
Bentheimer Core 1. They are two independent core geometries, not four cores.
All 12 fixed blocks and all three training seeds are retained. Comparisons
are descriptive across seeds; slices and paired schedules are not treated as
independent experimental cores. See `DATA_DICTIONARY.md` and `DATA_SOURCES.md`.

## Licence and citation

Author-original code is MIT-licensed (2026 Qisong Zhou). Author-generated
labels, weights and numerical resources are CC BY 4.0. Upstream datasets keep
their own licences and attribution. Raw CT images are not redistributed.
Cite `CITATION.cff`, this versioned release, and the relevant upstream dataset.
