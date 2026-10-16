# DynaRO

State-constrained 3D remaining-oil forecasting with dynamic oil-water contact.

DynaRO predicts the evolution of initially oil-filled voxels from an eight-class
semantic volume and a sequence of displacement measures. Predicted local
oil-water contact updates the next transition. Regional and initial-class
inventories and paired-schedule responses are read from the same oil field.

## Latest additions: v1.1.0

The [v1.1.0 release](https://github.com/Hiddenmn/DynaRO/releases/tag/v1.1.0)
adds the matched contact-mechanism controls and independent-core evaluation:
six final control states, 3,750 expert-reviewed eight-class labels, 480 compact
simulated trajectories, all 27 new model/seed/split result panels, and four
revised figure scripts and inputs. Download `DynaRO-v1.1.0-additions.zip` for
these resources; see its [instructions](extensions/v1_1_0/README.md).
The original DynaRO checkpoints remain the main model and are unchanged.
The addition uses the original model weights from v1.0.0 when running external
inference. Reaggregating its saved results and plotting need no new inference.

The sections below describe the preserved original study.

## Original study: v1.0.0

The complete reproduction package is the `DynaRO-v1.0.0.zip` asset on the
[v1.0.0 release](https://github.com/Hiddenmn/DynaRO/releases/tag/v1.0.0).
It contains code, data, model states and the inference-timing addendum.
`SHA256SUMS.txt` provides its checksum. The Git checkout contains the source
code and numerical records; large NumPy arrays and PyTorch states are supplied
in the release asset, not in Git history.

## Contents

- 80 expert-reviewed 128-cubed label volumes with the original 40/20/20 split.
- 28 final model states and two coarse-front pre-update states.
- Model, simulator, training, evaluation and plotting implementations.
- Fixed configurations and 6,400 trajectory-generation identities.
- Numerical and spatial inputs for all eight figures and eight tables,
  including the separate inference-timing measurements.

Raw grayscale CT is not redistributed. The source collection is
[Pak et al.](https://doi.org/10.6084/m9.figshare.c.4190738), with its
[dataset descriptor](https://doi.org/10.1038/sdata.2019.4).
The eight-class annotations and simulated outputs are derived research
materials, not annotations or measured future flow fields supplied by the
original data providers. See [DATA_SOURCES.txt](DATA_SOURCES.txt).

## Reproduce saved results

Unzip the complete release into a new directory. With Python 3.12:

```sh
cd reproduction
python -B verify_release.py --plans
python -B reporting/reproduce.py --tables-only --output-dir ../tables
python -m pip install -r reporting/requirements-3d.lock.txt
python -B reporting/reproduce_all.py --output-dir ../figures
cd ../timing
python -B summarize_timing.py timing.json --output rebuilt_results
```

The first command checks file integrity and metadata-only entry plans. The
other commands reconstruct saved tables, figures and timing summaries without
training a model or repeating TEST. Use output directories that do not already
exist. Column definitions and the figure-panel map are in
`reproduction/code/source_data/`.

## Model experiments

The mixed 25/50-step dynamic-contact study and the separate 25-step static-front
learning-rule study have different configurations. Their model/seed keys,
input routes and execution options are documented under `reproduction/`.
Complete TRAIN/TEST trajectory batches are regenerated from the supplied
plans and simulator; only the saved trajectories needed for figures are
included directly. Simulation steps are not calibrated physical time.

Original training used Linux, CUDA-enabled PyTorch and an RTX 4090. The separate
inference benchmark used Windows 11 and an RTX 5070. Portable full-CUDA
training has not been rerun end to end; six FNO/RTU initialization hashes differ
in the Windows CPU check, and their guards remain enabled. Saved-output
reproduction does not require that training route. Component documentation
records the original preparation snapshot; the release version and licences
are specified at this repository's root.

## Licence and citation

Author-original software is licensed under [MIT](LICENSE), copyright
2026 Qisong Zhou. Author-generated labels, model states, scientific metadata
and numerical resources are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt). Third-party materials retain their original
terms. See [LICENCE_SCOPE.txt](LICENCE_SCOPE.txt) for the division.

Please cite the versioned software/data release using [CITATION.cff](CITATION.cff).
