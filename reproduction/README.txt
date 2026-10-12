DynaRO reproduction materials - integrated local candidate
==========================================================

This directory brings together the seven prepared components. It is not yet a
public release, a licence grant or a claim of independent CUDA reproduction.
The component snapshots are retained unchanged; older component-level notes
about other components being separate should be read alongside this root guide.

Start here
----------
Using Python 3.12, from this directory:

  python -B verify_release.py
  python -B verify_release.py --plans

Both commands are read-only. The second additionally checks the documented
input/training/evaluation metadata routes with third-party packages disabled.
It does not load trajectory arrays or weights, construct a model, use a GPU,
generate data, train, score TEST, download files or contact a server.

Reproduce the submitted values and figures
-----------------------------------------
The reporting component is sufficient for all eight figures and seven tables.
It contains the saved measurements and the fixed label/feature/trajectory
arrays used for the figures, rather than rendering from finished screenshots.

  python -m pip install -r reporting/requirements-3d.lock.txt
  python -B reporting/reproduce_all.py --output-dir ../dynaro_report

Use a new output directory outside reporting/. Rendering is CPU-only, but is
separate from the default read-only verification. For tables without plotting
dependencies:

  python -B reporting/reproduce.py --tables-only --output-dir ../dynaro_tables

The 28 numerical source-data CSVs, panel map and column definitions are under
code/source_data/. They support exact-value inspection; reporting/ recreates
figures/tables from their underlying saved inputs. These serve different uses.
Do not infer new model-performance evidence from successful figure recreation.

Component map
-------------
code/model_core/       Original numerical model modules and five-model factory
code/source_data/      Numerical tables/figure values with units and panel map
weights/               28 tensor-only frozen final checkpoints
inputs/                80 reviewed initial labels, split, schedules and simulator
evaluation/            Evaluation of the original frozen checkpoints
training/              Recorded TRAIN configurations and portable execution control
retrained_evaluation/  Explicit evaluation route for NEW completed training outputs
reporting/             All figure/table inputs and CPU rendering entry points

Model and experiment routes
---------------------------
Prepare model dependencies separately from the plotting environment. The
model core contains requirements-cpu.txt for its checked synthetic CPU route.
Original research execution used Linux, CUDA-enabled PyTorch and RTX 4090;
each execution entry enforces its recorded runtime and configuration.

The following commands only print plans (shown on one line to work across shells):

  python -B inputs/reproduce_inputs.py --set train25
  python -B evaluation/evaluate.py --core code/model_core --weights weights --inputs inputs --model-key mixed_horizon/dynaro/20260811
  python -B training/train.py --core code/model_core --inputs inputs --evaluation evaluation --model-key mixed_horizon/dynaro/20260811
  python -B retrained_evaluation/evaluate_retrained.py --core code/model_core --weights weights --inputs inputs --training training --evaluation evaluation --model-key mixed_horizon/dynaro/20260811

Replace the model key with learning_rule_25/state_prioritized/20260811 for the
separate static-front 25-step study. Consult the component protocol for all
prescribed model/seed keys, not an automatically selected best checkpoint.

Full TRAIN/TEST trajectory batches are not duplicated in this package. The
inputs component supplies their fixed generation plans and simulator entry.
Only the illustrative saved trajectories needed for figures are included in
reporting/. An original trajectory directory or a separately generated full
batch is required before actual training/evaluation. Original and regenerated
file identities remain distinct. Use --help and the component README for the
required data directories, explicit --execute, new output, run-time limit and
pause/resume conditions. No numerical experiment is launched by this guide.

Scientific distinctions retained
-------------------------------
Mixed-horizon training has 25/50-step full-chain rollouts; its TEST trajectories
extend to 100 steps. The learning-rule comparison is a separate 25-step static-
front study with five-step gradient windows. These are not interchangeable.
The main TEST panel contains 600 pairs, five equal-weight spatial blocks and
three model seeds; it is based on one source core. The zero-error block is kept.

Site/response scores are unscaled fractions internally. Reporting multiplies
Brier/MSE by 1000 and inventory fractions by 100 (percentage points). Simulation
steps are not calibrated physical time. Oil classes retain their reviewed
operational label definitions and the simulator's fixed class-dependent rules.

What has been checked
---------------------
Saved-output reconstruction, original-value agreement, component integrity,
metadata routes, synthetic CPU model/optimizer fixtures, strict frozen-weight
loading and producer/consumer file formats have been checked locally. No new
research training, inference or TEST experiment was run during assembly.

The portable full-CUDA path has not been demonstrated end to end. Six FNO/RTU
initialization hashes do not match in the Windows/CPU check and remain blocked
by strict guards; do not replace their expected hashes to force a run. This
does not change the available original frozen checkpoints or their reported
scores. Plotting uses the recorded Windows environment; font/rasterizer changes
on another system can change pixels. No fresh Linux/macOS installation is claimed.

Release decisions
-----------------
Authors must confirm code/data scope, redistribution rights, licence, versioned
release citation and reviewer-access route before publication. No raw upstream
micro-CT archive is mirrored. Author-derived labels, checkpoints, simulated
outputs and source data are separately identified. The complete reporting
semantic stack is larger than a routine source-code file and should be treated
as a data asset, not silently added to a source-code commit.

The public source collection is doi:10.6084/m9.figshare.c.4190738 and its descriptor
is doi:10.1038/sdata.2019.4. These citations do not imply that the upstream authors
provided the present study's semantic annotations, simulator or model outputs.
No repository, release URL, licence or DOI is invented by this local assembly.
