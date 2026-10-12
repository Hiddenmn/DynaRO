DynaRO: saved-output figure and table reproduction

This component reconstructs the paper's eight figures and seven tables from
the enclosed saved measurements, labels, trajectories and feature arrays.
It does not train a model, load weights, generate trajectories or evaluate
TEST again. Training and checkpoint evaluation have separate entry points.

Quick start (Python 3.12)

  python verify_reporting.py
  python -m pip install -r requirements-3d.lock.txt
  python -B reproduce_all.py --output-dir ../my_report

Use a new output directory outside this component. Nothing is overwritten.
For the standard-library-only seven-table route:

  python -B reproduce.py --tables-only --output-dir ../my_tables

Individual entry points

  reproduce.py               Tables 1-3/S1-S4 and Figures 3, 4, S3
  reproduce_sampling.py      Figure 1: full semantic stack and nested regions
  reproduce_architecture.py  Figure 2: saved features and method data flow
  reproduce_3d.py            Figure 5: fixed central DEV region and contact groups
  reproduce_parent3d.py      Figure S2: complete corresponding DEV parent volume
  reproduce_train3d.py       Figure S1: fixed illustrative TRAIN trajectory

Each takes --output-dir. Defaults resolve inputs relative to the script.
No account, network connection, GPU, original workspace or manuscript is
needed after installing the plotting dependencies. Do not run Python with -O.

Data interpretation

The long-horizon TEST comparison uses 600 paired schedules in five blocks
and three model seeds, retaining the zero-error block. Learning-rule results
belong to the separate 25-step static-front experiment. Model seeds are not
independent cores. Reported errors use x1000 for Brier/MSE and x100 for
inventory fractions (percentage points). Uncertainty is sample SD across
the stated seeds; single-seed comparisons have no fabricated SD.

Volume cases retain the original fixed case/crop selection, full arrays,
camera, thresholds and display-only transforms. Figure 1 keeps full-volume
class colours, while subvolumes pool oil colour only for display. The five
initial oil classes still enter the model and inventory measurements.
Display coordinates are voxel indices, not inferred physical distances.
Caption-level case-selection and statistical definitions remain in the paper.

Portability and validation

All binary inputs and CSV measurements retain their original bytes. In JSON
metadata only absolute source-location strings become portable artifact names;
no number, identifier, array, record order or scientific setting is changed.
Each transformed file records original and packaged hashes in provenance.
The plotting and numerical code is unchanged except for the layout-check
import. The new project-specific checker inspects axes without modifying them.
It does not bundle an external helper with unconfirmed redistribution terms.
An optional trusted QA module may be selected with DYNARO_FIGURE_QA; it is not
a runtime dependency, and setting it explicitly permits executing that module.

  python -B -m unittest discover -s tests -v
  python -B -m unittest discover -s tests_3d -v

The plotting lock records the working Windows environment, separate from
training. Arial is used when available, with DejaVu Sans as fallback; the
architecture uses DejaVu Sans. No font is bundled. Package, font and rasterizer
differences can change pixels without changing the values. A successful local
run is not a claim of fresh Linux/macOS installation or independent retraining.

Source and release status

Underlying public micro-CT collection: doi:10.6084/m9.figshare.c.4190738;
dataset descriptor: doi:10.1038/sdata.2019.4. Author-generated semantic labels
and simulated/model-derived outputs are separate from the upstream raw data.
The sampling volume includes the author's semantic stack and one semantic TIFF
used to verify the original in-plane image size, not a mirror of the raw collection.

Local release candidate only: no upload, licence grant or publication is made
by preparation. Authors must confirm the code and derived-data redistribution
scope and licence before release. Dependencies retain their respective licences.
