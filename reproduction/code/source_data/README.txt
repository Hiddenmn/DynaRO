DynaRO manuscript source data
Local candidate, 9 October 2026; not published or licensed for redistribution

Contents and scope

This folder contains 28 byte-preserved CSV exports: printed and unrounded
versions of seven result tables, plus numerical data for Figures 3, 4, 5,
S1, S2 and S3. It supports inspection of reported numbers without loading a
model. It is not the complete training/evaluation or image-reproduction release.
The data were copied from the existing saved-output reproductions; packaging
does not retrain, resimulate, reevaluate TEST, smooth curves or select new cases.

manifest.json maps each file to a table or panel and records its dimensions,
SHA-256 and column definitions. data_dictionary.csv provides the same column
definitions in a spreadsheet-friendly form. CSVs use UTF-8, comma separators
and a header row. Quoted cells follow standard CSV escaping. No missing values
are allowed except empty raw_sample_sd in explicitly single-seed table rows.
Empty SD means not estimated; it is not zero.

Use Python 3.10 or newer, standard library only, from any working directory:

    python -I -B verify_source_data.py

The verifier reads files only. It checks CSV hashes, schemas, finite values,
matched-seed summaries, selected figure/table agreements, paired differences,
all five spatial blocks, inventory units and the fixed contact-cohort
denominators. It does not verify model accuracy independently or establish
fresh-environment training reproducibility. No external paths or downloads
are needed. No credentials, manuscript source, upstream images, neural-network
weights, semantic volumes, third-party drawing helpers or fonts are enclosed.

Study and display conventions

- The mixed 25/50-step study compares DynaRO (state_front), Static front
  (static_front), RTU (rtu), FNO (fno), and regional-mean exposure (coarse_front).
  Tables 1, 2, S1-S3 and Figures 3 and S3 use this study.
- Tables 3 and S4 and Figure 4 are a separate 25-step STATIC-FRONT learning-rule
  study. Its historical stateguard key means state-prioritized learning. It
  does not denote the dynamic-contact architecture in those files. The
  stateguard_error field in Figure S3 instead denotes DynaRO. These original
  column names are preserved to avoid silently changing saved data.
- Model-training seeds are 20260811, 20260812 and 20260813. They are not the
  simulator random streams. Semicolon-separated seeds and seed_values align.
  Single-seed PCGrad/state-prioritized rows use 20260811 only.
- Each seed scores the same 400 DEV or 600 TEST pairs in five spatial blocks.
  Summary errors weight blocks equally. Seeds repeat these same samples;
  pairs and blocks are not independent rock specimens. R2 is computed from
  pooled equally block-weighted observations within each seed, then summarized.
- Means and sample SDs use three seeds and ddof=1 unless a row states one seed.
  Error bars are not confidence intervals. No significance tests are encoded.
- Brier and MSE raw values are dimensionless, on the fraction scale. Multiply
  by 1000 for axes/tables labelled 10^-3. Inventory MAE raw values are fractions
  of initial class inventory; multiply by 100 for percentage points (pp).
  R2 has multiplier 1. inventory_mae_pp is ALREADY in percentage points.
- Display CSVs contain rounded text exactly matching the table exports;
  *_values.csv files retain the saved precision for numerical use. Some
  display column labels are flattened versions of typeset multiline headers.
- 'horizon' is the trajectory endpoint; 'steps' is the scored interval.
  Inventory metrics are terminal, not time-averaged. In Table S1, the steps
  field describes R2 scoring; terminal_class_mae is evaluated at horizon.
  The 75/100-step schedules extend the last action of the 50-step schedule;
  these are not equal-dose order-only contrasts with the 25-step schedules.
- Coordinates are voxel indices in (z,y,x) order. Simulator steps are not
  physical time. No physical length or time calibration is implied.

Figure-specific reading notes

Figure 3: a-b contain 100 steps for each of four models and two metrics.
Panel c reports the mean and sample SD of within-seed percentage reductions
over steps 51-100 against HOLDING THAT MODEL'S step-50 field. Positive means
continuation lowers error. This is different from a reduction versus another
model computed from the ratio of aggregate means. Panel d retains all blocks:
 B1 z128_y384_x256; B2 z256_y0_x128; B3 z384_y0_x128;
 B4 z384_y384_x0; B5 z512_y256_x256.
B4's zero errors and its 120 pairs are included, not removed as uninformative.

Figure 4: a=Development, b=Locked test. Each bar is one seed's percentage
reduction: 100*(fixed-weight error-state-prioritized error)/fixed-weight error.
The underlying scores are equal-block means in the separate static-front
25-step study. Negative reductions indicate higher error.

Figure 5: the fixed NPI-012 DEV case uses S5 for 30 steps, then S3 through
step 100, seed 20260811. The parent begins at (0,256,256); the central 64^3
region spans parent indices 32-95 on all axes. The parent was selected before
prediction by maximum simulator mobilization among 240 background-free DEV
trajectories; it is not a randomly representative example. Panel e uses only
five saved states (0,25,50,75,100), joined as visual guides. Its denominator
is the 35,740 initial regional oil voxels. Predictions use continuous sums,
not counts of a thresholded surface.

For Figure 5f, groups are fixed from simulator-remaining oil at step 50.
Six-neighbour water contact is computed on the FULL parent grid, before
cropping. The plotted groups are 3,581 water-contacting and 26,236 noncontacting
sites in inlet-connected pores. Both methods divide by those SAME fixed
simulator group sizes: predicted conversion is sum(p50-p75), not relative to
predicted remaining amount at step 50. The file also retains the unplotted
1,973-site disconnected group (zero conversion) and initial-class counts.
figure5_auxiliary_depth.csv is an unplotted slab export from the same case,
not another figure panel or independent finding.

Figure S1: selected TRAIN crop, z40-103,y16-79,x16-79, S3x20 then S5x30.
Initial oil is 18,396 voxels. The source column remaining_fraction is NOT the
plotted y value; cumulative conversion is 100*(1-remaining_fraction). All
51 states are included. Counts at 0,20,50 also support the volume captions.
Case selection ranked 1,117 eligible interior crops by central-region
conversion after retaining the most-mobilizing schedule per training state.

Figure S2: full 128^3 parent of Figure 5. Panel g has all 101 saved states;
panel h uses step-100 conversion in eight 16-voxel x slabs, normalized by each
slab's initial oil; panel i has Brier errors at steps 1-100. Saved floating-
point precision is preserved, including parent regional quantities exported
from the original float32 records. These are single-case DEV curves, not
aggregate TEST evidence.

Figure S3: a-b show comparator minus DynaRO over steps 51-100, averaged across
matched seeds WITHIN each pair. Positive gains favour DynaRO. Source errors
and gains are raw; multiply by 1000 for the plotted scale. All 600 pairs are
retained. Boxes describe median/IQR and minimum-maximum variation across
pairs, not confidence intervals. The corresponding 10,800 seed-level rows
are provided. Panel c contains class-specific step-100 inventory MAE for
each seed: percentage points after averaging schedules, pairs within blocks
and equally weighted blocks. Average the three seeds and use sample SD to
reproduce bars and error bars; do not compute an SD over the five classes.
The separately saved all-class MAE and the mean of exported class values
can differ in their last digits because the original evaluator reduces the
former in float32 before export. Both saved values are preserved; the
cross-display check allows 1e-9 in raw fraction (1e-7 percentage points).

What this source-data subset does not contain

Figures 1 and 2 are the spatial hierarchy and architecture, not numerical
charts. Their label arrays, example feature volumes and layout code are kept
in the separate local image-reproduction material. The volume panels in
Figures 5, S1 and S2 also require those arrays and rendering scripts; these
CSVs alone cannot reconstruct voxel geometry. The full reproduction release
still needs an approved label/weight/code file set and documented access.

Original images and attribution

Upstream X-ray images are provided by the original Figshare collection:
https://doi.org/10.6084/m9.figshare.c.4190738
Dataset descriptor: https://doi.org/10.1038/sdata.2019.4
Those source images are not enclosed. This study's eight-class semantic
labels are author-generated outputs, not labels supplied by that collection.
Class codes are 0=background,1=rock,2=water,3=film oil,4=corner oil,5=droplet
oil,6=columnar oil,7=cluster oil. Geometry panels pool classes 3-7 as oil.

Release status

The authors have selected public release before submission. This is a local
preparation, not a public deposit. The final file scope and licence remain
unconfirmed; no licence or DOI is assigned here. Creating a GitHub repository
does not make these research materials available. Author/creator metadata
and a versioned release citation must be completed for the actual release.
