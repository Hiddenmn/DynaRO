# File and variable dictionary

## Initial labels

`external/native_labels/<segment>/*.png`: uint8 class IDs, native CT pixel
coordinates and slice IDs. `external/labels/*.npy`: fixed model-grid inputs,
shape (128,128,128), axis order (z,y,x). Codes: 0 background, 1 rock, 2 water,
3 film oil, 4 corner oil, 5 droplet oil, 6 columnar oil, 7 cluster oil.
`external/crops.json` records native origins, shapes, counts and file hashes.

## Compact simulated futures

`external/targets/pairs.jsonl` lists all 240 paired schedule identities and
480 target files. Each member has 100 steps. S1-S5 encode the action library
in `code/actions.py`; schedules and random-stream IDs are explicit.

Each `external/targets/targets/*.npz` stores:

- `first_removal_step`: uint8 vector on flattened initial sites with class >=3.
  Zero means oil remains through step 100; values 1-100 are first displacement
  steps. Occupancy at t is `(removal == 0) | (removal > t)`.
- `displaced_by_class`: int32 array (100,5), per-step counts for initial classes
  3-7, not cumulative counts.
- `connected_pore`: uint8 128-cubed binary mask, six-neighbour pore connection
  to the x=0 inlet.
- `action_index`: int8 length-100 array, 0-4 corresponds to S1-S5.

## Predictions and metrics

Weights are plain dense CPU tensor dictionaries (`torch.load(...,
weights_only=True)`). The six contact controls each contain 91,936 parameters.
`sameinfo/config/` records initialization, order, optimizer and final-state
selection. `sameinfo/protocol.json` retains original DEV/TEST identities;
historical relative paths identify the corresponding v1.0.0-generation tree.

Each result JSONL row is one paired schedule on one fixed spatial block and
stream. `scores` contains 25/50/75/100-step evaluations; same-information
scores retain an extra arm key. The fields `all`, `unseen`, `tail` and `last`
are time intervals. At horizon 100 these are steps 1-100, 51-100, 76-100 and
100, respectively. Site-state Brier error is mean squared occupancy error.
Regional MSE compares remaining fractions in fixed cell/initial-class groups.
Response MSE compares the two schedules' regional responses. Terminal class
inventory MAE is a fraction of each initial class inventory; multiply by 100
for percentage points (pp). `*_by_step` fields retain the full curves.

Saved summaries first average paired schedules within each block, then give
blocks equal weight. External cores are reported separately. Across-seed SD
uses three seed values and denominator n-1. Missing-class values are JSON
`null`, never manufactured zeros. The source-core zero-response block remains
included in the original and matched-control aggregation.

## Figures

- `reporting/plot_external_figures.py`: Figure 4 (temporal Brier curves and
  terminal class-inventory errors) and Figure 5 (fixed two-core spatial cases).
- `reporting/plot_learning_compact.py`: Figure 6, original seedwise gains in
  the separate 25-step static-front learning-rule comparison.
- `reporting/plot_source_spatial.py`: Figure 7, original source-core DEV case
  and fixed contact-at-step-50 cohorts followed over steps 51-75.

The spatial cases show the central [32:96]^3 interior of the fixed parent
volume. Errors use prediction minus binary target on the same initial-oil
surface. Camera and signed colour limits are shared. The source example was
selected by original target mobilization before inference; it is an informative
DEV case, not a representative TEST estimate. External examples are fixed by
core, block, action order and seed, not by prediction accuracy.
