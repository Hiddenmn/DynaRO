DynaRO initial labels, fixed partitions and simulator inputs
Local release candidate - unpublished, licence and final sharing scope pending

This component contains the 80 reviewed semantic initial states used in the
study, not newly segmented labels. Each label file is an unchanged uint8 NumPy
array of shape [128,128,128] in (z,y,x) order. The 20 crop coordinates are shared
by four source stages of one carbonate core: Oil, WI, NPI-006 and NPI-012. They
are not 80 independent cores or time-registered simulator trajectories. Every
stage version of a spatial block stays in the same partition: 40 initial states
in TRAIN, 20 in DEV and 20 in TEST (10/5/5 spatial blocks).

Files and interpretation

- labels/<stage>/*.npy: exact original stage-block labels, including background
  and rock. None are trimmed to oil-only volumes or changed for appearance.
- label_manifest.json: file identity, crop origin, split, voxel counts and
  array hash. Coordinates are integer model-grid indices, not physical lengths.
- spatial_block_split.csv: the original frozen 20-block allocation, unchanged.
- label_schema.json: the class dictionary, array conventions and source context.
- action_protocol.json: the five actions and canonical 25-step paired schedules.
- trajectory_plans/: one line per existing trajectory identity. These metadata
  describe the original panels; their binary trajectory arrays are not included.
  Machine-specific source/output paths and scalar outcome summaries are omitted.
- src/patent22/: preserved project actions, source adapter, generator and split
  helper. The generator/split files are current snapshots, not inferred
  execution-time versions; their hashes are recorded. The new package initializer
  does not import the historical development neural model.
- src/simulator_core.py: numerical functions and constants extracted without
  alteration from the original displacement simulator. Its original whole-file
  hash and per-definition extraction record are retained in the manifest.
  Plotting, workstation-search and command-line experiment orchestration are
  excluded. SciPy connectivity is required, as in the canonical generation.
- src/long_horizon_schedules.py: original schedule definitions, unchanged.
- reproduce_inputs.py: default read-only generation plan; an explicit flag and
  a new separate output are required to generate trajectories. No generator runs
  merely by importing the modules or verifying this folder.

Labels and provenance

Class IDs are 0 background, 1 rock, 2 water, 3 film oil, 4 corner oil, 5 droplet
oil, 6 columnar oil and 7 cluster oil. The project-generated semantic slices were
expert reviewed. Expert counts and agreement statistics were not supplied.
Oil classes retain their operational, slice-derived meaning. The four stages
provided distinct initial conditions for simulator rollouts; no temporal
registration or independent physical displacement validation is implied.

The upstream micro-CT collection is DOI 10.6084/m9.figshare.c.4190738; its dataset
descriptor is DOI 10.1038/sdata.2019.4. These upstream records are not the source
of this study's eight-class annotations. Upstream grayscale images are not
mirrored. The parent label stacks used 845 ordered slices, resized in-plane
to 512 x 512 by nearest-neighbour interpolation before these crops. The present
component starts from those saved crops; it does not recreate annotations from
the public grayscale scans. Original full stacks and display caches are separate
from these model inputs. No absolute voxel-size calibration is assigned here.

Schedules and randomness

TRAIN25/DEV25/TEST25 use A*10 -> B*15 and B*15 -> A*10 for each of ten unordered
action pairs. TRAIN50 uses A*20 -> B*30 and B*30 -> A*20. TEST100 retains each
50-step prefix and holds its last action to step 100; TEST50 and TEST75 are
prefix views. The two TEST schedules are no longer equal-dose beyond step 50,
so their later contrast is a schedule response, not a pure equal-dose order effect.

Simulator random streams are not model-training seeds. The literal streams
P1-22-DEV-A/B occur in both TRAIN and DEV; partition membership is determined
by spatial blocks, not by this historical token. TEST uses P1-22-TEST-A/B/C.
At each step, seed = uint64_big_endian(SHA256(stream|block|stage|step)[:8]) mod
2^32. Both action orders share that random field. Inlet axis is x and base
transition probability is 0.06. The class-difficulty and action-proxy rules
are fixed simulator assumptions, not fitted physical displacement constants;
simulator steps are not calibrated physical time.

Verify and use

The tested local preparation environment is Python 3.12.13, NumPy 2.4.6 and
SciPy 1.17.1. This is not a claim that it reproduces historical trajectories
bitwise. From this folder:

  python -I -B verify_inputs.py
  python -I -B reproduce_inputs.py --set train25

The first command verifies every original label byte hash and array schema plus
the fixed metadata. The second only describes the requested generation panel.
Generating trajectories is additional numerical work; consult --help and the
run's fixed plan before explicitly enabling it. Local preparation did not run
any simulator on these labels, train a model or recompute TEST scores.

For a separately authorized generation run, use --execute --output with a new
directory outside this component. --limit selects only the fixed first N rows;
it is not a complete benchmark set and must not be reported as one. A full set
can require substantial CPU time and storage. Outputs contain data/*.npz, the
generation plan, an append-only per-file ledger, and a completion summary.
Interrupted outputs are retained as partial work; this entry does not resume
or replace them. Transfer the finished arrays to the appropriate training or
evaluation input layout separately. Generation alone does not evaluate a model.

An original_npz_sha256 identifies an original compressed container. Regenerated
container bytes can depend on serialization; assess numerical identity from
the stored arrays, not merely the container name. TRAIN50's historical manifest
hashed its sorted filenames, not their contents; no invented payload hash is
given for those files. All original records/checkpoints remain untouched.

This is not the entire public reproduction release. Frozen weights/model code,
full training/evaluation entry points, figure-generation inputs, author-approved
scope/licence and a versioned access route are separate components. Do not call
the data public before the approved assets are actually accessible.
