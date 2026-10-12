DynaRO frozen-checkpoint evaluation
Local component, unpublished; final release scope and licence pending

This entry connects the separate model-core, tensor-only weight and initial-
label/simulator components. It covers all 28 saved final checkpoints: fifteen
mixed-horizon predictors (five models, three seeds) and thirteen static-front
25-step learning-rule models. It does not train, tune or select a checkpoint.

The numerical scoring functions, tensor readers, interval definitions and
block aggregation are extracted unchanged from the preserved experiment
sources, including their decorators. The manifest records the original file
hashes and selected definition hashes. New code changes file organization and
execution control, not the score formulas. The 25-step learning-rule study
uses t/25; the mixed-horizon models use t/(t+25). TEST50 is the prefix of the
same TEST100 file, and 75/100 metrics retain their original post-50 intervals.

Inputs

--core: directory containing the prepared model_configs.json and src/.
--weights: directory containing all 28 tensor-only weights and their manifest.
--inputs: directory containing the 80 labels and fixed trajectory plans.
--model-key: one exact key from the weights manifest/protocol.json; no default
             best seed or automatically preferred checkpoint is selected.

From this directory, a plan needs only standard-library Python:

  python -I -B evaluate.py --core ../code/model_core --weights ../weights --inputs ../inputs --model-key mixed_horizon/dynaro/20260811

The paths above are example component placements. The plan opens text metadata
and hashes component files, but does not deserialize labels/weights/trajectory
arrays, import the model or initialize CUDA. It reports absent trajectory
payloads when their directories have not been provided. Use
learning_rule_25/state_prioritized/20260811 for the separate 25-step study;
the PCGrad comparison has only seed 20260811.

Supply --test25 and, for mixed-horizon models, --test100 to check that the full
1,200-file input sets are present. For --data-kind original (default), these
arguments name directories containing the original NPZ files. Each payload
hash is checked before its pair is scored. Recompressed or replaced containers
are not silently treated as the original frozen files.

For independently regenerated data, explicitly choose --data-kind regenerated
and point these arguments to complete batches made by reproduce_inputs.py.
The entry requires their completed generation summaries and hash ledgers,
the same label manifest, schedules and identities. Such a run is recorded as
regenerated-data evaluation; generation alone does not prove historical
numerical identity or authorize replacing the paper's original results.

Numerical execution is separate work. It requires --execute, a new --output
outside all input components, and positive --max-seconds. The limit is soft:
the current complete pair is saved before stopping. A STOP_REQUESTED file in
the output also stops at the next pair boundary. --resume accepts only this
entry's incomplete output with unchanged context and a valid ordered prefix;
completed results and interrupted partial JSON lines are never overwritten.
An existing RUNNING.lock requires checking its process before manual recovery.

Execution retains Linux, CUDA-enabled PyTorch, RTX 4090, float32, no TF32 and
two CPU threads. It never falls back silently to CPU TEST. Memory caps retain
the recorded per-study limits. The CPU verification environment is Python
3.12.14, PyTorch 2.8.0+cpu and NumPy 2.3.2; this is not a tested CUDA environment
or a claim of end-to-end score reproduction through this new entry.

Each command evaluates one explicitly chosen checkpoint on all 600 pairs.
Outputs are context.json, append-only per_pair.jsonl and summary.json.
Aggregation includes all five spatial blocks with equal weight, including the
zero-error block. Values remain unscaled fractions; the paper's 10^-3 display
factor and inventory percentage-point conversion are reporting operations.
Repeat the command for the prescribed model/seed keys when an evaluation is
separately authorized; do not select a better run by its TEST score.

This is the evaluation component, not the whole release. Training entries,
figure-generation routes, full release integration, author metadata, rights
and licence/access decisions are separate work. Local assembly and synthetic
checks do not run any of these original-data model evaluations or upload files.
