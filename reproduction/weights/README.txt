DynaRO frozen model weights - local release candidate

This folder contains 28 inference-state dictionaries extracted from the final
checkpoints used in the manuscript comparisons. It is not published or licensed.
The original checkpoints are preserved separately and are never overwritten.

Contents

- mixed_horizon/: DynaRO, Static front, Regional-mean exposure, FNO and RTU,
  each at seeds 20260811, 20260812 and 20260813; final update 400, TRAIN25/50.
- learning_rule_25/: the static-front predictor trained with response weight 0,
  fixed weight 50, Du-style cosine gate and state-prioritized learning, each at
  the three seeds; PCGrad at seed 20260811 only; final update 200, TRAIN25.
  These thirteen models are not the mixed-horizon dynamic-contact DynaRO model.
- manifest.json: model, study, seed, training horizon, final update, architecture,
  original checkpoint/parameter identities and exported tensor/file hashes.
- load_weights.py: explicit CPU-only loading and verification. No predictions,
  training, TEST evaluation, optimizer updates or checkpoint selection occur.

Use the model_core directory from the companion local code candidate and its
requirements-cpu.txt. Run from this folder:

  python -B load_weights.py --core ../DynaRO_release_candidate_20261009/model_core

In Python, with this folder on the import path:

  from load_weights import load_frozen
  model, metadata = load_frozen(core_path, 'mixed_horizon/dynaro/20260811')

The model is on CPU, in evaluation mode, with gradients disabled. Select a key
explicitly; there is no best-seed or TEST-dependent automatic selection. Other
model keys are listed in manifest.json. The caller must provide correctly
encoded input geometry and the measure sequence before making predictions.

File meaning and verification

Each .pt file is a plain mapping from model-state names to dense CPU tensors.
Names, shapes, dtypes and values are preserved exactly. Training-only optimizer
states, RNG states, logs, embedded machine paths and training configuration
objects are excluded; the concise architecture/study metadata are in the JSON.
Use torch.load(..., map_location='cpu', weights_only=True). The exported archive
bytes differ from the original checkpoint, but every exported tensor is checked
against it, and strict loading must reproduce the recorded parameter hash.

These files support frozen-model inference, not exact training resume. The
earlier Windows CPU versus historical Linux initialization mismatch for FNO/RTU
does not change frozen weights, but remains unresolved for retraining. Successful
loading is not a rerun of the paper's scores or evidence of cross-platform
bitwise prediction agreement. Labels, simulator/generation routes, complete
training/evaluation entries and figure inputs remain separate release components.

Licence, final file scope, release identifier and reviewer-access arrangements
require author confirmation before publication. No licence is assigned here.
