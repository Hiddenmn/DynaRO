Evaluation of new retraining outputs
===================================

This separate entry connects the prepared training driver to the same scoring
functions as the frozen-original evaluation. It does not retrain, select a
checkpoint, overwrite an original result or claim a new result is historical.
The local candidate is not published or licensed.

Without --execute, the command only checks metadata and file bytes. It imports
no numerical libraries, deserializes no model/trajectory arrays, allocates no
GPU and writes no result. A --run may be omitted for a configuration-only plan:

  python evaluate_retrained.py --core CORE --weights WEIGHTS --inputs INPUTS \
    --training TRAINING_COMPONENT --evaluation EVALUATION_COMPONENT \
    --model-key mixed_horizon/dynaro/20260811

CORE is the model_core directory, not its parent release-candidate directory.
WEIGHTS is the frozen-original tensor component: its metadata defines the
reference architecture and configuration. Its learned tensors are NOT loaded
by this route. Add --run COMPLETED_RETRAINING_DIRECTORY to inspect a new final
checkpoint produced by train.py. That directory must contain matching context,
completion summary and retrained_final.pt. Recovery/partial checkpoints and
changed training settings are refused. Original versus regenerated training
and evaluation trajectories retain separate provenance.

Numerical TEST evaluation is a separate explicit action. It also needs
--test25 TEST25_DIRECTORY, --test100 TEST100_DIRECTORY for mixed-horizon models
(omit --test100 for learning_rule_25), --execute, a new --output directory and
a positive --max-seconds. The recorded Linux/CUDA/RTX4090 requirement is retained;
there is no CPU fallback or server lifecycle operation. The limit is checked
between complete pairs. See the evaluation component for STOP_REQUESTED and
explicit same-context --resume controls.

The model is loaded from retrained_final.pt with restricted CPU deserialization,
finite-tensor checks, exact architecture/dtype matching and the final parameter
identity from its training summary. The original score functions, intervals,
inference decorator, pair order, equal-five-block aggregation and absent-class
handling are unchanged. The saved evaluation context and completion status
explicitly identify NEW retraining. No summary is labelled a frozen original
result and no performance is inferred from training completion.

Validation of this component uses source/metadata checks and invented software
fixtures only. No completed research retraining output was produced or scored
while preparing it. An available command and successful unit tests do not
establish full numerical CUDA reproducibility or paper-performance agreement.
