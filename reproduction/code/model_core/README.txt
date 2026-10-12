DynaRO model core

This directory contains the unchanged numerical model, readout, objective and
state-prioritized gradient definitions used by the study. The small src/dynaro.py
entry selects the recorded five-model comparison configurations explicitly.
It constructs untrained models; it does not load weights, train, evaluate a
dataset or operate a server. This local candidate is not the complete release.

Model mapping

DynaRO                  RegionalStateFrontSurrogate, direct readout, 90991 parameters
Static front            RegionalActionSurrogate, direct readout, 90991 parameters
Regional-mean exposure  CoarseStateFrontRemainingOil, 90991 parameters
FNO adaptation          width 16, four layers, modes 8/8/8, 4196289 parameters
RTU adaptation          base width 16, four heads, two transformer layers, 2103565 parameters

The low-level files retain historical names. StateGuard in those files names
the state-prioritized learning rule; it is not synonymous with a dynamic-contact
architecture. The separate 25-step learning-rule experiment uses Static front,
not DynaRO. The original FNO factory defaults include a width-24 candidate:
use src/dynaro.py for the width-16 manuscript comparison configuration.
FNO and RTU are paper-guided task adaptations, not the cited authors' downloaded
implementations or checkpoints. Their originating papers are linked in source.

Input interface

One initial semantic volume [Z,Y,X], class codes 0..7; inlet and inlet-connected
pore masks of the same shape (bool); action vectors [T,4] and encoded step times
[T,1] (float32). For the reported study times are t/(t+25), t=1..T. All tensors
must be on the same device as the model. The masks describe the initial geometry,
not a future target. This core directory does not recreate the reviewed labels,
geometry masks, action definitions, simulator or split panel; those belong to
the full data/protocol release.

Class dictionary: 0 background; 1 rock; 2 water; 3 film oil; 4 corner oil;
5 droplet oil; 6 columnar oil; 7 cluster oil. These labels are author-generated,
not supplied by the cited upstream image collection.

With src on the Python import path:
    from dynaro import make_model
    model = make_model("dynaro").eval()
    output = model(labels, inlet, connected, actions, times, return_fields=True)

The returned So_sequence has shape [T,Z,Y,X]. Compact mode returns state values
on the original oil support and group inventories, avoiding full-field output.
Remaining-oil probabilities stay in [0,1] and do not create oil outside the
initial oil support. These are model state constraints, not validation of a
full multiphase-flow solver or calibrated physical time.

CPU software checks

Use Python 3.12 and the CPU build of PyTorch 2.8.0. Install the CPU wheel as
shown in requirements-cpu.txt and install numpy 2.3.2 separately. Then run:

    python -I -B run_checks.py

This creates synthetic in-memory 32-cubed tensors, checks all five full-width
models, and runs the original seven synthetic model/gradient tests. It blocks
research-array/checkpoint reads and CUDA initialization. It performs no optimizer
update and does not generate new experimental evidence. The --forward-only
option omits the seven gradient/unit tests; it must not be reported as a full pass.

The constructors use caller-controlled random seeds. A new CPU initialization
is not promised to match a historical CUDA initialization byte for byte. In
particular, earlier cross-platform FNO/RTU initialization checks did not match
the saved Linux initial hashes. Existing trained results are unchanged. Full
experimental reproduction needs the recorded protocols, inputs and checkpoints
and a separate authorized CUDA run; the small synthetic checker cannot prove it.

Distribution status

This is a local, unpublished candidate. No software licence is assigned here.
The exact public files, licence, weights/labels and versioned release citation
require author approval. Do not publish this candidate as a complete benchmark.
