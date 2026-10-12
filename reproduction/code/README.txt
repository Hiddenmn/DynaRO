DynaRO local release candidate

Contents currently assembled

model_core/  The five manuscript model architectures, state/readout/objective
             code, explicit architecture configs and synthetic CPU checks.
source_data/ All seven tables and the numerical panels of six figures, with
             28 CSV files, panel mapping, units and a standalone verifier.

This directory is a staged subset, not the complete planned reproduction
release and not a public deposit. Model core sources and numeric CSV files are
copied byte for byte from the retained study material. The top-level manifest
records file integrity without private workstation paths or server details.

Run the standard-library integrity/privacy/import-closure check:
    python -I -S -B verify_candidate.py

Verify the numeric source data separately:
    python -I -S -B source_data/verify_source_data.py --root source_data

For the synthetic model checks, see model_core/README.txt. No training, TEST,
server operation or repository upload occurs during candidate assembly.

Still required for the complete release

- Reviewed author labels and their class/stage/crop/split metadata.
- Simulator/action definitions and deterministic trajectory-generation routes.
- All final weights and complete TRAIN/evaluation protocols, including the
  separate 25-step static-front learning-rule comparison.
- All eight figure-generation routes and their approved image/volume inputs.
- Author-confirmed sharing scope and licence, attribution/creator information,
  a versioned public release identifier and suitable reviewer access.

Historical source snapshots, private preparation reports and server-management
notebooks are intentionally not included. The figure-tooling helper is not in
this subset. Do not infer permission to distribute missing data/helpers, or
invent a licence, because the source code is present locally.

Upstream image collection: https://doi.org/10.6084/m9.figshare.c.4190738
Dataset descriptor: https://doi.org/10.1038/sdata.2019.4
Original images are not mirrored. The eight-class semantic labels are a separate
author-generated output. The data/code availability statement must describe
the actual approved release, not this unfinished candidate.
