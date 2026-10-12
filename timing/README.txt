DynaRO inference-timing addendum — local preparation, 13 October 2026

This addendum supports Results 4.1 and Supplementary Results S8/Table S5.
It is prepared locally and has not been published. The existing 9 October
reproduction package is preserved as a separate snapshot.

Measurement

Four frozen primary predictors (seed 20260811), five NPI-012 DEV volumes,
128 x 128 x 128 inputs, batch size one, float32 and horizons 25, 50 and 100.
The schedule uses S2 for the first 20 steps, then S3. Each combination has
two warm-up calls and five measured calls. Labels and weights are already
on the GPU, and CUDA synchronization brackets each wall-clock measurement.

The recorded environment is Windows 11, NVIDIA RTX 5070, PyTorch
2.11.0+cu128, NumPy 2.4.6 and SciPy 1.17.1. This is the inference-timing
environment, not the training environment. Complete settings are in timing.json.

Latency is the equal-weight mean of five block-specific median latencies.
Memory is the largest peak allocated PyTorch CUDA memory across the measured
calls and inputs. The timed region is the native model forward calculation,
including all-step initial-oil-site probabilities and cell/class inventories.
Input reading, connectivity preprocessing, transfers, weight loading, target
scoring and output-file writing are excluded. Repetition ranges describe
runtime variability, not independent experimental replication.

Files

timing.json                  Complete recorded benchmark: 60 combinations,
                             300 measured calls, input and weight hashes.
results/summary.csv          Twelve model/horizon summaries.
results/timed_repetitions.csv All 300 measured calls.
results/comparison.json      Ratios computed from unrounded summaries.
results/table_rows.tex       The four rows displayed in Table S5.
benchmark_inference.py       Original benchmark implementation, unchanged.
summarize_timing.py          Rebuild the small summaries from timing.json.
test_benchmark_inference.py  Original protocol and aggregation checks.
manifest.json               File hashes for this local addendum.

Rebuild summaries without loading a model or using a GPU:

  python -B summarize_timing.py timing.json --output rebuilt_results

The output directory must not already exist. This command needs only the
Python standard library. It does not repeat the measurements.

The optional benchmark implementation requires the separately prepared full
reproduction package. Without --run, it prints the input plan only:

  python -B benchmark_inference.py --reproduction PATH_TO_REPRODUCTION --phase benchmark

Repeating the measurement requires explicitly adding --run and --output with
a fresh directory, plus a CUDA environment and the approved weights and inputs.
No measurement or scientific experiment was repeated while preparing this copy.

Publication preparation

No raw CT, passwords, server configuration or private workstation paths are
included. The proposed licence division remains MIT for author-original code
and CC BY 4.0 for author-generated results. Final attribution and publication
are pending with the parent release; this addendum has not been silently
merged into its previously approved file manifest.
