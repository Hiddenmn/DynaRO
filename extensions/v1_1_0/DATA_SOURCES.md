# Data sources and transformations

Original carbonate-core data: Pak et al., dataset collection
https://doi.org/10.6084/m9.figshare.c.4190738 (CC BY 4.0), described in
https://doi.org/10.1038/sdata.2019.4. The original DynaRO labels, splits,
weights and study results are retained in v1.0.0.

Ketton / WF1: Scanziani, A., Singh, K. and Blunt, M. (2018), Water-wet
three-phase flow micro-CT tomograms, Digital Rocks Portal,
https://doi.org/10.17612/P7HT11.

Bentheimer Core 1: Jackson, S., Lin, Q. and Krevor, S. (2020), A large scale
X-ray micro-tomography dataset of steady-state multiphase flow,
Digital Rocks Portal, https://doi.org/10.17612/KT0B-SZ28.

The two external source records specify Open Data Commons Attribution 1.0
(https://opendatacommons.org/licenses/by/1-0/). Cite their original records when
reusing the derived annotations. The DynaRO authors' eight-class annotations
and simulated futures are new outputs. The eight-class annotations were
reviewed by experts.

Native PNG coordinates are unchanged. `external/native_labels.jsonl` maps
each PNG to the original raw slice filename, index, dimensions and source DOI.
No offsets from older four-phase segmentations are used. For each model input,
128 consecutive slices and a 256 by 256 in-plane window are selected at the
listed origin. Sampling uses y0+1+2j and x0+1+2i, producing a 128-cubed grid;
there is no axial resampling, rotation, class merging or boundary cleanup.
Coordinates are voxel indices. Digital-grid transfer is not a physical
length rescaling. Original CT pixels are cited at source and not mirrored.

Displacement action sequences, noise streams and crop selection are in
`external/protocol.json`. Pair members share each step's random field;
`code/external_numerics.py` preserves the seed derivation. The same generator
and the same five source-trained model families are used for both cores.
