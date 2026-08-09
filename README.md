# MRI/PET GeoMC pipeline

This repository contains one MRI/PET foundation-model pipeline:

\[
\boxed{
\text{MRI/PET observations}
\rightarrow
\text{shared SAT3D encoder}
\rightarrow
\text{coordinate interpolation}
\rightarrow
\text{reliability-weighted node feature aggregation}
\rightarrow
\text{geometry-guided cross-scale routing}
\rightarrow
\text{latent node field}
\rightarrow
\text{query decoder}
}
\]

The public method name is `shared_sat3d_geomc`, and the maintained experiment
is `geomc`. Historical model generations are not retained as selectable
branches in this clean repository.

The bounded research question is:

> Can one shared encoder learn useful MRI/PET representations while
> geometry-guided cross-scale routing improves how image features interact over
> the brain domain, without forcing the learned representation into a narrow
> spectral subspace?

The implementation is empirical-first. The finite-element geometry supplies a
fixed spatial coordinate system and multiscale operators. The neural network
still learns the feature representation, routing matrix, nonlinear updates,
and query prediction.

## 1. Scope

The current package implements:

- structural MRI and static PET observations;
- one shared trainable SAT3D image encoder;
- MRI-versus-PET observation metadata supplied to the shared token projector
  after SAT3D encoding, and target metadata supplied later to the query decoder;
- reliability-weighted aggregation of all available input observations;
- a fixed volumetric finite-element (FEM) brain domain;
- a numerically solved generalized eigenproblem;
- smooth resolvent bands and the exact orthogonal complement of the retained
  eigenspace;
- an unconstrained, input-dependent cross-scale routing matrix;
- one shared nonlinear update network for every scale;
- one query decoder;
- an exponential-moving-average (EMA) target encoder and masked Huber loss;
- within-modality, verified cross-modal, and multi-observation training
  examples;
- CPU and CUDA implementation tests, resumable server execution, and
  subject-level evaluation.

The package does **not** claim:

- that unpaired MRI and PET identify subject-specific cross-modal coupling;
- that the routing matrix is a literal connectome or neural wave;
- that a 30-subject single-seed run establishes a foundation model;
- that a fixed template FEM represents subject-specific cortical physics;
- that unknown scanner, sequence, tracer, dose, reconstruction, or PSF values
  can be inferred from an availability indicator;
- that a latent-prediction score alone guarantees downstream transfer.

## 2. End-to-end method

### 2.1 Observation records

Each available acquisition is stored as an independent `Observation` record.
The record binds:

- the model-ready image and brain-mask paths;
- subject, session, dataset, and acquisition identifiers;
- MRI or PET modality;
- frozen train, validation, or test split;
- source and preprocessing provenance;
- observation reliability;
- any acquisition metadata that was actually recorded, together with an
  explicit availability flag.

Unknown metadata remains unknown. It is not filled with a plausible value and
is not used as a silent data-selection rule.

The current neural input metadata is deliberately small. It contains MRI/PET
identity, observation reliability, and one acquisition-metadata-availability
indicator. Specific MRI sequence, PET tracer, scanner, dose, reconstruction,
and PSF values may be preserved in the observation manifest, but the current
network does **not** consume those values. Therefore this implementation must
not be described as tracer-specific or acquisition-parameter-conditioned.

### 2.2 Masking before the online encoder

The online encoder receives a single image channel in which hidden/query voxels
are zeroed before SAT3D. Normalization statistics are recomputed from visible
in-mask voxels only, so hidden intensities cannot affect normalization.

The visibility mask is not concatenated as a second SAT3D channel. Instead,
visible support is summarized as token coverage and carried separately with
token reliability into coordinate interpolation and node-feature aggregation.
This keeps masked support explicit downstream without changing SAT3D's
single-channel image input.

Masking only after feature extraction would be invalid: the encoder receptive
field could already have transferred hidden target information into nearby
visible tokens.

### 2.3 Shared SAT3D feature encoder

For observation \(o\), let \(Y_o\) be the model-ready image, \(M_o\) the
visible in-brain voxel mask, and \(m_o\) the small observation-metadata vector.
Masking and visible-only normalization happen before SAT3D:

\[
\widetilde Y_o=\mathcal N_{M_o}(M_o\odot Y_o),
\qquad
H_o=E_\theta(\widetilde Y_o),
\qquad
F_o=P_\theta(H_o,m_o).
\]

Here \(\mathcal N_{M_o}\) recomputes intensity statistics from visible in-mask
voxels, \(E_\theta\) is the shared SAT3D encoder, \(H_o\) is its feature map,
and \(P_\theta\) is the shared `TokenProjector`. Observation metadata does not
enter SAT3D; it is used only by the projector after encoding. MRI and PET share
both sets of parameters rather than selecting separate backbones. Target
metadata is used later by the `QueryDecoder` and is not passed into SAT3D or
the GeoMC core.

The server configuration initializes SAT3D from the audited external checkpoint
and trains the configured final stage. Synthetic tests use a small encoder so
the complete implementation can be checked without external assets.

The trainable SAT3D image encoder is called directly. A wrapper that forces
evaluation mode or applies `no_grad` is not a valid training path.

### 2.4 Coordinate-based token/node interpolation

SAT3D features lie on a regular token grid, whereas GeoMC operates on FEM
nodes. Fixed local interpolation maps token features to FEM nodes using their
physical coordinates:

\[
X_o = T_{\mathrm{token}\rightarrow\mathrm{node}}F_o.
\]

The interpolation uses nearby tokens, a distance kernel, token volume, and
visible-token coverage. Its weights are normalized to preserve a constant
feature where input support exists. A node with no input support receives zero.
The interpolation weights are fixed, but gradients still flow through the
interpolated feature values to SAT3D.

For the available input observations \(\mathcal O\), the node features are
combined with coverage and reliability weights:

\[
X_{\mathrm{in}}(n)=
\frac{
\sum_{o\in\mathcal O} w_o(n)X_o(n)
}{
\sum_{o\in\mathcal O}w_o(n)+\epsilon
}.
\]

This weighted mean is permutation invariant: reordering the same observations
does not change the aggregated node features.

### 2.5 Volumetric FEM brain domain

The real pipeline creates a fixed volumetric tetrahedral FEM from the registered
model mask. Let \(L\) denote the stiffness matrix and \(M\) the mass matrix. The
geometry stage solves

\[
L\psi_k=\lambda_kM\psi_k,
\qquad
\Psi^\top M\Psi=I.
\]

This is an actual numerical generalized eigensolve. The eigenvectors are not
learned positional embeddings, and the mesh is not rebuilt from feature
similarity during training.

This template FEM is a practical coordinate system for feasibility testing. A
result on this volumetric domain is not proof of a cortical-surface or
subject-specific neurophysical mechanism.

### 2.6 Resolvent frame with exact complement

For radius \(r\), the geometry-derived resolvent response is

\[
g_r(\lambda)=\frac{1}{1+r^2\lambda}.
\]

Several radii define smooth overlapping spectral bands. Their nonnegative
weights satisfy

\[
\sum_b s_b(\lambda)=1.
\]

The implementation uses square-root Parseval factors

\[
a_b(\lambda)=\sqrt{s_b(\lambda)}
\]

and includes the exact orthogonal complement outside the retained eigenspace.
Consequently,

\[
\sum_b A_b^\ast A_b=I,
\]

up to floating-point error. For a latent node field \(X\), the frame components
are

\[
Z_b=A_bX.
\]

The representation is therefore not projected permanently into a small
low-frequency basis. The exact complement and the full
non-spectrally-decomposed normalized input field remain available when retained
eigenmodes are insufficient.

### 2.7 Input-dependent cross-scale routing

At every FEM node and GeoMC block, the router receives ordinary learned
features derived from:

- the current node state;
- the aggregated input features at that node;
- mass-weighted global summaries of both;
- the relative energy of each frame component;
- component-wise similarity between state and input features.

It predicts a complete signed routing matrix

\[
R_n=I+\Delta R_n,
\qquad
\Delta R_n\in\mathbb R^{S\times S}.
\]

The final routing layer starts at zero, so the initial matrix is the identity.
During training, \(R_n\) is not forced to be positive, stochastic, orthogonal,
low-rank, contractive, or energy-decreasing.

The routed component and cross-scale update are

\[
\widetilde Z_{b,n}=\sum_jR_{n,bj}Z_{j,n},
\qquad
U_{b,n}=\widetilde Z_{b,n}-Z_{b,n}.
\]

One update network is reused for every frame component. It receives the current
component, cross-scale update, corresponding input component, the full
non-spectrally-decomposed normalized input field, and a learned scale embedding.
After applying the matching frame operator, the block returns one residual
update:

\[
X^{\ell+1}
=
X^\ell
+
\alpha_\ell
\sum_b A_b^\ast
F_\theta(
Z_b^\ell,
U_b^\ell,
X_{\mathrm{in},b},
X_{\mathrm{in}},
e_b
).
\]

Here \(X_{\mathrm{in}}\) is the full normalized input node field and
\(X_{\mathrm{in},b}=A_bX_{\mathrm{in}}\) is its component at scale \(b\).

This is one shared computation. There are no separate MRI, PET,
low-frequency, high-frequency, or biology branches.

### 2.8 Geometry embedding

Basis-invariant FEM descriptors are projected into the latent dimension and
added before the GeoMC blocks. With `geometry_embedding.type: spectral`, as in
the maintained configurations, these descriptors use spectral operator
diagonals and relative node mass. The optional `spectral_xyz` type additionally
includes template coordinates, while `none` disables this embedding. The
spectral descriptors do not expose arbitrary eigenvector columns or signs.

This component is an additive geometry embedding, not a probabilistic prior and
not a replacement for image-derived features. Its learned scale and its RMS
relative to the aggregated input features are reported so excessive reliance
on the fixed template can be detected.

### 2.9 Query decoder

After the latent node field is updated, fixed coordinate interpolation maps FEM
node features back to the SAT3D token locations. An MLP query decoder predicts
the target latent features. Target MRI/PET metadata is supplied to this decoder
but not to SAT3D or the GeoMC latent field; target intensity never enters the
online model.

For a fixed set of visible input observations, the latent node field is formed
before query decoding. Different target queries therefore use the same shared
subject representation rather than separate modality-specific cores.

### 2.10 EMA target and pretraining loss

The target encoder is an EMA copy of the online encoder. For selected query
tokens \(Q_T\), it produces stopped-gradient target features \(z_q\). The
online model predicts \(\widehat z_q\) from the allowed visible inputs.

The training objective is the reliability-weighted masked Huber loss

\[
\mathcal L
=
\frac{
\sum_{q\in Q_T}w_q\,\rho_{\mathrm{Huber}}(\widehat z_q-z_q)
}{
\sum_{q\in Q_T}w_q+\epsilon
}.
\]

There is no auxiliary generation, reconstruction-shape, or task-specific loss
in the maintained pretraining objective.

## 3. Paired and unpaired observations

All training examples use the same encoder, node-feature aggregation, GeoMC
blocks, query decoder, and loss. Data availability changes which source-target
relations can be sampled; it does not switch architectures.

| Available observations | Valid source-to-target relation | Information identified |
|---|---|---|
| MRI only | visible MRI to hidden MRI features | MRI marginal structure |
| PET only | visible PET to hidden PET features | PET marginal structure |
| Verified MRI/PET pair | MRI to PET or PET to MRI | subject-level cross-modal conditional information |
| Verified pair with both partly visible | multiple observations to hidden target queries | multi-observation fusion |
| Missing observation | omit that observation from the inputs | missing-modality robustness |
| Missing metadata | set its availability indicator to zero | robustness to metadata gaps |

Unpaired observations update the shared encoder, aggregation, geometry-guided
routing, and modality marginals. They do not reveal which PET volume belongs to
a particular MRI subject. Only verified links identify subject-level
cross-modal relations. The pipeline does not create pseudo-pairs to hide this
identifiability boundary.

The included server configuration uses a frozen 30-subject paired cohort only
as a bounded model test. It does not test the intended large-scale,
mostly-unpaired pretraining setting.

## 4. Mathematical constraints and learning capacity

The implementation strictly enforces only properties needed for correct data
handling or well-defined numerical operators:

- physical-coordinate mapping;
- the FEM generalized eigenproblem and mass-orthonormal eigenmodes;
- Parseval frame closure and the exact complement;
- provenance, masking, and target-leakage prevention;
- subject-level evaluation and train-only fitting of any latent-space
  alignment.

The learned representation is **not** required to obey:

- a fixed resolvent response at every layer;
- hard low-frequency truncation;
- positive or conservative routing weights;
- a low-rank state;
- energy-decreasing updates;
- a fixed-point or recurrent-dynamics equation;
- a literal biological meaning for each latent channel.

Geometry supplies a specific multiscale routing mechanism, while the network
retains capacity to ignore, reverse, or locally modify that routing when image
features require it.
## 5. Configuration

Three configurations are maintained:

- [`configs/smoke.yaml`](configs/smoke.yaml): deterministic CPU implementation
  test with synthetic inputs and a small encoder;
- [`configs/local_cuda_smoke.yaml`](configs/local_cuda_smoke.yaml): the same
  synthetic implementation test on CUDA with BF16;
- [`configs/server.yaml`](configs/server.yaml): the bounded 30-subject SAT3D
  server run.

The public configuration uses direct names. For example:

```yaml
model:
  latent_dim: 32
  metadata_dim: 4
  geomc_hidden_dim: 64
  geomc_scale_embedding_dim: 8
  geomc_residual_scale: 0.10
  geomc_blocks: 3

geometry_embedding:
  type: spectral
  hidden_dim: 32
  initial_scale: 0.05
  trainable_scale: true

inputs:
  max_observations: 2
  include_target_observation_probability: 0.50
  aggregation: reliability_weighted_mean

training:
  relation_sampling_weights:
    mri_to_mri: 0.35
    pet_to_pet: 0.35
    mri_to_pet: 0.15
    pet_to_mri: 0.15

experiments:
  - id: geomc
    verified_pair_retention_fraction: 1.0
    pairing_mode: verified
    seeds: [0]
```

Only one model implementation is selectable. Terms such as frame
factorization, routing parameterization, and interpolation method are recorded
in the model specification and reports; they are not presented as artificial
experiment choices when no alternative implementation exists.

`evaluation.comparisons` is empty in the maintained configuration. Scientific
controls should be introduced later as explicit matched experiments rather than
hidden historical branches.

## 6. Pipeline stages

`full` executes the complete plan:

| Stage | Purpose |
|---|---|
| `00.preflight` | resolve configuration, paths, software, and resource facts |
| `01.observations` | build observation records, verified links, and frozen splits |
| `02.asset_snapshot` | bind external inputs by content hash |
| `03.registration_qc` | compute automatic nonblocking registration diagnostics |
| `04.geometry` | build the FEM, solve the eigenproblem, and construct the resolvent frame |
| `05.model_check` | verify shapes, frame dimensions, forward propagation, and gradients |
| `06.end_to_end_smoke_test` | run one real forward/backward step and record device, memory, and timing |
| `07.small_sample_overfit_check` | test whether the model can fit a very small training subset |
| `08.experiment.geomc.seed0` | train the maintained experiment |
| `09.evaluate` | run subject-level evaluation in the checkpoint and fixed-reference latent spaces |
| `10.report` | summarize execution, results, limitations, and diagnostics |

Scientific thresholds are report-only. A weak QC result, smoke-test metric, or
negative model result is scientific evidence to interpret; it is not a reason
to skip unrelated requested computations. Only an execution error that makes a
dependent computation impossible stops that dependency.

Every completed stage writes `stage_record.json`. The record binds the stage
inputs, outputs, source tree, resolved configuration, execution plan, status,
and content hashes. Reusing a launch directory is safe only when those values
still match.

## 7. Installation

Python 3.10-3.12 is required. The pinned server environment targets CUDA 12.8.

For an existing compatible environment:

```bash
python -m pip install --requirement requirements.txt
python -m pip install --no-deps --editable .
python -m pip check
```

For development:

```bash
python -m pip install --editable '.[dev]'
```

The bootstrap script creates or reuses `.venv_geomc`, verifies the pinned
requirements, refreshes the editable package, checks that source, installed,
and imported package versions agree, and then starts the full run.

## 8. Local implementation tests

Inspect the plan without running it:

```bash
python run_all.py plan --config configs/smoke.yaml
```

Run and independently verify the complete CPU test:

```bash
python run_all.py full \
  --config configs/smoke.yaml \
  --launch-dir artifacts/smoke/geomc_smoke
python run_all.py verify artifacts/smoke/geomc_smoke
```

Run the corresponding local CUDA test:

```bash
python run_all.py full \
  --config configs/local_cuda_smoke.yaml \
  --launch-dir artifacts/smoke/local_cuda_smoke
python run_all.py verify artifacts/smoke/local_cuda_smoke
```

A completed synthetic test establishes implementation integration only. It is
not scientific evidence and does not establish full-volume server memory or
training performance.

The command-line interface also exposes direct environment and GPU checks:

```bash
python run_all.py environment-check --config configs/server.yaml
python run_all.py gpu-smoke-test --config configs/server.yaml
```

## 9. Server execution

Copy the repository to the server with the project's normal source-sync method.
Do not copy local virtual environments, caches, derivatives, or artifacts as
source code. This repository does not include a Windows synchronization script.

On the server, run:

```bash
bash scripts/bootstrap_and_run_server.sh
```

Create a new run instead of resuming the configured directory:

```bash
bash scripts/bootstrap_and_run_server.sh \
  --launch-dir /fs04/scratch2/ea78/sizhew/mri_pet_geomc_pipeline_2026-08-09/artifacts/geomc_run_name
```

To select a private server override without committing it:

```bash
MRI_PET_GEOMC_CONFIG="$PWD/configs/server.local.yaml" \
  bash scripts/bootstrap_and_run_server.sh
```

`configs/server.local.yaml` is ignored by Git. It should contain only
machine-specific path changes; the scientific settings should remain explicit
and reviewable.

## 10. Evaluation

### 10.1 Aggregate metrics by subject

Patches, views, query tokens, and relation examples from one subject are not
independent subjects. The evaluator first aggregates within each subject and
only then computes cohort statistics and bootstrap intervals.

### 10.2 Fixed reference latent space

The initial EMA target encoder defines a fixed reference latent space.
Similarity Procrustes alignment is fitted only on training subjects and then
held fixed for validation and test. This prevents every checkpoint from being
scored in an unrelated latent orientation and scale.

The report keeps checkpoint-specific EMA metrics separate from fixed-reference
metrics. In the current single-model pipeline, the fixed reference is an
absolute diagnostic; it is not a cross-model ranking by itself.

### 10.3 Representation and mechanism diagnostics

The report includes:

- latent channel variance and RMS amplitude;
- hard-threshold rank and entropy effective rank;
- spatial-mean energy fraction;
- Parseval frame reconstruction and additive-energy closure;
- routing-matrix deviation from identity and off-diagonal scale mixing;
- node-wise variation of the routing matrix;
- residual-update RMS across GeoMC blocks;
- geometry-embedding RMS relative to aggregated observation features;
- input coverage and reliability statistics.

These metrics help detect collapse, excessive smoothing, unused routing, or a
shortcut through the fixed geometry embedding. They do not independently prove
biological validity.

## 11. Data and engineering invariants

### Observation provenance

- Subject, session, MRI/PET modality, source file hash, transforms, brain mask,
  and query mask remain bound through the pipeline.
- Pseudo-pairs are disabled.
- Cross-modal examples require a verified subject link.
- Train, validation, and test subjects are assigned before relation sampling.
- A model-ready observation can point to an upstream preprocessing record. The
  new data API calls this `preprocessing_record_path`; legacy manifest field
  names are accepted only while importing old records.

### PET preprocessing

- PET model inputs are produced by one auditable resampling path.
- The current FDG values are relative intensities, not absolute quantitative
  metabolism.
- Specific scanner, dose, reconstruction, tracer, or PSF values do not enter the
  current model even if some are retained for provenance.

### Registration quality checks

- Registration checks are automatic and nonblocking.
- Statistical flags remain available for stratified interpretation and possible
  reliability weighting.
- There is no manual-signature gate and no rule that cancels the remaining
  experiment because one diagnostic is weak.

### Resource checks

- The end-to-end smoke test records the actual device, precision, allocated
  memory, and timing.
- Arbitrary minimum RAM, disk, and VRAM thresholds are zero by default.
- A real allocation or filesystem failure remains an execution error.

### Reproducibility

- The resolved configuration is written into the launch directory.
- Source, plan, configuration, inputs, outputs, and checkpoints are content
  hashed.
- Checkpoints include optimizer, scheduler, EMA, and random-number-generator
  state required for controlled resumption.
- Validation-best metadata and fixed-final checkpoints have distinct meanings
  and cannot be exchanged silently.

## 12. Repository layout

```text
mri_pet_geomc_pipeline_2026-08-09/
|-- configs/
|   |-- local_cuda_smoke.yaml
|   |-- server.yaml
|   `-- smoke.yaml
|-- scripts/
|   |-- bootstrap_and_run_server.sh
|   `-- run_server_full.sh
|-- src/mri_pet_geomc/
|   |-- data/
|   |-- evaluation/
|   |-- field/
|   |-- geometry/
|   |-- model/
|   `-- training/
|-- tests/
|-- .gitignore
|-- pyproject.toml
|-- README.md
|-- requirements.txt
`-- run_all.py
```

Generated `artifacts/`, `cache/`, `derivatives/`, Python caches, editable-install
metadata, and virtual environments are not repository source.

## 13. Testing

Run the complete unit suite:

```bash
python -m pytest -q
```

A terminology refactor must be checked at several levels:

1. import and Python syntax checks;
2. all unit tests;
3. resolved-configuration validation;
4. fixed-coordinate interpolation and Parseval-frame numerical checks;
5. one deterministic forward/backward test;
6. one complete synthetic pipeline run followed by independent verification.

Renaming a class or configuration key is not scientific evidence of model
quality. It also changes Python imports, report schemas, and potentially model
state-dictionary keys, so an old GeoMC checkpoint must not be assumed to load
strictly unless compatibility is tested explicitly. The external SAT3D
checkpoint remains a separate initialization asset.

## 14. Interpreting a feasibility result

A useful bounded result requires, within the declared protocol:

- stable training without representation collapse;
- lower held-out latent-prediction loss;
- nontrivial but non-dominating geometric routing;
- preserved local and spatially varying latent information;
- no evidence that the fixed geometry embedding alone explains performance;
- stable behavior across subjects and source-target relations.

Even a positive result only justifies promotion to larger pretraining and
matched controls. A foundation-model claim additionally requires large
mostly-unpaired pretraining, multiple downstream tasks, frozen and few-shot
transfer, external cohorts, missing-modality tests, and multiple seeds.

If later matched controls show that geometry does not improve transfer or
prediction, the correct conclusion is that this routing mechanism was not
useful under those conditions. Biological motivation alone is not sufficient.

## 15. Troubleshooting

### A launch unexpectedly resumes

`run.launch_dir` is stable by design. Pass a new absolute `--launch-dir` for an
independent run. Do not delete a valid run merely to avoid reuse of a matching
stage record.

### Verification reports a source mismatch

Verification compares the current source tree with hashes saved in the launch.
Restore the exact source used for that run or create a new launch directory. Do
not edit stage records to force a match.

### SAT3D appears frozen

Confirm that training calls the image encoder directly, the configured
trainability policy marks the expected parameters trainable, and those
parameters are present in the optimizer with nonzero gradients.

### A registration flag appears

Inspect the stored registration and mask-overlap diagnostics. A statistical
flag is not a manual-review request and does not automatically invalidate every
subject or stop later computations.

### The CUDA test succeeds but the full run fails

The small CUDA test checks device integration, not the memory footprint of the
real SAT3D encoder, FEM node count, query count, or training schedule. Use the
real end-to-end smoke-test record to diagnose full-run memory and timing.

### The server path differs

Copy `configs/server.yaml` to the ignored `configs/server.local.yaml`, change
only machine-specific paths, and select it with `MRI_PET_GEOMC_CONFIG`.