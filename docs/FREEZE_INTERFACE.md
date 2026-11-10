# Frozen forecasting family and calibration interface

In version 0.1.0, the freeze stage binds the three selected computational forecasters and the complete
calibration protocol into a self-contained snapshot. It executes the saved
models and verifies their development provenance without refitting them,
generating calibration observations, or evaluating their oracle population
risks. **A successful freeze is not a statistical risk certificate.**

The implementation is [freeze.py](../kahm_demo/freeze.py). The
[scientific contract](../kahm_demo/specification.json) governs the scientific scope;
[READOUT_DESIGN.md](READOUT_DESIGN.md) describes the completed development search.

## 1. Public API and commands

| API | Purpose |
| --- | --- |
| `freeze_forecasters(bundle, encoders, models, output)` | Verify the selected development family and commit a fresh snapshot |
| `verify_frozen_bundle(freeze_path, *, dataset_bundle=None)` | Verify the snapshot, execute its models, and check the fixed calibration protocol |
| `load_frozen_forecaster(freeze_path, candidate_id, *, dataset_bundle=None)` | Return the already verified model instance, bound to its freeze and protocol identities |

All paths may be supplied as `pathlib.Path` objects. The record must be named
`freeze.json`. If `dataset_bundle` is omitted, verification uses the archived
development data. Supplying it additionally checks that the external bundle has
the same development and reservation identities. Neither mode reads calibration
outcomes, even when the external bundle later contains calibration data.

From the project root, after completing development and model selection:

```bash
python -m kahm_demo freeze-forecasters --bundle data/primary --encoders encoders/primary --models models/primary --output frozen/primary
python -m kahm_demo verify-freeze --freeze frozen/primary/freeze.json --bundle data/primary
python -m kahm_demo forecast-frozen --freeze frozen/primary/freeze.json --candidate kahm_local --history 0.25 0.75
```

Freeze creation requires a fresh output directory outside the source data,
encoder, and model bundles. It rejects an already materialized calibration
stream. It builds and verifies a staging snapshot before committing it, and
checks again that calibration has not appeared before the commit. It never
replaces an existing freeze or changes the original development artifacts.

The Python prediction interface is:

```python
from pathlib import Path
from kahm_demo.freeze import load_frozen_forecaster

frozen = load_frozen_forecaster(
    Path("frozen/primary/freeze.json"),
    "kahm_local",
    dataset_bundle=Path("data/primary"),
)
print(frozen.predict_one([0.25, 0.75]))
print(frozen.inspect_one([0.25, 0.75]))
print(frozen.predict([[0.25, 0.75], [0.25, 0.75]]))
print(frozen.freeze_record_sha256)
print(frozen.calibration_protocol_sha256)
```

The returned `FrozenForecaster` wraps the same in-memory `ForecastModel` whose
fingerprint and outputs were checked. Its prediction methods retain the
canonical one-history computation, finite checks, clipping, and recorded encoder
fallbacks. Loading does not substitute a newly fitted model.

## 2. Snapshot contents and status authority

| Path within a snapshot | Contents |
| --- | --- |
| `freeze.json` | The authoritative frozen-family record |
| `models/{raw,kahm_mem,kahm_local}.json` | Byte-identical copies of the three selected development artifacts |
| `predictions/{raw,kahm_mem,kahm_local}.json` | Their canonical predictions on all 16 supported histories |
| `development/data/` | Development manifest, training and validation observations, and the untouched calibration reservation |
| `development/encoders/` | Both fitted encoder banks and their supporting artifacts |
| `development/forecasters/` | Complete development bundle, including all 225 search states, their predictions and exact empirical losses, and the recorded selection |
| `sources/kahm_demo/` | Archived package source and JSON resources |
| `calibration_protocol.json` | The exact fixed protocol resource for the later calibration stage |

The source artifacts retain their original development statuses. In particular,
all 225 search states remain `development_candidate`, and the three selected
model artifacts remain `development_selected_not_frozen` inside their unchanged
JSON files. **The enclosing verified `freeze.json` establishes frozen status.**
It records `status: frozen` for the family and each of its three members, and
binds those members to their exact selected artifact bytes and fitted-state
fingerprints. Relabeling the development artifacts is unnecessary and would destroy their
byte identity with the development record.

The snapshot contains no calibration observations. It must not be used as the
output data bundle for later calibration: both public calibration APIs reject
any resolved calibration directory inside the freeze record's parent tree,
including a path routed there through a symlink. Preserve the snapshot unchanged;
write later calibration data, receipts, and reports outside it.

## 3. Record identities and verification

The record uses `schema: kahm.model_freeze.v1` and
`model_preflight_schema: kahm.frozen_forecast_bundle.v1`. Its required fields bind:

- The scientific contract, run, dataset manifest, and calibration reservation.
- The canonical inference method, numerical runtime, archived sources, and fixed
  execution-source roster.
- Exactly the ordered candidates `raw`, `kahm_mem`, and `kahm_local`, including
  their artifacts, fitted-state identities, winning trial IDs, canonical
  prediction tables, clipping/fallback diagnostics, and scalar/batch agreement.
- The complete development snapshot, search count, and shared-bank selection.
- The calibration protocol's exact resource hash, reserved stream, and episode
  count.
- A complete payload-file inventory and a digest of the record payload.
- The declarations that no refit, oracle population-risk evaluation, or
  statistical certification occurred, and that calibration was reserved at
  freeze creation.

Artifact paths are relative to the snapshot root and must remain within it.
Inventory entries contain path, byte count, and SHA-256. Missing, additional, or
changed files invalidate the recorded inventory. Symlinks are forbidden within
the snapshot. A path component named `calibration` is rejected before inventory
hashing so outcome files cannot be introduced and read as snapshot payload.

The top-level `record_sha256` hashes canonical record content with that field
excluded. The separate `freeze_record_sha256` reported by verification hashes the
actual `freeze.json` bytes. Later calibration receipts bind to that exact file
identity.

Full verification performs more than hash comparisons. It reloads and executes
all 225 archived development candidates, checks their training moments and ridge
equations, recomputes exact training and validation losses, and repeats the
recorded selection. It then executes all three frozen members on the full
16-history support, verifies their canonical tables, and checks scalar/batch
agreement with a repeated history. It confirms that the two KAHM members embed
one identical selected encoder bank and that their artifacts remain identical to
the winning development states.

The archived data snapshot contains only its declared development files and
reservation. An external data bundle is checked through its development and
plan loaders. Declared and resolved paths are checked before those loaders run,
so a malformed manifest cannot redirect a development or plan read into its
calibration directory.

## 4. Installed source and runtime enforcement

Verification executes the installed implementation on the archived numerical
state. It does **not** import or execute arbitrary code from the snapshot.
The fixed `EXECUTION_SOURCE_FILES` roster in `freeze.py` covers the prediction,
serialization, data, sampling, contract, protocol, selection, and verification
dependencies. Every installed file in that roster must match the archived bytes.
The roster is fixed by the implementation, not selected by an incoming record.
The saved canonical method and complete numerical runtime record must also match.

The package initializer and CLI are archived for provenance but are outside the
execution-source roster. Both the installed and archived initializer must parse
as an optional docstring followed only by a nonempty literal `__version__`
assignment. This allows a metadata-only release-label change while preserving
the enforced execution sources and runtime. A changed version string does not
permit changed prediction or verification code. The archived files and recorded
release metadata remain immutable parts of the snapshot inventory.

Source, runtime, and replay mismatches stop verification. There is no bypass that
accepts a different implementation merely because its predictions appear close.
Exact replay concerns this computational predictor; it does not certify
numerical equivalence to an ideal real-arithmetic Gaussian KAHM map.

## 5. Fixed calibration protocol

[calibration_protocol.json](../kahm_demo/calibration_protocol.json) has schema
`kahm.calibration_protocol.v1`, protocol ID
`iid_three_forecaster_exact_mse_upper_v1`, and protocol version `1.0.0`.
Its exact SHA-256 is declared by `CALIBRATION_PROTOCOL_SHA256` in
`kahm_demo/calibration_protocol.py` and copied into each freeze record. This
identity is specific to the actual protocol bytes in this edition.

`calibration_protocol.load_calibration_protocol()` first checks the exact
resource bytes against that digest, then verifies consistency with the
scientific contract. `calibration_protocol_sha256()` returns the declared identity.
Freeze verification checks both the archived protocol copy and the installed
resource. Altering a numerical choice or a prose rule changes the resource
identity; the protocol is not a runtime-tunable configuration.

| Fixed element | Protocol |
| --- | --- |
| Certified object | Actual frozen deterministic computational predictor |
| Loss | Scalar squared error in `[0,1]`, without variance normalization |
| Calibration | One shared set of 4,096 iid episodes, independent of development; every occurrence retained |
| Family | RAW, KAHM MEM, KAHM LOCAL |
| Failure allocation | `1/60` per candidate; `1/20` total |
| Simultaneous coverage target | `19/20`, conditional on development and the stated premises |
| Primary tolerance | Exact MSE threshold `1/10` |
| Empirical mean | Exact rational losses of the actual binary64 predictions and exact benchmark targets |

For exact empirical mean $q$, the prescribed endpoint is the upper binary-KL
inverse

$$
k_+\!\left(q,\frac{\log 60}{4096}\right)
=\sup\left\{p\in[q,1]:
\operatorname{kl}(q\Vert p)\le\frac{\log60}{4096}\right\}.
$$

The protocol also fixes how certification must obtain a conservative numerical endpoint.
It prescribes exact integer/rational arithmetic, range-reduced logarithm
intervals from the atanh series, and directed dyadic enclosures at 128, 256, and
512 bits. Bisection changes an endpoint only after an interval proves the sign
of `4096*kl(q||p)-log(60)`. The target bracket width is `2^-80`, with at most 80
successful halvings and explicit work limits. Independent endpoint checks must
verify the saved bracket and inequalities without invoking the root search.
The `q=0`, `q=1`, and upper boundary `H=1` cases have explicit conventions.

The returned upper endpoint remains an exact rational. Its 18-place decimal
string is rounded upward; the acceptance decision uses the exact comparison
`U <= 1/10`. An unresolved sign, failed identity or numerical check, or exceeded
work limit produces `certificate_unavailable`; it does not trigger a replacement
sample or an automatically looser reported endpoint. Every candidate remains in
the report, and unused confidence allocation is not reassigned after outcomes.

The model freeze fixes these requirements. The certification stage implements them in the separate
certification modules described in [CERTIFICATION_DESIGN.md](CERTIFICATION_DESIGN.md).
The protocol module itself performs validation only. The statistical
statement requires iid episodes independent of development;
deterministic seed labels do not prove that premise. The self-contained proof
is in [FITTED_RISK_GUARANTEE.md](FITTED_RISK_GUARANTEE.md): Sections 2–4
establish statistical coverage and Sections 5–6 establish the conservative
numerical endpoint. Numerical availability and requirement acceptance are not
additional conditioning events in that coverage statement.

## 6. Public calibration gates

The data APIs are

```python
materialize_calibration(bundle: Path, freeze_path: Path) -> dict
load_calibration(bundle: Path, freeze_path: Path) -> Dataset
```

Their private `_model_freeze_preflight` lazily calls
`verify_frozen_bundle(freeze_path, dataset_bundle=bundle)`. There is no public
bypass flag. Both APIs keep calibration outside the frozen tree and run this
**full model preflight before the structural data-layer checker**. This order
lets the full verifier reject unsafe declared data paths before the legacy
structural loader reads a plan.

Materialization first rejects an existing calibration path, including a dangling
symlink, without inspecting its outcomes. After the snapshot-location guard,
full preflight, structural check, and development integrity checks succeed, it
samples the one reserved stream into a staging directory. A separate receipt
binds its data hash and count to the exact freeze, dataset manifest, calibration
plan, and three candidates. Existing calibration cannot be replaced.

Loading runs the snapshot-location guard and full preflight, then structural
validation. It reads the outcome-free receipt and requires its exact freeze hash
to match before any generic integrity check can read episode outcomes. It then
checks the data hash and episode structure and returns the recorded data path.
A different otherwise-valid freeze cannot authorize access to those outcomes.

The public materialization command writes to the external bundle:

```bash
python -m kahm_demo materialize-calibration --bundle data/primary --freeze frozen/primary/freeze.json
```

The assessment workflow invokes that gate after committing its numerical
implementation and assessment plan. Once calibration has been materialized,
it must not be replaced. When a risk assessment already exists, inspect it with:

```bash
python -m kahm_demo verify-certificate --certificate certificates/primary/certificate.json --bundle data/primary --freeze frozen/primary/freeze.json
```

Using `--bundle frozen/primary/development/data` is rejected because it would
mutate the sealed snapshot. The read-only `verify-freeze` API may use that
archived data; the outcome-writing and outcome-loading APIs may not.

`validate_freeze_record` remains a standalone structural checker.
`verify_bundle` remains an integrity-only data checker and may inspect
already-materialized episode rows. Neither function by itself is a successful
model preflight. The public calibration paths enforce the full gate; isolated
data-layer unit tests patch that private gate explicitly for their structural
constant fixtures, while integration tests exercise actual frozen models.

## 7. Meaning and limits of a freeze

A verified freeze fixes which computational functions and calibration protocol
will be assessed. It preserves selected parameters, the shared bank, canonical
predictions, development evidence, and evaluation choices before primary
calibration. It does not report a confidence endpoint or assert that the MSE
requirement has been met.

File hashes and workflow checks detect inconsistent artifacts. They are not
digital signatures, do not make an independently editable directory tamper-proof,
and do not authenticate execution chronology. Later changes to a predictor,
protocol, or required execution dependency must follow the recorded change
control; the existing freeze cannot silently be relabeled to cover them.

The Python freeze verifier returns `statistical_certificate_issued: false`.
This field describes what freeze
verification itself establishes. It does not inspect a later certificate bundle.
The current `verify-freeze` CLI exposes that scope as
`risk_certificate_checked: false`. The separate `verify-certificate` command
establishes the status of the risk assessment.
