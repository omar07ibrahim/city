# UrbanLens

UrbanLens is a planned geospatial machine-learning system built around an
audited snapshot of world-city data. The repository is currently at **phase
0**: the original CSV is preserved byte-for-byte while its provenance, license,
grain, known quality constraints, and source-bound evidence visuals are made
explicit. No model, API, or web map is claimed yet.

## Snapshot at a glance

| Property | Verified value |
| --- | ---: |
| Data rows | 44,691 |
| Columns | 11 |
| Candidate primary key | `id` |
| Unique, non-empty IDs | 44,691 |
| File size | 4,734,682 bytes |
| SHA-256 | `de941def7faca87c0911abb79c3cbd07672887fd486a9b7bea6c48c12ce0cf18` |
| Original repository commit | `86eec43a0951cad32d84adac1ad3fb48788c6afc` |

The intended grain is one source city record per source-provided `id`. The
schema and identifier pattern are consistent with the inferred SimpleMaps
source described below, but the repository does not claim authenticated
release provenance. The checked-in schema is:

```text
city, city_ascii, lat, lng, country, iso2, iso3,
admin_name, capital, population, id
```

## Provenance and attribution

The file is consistent with the
[SimpleMaps World Cities Basic Database](https://simplemaps.com/data/world-cities)
schema and with its 1.76-era snapshot. The upstream
[release history](https://simplemaps.com/data/releases) dates version 1.76 to
March 31, 2023.

The original upload did not include the upstream archive, version marker, or
download receipt, so the exact version is an evidence-based inference rather
than authenticated provenance. The file hash above is the authoritative
identity for this repository snapshot.

SimpleMaps states that the Basic World Cities Database is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
See [DATA_LICENSE.md](DATA_LICENSE.md) for the attribution statement and scope.

## Data-quality baseline

The initial read-only profile found:

| Check | Result | Interpretation |
| --- | ---: | --- |
| Duplicate IDs | 0 | `id` is a viable row key for this snapshot |
| Blank IDs | 0 | key completeness passes |
| Invalid latitude / longitude | 0 / 0 | all coordinates are in geographic range |
| Negative or non-numeric population values | 0 | populated values pass the basic domain rule |
| Missing population | 307 (0.69%) | downstream models need an explicit missing-value policy |
| Missing admin name | 316 (0.71%) | regional joins cannot assume complete coverage |
| Blank capital classification | 33,553 (75.08%) | blank means “not marked as a capital,” not corrupted data |
| Repeated city/country/admin rows beyond the first | 301 | names are not a safe key |
| Repeated coordinate rows beyond the first | 136 | coordinates are not a safe key |
| Exact duplicates excluding `id` | 0 | repeated names/coordinates still differ in another field |
| ISO2 codes mapping to multiple ISO3 codes | 0 | the code-pair relationship is structurally consistent |
| Exact country-label variants within an ISO2 | 1 | normalize labels deliberately before grouping |

These checks establish a baseline, not a guarantee of geopolitical truth,
population freshness, or entity-resolution correctness. Country labels and
administrative assignments follow the upstream source's conventions. The
population field is an estimate and is not available for every city.

## Source-bound evidence visuals

Every visual below is generated from the fresh aggregate audit; none is a
hand-made mockup. Generation fails before updating files unless a new
`python3 -m urbanlens.audit train.csv` result equals the checked manifest
byte-for-byte.

### Comparable row-derived observations

![Horizontal bar chart of four row-derived data-quality observations](docs/visuals/data-quality-observations.svg)

Missing admin name is the largest of this predeclared comparable set at 316
rows (0.71%), followed by missing population at 307 (0.69%), repeated
city/country/admin excess at 301 (0.67%), and repeated-coordinate excess at
136 (0.30%). All rates use the same 44,691-row denominator; categories can
overlap and must not be summed as affected rows.

### Implemented audit and freshness workflow

![Phase-0 workflow from captured source bytes through the freshness check](docs/visuals/phase-0-workflow.svg)

The diagram reflects the current executable path: unchanged no-follow source
capture, bounded CSV parsing, contract and aggregate metrics, canonical JSON,
and a byte-for-byte freshness check. It deliberately does not show a map,
model, API, or live-data pipeline because those do not exist yet.

### Actual CLI check result

![Actual successful audit-check command, streams, and exit code](docs/visuals/audit-cli-result.svg)

This terminal view is generated from the actual subprocess invocation, not
typed output. Machine paths and timestamps are excluded so independent runs
remain byte-for-byte comparable.

The analytical decisions, palette, exclusions, provenance, and QA surface are
documented in the [phase-0 visual evidence contract](docs/chart-contract.md).

## Run the reproducible audit

The phase-0 audit is now an executable, dependency-free contract. It reads the
real checked-in CSV and emits a deterministic aggregate manifest:

```bash
python3 -m urbanlens.audit \
  --check-manifest artifacts/data_quality/train.quality.json
python3 -m unittest discover -s tests -v
```

A fresh manifest can be printed to standard output or written atomically:

```bash
python3 -m urbanlens.audit train.csv
python3 -m urbanlens.audit train.csv \
  --output artifacts/data_quality/train.quality.json
```

Regenerate the three SVGs only after the freshness gate passes, then verify
that the checked artifacts match a new in-memory render:

```bash
python3 -m urbanlens.visuals --write
python3 -m urbanlens.visuals --check
```

The parser fails closed above stable safety limits: 16 MiB of source bytes,
50,000 data rows, 32 columns, and 4,096 characters per physical line. The
frozen snapshot is 4,734,682 bytes, 44,691 rows, 11 columns, and at most 187
characters per physical line. Multi-line CSV records are outside this
snapshot's contract. Checked manifests must be unchanged regular files of at
most 1 MiB; symlinks, FIFOs, and devices are rejected.

For `--output`, the destination directory is opened once without following a
final symlink and held by file descriptor through the audit and atomic rename.
This prevents a concurrent parent-path replacement from redirecting the
manifest write.

The checked
[`train.quality.json`](artifacts/data_quality/train.quality.json) is bound to
the CSV SHA-256 and contains schema, completeness, uniqueness, domain, and
cross-field evidence. It intentionally contains no timestamp or absolute path,
raw source row, or source label, so independent runs are byte-for-byte
comparable. Row-derived rates are integer parts per million rather than
floating-point values; non-row observations do not publish a false row rate.

The [manifest v2 contract](docs/data-quality-manifest-v2.md) documents the
schema, invariants, canonical encoding, and exit codes. A passing audit means
the frozen file satisfies this structural contract; it does not certify that
the historical source values are current or geopolitically authoritative.

## Portfolio roadmap

Development will proceed in reviewable, reproducible slices:

1. deterministic ingestion, schema validation, and a committed quality
   manifest bound to the CSV hash (**implemented**);
2. source-derived quality visuals with byte freshness, a real CLI capture, and
   an implemented-workflow diagram (**implemented**);
3. Haversine and spatial-index nearest-city baselines with a typed CLI;
4. population imputation with geographic features, spatial cross-validation,
   simple baselines, and conformal uncertainty intervals;
5. frozen regional evaluation slices with RMSLE, MAE, interval coverage, and
   latency/throughput benchmarks;
6. a FastAPI service and MapLibre interface for search, nearby-city retrieval,
   predictions, and uncertainty;
7. richer reproducible product and model visuals: coverage and missingness
   maps, residual and calibration plots, Playwright interface screenshots, and
   a short end-to-end demo.

Every generated result will record its source hash, code revision, parameters,
and seed. Visuals will be regenerated by repository commands and freshness
checked instead of being hand-made mockups.

## Current limitations

- This repository currently contains a dataset snapshot and documentation, not
  a runnable ML product.
- The exact upstream package version is inferred, not cryptographically proven.
- The data is historical and must not be described as current.
- No code license has been selected yet; the data license does not
  automatically license future project code.
