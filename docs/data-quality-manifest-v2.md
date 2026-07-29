# Data-quality manifest contract, version 2

`artifacts/data_quality/train.quality.json` is an aggregate, machine-readable
audit of the exact `train.csv` snapshot. It contains no sampled records,
timestamps, machine-specific paths, credentials, or personal data.

## Canonical encoding

The generator emits UTF-8 JSON with object keys sorted lexicographically, a
two-space indent, LF line endings, and exactly one final newline. JSON
non-finite numbers are forbidden. Arrays retain the deliberate order defined
by the generator. These rules make a fresh result directly comparable as
bytes:

```bash
python3 -m urbanlens.audit \
  --check-manifest artifacts/data_quality/train.quality.json
```

The manifest is bound to the dataset through `dataset.sha256`,
`dataset.byte_size`, and `dataset.data_rows`. It records only the dataset
basename, never an absolute path.

## Required top-level objects

| Object | Purpose |
| --- | --- |
| `manifest` | Schema identifier/version, tool version, canonicalization rule |
| `dataset` | Immutable content identity and parsed row count |
| `schema` | Ordered CSV header and row-width evidence |
| `grain` | Candidate key and intended record grain |
| `quality` | Contract checks, aggregate metrics, observations, overall status |
| `scope` | Explicit boundaries on what the audit can establish |

Every item in `quality.contract_checks` has a stable `id`, an `actual` value,
an `expected` value, and a `pass` or `fail` status. `quality.status` is `pass`
only when every contract check passes. A passing status establishes structural
fitness for the next project phase; it does **not** claim that the historical
population estimates or geopolitical labels are current or authoritative.

Version 2 makes rate denominators explicit by semantics: missingness and
row-derived observations use `quality.metrics.profiled_rows`. Observations
whose counts describe code/label relationships rather than rows omit
`row_rate_ppm`. It also replaces the inferred provider-specific grain wording
with the evidence-bounded phrase “source-provided id.”

## Metric invariants

- Counts are non-negative integers.
- Missingness and row-derived observation rates use integer parts per million
  (`row_rate_ppm`) with `profiled_rows` as their denominator, so malformed-width
  exclusions cannot silently understate a rate and no platform-dependent
  floating-point formatting enters the manifest.
- Counts over ISO codes or distinct label variants are not row counts and
  therefore do not carry `row_rate_ppm`.
- Duplicate `duplicate_row_excess_count` means rows beyond the first within
  each repeated group; `rows_in_duplicate_groups` includes every row in those
  groups.
- Invalid population counts exclude blanks because this source explicitly
  permits missing population; populated values must be finite and
  non-negative.
- Latitude and longitude must be finite and within `[-90, 90]` and
  `[-180, 180]`.
- ISO codes must be uppercase ASCII alphabetic strings of length two and
  three. Each ISO2 must map to a single ISO3.
- The `id` candidate key must be non-blank, unique, and a ten-digit ASCII
  string for this frozen snapshot.
- `profiled_rows` excludes malformed-width rows. Any such exclusion also fails
  the snapshot contract.

Observations describe known sparsity or non-key repetitions without silently
turning them into failures. They are still operationally important for future
joins and models.

## Resource and filesystem safety

- Dataset capture is limited to 16 MiB. Parsing is additionally limited to
  50,000 data rows, 32 columns, and 4,096 characters per physical line.
- The frozen snapshot contract permits one CSV record per physical line.
  Multi-line records fail closed rather than allowing one bounded byte stream
  to expand into an unbounded parser object graph.
- Dataset and checked-manifest leaves must be unchanged regular files opened
  with no-follow and non-blocking flags. Checked manifests are limited to
  1 MiB; symlinks, FIFOs, devices, and oversized files are rejected.
- Open-file identity checks include device, inode, mode, owner, link count,
  size, modification time, and change time. Restoring a file's size and
  modification time after an in-place rewrite therefore does not make the
  capture appear unchanged.
- An output parent is opened once with directory and no-follow flags before
  the audit. Temporary creation and atomic replacement use only names relative
  to that held directory descriptor, so a later parent-path swap cannot
  redirect the write.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Audit contract passes; in check mode the manifest also matches |
| `1` | Contract failure, stale manifest, or non-canonical manifest bytes |
| `2` | Invalid command-line usage (provided by `argparse`) |
| `3` | Dataset or checked-manifest input is missing, unreadable, malformed, non-regular, or over a safety limit |
| `4` | Output is unsafe or cannot be written |

An output manifest is written even when a parsed dataset fails its contract,
so CI can preserve aggregate diagnostic evidence. Input failures do not
produce a partial manifest.
