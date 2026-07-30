# Exact spatial-search contract

This document defines UrbanLens phase-1 nearest-city semantics. The search
layer is an exact accelerator over the repository's frozen historical
world-cities snapshot; it is not an approximate geocoder.

## Scope and identity

The repository snapshot remains the authority for reproducibility:

| Field | Contract |
| --- | --- |
| Source | `train.csv` |
| SHA-256 | `de941def7faca87c0911abb79c3cbd07672887fd486a9b7bea6c48c12ce0cf18` |
| Bytes | 4,734,682 |
| Source records | 44,691 |
| Audit schema | `urbanlens.data-quality-manifest` v3 |
| Coordinate scale | E7, or 10,000,000 integer units per degree |
| Earth radius | 6,371,008.8 metres |
| Index | `balanced-unit-sphere-kd-tree` v1 |

The loader captures the CSV once through the bounded, no-follow audit path.
Audit checks, normalization, and index construction consume that same
immutable in-memory capture. There is no audit-then-reopen gap. A build fails
closed if the byte identity, ordered schema, row count, source IDs, coordinate
grammar, or aggregate audit contract differs.

SHA-256 identifies repository bytes. It does not authenticate the inferred
upstream release, prove that values are current, anonymize the data, or make
geopolitical labels authoritative.

## Coordinate boundary

Coordinates cross the spatial boundary as strict ASCII decimal strings:

```text
-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,7})?
```

The parser rejects booleans, numeric Python objects, surrounding whitespace,
leading `+`, exponent notation, separators, Unicode digits, non-finite
spellings, leading-zero ambiguity, and more than seven fractional places. It
checks geographic bounds before publishing a `GeoPoint`.

Canonicalization is semantic rather than cosmetic:

- every signed zero becomes integer zero;
- longitude `+180` becomes `-180`;
- longitude becomes zero at exact `+90` or `-90` latitude;
- equality and hashing use signed E7 integers, never formatted floats.

The source has 44,554 canonical numeric locations for 44,691 records. Its 106
co-location groups contain 137 records beyond the first, including one group
hidden by different decimal spellings in the raw CSV. UrbanLens preserves all
records. Co-location is not evidence that two rows describe the same entity,
and `k` counts source records rather than unique coordinate pairs.

## Distance and rank semantics

A canonical point at latitude `φ` and longitude `λ` is projected to the unit
sphere:

```text
x = cos(φ) cos(λ)
y = cos(φ) sin(λ)
z = sin(φ)
```

For unit vectors `p` and `q`, the search key is squared chord distance:

```text
c² = (px - qx)² + (py - qy)² + (pz - qz)²
```

Great-circle angle `θ` and chord length are related by
`c = 2 sin(θ / 2)`. On `θ ∈ [0, π]`, that transformation is monotone, so
ordering by chord distance selects the same neighbors as ordering by spherical
great-circle distance.

The exact declared rank key is:

```text
(binary64 squared_chord_distance, source_id)
```

No epsilon is used to invent tie groups. Numerically identical locations have
distance zero and sort by source ID. An unrounded distance key decides
membership and order; the receipt's integer-metre value is display evidence
only.

Human-facing distance converts chord length back to great-circle metres with
the fixed mean-Earth sphere and a clamped inverse-sine calculation. A separate
clamped, `atan2`-based Haversine implementation checks edge-case equivalence.
The value is a spherical straight-line distance, not an ellipsoidal survey
distance or a route length.

## Deterministic index construction

Construction starts from records sorted by source ID. Each recursive subtree:

1. finds the unit-vector axis with the greatest span, breaking exact span ties
   as `x`, then `y`, then `z`;
2. sorts by the selected coordinate and source ID;
3. chooses the lower median;
4. recursively builds both children; and
5. stores subtree size, depth, and a three-dimensional axis-aligned bounding
   box.

Duplicate coordinates remain separate nodes. A pre-order digest covers the
algorithm version, split axes, source IDs, and canonical E7 coordinates. The
same supported Python runtime, source bytes, and algorithm version therefore
produce the same topology, query order, and structural diagnostics regardless
of input order or `PYTHONHASHSEED`.

Building with a per-level sort costs `O(n log² n)` time and `O(n)` resident
index memory. Recursion depth is `O(log n)` for the median-balanced tree.

## Exact pruning argument

For a query vector, the squared Euclidean distance to a subtree bounding box
is a lower bound on the squared distance to every point inside that subtree.
UrbanLens rounds each positive gap, square, and accumulated term downward with
`math.nextafter`. It rounds the current worst accepted distance upward.

A subtree is pruned only when:

```text
conservative_lower_bound > upward_rounded_worst_distance
```

The inequality is deliberately strict. Equality must still be explored
because a record at the same distance with a smaller source ID can change the
declared result.

An ordinary query is expected to visit a small fraction of this snapshot, but
the honest worst case is `O(n)`. The candidate set is bounded by `k ≤ 100`.
Every query publishes deterministic accounting:

```text
evaluated_records + pruned_records = indexed_records
```

`visited_nodes` counts nodes whose pruning decision was inspected; a pruned
node can represent an entire subtree.

## Oracle verification

The reference path performs a full scan over all canonical records, uses the
same public rank contract, and performs no tree pruning. With `--verify`, the
accelerated and reference paths must return identical ordered rank keys. Any
disagreement produces a verification failure and no passing receipt.

The oracle is an independent traversal, not an independent geodesy model.
This distinction matters: it detects tree construction, bounding, pruning,
and ordering errors without pretending that two implementations of the same
floating-point formula constitute physical ground truth.

## Receipt and disclosure contract

One CLI invocation accepts one query and emits one bounded receipt. A
successful canonical JSON receipt may intentionally disclose only:

- frozen source identity and audit schema;
- stable index metadata and topology digest;
- the normalized query explicitly supplied by the caller;
- up to 100 allowlisted result projections;
- integer display distance and exact-search accounting; and
- whether the full-scan oracle ran and matched.

JSON uses sorted keys, two-space indentation, ASCII escaping, finite values,
LF line endings, and one final newline. Receipts are limited to 256 KiB before
anything is written. They contain no timestamp, absolute path, hostname,
username, process ID, file metadata, telemetry, cache key, or raw binary64
rank value.

Error output uses stable generic messages. It does not echo a rejected path,
coordinate token, source label, source ID, or underlying operating-system
exception. Input, verification, and serialization failures occur before the
first receipt byte is written. Receipts bypass the text encoding configured by
`PYTHONIOENCODING` and write their already-bounded canonical bytes directly.
An underlying output sink can still accept a prefix and then fail; exit code 4
marks that stream as incomplete, and consumers must reject it unless the whole
payload parses as canonical JSON.

## Resource bounds

| Resource | Bound |
| --- | ---: |
| Captured CSV | 16 MiB |
| Parsed source rows | 50,000 |
| Index records | 50,000 |
| Query count per CLI process | 1 |
| Requested `k` | 1–100 |
| Receipt bytes | 256 KiB |
| Candidate state | `O(k)` |
| Full-scan verification | at most 50,000 distance evaluations |

The implementation never creates a pairwise distance matrix and exposes no
user-controlled index resolution or recursion bound.

## QA matrix

Automated tests cover:

- strict coordinate grammar, exact E7 retention, signed zero, dateline and
  pole canonicalization;
- identical, tiny, quarter-circumference, antimeridian, pole, and antipodal
  spherical distances;
- duplicate coordinates and deterministic equal-distance source-ID ties;
- randomized AABB lower-bound checks against enclosed points;
- seeded differential queries against the full-scan oracle;
- shuffled input and multiple hash seeds;
- forged frozen models, duplicate IDs, resource limits, and invalid labels;
- the real snapshot's identity, record count, coordinate multiplicities, and
  single-capture construction;
- canonical receipt bytes, disclosure allowlist, output bound, and
  fail-closed CLI behavior.

## Nonclaims

Nearest means smallest spherical straight-line distance in this historical
snapshot. It does not mean largest, most relevant, currently inhabited,
administratively related, reachable, or closest by road, rail, sea, or air.
Source coordinates can be rounded centroids and are not survey-grade
locations. The fixed sphere is not an ellipsoid, elevation model, or routing
network.

Binary64 trigonometric results are deterministic within the supported runtime
used for checked evidence. UrbanLens does not claim bit-identical `libm`
results across every operating system and architecture; integer E7 inputs,
integer display metres, topology digests, differential tests, and checked
artifacts make that boundary explicit.
