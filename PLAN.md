# ffc full modern Fortran roadmap

Updated 2026-10-03. This is the sole execution roadmap for ffc and its
compiler dependencies. Give an implementing agent `AGENTS.md` and this file.
The detailed provider contract is [fo Gremlin PLAN](https://github.com/lazy-fortran/fo/blob/main/PLAN.md).
Read the project map, active task, and verifier below first; follow the ordered
queue using an external serial or parallel controller and Gremlin background testing.
GitHub issues define individual changes; they do not define a competing roadmap.

## Goal and completion contract

**Make ffc compile, link, and correctly execute all modern standard Fortran,
through Fortran 2023.** This includes the standard facilities introduced in
Fortran 90, 95, 2003, 2008, 2018 and 2023, their valid combinations, separate
compilation, and ISO parallel features: coarrays, images, synchronization,
collectives, atomics, locks, events, teams and notifications. A late feature
remains required; neither a small corpus nor today's supported subset limits
the destination. Diagnose invalid programs according to the selected language
mode and never silently lower a supported program to wrong output.

The authoritative language references are the relevant standard editions and
[J3/24-007](https://j3-fortran.org/doc/year/24/24-007.pdf) for the F2023 audit.
[docs/F2023_DELTA.md](docs/F2023_DELTA.md) indexes additions, not the entire
language. A reference compiler is an oracle with known gaps: use a second
implementation or a clause-derived independent model where it is insufficient.
Processor-dependent kinds, formatting, clocks and error messages require their
specified contract rather than arbitrary byte equality.

Accepted Lazy Fortran modes and extensions are an additional target, recorded
in `../standard`. Draft generics, traits and Synthesis proposals become
implementation obligations when their normative contract is accepted. They do
not delay standard Fortran work or count toward standard conformance. OpenMP,
OpenACC, vendor syntax and accelerator-specific backends are separate extension
work. MPI is a library integration target; ordinary interoperable/external calls
must still compile. Historical deleted constructs remain explicit compatibility
modes where promised. No ISO parallel facility is an exclusion.

Completion requires both language-family coverage and one final, frozen,
comprehensive verification version: every in-scope corpus case has an independent
compile/run, compile-and-consume, or rejection oracle; zero wrong answers,
crashes, hangs, unexpected acceptance/rejection, XFAIL, XPASS, FLAKY, unexplained
SKIP, NOREF, timeout or OOM remain. The full maintained suite, rejection gate,
separate-compilation/ABI, platform, sanitizer and performance gates must pass.
Documented external extensions and invalid harness-only cases remain classified
and visible outside the positive-language denominator. Corpus green alone does
not prove all Fortran; audit every standard family and its interactions.

## Project map and authoritative state

| Repository | Responsibility | Planning input baseline on 2026-10-03 |
| --- | --- | --- |
| [ffc](https://github.com/lazy-fortran/ffc) | Driver, lowering, descriptors, Fortran ABI, runtime, object/executable emission | `193dd9d` |
| [fortfront](https://github.com/lazy-fortran/fortfront) | Backend-neutral parsing, semantics, binding identities, typed public queries, diagnostics | `42e6303` |
| [liric](https://github.com/krystophny/liric) | Public session API, verification, native/LLVM backend serialization and emission | `9eff096` |
| [fo](https://github.com/lazy-fortran/fo) | Dependency DAG, caching, builds, named test routing, result transport | `2e781fd` |
| [standard](https://github.com/lazy-fortran/standard) | Normative language/proposal and grammar contracts | `156977b` |

All five inputs were clean `main` checkouts after fetching and fast-forward
pulling. These are documentation input revisions, not new validation claims.
Dependency ROADMAP files retain local responsibilities and refer here for
compiler order and test scheduling. External LFortran and GCC checkouts supply
pinned corpora; this roadmap does not authorize modifying those repositories.

Read [docs/SUPPORT_CONTRACT.md](docs/SUPPORT_CONTRACT.md) for current supported
behavior, the [runtime ABI](docs/RUNTIME_ABI.md), [array descriptor ABI](docs/ARRAY_DESCRIPTOR_ABI.md),
and [character descriptor ABI](docs/CHARACTER_DESCRIPTOR_ABI.md) when relevant.
Read [docs/CONFORMANCE.md](docs/CONFORMANCE.md) for harness formats and commands.
Generated `docs/PARITY_STATUS.md`, `test/conformance/parity_dashboard.tsv` and
`parity_epoch.json` are measurements, never alternative plans. Current project
contracts and verifiers govern implementation; this roadmap governs destination
and scheduling. Update affected contracts with the implementation that changes
behavior. An unsupported contract entry is current truth, not a permanent scope
exclusion.

### Latest observed checkpoint

The last comprehensive local observation is **2026-10-02**, at code checkpoints
ffc `73fc71f`, FortFront `2b6bc1a`, fo `50de371`, LIRIC `9042548`, standard
`160032a`. It is historical evidence for those versions, not today's full green.

- Maintained compiler suite: **502 pass, 6 fail, 508 names**, 509.16 seconds.
- Rejection gate: **919 files, 706 accepted, 213 rejected**, zero new rejections.
- FortFront: 767/767; frontend behavior/namespace oracles 56/56 and 34/34;
  sanitizer checks 4/4. fo pipeline/CLI checks passed; MCP checks 79/79.
- Dispatcher consolidation preserved 500 earlier public names and 497 test
  bodies; an independent 67-case conversion oracle preserved verdicts. Measured
  worker build footprint: 350,501,754 bytes; one-case edit: two relinks, 51.89 s.
- Production `.inc` removal is complete (`0676861`); do not restart the retired
  mechanical folding plan. Preserve zero includes and the real submodule DAG.
- BOZ formatting, named EXIT/CYCLE, namespace validation, character routing,
  MERGE/MASK/ABS reductions, bounded TRANSFER and allocation-bound slices have
  executable evidence; broader families remain incomplete.
- The checked-in four-suite scoreboard is stale. Its existing epoch pins an
  older ffc revision (`0365101`); previous successful refreshes are archived
  observations. Both external corpus directories now exist; old missing-corpus
  blockers are obsolete.

| Failing maintained test | Last observed failure to reproduce on a new build |
| --- | --- |
| `test_session_class_star_assumed_shape_compiler` | Missing real(8) refusal; reaches an undefined procedure at link time. |
| `test_session_class_star_rank2_assumed_shape_compiler` | Missing default-integer/real(8) refusal; undefined procedure at link time. |
| `test_session_pdt_inheritance_compiler` | Unsupported scalar type in inheritance. |
| `test_parity_dashboard` | Stale revision/source provenance. |
| `test_fortfront_corpus_conformance` | Hardcoded 517/264 totals disagree with 646/267 selections; observed reports had zero FAIL/XPASS but were not a locked epoch. |
| `test_conformance_gauntlet_smoke` | A case in XFAIL and NOREF did not receive its expected rejection. |

Evidence lives under `/var/tmp/lazy-fortran-finish-20261002`: `manifest.json`,
`suite-evidence.json`, `suite-relink-evidence.json`, frozen patch digests,
`logs/ffc-suite-final-frozen.log`, `evidence/final-test-verdicts.json`,
`logs/rejection-final.log` and independent parity receipts. Older `perf/...`
and `logs/...` references resolve under `/var/tmp/ffc-goal`. Git history retains
superseded planning narratives and resolved exploration; do not copy them into
another active plan. Missing scratch evidence requires fresh observation, not
invented results. Keep exact source/artifact identities for every new receipt.

## Active task and work selection

**Execution order:** (1) make resident Fo Gremlin usable with named focused
gates, build the exact candidate B, and verify its actual public behavior before
adopting it; (2) use dogfooding to remove measured Fo cruft, duplication, dead
code, metadata tests and stale docs; (3) deliver the Fo capabilities required
for FFC's FPM-only Gremlin route under [Fo #200](https://github.com/lazy-fortran/fo/issues/200);
(4) reproduce and repair the three current FFC failures; (5) implement
standalone Fo-native FPM semantics, using a generic TOML library if useful but
no imported FPM-specific modules; and (6) implement native CMake/CTest support
for the required ITpPlasma profiles. Do not make full Fo architecture/store/test
migration, CI, or benchmarks prerequisites for dogfooding or the FPM-only FFC
route. Fo's focused provider targets identify named scope, not complete impact
or proof of public candidate-B behavior.

After the FPM-only route and its public gates pass, reproduce and repair the
three current FFC failures to battle-test the integration. Then take the earliest
unblocked compiler substrate slice. Read-only review and independent Fo source
slices may continue in parallel. The Fo Gremlin discussion PR #146 is merged.
Experimental agent scheduler PR #147 was abandoned and closed without merge.
The workspace master PLAN/AGENTS govern execution mode; this paragraph orders
the Fo-to-FFC handoff without replacing the full compiler roadmap.

The controller records active issue(s), dependencies, declared file/API ownership,
worker/model capacity, source generation and targeted verifier in the fo task
state. The authoritative compiler roadmap remains below; the six failures in
the dated checkpoint are historical evidence, while the three current failures
are the immediate FFC repair target. Speed work must serve this sequence rather
than postpone language support.

### Earlier Fo provider issue map (reference, not an FFC gate)

These provider issues describe the earlier shared-engine plan. Their open
acceptance remains useful when a current Fo task needs it, but completing this
entire table is not a prerequisite for the FPM-only FFC route or compiler work.
Use the execution order above and Fo's current plan/issue state for active
priorities.

| Order | Atomic provider task | Observable acceptance |
| --- | --- | --- |
| 1 | [fo #138](https://github.com/lazy-fortran/fo/issues/138): shared targeted/affected/random selection and distinct shuffle policy | Same CLI/MCP selection, exact replay, mandatory targets retained, rotating coverage debt. |
| 2 | [fo #139](https://github.com/lazy-fortran/fo/issues/139): bounded owned-process-tree cancellation | Obsolete children stop within grace; unrelated lanes and sentinel survive. |
| 3 | [fo #140](https://github.com/lazy-fortran/fo/issues/140): durable per-case verdict journal | Completed pass/fail survives cancellation/restart; unfinished work stays unknown. |
| 4 | [fo #141](https://github.com/lazy-fortran/fo/issues/141): continuous latest-built-generation supervisor | Frozen inputs match artifacts; last compilable generation continues through failed builds; replacement preempts only its lane. |
| 5 | [fo #142](https://github.com/lazy-fortran/fo/issues/142): MCP/CLI/background lifecycle parity | Autonomous progress, reconnectable run/event IDs, bounded wait-for-failure, reproduction and stop. |
| 6 | [ffc #798](https://github.com/lazy-fortran/ffc/issues/798): ffc project adapter | Targeted dispatcher and corpus adapters use fo's engine and correct generations. |

In this earlier map, orders 1–3 were independent provider slices; later rows
consume those foundations. Existing [fo #119](https://github.com/lazy-fortran/fo/issues/119)
lossless JSON, [#130](https://github.com/lazy-fortran/fo/issues/130) visibility/log
retention, [#134](https://github.com/lazy-fortran/fo/issues/134) timeout diagnostics,
[#135](https://github.com/lazy-fortran/fo/issues/135) dependency freshness and
[#131](https://github.com/lazy-fortran/fo/issues/131)/[#132](https://github.com/lazy-fortran/fo/issues/132)
dispatcher identity are prerequisites wherever a reproduced defect blocks these
acceptance cases. Verify existing fixes before closing them; do not duplicate
their obligations in another subsystem. Pure global output/configuration changes
are assigned before workers depending on their contract, never raced against them.

The following CLI and primitive notes belong to the earlier provider map; use
Fo #200 and current Fo public help for active scope and implemented behavior.

The current fo CLI already provides `fo test --random N --seed S`,
`--only-changed`, `FO_JOBS`, caches and project locks. Its MCP has async check
start/status/diagnostics/cancel. Those primitives are not the complete protocol:
watch blocks in check, rerun-pending waits for old completion, async cancellation
signals only the parent, native team reporting is delayed, and MCP test omits
selection fields. Reuse the primitives and repair their gaps.

**Proposed public interface, unavailable until the provider issues land:**
`fo gremlin start`, `fo gremlin status --json`,
`fo gremlin wait --failure --json`, `fo gremlin failures --json`,
`fo gremlin reproduce ID`, `fo gremlin stop`, with explicit lane/generation,
`--random N`, `--seed S`, `--shuffle`, budgets and process capacity. A
`fo test --continuous` alias may share the same engine. Extend fo's existing
single-tool MCP with equivalent gremlin actions and fields, not a second daemon
or token-consuming agent scheduler. The service must progress while clients
reason, compact context or disconnect. Bounded/event-driven waits avoid polling.
CLI and MCP must advertise implemented capabilities and reject unsupported ones.
They expose the same feature set through shared engine code. After a fo fix, use
the updated CLI immediately when MCP cannot reload; never make MCP availability
a prerequisite or duplicate the engine in ffc. Fix fo whenever build, cache,
selection, supervision, CLI/MCP or result transport misbehaves.

### Work modes

Both modes use one controller for authoritative state, assignment, integration,
commits, issue updates and promotion. Workers may commit freely on their own task branches and return commits or
patches plus frozen receipts; they never merge or push main themselves.

- **Serial mode:** the main session only, without subagents; a configured local
  model may be that session, with one background campaign. Implementation never waits for the whole
  suite. Use this when the local model cannot sustain concurrent requests.
- **Parallel mode:** a luna coordinator assigns independent ready tasks that
  fill configured capacity with luna workers. There is no hardcoded team size: use
  `max_workers` and provider/resource limits. Unavailable slots due to real
  dependency or ownership conflicts must be visible. Capacity limits worker
  count separately from compiler/test processes, memory and disk.
  Use max reasoning effort for luna implementation workers where available and
  record runtime-resolved model/effort per task.

Agent dispatch, task DAGs, worktrees, model choice and integration belong to the
external controller. fo provides build, test and Gremlin feedback only. Serial
task escalation uses the GPT skill with Sol; parallel escalation uses a
native Sol worker for only the repeatedly failing task. Stop the previous writer
and transfer frozen evidence after two substantive failed repair attempts.
Workers receive this PLAN, their exact issue/base,
ownership, verifier and the output-size rule from AGENTS.

Use one writable Git worktree per implementation/fix worker and a protected
integration worktree controlled by the coordinator. Read-only research/review
needs no extra worktree. Assign independent file/API domains and frozen sibling
dependency bundles. Independent build trees may reuse the validated shared fo
content-addressed cache; a cache hit requires the complete unchanged action key. A source file, shared ABI, module facade, registry
or generated manifest has one writer at a time. A regression fixer may get a
separate worktree, but overlapping patches still need controller sequencing.
Alternative implementations also get distinct task worktrees. Parallel workers
cannot race-edit the shared checkout or silently overwrite a
library/tool installation another campaign uses. Cross-repository API changes
follow additive provider, migrated consumer, default switch, old-path retirement.

Each worker owns a lane_id and local generation; integrated-main owns another.
A new A build cancels only A's old campaign, never B's or integrated-main's.
The controller tests a temporary candidate consisting of integration D plus
worker result A before advancing integration: build that exact combined commit
or hashed patch and run regression reproducers plus the union of affected tests.
A failing candidate goes back to its worker and leaves integration unchanged.
A passing targeted gate advances integration and starts its discovery campaign;
it does not claim the full suite passed. Integrate one candidate at a time, then
promote explicit reviewed changes to main. Worker campaigns run known failures, affected tests and randomized subsets.
Integration campaigns run regression reproducers, the affected-test union,
randomized discovery, and eventually complete frozen verification. Worker
evidence supports review; it is not automatically evidence
for the rebased/integrated artifact. Confirmed regressions get the next available
compatible worker slot; unrelated independent work can continue. In serial mode
the regression repair is the next task. Reserve resources for integration and
regression repair instead of saturating the host with nested test pools.

## Continuous randomized testing

**Gremlin mode** is the project name for **continuous randomized testing**:
continuous speculative regression testing with randomized scheduling and
latest-version preemption. Speculation means the editor proceeds while a
background campaign checks the last compilable snapshot; the external controller
determines implementation worker concurrency. Random sampling selects existing tests; a distinct shuffle policy randomizes
their dispatch order. Record actual starts/completions when workers run in parallel. This is not
fuzzing unless a separate task generates or mutates inputs.

[Saff and Ernst's ISSRE 2003 paper](https://homes.cs.washington.edu/~mernst/pubs/wasted-time-issre2003.pdf),
sections 5.3–5.4, describes background tests, random ordering without repetition,
recent-failure priority and restarting tests on the next compilable version.
[Infinitest](https://infinitest.github.io/doc/index) reruns tests affected by
source changes. [pytest-randomly](https://github.com/pytest-dev/pytest-randomly)
records seeds for reproducible shuffled ordering. [GitHub Actions concurrency](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency)
can cancel superseded workflows; ffc CI already uses cancel-in-progress.
These support the components. The budget and coverage policy below are our
project choices, not a universally named or proven optimal algorithm.

### Latest-version preemption

1. Build version Vn, identify its inputs/artifacts, run its narrow reproducer
   and positive/negative neighbor tests, and start that lane's background campaign.
   Continue preparing the next fix immediately; do not wait for a full suite.
2. Each source, dependency, manifest, harness or compiler change that affects
   results invalidates the campaign inputs. Editing can continue while Vn runs;
   never mutate the inputs or binaries Vn is using. Build and test concurrency obey shared resource admission, separately from model-worker capacity.
3. When Vn+1 successfully compiles, inspect Vn's already published failures,
   persist their receipts and its untested/cancelled selection, cancel its owned
   process group, and immediately start testing Vn+1. Do not finish an obsolete
   suite. Failed builds do not create a test version: keep testing the last compilable generation while repairing the candidate build.
4. Campaign life is the minimum of the next built-version arrival, selected
   subset completion and wall budget. Default discovery budget: **60 seconds**,
   **at most 32 corpus cases**, one corpus compiler process per campaign by default, **5 seconds per case**.
   Record justified overrides for expensive tests; always retain preemption.
   If the subset completes while editing continues, start a new seeded subset
   against that same immutable version. A full corpus is never an iteration gate.
5. Use a recorded dedicated runner PID/process group. TERM the owned group,
   allow at most two seconds for shutdown, then KILL surviving owned children.
   Never use broad `pkill`, kill unrelated builds, or leave old descendants
   consuming resources. The controller must emit a receipt before cancellation;
   notification/cancellation latency is measured, not assumed instantaneous.

### Regression response

An old snapshot turning red **does not interrupt an uncommitted edit**. Record
its version, test/configuration, seed, executed-order prefix, environment,
oracle and failure signature. Put that exact reproducer first on the newest
compilable version; preserve predecessor tests when an order dependency is
suspected. The priority order is current confirmed regressions, edit-impacted
tests, historically failure-prone tests, then randomized unexplored tests.

If the reproducer passes on the newest version, continue and retain the old
failure as history. If it fails reproducibly there, finish the current coherent
edit and assign the repair or minimal revert ahead of unrelated work in that lane.
Parallel mode can assign a dedicated compatible fixer while other independent
workers continue; serial mode takes the repair next. Never turn a current reproducible regression into background
debt. Intermittent failures get seeded/order replay and environment checks;
track product nondeterminism and test pollution explicitly rather than hiding
failures. After repair, rerun the reproducer and return it to the normal pool.
A broken shared build gets repair/revert priority before unrelated promotion.

### Random selection and coverage debt

Each campaign combines its mandatory reproducers with a small corpus subset.
`--random N` controls membership; `--shuffle` controls dispatch order. A seeded
selection alone must never be described as a guaranteed sequential execution order.
Stratify across FortFront F90/LF, LFortran, GCC and language/negative-test families.
Record inventory and exclusions, seed, selected paths and exact order. Shuffle
without replacement within a subset and change the seed across campaigns.
Reserve roughly one quarter of discovery slots for least-recently completed or
repeatedly cancelled cases; fill the remainder from recent failures, impacted
families and random exploration. Rotate suite/feature starting slots so short
runs do not always exercise the same prefix. The 32-case discovery cap excludes
the mandatory narrow reproducer set; keep that set small and independently timed.

Track last attempted/completed version, duration, failure history and cancellation
count per case. Give slow/starved cases early slots and an explicit bounded
budget when ordinary runs cannot finish them. Preemption still wins. Report
remaining coverage debt; a stream of partial versions cannot guarantee full
coverage. Shuffle isolated tests to vary coverage; detecting pollution also
requires replaying the affected shared-state order rather than assuming process
isolation proves independence. History guides scheduling, never oracle outcomes.

### Receipts and result meanings

A version receipt includes every involved base commit, worktree patch SHA256
(including untracked inputs), binary and runtime-library hashes, corpus pins,
harness/manifest hashes, reference compiler identity, flags, locale, stdin,
working directory, seed, selected-list/order hash and owned process identity.
Freeze review inputs as exact base commits plus patch digests when uncommitted.
Publish completed per-case outcomes and stdout/stderr/exit information promptly.

| Outcome | Meaning |
| --- | --- |
| Observed pass | Completed independent oracle passed on this exact version. |
| Untested / cancelled | No completed verdict for this version; partial execution proves nothing. |
| Known failure | Same-case baseline evidence or an owned unsupported gap explains the result. |
| Candidate regression | Newly observed failure awaiting comparable baseline/latest-version reproduction. |
| Confirmed regression | Previously supported behavior fails reproducibly on the newest version. |
| Flaky / timeout / infrastructure failure | Execution needs diagnosis; none is a behavioral pass. |

Old failures survive aggressively as reproduction priorities. Old passes remain
historical unless the relevant source/dependency/test/oracle closure is proven
unchanged. Default to rerunning; a matching test name is insufficient. Never
combine different compiler versions into one purported full green result.

### ffc corpus, binary and disk policy

Preserve the existing consolidated test dispatcher: hundreds of original named
cases share compiler code and run in isolated case processes. The observed
consolidation worker footprint was about 350 MB with two relinks after one-case
edit; it did not meet every latency target. **Never restore hundreds of duplicated
large statically linked test executables.** ffc's heavy code/library linking is a
separate cost center from runtime corpus comparisons and must be measured.

The slowest wrappers include `test_fortfront_corpus_conformance`,
`test_conformance_gauntlet_smoke` and dashboard/provenance operations. For
continuous discovery, register leaf corpus case IDs or tiny completed batches
through fo's project adapter, not a random whole-wrapper selection that consumes
each generation's budget and loses unfinished output. Preserve original public
wrapper names, invalid/NOREF checks and their complete milestone verification;
a sampled leaf result is not evidence that the full wrapper passed.

Adapter rules encode ffc source modes, corpus pins, expected dispositions,
independent gfortran/twin/clause oracles, stdin/dependency closure and runtime
artifacts. Generic lifecycle/cache/selection/parallel/disk logic stays in fo.
A project hook/configuration is allowed when needed, but it delegates to shared
fo state and receipts rather than becoming another test scheduler. Fix fo when
its existing dispatcher or missing adapter capability causes inefficiency.

Reuse valid reference outputs and compiled actions, share immutable Git/CAS
content and private mutable build materialization, and avoid copying whole
LFortran/GCC corpora per worker/generation. Preserve applicable debug-payload
reduction, linker DCE, zero production includes/order shims and stale-profile
hygiene. Optional shared development linking lands only with measured ABI/toolchain
benefit; it is not a second permanent compiler ABI. Track compiler binary count,
link count/time, retained bytes, peak RSS and actual nested corpus process count.
Current/last-compilable/pinned reproducer data are separate from collectible
inactive generated profiles. Never clean valuable worker source or active evidence.

The generic [fo project matrix](https://github.com/lazy-fortran/fo/blob/main/PLAN.md#project-matrix)
includes focused ffc cases early, then its full pinned corpus at milestones,
plus all maintained fo projects and selected external fpm projects. fo
[#144](https://github.com/lazy-fortran/fo/issues/144) atomic artifact publication
and [#145](https://github.com/lazy-fortran/fo/issues/145) matrix evidence are
provider requirements. A cached or existing static archive must be complete and
correctly keyed before a linker can consume it; pathname existence is insufficient.

### Existing harness adapter

After the fo enabling stage, register the following existing ffc operations
with its project policy. Build and named tests use the normal fo interface:

```bash
export LIBRARY_PATH="$PWD/../liric/build"
fo build
fo test test_session_stop_code_compiler
FO_JOBS=1 fo test --random 12 --seed 1729
```

These one-shot commands do not implement continuous supervision. Snapshot corpus
campaigns must invoke a frozen matching ffc harness checkout, with a frozen
runtime and dependency vector; its editable worker checkout is separate:

```bash
scripts/conformance_gauntlet.sh --suite lfortran \
  --ffc "$FFC_SNAPSHOT" --jobs 1 \
  --files-from "$RUN_DIR/batch-0001.files" \
  --report "$RUN_DIR/batch-0001.jsonl" \
  --observations "$RUN_DIR/batch-0001.observations.jsonl" \
  --ref-cache "$REFERENCE_CACHE" --timeout 5
```

Use seeded shuffled suite-relative lists. Gauntlet `--sample` randomizes membership
but sorts execution back into corpus order. Current gauntlet publishes only at
batch completion and deletes unfinished staging on cancellation: complete tiny
batches, preferably one case when exact retention matters, until fo's durable
per-case adapter is available. Previously completed batches survive; current
batch interruption remains untested. Give every batch distinct paths/scratch.
Never run background `fo test` in the editable build tree while another worker
mutates/rebuilds it. Project locking and immutable materialization still apply.

`--ffc` and `--require-provenance` are mutually exclusive. Snapshot discovery
reports are not release epochs. The legacy report comparer requires identical
worktree/suite identities; respect its comparability checks, or reproduce with
a controlled matching environment/independent oracle. Do not bypass provenance
checks simply to obtain a favorable delta. The fo adapter owns compatibility
validation and must repair missing capabilities rather than ffc scripting around
an unreliable driver. Required corpora must exist and have nonempty inventories.

## Ordered compiler roadmap

After the Fo resident/FPM-only route passes its named focused checks and actual
candidate-B public behavior—including Fo #200—reproduce and repair the three
current FFC failures. Then follow the phases below; independent ready slices may
run in parallel in disjoint worktrees. Respect provider/consumer
and ABI dependencies rather than treating every open issue as simultaneously
ready. Every phase ends with updated contracts, owned issue dispositions and a
current integrated-generation test campaign; full final verification runs on a
frozen milestone version. A feature-family row is a complete destination, not a
claim that the existing issue already covers every missing combination.

### 1. Regressions, truth and immediate correctness

Reproduce the six maintained failures, separate source defects from harness or
metadata defects, and file/narrow one issue per observed signature. Prioritize
wrong answers, compiler/runtime crashes, hangs and invalid-program acceptance.
Reproduce outstanding DATA stores (#751/#752), OPEN variable specifiers (#628),
nested-tail lowering (#626), expression-call failures (#671), FORALL snapshot
semantics (#673), nondeterminism (#649), runtime hash overflow (#758) and
load-dependent timeouts (#478/#531/#754). Existing slices may already fix an
issue; require exact current-scope evidence before closing it.

Rebaseline stale corpus umbrellas #576/#609 by current named cases; no historical
failure count is a current denominator. #532/#540 refresh owned FAIL/XFAIL/XPASS
metadata using observations, never row deletion. Preserve per-file negative
validity oracles and near-miss positives. Diagnostic work includes FortFront
#2883/#2897/#2951/#2970 plus source spans, correct classes and rejection-stage
behavior. Infrastructure failures remain visible, not feature passes.

### 2. Canonical descriptors, storage and lifetimes

Finish descriptor views/sections (#337), runtime bounds, negative/variable strides,
noncontiguous and higher-rank sections, gathers/scatters (#399), character
dummies/results (#348), allocation, pointer remapping, optional/value passing,
reallocation and alias-safe temporary ownership. Complete the ABI beyond its
current seven dimensions to standard rank 15 and assumed-rank/SELECT RANK.
Eliminate competing inline derived-component layouts; migrate all component
arrays and character owners to the canonical model. Retain the legitimate
caller/callee runtime-shape producer pair; closed/mechanically retired paths are
not justification for deleting valid descriptor producers.

Restore all standard ALLOCATE/DEALLOCATE status/error semantics, including an
unallocated DEALLOCATE error; implement SOURCE/MOLD/STAT/ERRMSG, automatic arrays,
allocatable/pointer components/results/dummies and full character kind/length
semantics beyond current default-character-kind support. Descriptor metadata
must survive separate compilation and parameterized types. Use published
ownership/descriptors and matching behavioral lifetime/alias/error oracles.

### 3. Binding, procedures and module ABI

Finish authoritative binding identity across host, USE, BLOCK and ASSOCIATE
(#584), two-pass specification/scope resolution, forward references and correct
shadowing. Lower all permitted external/internal/module/submodule program units,
ENTRY (#456), procedure-only objects (#582), explicit/abstract interfaces,
optional/keyword/value/pointer/class arguments (#579), scalar procedure dummies
(#467), procedure pointers (#461), NOPASS components (#522), recursion,
pure/elemental procedures and generic resolution (#415).

Version `.fmod` metadata for binding identities, module variables, types/PDTs,
dummy names/results, generics, parent submodules, target/runtime ABI and
invalidation. Test producer/consumer compile-link-run plus schema incompatibility.
Nested internal procedures must be diagnosed according to their actual standard
legality; implement permitted nesting and keep extension nesting separate.
Cross-unit CLASS(T) ABI and dispatch are authorized roadmap work, not a standing
extra-approval barrier. Land additive metadata before migrated lowering clients.

### 4. Expressions, arrays, derived types and polymorphism

One type/kind/rank-directed scalar/intrinsic dispatcher (#453) and one descriptor
array-expression engine serve arithmetic/conversions, comparisons, elemental
calls, constructors/implied-DO (#459), declarations (#458), whole-array and
section assignment, WHERE/FORALL/DO CONCURRENT locality/reduction, reductions,
PACK/UNPACK/SPREAD/RESHAPE/MATMUL and full TRANSFER (#435). Correct aliasing,
shape, lower bounds and evaluation count are independent acceptance properties.

Complete derived scalar values (#449), nested/array/allocatable/pointer components,
constructors/default initialization, PDT inheritance, extension, polymorphic
arrays (#422), dynamic SELECT TYPE (#419), vtables/type-bound generic dispatch,
defined operations/assignment (#462/#465), finalization and allocatable lifetime.
A bounded existing scalar/rank-one slice does not close its full family.

### 5. Control flow, I/O, intrinsics and interoperation

Complete IF/CASE/RANK/TYPE, DO/DO CONCURRENT, BLOCK/ASSOCIATE, labels, GOTO,
EXIT/CYCLE (#455), RETURN/STOP/ERROR STOP and expression side effects. Preserve
valid fixed/free source forms, continuations, standard legacy storage/declarations,
DATA/COMMON/EQUIVALENCE/BLOCK DATA and all permitted language-mode combinations.
Named-loop fixes are delivered; broader control-transfer scope still needs audit.

Complete standard list/formatted/unformatted/internal/external record I/O,
format edits and positioning (including trailing X), files/units, OPEN/CLOSE/
INQUIRE, NAMELIST (#460), stream/direct/sequential/asynchronous/WAIT/FLUSH and
error/EOF/EOR/IOSTAT/IOMSG behavior. Replace local ad hoc lowering with the
Fortran runtime contract. Test exact required behavior without equating arbitrary
processor-dependent list formatting with universal standard semantics.

Complete every standard intrinsic and intrinsic module: numeric/complex/logical,
character, bit, array inquiry/transformation/reduction, ISO_FORTRAN_ENV, IEEE
arithmetic/exceptions and ISO_C_BINDING. C interoperability includes BIND(C),
VALUE, interoperable derived types/enums, data/procedure pointers and the C
array/character descriptor surface. Extend kind support according to processor
capability, with explicit unsupported-kind diagnostics rather than silent narrowing.

### 6. Fortran 2023 additions

The [delta audit](docs/F2023_DELTA.md) supplies exact reproductions, clauses and
producer/consumer dependencies. All 51 audit entries remain accounted for; five
previous ISO-parallel exclusions are now required implementation work. The
original scope-audit #473 is complete; feature issues remain open.

| Feature group | Producer / consumer work |
| --- | --- |
| Source budgets; TYPEOF/CLASSOF; RANK and array bounds | FortFront #3022/#3023/#3024/#3027/#3028 |
| Interoperable/noninteroperable named enums and BOZ | FortFront #3025/#3026; ffc #768/#769/#794/#795 |
| Multiple @ subscripts/triplets | FortFront #3029; ffc #793 |
| Conditional expression and conditional actual | FortFront #3030/#3031; ffc #791/#792 |
| SIMPLE, REDUCE locality | FortFront #3032/#3033, followed by independent native semantics |
| NAMELIST privacy, deferred output/allocation messages | ffc #767/#770/#771/#772/#773 |
| Vector allocation/pointer bounds | ffc #774/#775 |
| AT and leading-zero controls | FortFront #3034; ffc #776/#777 |
| New intrinsic identities, degree/pi trig, logical kind, SPLIT/TOKENIZE, clock/kind constants | FortFront #3035; ffc #778–#784 |
| C string interop and C_F_POINTER LOWER | ffc #785/#786/#787/#796 |
| IEEE extrema and pure/simple rounding/underflow/mode/status operations | ffc #788/#789/#790/#797 |
| Coarray components, notifications, collective image-local errors | Parallel phase below; ffc #799–#802 and FortFront #3036 |

For absent consumer tasks, file the smallest atomic follow-up after proving the
producer contract; do not close a producer and pretend native behavior follows.

### Standard parallel features

This phase is mandatory even if most corpus tests are sequential. Start with
[FortFront #3036](https://github.com/lazy-fortran/fortfront/issues/3036) public
codimension/image-selector metadata and [ffc #799](https://github.com/lazy-fortran/ffc/issues/799)
scalar single-image allocation/inquiry. Single-image success is a foundation,
not complete coarray support. [ffc #800](https://github.com/lazy-fortran/ffc/issues/800)
defines the versioned multi-image runtime ABI and splits executable children
before closing; bind transport/runtime through an explicit compiler-neutral API.

Then implement scalar/array/derived coarrays and coarray components, cobounds,
coindexed reads/writes, image lifecycle, segment ordering, SYNC ALL/IMAGES/MEMORY,
locks, atomics, events, collectives, teams/change-team and stopped/failed-image
status propagation. Each child needs deterministic two/multi-image trace oracles,
valid/invalid neighbors, timeout detection and lifetime/error coverage. Never
substitute single-image constants for multi-image semantics.

F2023 [#801](https://github.com/lazy-fortran/ffc/issues/801) owns NOTIFY_TYPE,
NOTIFY= and NOTIFY WAIT; [#802](https://github.com/lazy-fortran/ffc/issues/802)
owns image-local collective error outcomes after collective foundations. Expand
producer queries with narrow child issues where needed. All parallel additions
remain unsupported until their runtime and independent oracles are implemented.

### 7. Accepted Lazy Fortran and proposals

Preserve standard-mode semantics while completing accepted strict defaults,
infer-mode scripts/first-assignment inference/walrus, arrays and specialization.
Monomorphization #437 and versioned cross-module specialization #433 consume
stable producer identities. Correct walrus redeclaration and signed/unsigned
extension mixing without counting those extensions as F2023 features.

Accepted generics/template/requirement/instantiate/traits contracts get producer,
native consumer and serialization tests. Standard proposals #745/#753 are design
inputs until accepted. Synthesis remains standard #756 → FortFront #2976 → ffc
#632 → fo #120 after normative acceptance; do not restart speculative syntax in
parallel with an unaccepted language contract.

### 8. Backends, performance and release

LIRIC [#533](https://github.com/krystophny/liric/issues/533) trustworthy producer
artifacts precede [#523](https://github.com/krystophny/liric/issues/523)
dominance/serialization evidence. [#535](https://github.com/krystophny/liric/issues/535)
removes unnecessary dependency debug payload. Keep direct session APIs; native
fast compilation is the development default, LLVM optimized code a separate
measured path. No private LLVM/MLIR/HLFIR/text-IR backend revival.

Maintain zero production .inc/order shims, smaller typed services and frozen
lowering-context limits; mechanical folding is done. Measure incremental edit,
no-op build, link count, compiler latency/runtime performance and disk retention.
Earlier targets remain direction: no-op under one second, single edit under
60 seconds, maintained full suite under 120 seconds, build footprint under 2 GB.
Observed results have not met all targets; background subset preemption limits
ordinary iteration cost while real performance work proceeds. Optional shared
development linking requires measured benefit and ABI/platform support.

Freeze and verify GNU/LLVM/NVHPC/platform/sanitizer/separate-compilation matrices
promised by the project. Do not touch protected hosts without the user's named
host request. A final release requires comprehensive exact-version evidence,
current issue owners and a truthful support contract, not a clean random sample.

## Verifier and delivery protocol

Every semantic slice needs a focused independent executable oracle, invalid
neighbors and boundary/interaction cases. Existing parity tools under `tools/`
cover scalar semantics, logical/BOZ output, format edits, names/constructs,
character allocation/dummies/substrings, derived access, vector gather/scatter,
allocation, descriptor reductions, MASK/MERGE/ABS, implied-DO I/O and string
search. Preserve their controlled falsification evidence and byte-exact checks
where the language guarantees it; do not create twenty repeated identical rows
as a substitute for independent behavior. Pure planning changes need prose/link
validation, not compiler tests that cannot observe them.

Iteration promotion needs a coherent build, targeted oracles on the combined
candidate and inspection of already completed background regressions. Once the
exact integrated generation is `local_gate_green` with no known current
regression, the controller commits and pushes it to `main` immediately. It does
not wait for the full random campaign, GitHub CI or unrelated review. CI audits
the latest main asynchronously and may cancel superseded runs. A current
reproducible regression gets immediate repair/revert priority; unrelated workers
may continue, but unrelated main promotions normally pause. Never weaken/delete tests, widen XFAIL,
drop negative cases or change scope to manufacture green. Contract updates and
original test identities remain mandatory. Cross-repo ABI changes get producer,
consumer and combined interaction evidence before promotion; a full exact-version
suite is a milestone/release check, running in background and preempted by newer
fixes until the milestone version is final.

Useful committed checks:

```bash
scripts/fetch_corpora.sh --verify-only
scripts/check_lowering_context_freeze.sh
python3 tools/assert_no_legacy_runtime_shape.py
scripts/audit_manifest_owners.sh
FO_JOBS=1 FO_TEST_REPORT_NAMES=1 fo test --all
scripts/corpus_rejection_gate.sh \
  --corpus "$FFC_FORTFRONT_DIR/examples" --ffc "$FFC_SNAPSHOT" \
  --out "$RUN_DIR/rejections.tsv" \
  --baseline test/fixtures/corpus_rejection_baseline.tsv
```

Use isolated `/var/tmp` scratch and explicit frozen artifact paths. Verify both
positive acceptance and invalid-program rejection, compare names/signatures
rather than only counts, and classify missing or empty corpora as unavailable
coverage. The descriptor guard and context ceiling are architecture verifiers,
not substitutes for behavioral oracles. Use current Fo public commands and
report baseline findings precisely; fix fo workflow defects first, using the
CLI if attached MCP is stale. Bare `fo` resident start/attach and the legacy
pipeline under `fo verify` are planned transitions, not current command claims.
Do not route around the driver.

For a final comprehensive epoch, use one clean fixed ffc/dependency bundle and
all four nonempty pinned suites, then lock and generate the dashboard. Run the
following gauntlet command once per suite, with SUITE set to fortfront-f90,
fortfront-lf, lfortran and gfortran-dg:

```bash
scripts/conformance_gauntlet.sh --suite "$SUITE" --jobs 1 \
  --require-provenance --report "$EPOCH_DIR/$SUITE.jsonl" \
  --observations "$EPOCH_DIR/$SUITE.observations.jsonl"
python3 scripts/lock_conformance_epoch.py \
  --report fortfront-f90="$EPOCH_DIR/fortfront-f90.jsonl" \
  --report fortfront-lf="$EPOCH_DIR/fortfront-lf.jsonl" \
  --report lfortran="$EPOCH_DIR/lfortran.jsonl" \
  --report gfortran-dg="$EPOCH_DIR/gfortran-dg.jsonl" \
  --output "$EPOCH_DIR/parity_epoch.json"
```

`generate_parity_dashboard.sh` consumes the same four reports plus `--epoch-lock`,
`--snapshot` and `--output`. Only an exact union of disjoint unsampled,
provenance-verified shards for that same version can form a full epoch. Snapshot
sampling from different versions never qualifies. Reclassification from saved
observations need not recompile, but oracle/input changes invalidate them.

After each meaningful delivery, rewrite the active stage, capability claims and
issue dispositions in this PLAN and fo's provider plan. Close/narrow issues with
exact receipts; historical completion does not imply current corpus green.
Workers may commit task branches; only controller integrates/promotes. Stage
explicit paths, preserve unrelated edits, commit/push main regularly, then fetch
and fast-forward pull involved repositories and verify convergence. One short
adversarial review checks whether the change fixes the cause, whether its oracle
can still pass with the defect, and whether analogous sites remain; repair
blocking findings before promotion. Keep output/tool calls below about 300 lines.

## Issue index and maintenance

The rows below enumerate the pre-implementation open ffc queue as of 2026-10-03,
plus the new Gremlin integration and ISO-parallel issues. They are ownership
pointers, not evidence that all historical descriptions remain current. Query
live issue state before assigning; update this table as behavior moves.

| Issue | Work / current disposition |
| --- | --- |
| [#337](https://github.com/lazy-fortran/ffc/issues/337) | [arraydesc-04] represent array sections as descriptor views |
| [#339](https://github.com/lazy-fortran/ffc/issues/339) | [arraydesc-07] retire legacy runtime-shape metadata |
| [#348](https://github.com/lazy-fortran/ffc/issues/348) | [chardesc-03] pass character dummies and results by descriptor |
| [#399](https://github.com/lazy-fortran/ffc/issues/399) | [vector-subscript-01] lower vector subscripts as gather views |
| [#415](https://github.com/lazy-fortran/ffc/issues/415) | [fmod2-03] serialize rank-aware generic specifics |
| [#419](https://github.com/lazy-fortran/ffc/issues/419) | [poly-03] lower runtime SELECT TYPE guards |
| [#422](https://github.com/lazy-fortran/ffc/issues/422) | [poly-06] pass and allocate polymorphic arrays |
| [#433](https://github.com/lazy-fortran/ffc/issues/433) | [lf-monomorph-fmod-01] stabilize cross-module Lazy specializations |
| [#435](https://github.com/lazy-fortran/ffc/issues/435) | [transfer-02] lower sized and array-valued TRANSFER |
| [#437](https://github.com/lazy-fortran/ffc/issues/437) | [lf-monomorph-01] emit one-unit Lazy procedure specializations |
| [#449](https://github.com/lazy-fortran/ffc/issues/449) | [derived-value-01] lower plain scalar derived values |
| [#453](https://github.com/lazy-fortran/ffc/issues/453) | [intrinsic-dispatch-01] centralize scalar intrinsic calls |
| [#455](https://github.com/lazy-fortran/ffc/issues/455) | [control-transfer-01] lower structured branch targets |
| [#456](https://github.com/lazy-fortran/ffc/issues/456) | [entry-01] lower procedure ENTRY points |
| [#458](https://github.com/lazy-fortran/ffc/issues/458) | [array-decl-core-01] migrate core array declarations |
| [#459](https://github.com/lazy-fortran/ffc/issues/459) | [array-constructor-01] lower typed array constructors |
| [#460](https://github.com/lazy-fortran/ffc/issues/460) | [namelist-write-01] write scalar NAMELIST groups |
| [#461](https://github.com/lazy-fortran/ffc/issues/461) | [proc-pointer-core-01] complete scalar procedure-pointer state |
| [#462](https://github.com/lazy-fortran/ffc/issues/462) | [defined-assign-core-01] recurse through defined assignment components |
| [#465](https://github.com/lazy-fortran/ffc/issues/465) | [defined-assign-array-01] lower elemental defined assignment over arrays |
| [#467](https://github.com/lazy-fortran/ffc/issues/467) | [proc-dummy-core-01] lower scalar procedure dummy arguments |
| [#475](https://github.com/lazy-fortran/ffc/issues/475) | [lint-gate-01] clear the 27 pre-existing fo lint warnings and make the CI lint gate blocking |
| [#478](https://github.com/lazy-fortran/ffc/issues/478) | [repro-03] a corpus case at half the timeout budget makes the snapshot load-dependent |
| [#522](https://github.com/lazy-fortran/ffc/issues/522) | [proc-component-02] unblock pr66465: NOPASS procedure components and PRINT of ASSOCIATED |
| [#531](https://github.com/lazy-fortran/ffc/issues/531) | [corpus-regress-01] benchmark_5000_lines.f90 regressed to FAIL in the maintained fortfront-f90 corpus |
| [#532](https://github.com/lazy-fortran/ffc/issues/532) | [corpus-xpass-01] reclassify 65 XPASS cases and regenerate the parity dashboard |
| [#540](https://github.com/lazy-fortran/ffc/issues/540) | [corpus-manifest-01] refresh stale fail_owners entries for files that now pass |
| [#576](https://github.com/lazy-fortran/ffc/issues/576) | [corpus-segv-01] test_fortfront_corpus_conformance fails on main with 12 cases, several segfaulting at runtime |
| [#579](https://github.com/lazy-fortran/ffc/issues/579) | [proc-args-02] accept valid actual arguments for POINTER, CLASS and scalar dummies |
| [#581](https://github.com/lazy-fortran/ffc/issues/581) | [reject-false-01] stop rejecting valid specification parts and statements |
| [#582](https://github.com/lazy-fortran/ffc/issues/582) | [external-unit-02] compile procedure-only translation units to objects |
| [#584](https://github.com/lazy-fortran/ffc/issues/584) | [scope-06] resolve declarations across host, ASSOCIATE and module-export boundaries |
| [#609](https://github.com/lazy-fortran/ffc/issues/609) | [corpus-gaps-01] seven fortfront-f90 corpus cases fail to compile on unimplemented features |
| [#626](https://github.com/lazy-fortran/ffc/issues/626) | Statement after a nested DO loop is dropped from the emitted code |
| [#628](https://github.com/lazy-fortran/ffc/issues/628) | OPEN: FILE= and STATUS= given as variables are lowered as literal text |
| [#632](https://github.com/lazy-fortran/ffc/issues/632) | Support Fortran Synthesis as an optional native frontend path while preserving standard-Fortran lowering |
| [#649](https://github.com/lazy-fortran/ffc/issues/649) | [corpus-flake-02] two genuine FLAKY cases: nondeterministic MINLOC binary and random-branch comparison |
| [#669](https://github.com/lazy-fortran/ffc/issues/669) | [char-elem-substring-01] a substring of a character array element does not lower |
| [#671](https://github.com/lazy-fortran/ffc/issues/671) | [expr-call] contained function call as an expression operand is skipped or hangs |
| [#673](https://github.com/lazy-fortran/ffc/issues/673) | FORALL with a conflicting assignment prints a wrong answer |
| [#751](https://github.com/lazy-fortran/ffc/issues/751) | DATA on COMMON-bound array loses value: stores land before COMMON rebinding |
| [#752](https://github.com/lazy-fortran/ffc/issues/752) | DATA integer(8) scalar prints low 32 bits (64-bit DATA store width) |
| [#754](https://github.com/lazy-fortran/ffc/issues/754) | corpus conformance suite: fortfront-f90 exceeds test wall budget; read_statement.lf EOF run has no lazy oracle |
| [#758](https://github.com/lazy-fortran/ffc/issues/758) | UBSan: FNV-1a hash relies on signed int64 overflow (ffc_runtime_link.f90:114) |
| [#760](https://github.com/lazy-fortran/ffc/issues/760) | named DO nested >=2 levels below unnamed DOs is rejected as 'Unrecognized statement' |
| [#762](https://github.com/lazy-fortran/ffc/issues/762) | allocate(a(2:4)) non-unit lower bound refused on rank-1 allocatable (contract promises rank-1 allocate + bounds access), refused with a misleading "integer expressions" message |
| [#763](https://github.com/lazy-fortran/ffc/issues/763) | dotted relational operators mis-split as real literals: print *, 1.lt.2 emits 1.00000000 instead of T |
| [#765](https://github.com/lazy-fortran/ffc/issues/765) | B, O, Z edit descriptors unsupported; measured gfortran spec incl. Z blank-padding and width-overflow stars |
| [#766](https://github.com/lazy-fortran/ffc/issues/766) | MASK= unsupported on sum/maxval/minval (diagnostic blames correct arity); count(mask=) and merge unsupported |
| [#767](https://github.com/lazy-fortran/ffc/issues/767) | [f2023-namelist-01] allow PUBLIC groups with PRIVATE objects |
| [#768](https://github.com/lazy-fortran/ffc/issues/768) | [f2023-boz-real-01] reinterpret BOZ bits in real initialization and assignment |
| [#769](https://github.com/lazy-fortran/ffc/issues/769) | [f2023-boz-ctor-01] accept BOZ elements in typed integer and real constructors |
| [#770](https://github.com/lazy-fortran/ffc/issues/770) | [f2023-errmsg-01] allocate deferred-length allocation error messages |
| [#771](https://github.com/lazy-fortran/ffc/issues/771) | [f2023-iomsg-01] allocate deferred-length I/O messages |
| [#772](https://github.com/lazy-fortran/ffc/issues/772) | [f2023-internal-write-01] allocate a deferred scalar internal file |
| [#773](https://github.com/lazy-fortran/ffc/issues/773) | [f2023-intrinsic-char-out-01] allocate deferred character intrinsic outputs |
| [#774](https://github.com/lazy-fortran/ffc/issues/774) | [f2023-allocate-bounds-01] expand vector allocation bounds |
| [#775](https://github.com/lazy-fortran/ffc/issues/775) | [f2023-pointer-bounds-01] expand vector pointer bounds and remapping |
| [#776](https://github.com/lazy-fortran/ffc/issues/776) | [f2023-at-edit-01] emit trimmed character fields for AT |
| [#777](https://github.com/lazy-fortran/ffc/issues/777) | [f2023-leading-zero-01] honor OPEN WRITE and LZ leading-zero controls |
| [#778](https://github.com/lazy-fortran/ffc/issues/778) | [f2023-degree-trig-01] lower degree-based trigonometric intrinsics |
| [#779](https://github.com/lazy-fortran/ffc/issues/779) | [f2023-pi-trig-01] lower half-revolution trigonometric intrinsics |
| [#780](https://github.com/lazy-fortran/ffc/issues/780) | [f2023-logical-kind-01] support runtime SELECTED_LOGICAL_KIND |
| [#781](https://github.com/lazy-fortran/ffc/issues/781) | [f2023-split-01] lower incremental SPLIT token positions |
| [#782](https://github.com/lazy-fortran/ffc/issues/782) | [f2023-tokenize-01] lower TOKENIZE result arrays and token owners |
| [#783](https://github.com/lazy-fortran/ffc/issues/783) | [f2023-system-clock-01] enforce argument-kind constraints and document clock selection |
| [#784](https://github.com/lazy-fortran/ffc/issues/784) | [f2023-iso-env-kinds-01] define LOGICAL storage and REAL16 kind constants |
| [#785](https://github.com/lazy-fortran/ffc/issues/785) | [f2023-f-c-string-01] append NUL with the specified ASIS trimming policy |
| [#786](https://github.com/lazy-fortran/ffc/issues/786) | [f2023-c-f-strpointer-cptr-01] preserve canonical character TARGET storage in C string views |
| [#787](https://github.com/lazy-fortran/ffc/issues/787) | [f2023-c-f-pointer-lower-01] apply the optional LOWER argument |
| [#788](https://github.com/lazy-fortran/ffc/issues/788) | [f2023-ieee-extrema-01] lower IEEE MAX MIN and magnitude extrema |
| [#789](https://github.com/lazy-fortran/ffc/issues/789) | [f2023-ieee-number-extrema-01] implement revised IEEE numeric extrema semantics |
| [#790](https://github.com/lazy-fortran/ffc/issues/790) | [f2023-ieee-arithmetic-state-01] provide pure and simple IEEE rounding and underflow procedures |
| [#791](https://github.com/lazy-fortran/ffc/issues/791) | [f2023-conditional-expr-02] lower selective conditional expressions |
| [#792](https://github.com/lazy-fortran/ffc/issues/792) | [f2023-conditional-arg-02] lower selected actual addresses and absence |
| [#793](https://github.com/lazy-fortran/ffc/issues/793) | [f2023-multiple-subscript-02] lower @ index and section groups |
| [#794](https://github.com/lazy-fortran/ffc/issues/794) | [f2023-enum-type-02] lower named interoperable enum values |
| [#795](https://github.com/lazy-fortran/ffc/issues/795) | [f2023-enumeration-02] lower ordered enumeration values and operations |
| [#796](https://github.com/lazy-fortran/ffc/issues/796) | [f2023-c-f-strpointer-array-01] associate views of TARGET C_CHAR arrays |
| [#797](https://github.com/lazy-fortran/ffc/issues/797) | [f2023-ieee-exceptions-state-01] provide pure and simple IEEE mode and status snapshots |
| [#798](https://github.com/lazy-fortran/ffc/issues/798) | Integrate ffc with fo Gremlin mode |
| [#799](https://github.com/lazy-fortran/ffc/issues/799) | Scalar single-image coarray allocation/inquiry |
| [#800](https://github.com/lazy-fortran/ffc/issues/800) | Multi-image runtime ABI and executable split |
| [#801](https://github.com/lazy-fortran/ffc/issues/801) | F2023 notifications |
| [#802](https://github.com/lazy-fortran/ffc/issues/802) | Image-local collective errors |

Completed planning/tooling deliveries: [#473](https://github.com/lazy-fortran/ffc/issues/473)
original F2023 audit/split, and [#663](https://github.com/lazy-fortran/ffc/issues/663)
committed rejection gate. Their completion claims are limited to those
acceptance scopes. Feature gaps and current dashboard refresh remain open.

FortFront open producers: #2883/#2897/#2951/#2970; #3018/#3019/#3020/#3021
require current-scope confirmation; #3022–#3036 cover F2023 and coarray metadata.
fo enabling/provider issues are linked above; #117/#129 and adjacent tooling
#56/#59/#62 remain local work rather than unbounded prerequisites for compiler
features. LIRIC owns #523/#533/#535. Standard's open proposals remain normative
inputs and are queried at their source, never silently promoted to accepted
language. Do not close a manifest owner without migrating its still-failing
owned cases to exact open child issues.

Retired competing ffc plans: ROADMAP.md, BACKLOG.md, DESIGN.md,
docs/PARITY_PLAN.md and the resolved character-allocation exploration. Their
remaining scope/architecture/verification requirements are integrated here;
Git history retains the old evidence narratives. CLAUDE.md points to AGENTS.md.
Technical contracts, operational harness documentation and generated measurements
remain separate references, not alternative work orders.
