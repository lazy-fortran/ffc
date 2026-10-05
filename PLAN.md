# FFC compiler goals

Updated 2026-10-05. This is the compiler delivery plan; issues specify observable
success. Apply [goals and architectural freedom](https://github.com/lazy-fortran/fo/blob/main/doc/GOAL_DRIVEN_DEVELOPMENT.md).
Choose the smallest adequate design and revise it when evidence requires.
Internal representations, module lists and phase/extraction recipes are open.
Existing accepted language and public API/ABI contracts remain authoritative.

## Destination

Compile, link and correctly execute all modern standard Fortran through
Fortran 2023, including valid interactions, separate compilation and every ISO
parallel facility. Diagnose invalid programs accurately. No silently wrong
result, crash or hang is acceptable merely because a feature is difficult.

The target includes Fortran 90/95/2003/2008/2018/2023: arrays and descriptors,
character, derived types/PDTs, polymorphism, procedures/generics, I/O, intrinsics,
interoperability and the standard parallel facilities. Coarrays, images,
synchronization, collectives, atomics, locks, events, teams and notifications
remain required. Today's supported subset and corpus size do not limit this goal.

Accepted Lazy Fortran modes are an additional target. Unaccepted Synthesis,
traits/generics and other proposals do not silently become standard obligations.
Vendor syntax, OpenMP/OpenACC and accelerator-specific behavior remain explicit
extensions; promised compatibility modes retain their own scope.

## Current delivery

Use the actual resident Fo loop while improving it. Fo
[#200](https://github.com/lazy-fortran/fo/issues/200) owns current gate/driver
evidence; its initial four-case gate passed, then background coverage found
current defects and correctly cleared readiness. Scoped repairs are active. Only reproduced FFC
consumer blockers delay the FFC handoff, not the full Fo cleanup backlog.

1. Make the current FPM project and public shared test dispatcher work reliably
   through Gremlin, preserving original case names and avoiding duplicated
   static test binaries. [#798](https://github.com/lazy-fortran/ffc/issues/798)
   owns the consumer goal.
2. Reproduce and repair up to three confirmed current failures through that
   loop. Historical CLASS(*) assumed-shape/PDT cases are starting candidates;
   replace already repaired cases with the next actual failure.
3. Continue useful language-family increments and cross-feature repairs with
   focused gates and finite background coverage.
4. Substantially reduce compiler/tool/test/documentation volume through
   [Fo #205](https://github.com/lazy-fortran/fo/issues/205), preserving supported
   behavior. Consolidation/deletion should accompany delivery rather than become
   another upfront architecture pass.

2026-10-05 initial pilot evidence: at FFC base
`3af3123b8342bc47554204c98667076d7813883e`, pinned Fo driver SHA256
`d4c5eb11749481738910be9110a4d8a497c14144b8b33d265eea3f2ce2a91a7a` built
generation `821fdb57b399ba60aca999654510c810b6c268f145c9c55a82416a656a20a87b`
and passed `test_session_stop_code_compiler` (gate 1/1). The early random sample
completed 12 passes and failed `test_session_external_only_unit_compiler` at
exit 90 before invoking FFC: its relative executable lookup cannot see app
artifacts from the private Gremlin execution view. The named case passes in the
normal checkout (0.27 s), so this is a Fo execution-view blocker rather than a
confirmed compiler failure.

The execution-view blocker is resolved by Fo commit
`b3a387f4990d724fc6ca67041c62d0dcd3f2563e`, promoted with plan update
`100fcca4719a346748388e216ecda57fc13d06b8`. Candidate driver SHA256
`e83f708b1874fe7b403e3f6cd23946920026241203ac8a22f838e90123ffd711` stages
the direct app output in Gremlin's private test view. FFC declares the two
runtime files consumed by its CMake/runtime tests in `fpm.toml`; the runtime
archive test now passes its configured LIRIC build directory to CMake.

The follow-up FFC candidate started from base
`3ea3dd4da3770c4c2de8cc2be46ad0db5321af78`, built generation
`a658abbfeeab98ea53730b88f4ffc257620dad8ab953dfa375c40b2dc9e81715`, and ran
through Fo session `1289217-1791170558-274235797` on lane
`ffc-798-final-candidate-20261005`. All five required cases passed:
`test_session_external_only_unit_compiler`, `test_session_stop_code_compiler`,
`test_runtime_archives`, `test_session_runtime_archive_compiler`, and
`test_runtime_link_compiler`. The seeded lane also completed 48 additional
distinct cases; all 53 outcomes passed with zero failures. It was stopped with
456 of 509 inventory cases still unknown, so this is a green focused gate and
pilot sample, not full-suite verification. Continue current-failure discovery
from this generation before moving to standalone FPM work.

The historical 2026-10-02 maintained observation was 502 PASS/6 FAIL/508 names
and a 919-file rejection gate with zero new rejections. These are exact old
observations, not current complete green. Reproduce historical failures on the
newest candidate before assigning their repair or changing their classification.

## Ownership and contracts

| Provider | Desired responsibility |
| --- | --- |
| FFC | Correct compiler commands, lowering/runtime/ABI and consumer output |
| FortFront | Correct source interpretation, semantics, public facts and diagnostics |
| LIRIC | Correct public session/backend behavior, verification and emitted artifacts |
| Fo/Fx | Fast, correct development execution, shared reuse and exact evidence |
| standard | Accepted language contracts and independently reviewable proposals |

These are ownership boundaries, not mandatory internal layouts. Repair an
observed defect in its owning provider and recheck the original consumer.
Do not add a permanent consumer workaround for a reproducible provider defect.

Read the relevant accepted contracts when a task affects them:
[support](docs/SUPPORT_CONTRACT.md), [runtime ABI](docs/RUNTIME_ABI.md),
[array ABI](docs/ARRAY_DESCRIPTOR_ABI.md),
[character ABI](docs/CHARACTER_DESCRIPTOR_ABI.md) and
[conformance evidence](docs/CONFORMANCE.md). Change a public contract deliberately
with compatible migration/versioning and producer-consumer evidence where needed.

## Remaining language goals

| Family | Observable outcome |
| --- | --- |
| Source, scope and declarations | Valid forms preserve identity; invalid neighbors get precise diagnostics |
| Control flow | All required branches, loops, ENTRY and returns execute with correct side effects |
| Arrays | Standard rank/bounds/shape/sections/constructors/assignment/inquiry preserve values and aliasing |
| Character and I/O | Length, ownership, formatting, units, status and messages obey the selected standard |
| Derived/PDT/polymorphic values | Construction, allocation, dispatch, assignment and finalization are correct |
| Procedures and generics | Arguments, optionality, pointers, bindings and separate modules retain their interfaces |
| Intrinsics and IEEE | Required kinds, ranks, special values, flags and numerical behavior are correct |
| C interoperability | Published interfaces, pointer views and data/calling representations interoperate |
| ISO parallel | Required image-local/multi-image behavior, communication and failure/status semantics work |
| Accepted Lazy modes | Accepted inference/specialization semantics are reproducible and separately verified |

Issues cover scoped pieces of these goals. A slice may declare current unsupported
neighbors, but cannot remove the family from the full standard destination.
Fortran 2023 additions are indexed in [F2023_DELTA.md](docs/F2023_DELTA.md);
that index supplements rather than replaces the full language audit.

## Verification and publication

Use a named independent positive/negative oracle for the changed behavior and
the exact integrated compiler/tool identities. A reference compiler is useful
where it implements the feature; otherwise use the standard clause and an
independent model/implementation. Processor-dependent results require their
specified contract rather than arbitrary byte equality.

The controller commits/pushes locally verified increments promptly. Workers
return evidence and never promote main. Do not wait for full corpus coverage or
GitHub CI while implementation remains available. Old failures are first
reproduced against current code; confirmed current regressions receive priority.

Every generation retains its own evidence. Finite remaining coverage continues
after the focused gate and obsolete work can be superseded. Interrupted cases
remain untested; a marker or compilation cache hit is not PASS.

Declare full completion only on a frozen version with complete language-family
and interaction evidence, truthful maintained/corpus/rejection/ABI/platform
results and no unexplained FAIL/XFAIL/XPASS/FLAKY/SKIP/NOREF/timeout/OOM within
the claimed scope. Classify external extensions and invalid harness cases
explicitly. Performance audits remain independent advisory work.

## Historical evidence

[The pre-revision plan](https://github.com/lazy-fortran/ffc/blob/c204ad27cf3944878f90c634a23172b8b99bc78d/PLAN.md)
retains prior measurements, failing names and earlier architecture assumptions.
Generated dashboards are observations of their pinned epochs, never competing
plans. Keep active issues/current evidence concise and update them when work moves.
