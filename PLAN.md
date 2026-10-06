# FFC compiler goals

Updated 2026-10-06. This is the compiler delivery plan; issues specify observable
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

Current focus: make Fo's resident Gremlin workflow fast and dependable on
`main` before further FFC implementation. The latest FFC compiler/test code
is at `983e0b3`; the latest
six-case focused gate passed, while 503 of 509 inventory cases remain unknown.
The same-generation follow-on session restarted a 516-node build and was
stopped before testing. Fo `22f6dd3` removed the forced single compile worker;
the installed driver SHA256 is
`5bac5856a5e9425b804ec3a46f75708d8dfed195fa5fa79bd3cc755878e5b413`.
A fresh public Gremlin fixture observed overlapping compiler processes with
`FO_JOBS=2` and passed its required case. Warm action restoration remains a
separate Fo speed issue. Recheck the FFC consumer after that repair before
resuming compiler work. The unrelated character-prefix edit remains preserved.

1. Recheck the current FPM project and public shared test dispatcher through
   the fixed Fo driver, preserving original case names and avoiding duplicated
   static test binaries. [#798](https://github.com/lazy-fortran/ffc/issues/798)
   owns the consumer goal.
2. After Fo is usable, resume bounded current-generation discovery and repair
   the next confirmed current compiler failure. Two class-star real-kind
   compiler defects are confirmed and repaired; no third failure is confirmed.
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

2026-10-05 follow-on evidence: on FFC base
`682cdf2e4e962ec6aa41d5b93a3a89dda9ba910b`, pinned Fo driver SHA256
`4db8ba18608c0375506cc5e10c3d14fa98fcd28e812b44f508dfc4e7c0547fa2` built
generation `855a3748a2e10455cadb3b67ecefd9fe05dc536a31696d94e05cde2a3fa11708`.
Session `593714-1791225341-736750343` passed all three required cases:
`test_session_pdt_inheritance_compiler`, `test_parity_dashboard` and
`test_conformance_epoch_lock`, then reached quiescence with zero failures.
The production dashboard snapshot now matches current FortFront and LIRIC
revision/tree provenance; corpus-file hashes were unchanged, so the recorded
suite results were retained.

A separate earlier generation, `a45f02dbf3c9d64bbe8e5d33731e664a37d42ce9e0559e668e72e0bfbb4cc94b`,
completed 112 distinct ordinary cases plus its two gate cases with all 114
receipts passing in session `546835-1791224294-328783100`. It included passes for
`test_session_class_star_rank2_assumed_shape_compiler` and
`test_session_pdt_constant_compiler`. This sample predates the dashboard
provenance refresh and remains separate evidence. No current FFC compiler
failure has been confirmed yet.

2026-10-05 promoted-main discovery: FFC commit
`958e4cb9000510ff62df61f7bfef5c18f3f94d18` plus the preserved worktree patch
SHA256 `fe8ded3da0aedbb526135601778b194e8fa5b3aeaefb0997d89deb20b8de4218`
built Fo generation
`f4d90ca468d1df2245e0892d72efc36bad1cb95f123f94433c7411eb361e9142` with
driver SHA256
`4db8ba18608c0375506cc5e10c3d14fa98fcd28e812b44f508dfc4e7c0547fa2`.
Session `617029-1791225994-179028879` passed the build, both required gate
cases (`test_parity_dashboard`, `test_conformance_epoch_lock`) and 37 additional
distinct cases: 39/39 test receipts passed, zero failed, with one in-flight case
cancelled at cooperative stop. The 509-case inventory has 470 unknown cases
remaining. This is current promoted-main discovery, not full-suite verification;
continue bounded discovery before moving to standalone FPM work.

2026-10-05 Gremlin test-input repairs: session
`641046-1791226584-288695610` on generation
`6892a6e6d2010a59100317efa9d3c36ee1ad4b7f1b63045bfe79f211461920b3` found
current failures in `test_conformance_report_worktree` and
`test_conformance_isolation`; targeted session
`671003-1791227142-066012299` reproduced both. The Fo execution view omitted
FortFront's sibling `examples/f90` corpus, which those tests had assumed, and
the worktree test also used an undeclared comparison script. These were FFC
test-closure failures, not compiler or Fo defects. FFC commit
`b68e807` adds a generated local corpus fixture and declares
`scripts/compare_conformance_reports.sh` in `fpm.toml`.

Before promotion, the exact worktree patch SHA256 was
`1e6bfc0e19fb2fd7692cd7d169053c7dbcf470b3d01bd10f03b78d92446ef254` on base
`c710c75aacc0a9a7fa0f475c3bf65f4b850da5cb`. Pinned Fo driver SHA256
`4db8ba18608c0375506cc5e10c3d14fa98fcd28e812b44f508dfc4e7c0547fa2` built
generation `bda0db1c79647553f1290c6e8ac2e55deb1d1b2b434985804672f1ddcedd315c`.
Session `740841-1791228145-245395678` passed all four required cases:
`test_conformance_report_worktree`, `test_conformance_isolation`,
`test_parity_dashboard` and `test_conformance_epoch_lock`; it reached quiescence
with the local gate green and zero failures. Continue current-failure discovery
from the promoted FFC candidate.

The next current-generation failure was
`test_conformance_flake_detection` in session
`785003-1791228864-526493689` on generation
`a2e745d9603d9cbd129e8059f990a0fef3d6c510c1735c1d24418e20e950c77f`. Its
repeat-run fixture also assumed the absent FortFront corpus. FFC commit
`2e68dce` gives it a generated local source and sets `FFC_FORTFRONT_DIR` for
both repeated-run paths. On base `83af90a371fe1bfeb22615739be6ec2258b383e4`
plus worktree patch SHA256
`0d3c7ab4aa690620193c4402dbec68e5591b050360e6e28af0b9027c775ce88f`, pinned
Fo driver SHA256
`4db8ba18608c0375506cc5e10c3d14fa98fcd28e812b44f508dfc4e7c0547fa2` built
generation `fdcbf6ea734c80ce04d661aa4b1dfe3eae1d9f3f73623fee571cb8e38d63433d`.
Session `821459-1791229318-423056034` passed all five required cases: the
three repaired conformance tests, parity and epoch-lock.

After promotion, source base `2e68dce7099d7e1ca05a22b4f6f496e880589ebd` plus
the preserved character-prefix worktree patch SHA256
`fe8ded3da0aedbb526135601778b194e8fa5b3aeaefb0997d89deb20b8de4218` built
generation `29a28a3a2f5928f69f21c1b4dc1d0d481cf1c9ef9b834dca29b6762958210ac4`.
Session `861760-1791229714-077265582` passed all five gate cases and 38
additional distinct cases: 43/43 receipts passed, zero failed, with one
in-flight case cancelled at cooperative stop. The 509-case inventory has 466
unknown cases remaining. The three confirmed failures to date were FFC
conformance-test input closure defects, all repaired; no compiler defect or Fo
Gremlin implementation defect has been confirmed. Continue bounded discovery;
this is not full-suite verification.

The next post-plan candidate used FFC base
`27b84f4f6463f1a841022ac8b344752de92b71f8` plus the preserved character-prefix
worktree patch SHA256
`fe8ded3da0aedbb526135601778b194e8fa5b3aeaefb0997d89deb20b8de4218`. Pinned
Fo driver SHA256
`4db8ba18608c0375506cc5e10c3d14fa98fcd28e812b44f508dfc4e7c0547fa2` built
generation `9012fa1f5b1cef4dd7c3c0ef19ff44e3311b3b1596ed60dd422805e47a8bdbbd`.
Session `909369-1791230353-163739632` passed the five required cases and 36
additional distinct cases: 41/41 test receipts passed, zero failed, and one
in-flight case was cancelled on cooperative stop. The 509-case inventory has
468 unknown cases remaining. This is a separate generation from the prior
43-case sample; do not combine generation receipts into a full-suite claim.
Continue bounded discovery on the promoted code.

The historical 2026-10-02 maintained observation was 502 PASS/6 FAIL/508 names
and a 919-file rejection gate with zero new rejections. These are exact old
observations, not current complete green. Reproduce historical failures on the
newest candidate before assigning their repair or changing their classification.

2026-10-06 class-star real-kind repair: FFC base
`cb7f53207a708d54b0ca22464fee1cee73cb72ce` plus owned worktree patch SHA256
`1cd1c7614e928ebea37f0a0c66080fa4f0b3a0f279a03cddf74568d09bc6319b` and the
preserved character-prefix test patch SHA256
`fe8ded3da0aedbb526135601778b194e8fa5b3aeaefb0997d89deb20b8de4218` produced
Fo generation `554b865b7733092988653852276897de6e1a2b07cbc7467bb46ff510886a2839`.
Its exact combined worktree patch SHA256 was
`585f2f43a9ef617df6a8ccae0207cd703cd8249e19e2141ff1c93ec90f614c34`; the
character-prefix change is unrelated and remains unstaged. Pinned Fo driver
SHA256 `c3e41b5bc5f6bfb5550832269092b69a53dedf47c62e21fa8fc1984153fc8b5d`
passed the four required cases in session `2803218-1791240336-557229957`:
`test_session_class_star_rank2_assumed_shape_compiler`,
`test_session_class_star_assumed_shape_compiler`,
`test_session_select_type_compiler`, and `test_fortfront_corpus_conformance`.
The select-type case includes a gfortran differential for default-real and
real(8) scalar actuals. The gate is green with zero failures; the 509-case
inventory is not fully verified.

The pilot reproduced two compiler defects: class-star assumed-shape lowering
refused default-real array actuals such as the typed constructor in
FortFront's `issue_2455_array_constructor_arg.f90`, and scalar class-star
literal tagging confused default real with explicit `real(8)`. Lowering now
assigns distinct intrinsic ids, accepts F32 array descriptors, and derives a
scalar literal's actual kind from the shared literal-kind resolver. Continue
bounded discovery; do not combine receipts from the superseded failing
generations with the final green gate.

2026-10-06 rank-3/rank-4 coverage extension: on FFC base
`571b2d383eb9124f3b8d0c5df98c1114ebf72b4b`, class-star test patch SHA256
`bc4d796c8b86861fe37bce9865235696d8020ac7e7ebc2cd7be3647127b5ba4f` plus
the preserved unrelated character-prefix patch SHA256
`fe8ded3da0aedbb526135601778b194e8fa5b3aeaefb0997d89deb20b8de4218` had
combined worktree patch SHA256
`ba6920a616c788694870f08e074c99607a25fc7deb60f4c79fcc6875a283ede4`.
Pinned Fo driver SHA256
`95222d18cff2b3dfe82dce41827ddcdae202465fe7680d0a9afca5db2a6c9280` built
generation `ab3c07f9e67c44fc17055ec2f875d4af3fb5fb4af03732e88c7f461691b61932`.
Session `2533658-1791303354-375876968` reached quiescence with all six required
cases passing, including a gfortran differential for default-real rank-3 and
rank-4 assumed-shape descriptors. The six case commands took 57.22 s in total;
Fo reported 516 build nodes for the candidate generation, which dominated the
wall time. This does not establish redundant work. The 509-case inventory is
still incomplete: six cases passed and 503 remain unknown. This verified
increment was pushed to FFC `main` as commit `b2ef900`.

Follow-on sample session `2828927-1791305007-708136295` used the same candidate
generation with `input_changed=false`. Its build progress restarted at 0/516
and reached 30/516 within about a minute; the owner was cooperatively stopped
before any cases ran. This confirms expensive repeated graph work but does not
distinguish cache restoration from compilation. The interrupted session adds
no case outcomes. Fo has since removed the forced `FO_JOBS=1` override, but
this FFC generation has not yet been retested with the corrected driver. Measure
its restore/compile split and recheck the six-case consumer gate before
resuming bounded discovery.

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
