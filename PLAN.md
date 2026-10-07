# FFC compiler goals

Updated 2026-10-07. This is the compiler delivery plan; issues specify observable
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

2026-10-07: FFC `55d5c1a` removes obsolete runtime shape tools, backend
aliases and stale documentation while retaining the shared dispatcher and
supported compiler behavior. Its task branch passed four focused cases. The
exact combined main candidate, including the separately preserved
character-prefix worktree edit, passed five focused cases with zero failures
and `local_gate_green=true` in Fo Gremlin session
`603216-1791358597-803198692`, generation
`352ec69e45358f9d3a1dd20fcd432619f20cca34ad317bcdb68077c842201539`.
The tested Fo driver SHA256 is
`81c096068873c99e0211c1754ab2a122092c0df6817c8b1c4f325b669aa51105`.
All 516 build actions passed. Full coverage remains open: 504 of 509 cases
have no current-generation result. The unrelated character-prefix edit remains
uncommitted and preserved.

FFC `3d28674` moved its four remaining imports from FortFront's broad facade
to focused modules. Its exact combined candidate, including the separate
character-prefix edit, passed four focused cases with zero current failures
and `local_gate_green=true` in session
`648587-1791359646-501616280`, generation
`ca425a71386b3a45e0f8cef568a3c58775de086e1e17180547236df8e5a7c757`.
The earlier five-case receipt belongs to its own generation; these are not a
single full-suite green. This generation has 505 of 509 cases untested.

FortFront `8eced2d` also removed its internal AST facades and compatibility
inheritance layer, retained the live indexed arena and declared example test
inputs. All FortFront FPM test targets compiled; its exact Fo Gremlin candidate
passed 13/13 focused cases. FFC `c9cfdfe` now imports the control AST nodes
from their owning modules. Against that final FortFront candidate, isolated FFC
session `1077304-1791367095-211589100` passed
`test_session_intrinsics_extra_compiler`,
`test_session_maxloc_minloc_rank234_compiler` and
`test_session_stop_code_compiler` (3/3, zero failures) in generation
`f6e57e6851e4441e1e665f34a3ba2866014ef53f6ac02388917fe2946e97a50f`
under exact Fo driver SHA256
`faa5dd3cc2e53990bb92dff06e37bbe76264b8af1acf00c97c00aea0a8b5589c`.
Full FFC coverage remains open. The unrelated primary-checkout
character-prefix edit remains uncommitted and preserved. Resume bounded
current-generation compiler discovery and repair the next confirmed failure
through the resident Fo loop.

2026-10-07 Gremlin pilot follow-up: a 32-case sample on FFC `1d07b80`
reached 65 current-generation PASS receipts and one timeout in
`test_conformance_gauntlet_smoke`. The case has a documented wall path longer
than Fo's default 100-second cap. Raising the project wall cap to 240 seconds
exposed a separate test-closure failure: the smoke and full FortFront corpus
cases looked for a sibling checkout after Gremlin had materialized the declared
corpus in its private execution view. Both cases now resolve that declared
input from the view's `TMPDIR`. Under Fo driver SHA256
`09d0e252eca6875bb3c3ffb435be1fbc04f7eeb0674421cf57198b4d5742b885`,
session `2361525-1791371872-652641592` passed both cases (2/2), with
`local_gate_green=true`, zero failures, and generation
`2b905e37f9470e67979828e754fad6019a54ac54bb3db3a508b70710c8b47645`.
This is a focused gate, not a 509-case full-suite result.

2026-10-07 third current compiler defect: on FFC base
`8d82881aef812146f733c169558aebdcbfa1b74d`, a reduced program from
the maintained `associate_79.f90` failure compiled and exited successfully
with gfortran 16.2.1, while FFC rejected `kind(z%re)` with
`kind argument is not a literal or named entity`. Plain `%re/%im` values
compiled and ran, isolating the missing compile-time KIND query. The owned
source/test patch SHA256 was
`acce89526f0a58435d7ae80b0a4105f30af870ffd633cacab5501bed27e25159`.
FFC now resolves real and imaginary complex components to their real kind and
folds KIND over such components. The shared dispatcher case checks both
complex(4) and complex(8), an ASSOCIATE alias and runtime values; gfortran's
independent oracle exited 37 as expected. Direct `fo test --all
test_session_associate_compiler` passed 1/1. Exact Fo driver SHA256
`5d6ba22ff566db1a4a24c643aafcd5045f7153a9d951d9ab50707469643a53d7`
passed the same focused case in Gremlin session
`3175876-1791373469-238501054`, generation
`ca23274aaddd064eec222e37e316a7fe5b5719e6328d25d7b20b5b2ab1a86803`,
with `local_gate_green=true`, zero failures. The original corpus case also
uses a complex function result as its selector.

2026-10-07 follow-on compiler repair: on FFC base
`ad01eacacd2fb7917ed5f0f1e65cb29264b09f90`, an ASSOCIATE selector that
called a contained complex function still failed with `unsupported scalar
function call or array expression`. The source/test patch SHA256 was
`3f188025ed2761b642181d6346eca86426feab507e8134e5a6db1bd68a809d14`.
FFC now binds the function's complex result buffer to the associate name for
both supported complex widths, with the existing reference-argument and
copyback path. An exact two-width, two-function program exits 37 under both
gfortran 16.2.1 and FFC. Direct Fo tests passed the ASSOCIATE case and the
existing complex function-result case (2/2). Exact Fo driver SHA256
`5d6ba22ff566db1a4a24c643aafcd5045f7153a9d951d9ab50707469643a53d7`
passed both Gremlin cases in session `3211697-1791374303-960695420`,
generation `521e206906983030fde1f626e385e2403bc2931cddeb36685e7a4b34da5f34f4`,
with `local_gate_green=true` and zero failures.

2026-10-07 complex RESULT kind repair: on FFC base
`7795172d937191a8693bcb50ca3a564e35d500e7`, two contained functions
with the same RESULT name but different declared complex kinds caused FFC to
reject the second with `duplicate complex declaration: z`; gfortran compiled
and ran both. FortFront supplied a bare `COMPLEX` return type for the second
function, so FFC now recovers the kind from its body result declaration. The
source/test patch SHA256 was
`da1bde5f3a8732e2a9aab30e1764de495fc091d9c1beca791bbcb82cab25725f`.
The direct gfortran differential case and the affected ASSOCIATE case passed
2/2 under Fo. Exact Fo driver SHA256
`5d6ba22ff566db1a4a24c643aafcd5045f7153a9d951d9ab50707469643a53d7`
passed both Gremlin cases in session `3250682-1791375003-372954126`,
generation `0670496296a7d6240011dca58774a4b9656f6aec0bbb2bf8f1ca80256ba13e0c`,
with `local_gate_green=true` and zero failures. The complete GCC
`associate_79.f90` still fails in FFC on complex `SIN`; a six-line complex
`SIN(z)` program reproduces `unsupported scalar intrinsic: sin` while gfortran
runs it. Keep its corpus owner open for complex intrinsic lowering and complete
the 509-case FFC suite later.

1. Keep the FPM project and public shared test dispatcher working through Fo,
   preserving original case names and avoiding duplicated static test binaries.
   [#798](https://github.com/lazy-fortran/ffc/issues/798) owns this consumer goal.
2. Continue bounded current-generation discovery. Two class-star real-kind
   defects, the complex-component KIND defect and the contained complex
   function-result ASSOCIATE selector and explicit complex RESULT kind are
   repaired. Implement complex `SIN` and verify the complete maintained
   `associate_79.f90` case before closing its corpus owner.
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
unknown cases remaining. At this 2026-10-05 checkpoint, the three confirmed
failures were FFC conformance-test input closure defects, all repaired; no
compiler defect had yet been confirmed. The later 2026-10-06 pilot below
confirmed and repaired two separate compiler defects. This checkpoint is not
full-suite verification.

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
