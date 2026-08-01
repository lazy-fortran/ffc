program test_session_submodule_compiler
    use ffc_test_support, only: expect_exit_status, expect_error_contains
    implicit none

    logical :: all_passed
    ! Parent, submodule, and caller for the three-file separate-compilation
    ! fixtures (#297).
    character(len=*), parameter :: parent_source = &
        'module fmod297_parent'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    interface'//new_line('a')// &
        '        module function twice(x) result(r)'//new_line('a')// &
        '            integer, intent(in) :: x'//new_line('a')// &
        '            integer :: r'//new_line('a')// &
        '        end function twice'//new_line('a')// &
        '    end interface'//new_line('a')// &
        'end module fmod297_parent'
    character(len=*), parameter :: submodule_source = &
        'submodule (fmod297_parent) fmod297_impl'//new_line('a')// &
        'contains'//new_line('a')// &
        '    module function twice(x) result(r)'//new_line('a')// &
        '        integer, intent(in) :: x'//new_line('a')// &
        '        integer :: r'//new_line('a')// &
        '        r = 2 * x'//new_line('a')// &
        '    end function twice'//new_line('a')// &
        'end submodule fmod297_impl'
    character(len=*), parameter :: submodule_source_unrestated = &
        'submodule (fmod297_parent) fmod297_impl'//new_line('a')// &
        'contains'//new_line('a')// &
        '    module procedure twice'//new_line('a')// &
        '        r = 2 * x'//new_line('a')// &
        '    end procedure twice'//new_line('a')// &
        'end submodule fmod297_impl'
    character(len=*), parameter :: caller_source = &
        'program main'//new_line('a')// &
        '    use fmod297_parent, only: twice'//new_line('a')// &
        '    stop twice(21)'//new_line('a')// &
        'end program main'

    print *, '=== submodule compiler test ==='

    all_passed = .true.
    if (.not. test_restated_module_function()) all_passed = .false.
    if (.not. test_separate_module_subroutine()) all_passed = .false.
    if (.not. test_separate_module_function()) all_passed = .false.
    if (.not. test_generic_interface_body_specific()) all_passed = .false.
    if (.not. test_parent_module_not_found()) all_passed = .false.
    if (.not. test_caller_links_deferred_module_procedure()) all_passed = .false.
    if (.not. test_plain_interface_is_not_exported()) all_passed = .false.
    if (.not. test_submodule_compiles_as_its_own_unit()) all_passed = .false.
    if (.not. test_submodule_parent_diagnostics()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: single-file submodules lower against the parent module'

contains

    logical function test_restated_module_function()
        ! #292: a submodule restates and implements a module procedure declared
        ! by an interface body in the parent module; the program calls through.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    module function f(x) result(y)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '      integer :: y'//new_line('a')// &
            '    end function'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            'submodule (m) s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module function f(x) result(y)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '    y = 2*x'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program p'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop f(21)'//new_line('a')// &
            'end program'

        test_restated_module_function = expect_exit_status( &
            source, 42, '/tmp/ffc_session_submod_restated_test')
    end function test_restated_module_function

    logical function test_separate_module_subroutine()
        ! #292: the separate `module procedure P` body inherits its dummy
        ! declarations from the parent interface, so an out argument resolves.
        character(len=*), parameter :: source = &
            'module m2'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    module subroutine setval(x)'//new_line('a')// &
            '      integer, intent(out) :: x'//new_line('a')// &
            '    end subroutine'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            'submodule (m2) s2'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module procedure setval'//new_line('a')// &
            '    x = 9'//new_line('a')// &
            '  end procedure'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program p2'//new_line('a')// &
            '  use m2'//new_line('a')// &
            '  integer :: k'//new_line('a')// &
            '  call setval(k)'//new_line('a')// &
            '  stop k'//new_line('a')// &
            'end program'

        test_separate_module_subroutine = expect_exit_status( &
            source, 9, '/tmp/ffc_session_submod_sep_sub_test')
    end function test_separate_module_subroutine

    logical function test_separate_module_function()
        ! #292: the separate form omits the function signature; the parent
        ! interface supplies both the result kind and the dummy declarations.
        character(len=*), parameter :: source = &
            'module m3'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    module function triple(n) result(r)'//new_line('a')// &
            '      integer, intent(in) :: n'//new_line('a')// &
            '      integer :: r'//new_line('a')// &
            '    end function'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            'submodule (m3) s3'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module procedure triple'//new_line('a')// &
            '    r = 3*n'//new_line('a')// &
            '  end procedure'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program p3'//new_line('a')// &
            '  use m3, only: triple'//new_line('a')// &
            '  stop triple(4)'//new_line('a')// &
            'end program'

        test_separate_module_function = expect_exit_status( &
            source, 12, '/tmp/ffc_session_submod_sep_fn_test')
    end function test_separate_module_function

    logical function test_generic_interface_body_specific()
        ! #292: a named generic interface whose specific is a module-procedure
        ! interface body (`module subroutine impl(...)`) implemented in the
        ! submodule; a call through the generic name resolves to that specific.
        character(len=*), parameter :: source = &
            'module mg'//new_line('a')// &
            '  interface gen'//new_line('a')// &
            '    module subroutine impl_sub(n)'//new_line('a')// &
            '      integer, intent(inout) :: n'//new_line('a')// &
            '    end subroutine'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            'submodule (mg) mgs'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module procedure impl_sub'//new_line('a')// &
            '    n = 3'//new_line('a')// &
            '  end procedure'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program pg'//new_line('a')// &
            '  use mg'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 2'//new_line('a')// &
            '  call gen(n)'//new_line('a')// &
            '  stop n'//new_line('a')// &
            'end program'

        test_generic_interface_body_specific = expect_exit_status( &
            source, 3, '/tmp/ffc_session_submod_generic_body_test')
    end function test_generic_interface_body_specific

    logical function test_parent_module_not_found()
        ! #292: a submodule whose parent module is absent from the compilation
        ! set is rejected with a targeted diagnostic.
        character(len=*), parameter :: source = &
            'submodule (nonexistent_parent) orphan'//new_line('a')// &
            'contains'//new_line('a')// &
            '  module subroutine do_thing(x)'//new_line('a')// &
            '    integer, intent(inout) :: x'//new_line('a')// &
            '    x = 7'//new_line('a')// &
            '  end subroutine'//new_line('a')// &
            'end submodule'//new_line('a')// &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 1'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program'

        test_parent_module_not_found = expect_error_contains( &
            source, 'submodule parent module not found', &
            '/tmp/ffc_session_submod_orphan_test')
    end function test_parent_module_not_found

    logical function test_caller_links_deferred_module_procedure() result(ok)
        ! A module procedure whose interface the parent declares and whose body
        ! a submodule supplies must reach a separately compiled caller: the
        ! caller sees only the parent's .fmod, and can resolve and link the
        ! procedure only because the artefact carries that deferred interface
        ! (#297).
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: module_source = &
            'module fmod297_parent'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        module function twice(x) result(r)'//new_line('a')// &
            '            integer, intent(in) :: x'//new_line('a')// &
            '            integer :: r'//new_line('a')// &
            '        end function twice'//new_line('a')// &
            '    end interface'//new_line('a')// &
            'end module fmod297_parent'//new_line('a')// &
            'submodule (fmod297_parent) fmod297_impl'//new_line('a')// &
            'contains'//new_line('a')// &
            '    module procedure twice'//new_line('a')// &
            '        r = 2 * x'//new_line('a')// &
            '    end procedure twice'//new_line('a')// &
            'end submodule fmod297_impl'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod297_parent, only: twice'//new_line('a')// &
            '    stop twice(21)'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod297_deferred', dir)
        separate_status = run_separate_compilation(dir, module_source, &
                                                   program_source)
        same_status = run_same_unit_compilation(dir, module_source, &
                                                program_source)
        if (same_status /= 142) then
            print *, 'FAIL: same-unit deferred module procedure status ', &
                same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= same_status) then
            print *, 'FAIL: separately compiled caller status ', &
                separate_status, ' differs from same-unit ', same_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_caller_links_deferred_module_procedure

    logical function test_plain_interface_is_not_exported() result(ok)
        ! A plain interface block declares an external procedure with no module
        ! mangling, so it must not be published as a module procedure of the
        ! enclosing module: a caller that resolved it there would link against
        ! a symbol the module never defines.
        character(len=:), allocatable :: dir
        integer :: separate_status
        character(len=*), parameter :: module_source = &
            'module fmod297_external'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface'//new_line('a')// &
            '        integer function outside(x)'//new_line('a')// &
            '            integer, intent(in) :: x'//new_line('a')// &
            '        end function outside'//new_line('a')// &
            '    end interface'//new_line('a')// &
            'end module fmod297_external'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod297_external, only: outside'//new_line('a')// &
            '    stop outside(1)'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod297_plain', dir)
        separate_status = run_separate_compilation(dir, module_source, &
                                                   program_source)
        if (separate_status /= 92) then
            print *, 'FAIL: plain interface was exported as a module '// &
                'procedure, status ', separate_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_plain_interface_is_not_exported

    logical function test_submodule_compiles_as_its_own_unit() result(ok)
        ! Parent, submodule, and caller compile as three independent ffc
        ! invocations and link. The submodule unit never sees the parent's
        ! source: it binds to the interface the parent's .fmod carries, and
        ! emits its body under the parent's mangled symbol (#297).
        character(len=:), allocatable :: dir
        integer :: status
        integer :: exit_stat, cmd_stat

        ok = .false.
        call make_scratch_dir('fmod297_threefile', dir)
        if (.not. write_source(dir//'/par.f90', parent_source)) return
        if (.not. write_source(dir//'/sub.f90', submodule_source)) return
        if (.not. write_source(dir//'/main.f90', caller_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c par.f90 -o par.o >>log 2>&1 || exit 91; '// &
            '"$exe" -c sub.f90 -o sub.o >>log 2>&1 || exit 92; '// &
            '"$exe" main.f90 par.o sub.o -o main >>log 2>&1 || exit 93; '// &
            "./main; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        status = exit_stat
        if (cmd_stat /= 0) status = 90
        if (status /= 142) then
            print *, 'FAIL: three-file submodule build status ', status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_submodule_compiles_as_its_own_unit

    logical function test_submodule_parent_diagnostics() result(ok)
        ! Every way a separately compiled submodule can fail to match the
        ! parent interface it claims is diagnosed against the artefact, rather
        ! than emitted under a symbol callers would then miscall.
        character(len=:), allocatable :: dir

        ok = .false.
        call make_scratch_dir('fmod297_diag', dir)
        if (.not. write_source(dir//'/par.f90', parent_source)) return
        if (.not. compile_parent_object(dir)) then
            print *, 'FAIL: parent object did not compile'
            call show_log(dir)
            return
        end if
        if (.not. expect_submodule_error(dir, &
            'submodule (fmod297_absent_parent) s'//new_line('a')// &
            'contains'//new_line('a')// &
            '    module function twice(x) result(r)'//new_line('a')// &
            '        integer, intent(in) :: x'//new_line('a')// &
            '        integer :: r'//new_line('a')// &
            '        r = 2 * x'//new_line('a')// &
            '    end function twice'//new_line('a')// &
            'end submodule s', 'parent module not found')) return
        if (.not. expect_submodule_error(dir, &
            'submodule (fmod297_parent) s'//new_line('a')// &
            'contains'//new_line('a')// &
            '    module function nosuch(x) result(r)'//new_line('a')// &
            '        integer, intent(in) :: x'//new_line('a')// &
            '        integer :: r'//new_line('a')// &
            '        r = x'//new_line('a')// &
            '    end function nosuch'//new_line('a')// &
            'end submodule s', 'is not declared by module')) return
        if (.not. expect_submodule_error(dir, &
            'submodule (fmod297_parent) s'//new_line('a')// &
            'contains'//new_line('a')// &
            '    module function twice(x, y) result(r)'//new_line('a')// &
            '        integer, intent(in) :: x'//new_line('a')// &
            '        integer, intent(in) :: y'//new_line('a')// &
            '        integer :: r'//new_line('a')// &
            '        r = x + y'//new_line('a')// &
            '    end function twice'//new_line('a')// &
            'end submodule s', 'does not match the dummy argument count')) return
        if (.not. expect_submodule_error(dir, submodule_source_unrestated, &
            'must restate its interface')) return
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_submodule_parent_diagnostics

    logical function compile_parent_object(dir) result(ok)
        character(len=*), intent(in) :: dir
        integer :: exit_stat, cmd_stat

        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c par.f90 -o par.o >>log 2>&1'//"'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        ok = cmd_stat == 0 .and. exit_stat == 0
    end function compile_parent_object

    logical function expect_submodule_error(dir, source, fragment) result(ok)
        ! Compiling this submodule alone must fail with a diagnostic naming the
        ! given fragment.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: fragment
        integer :: exit_stat, cmd_stat, grep_stat

        ok = .false.
        if (.not. write_source(dir//'/bad.f90', source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c bad.f90 -o bad.o >bad.log 2>&1'//"'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat == 0) then
            print *, 'FAIL: submodule was accepted, expected: ', fragment
            return
        end if
        call execute_command_line('grep -q "'//fragment//'" '//dir//'/bad.log', &
                                  exitstat=grep_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. grep_stat /= 0) then
            print *, 'FAIL: diagnostic did not mention: ', fragment
            call execute_command_line('cat '//dir//'/bad.log')
            return
        end if
        ok = .true.
    end function expect_submodule_error

    integer function run_separate_compilation(dir, mod_source, prog_source) &
            result(status)
        ! Compile the module (with its submodule) in one ffc invocation, then
        ! the program in a second, independent invocation that can only learn
        ! the module's procedures from the .fmod artefact. Returns 90 when no
        ! ffc binary was found, 91/92 when a compilation failed, and
        ! 100 + exit status when the program ran.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: mod_source
        character(len=*), intent(in) :: prog_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_source(dir//'/m.f90', mod_source)) return
        if (.not. write_source(dir//'/p.f90', prog_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c m.f90 -o m.o >>log 2>&1 || exit 91; '// &
            '"$exe" p.f90 m.o -o p >>log 2>&1 || exit 92; '// &
            "./p; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_separate_compilation

    integer function run_same_unit_compilation(dir, mod_source, prog_source) &
            result(status)
        ! Compile the same module, submodule, and program as one unit, so the
        ! separate result can be held against it.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: mod_source
        character(len=*), intent(in) :: prog_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_source(dir//'/same.f90', mod_source//new_line('a')// &
                               prog_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" same.f90 -o same >>log 2>&1 || exit 92; '// &
            "./same; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_same_unit_compilation

    subroutine make_scratch_dir(tag, dir)
        ! A scratch directory of this run's own, so concurrent builds of other
        ! worktrees never share it (ffc #547).
        character(len=*), intent(in) :: tag
        character(len=:), allocatable, intent(out) :: dir
        character(len=32) :: stamp
        integer :: values(8)

        call date_and_time(values=values)
        write (stamp, '(I0,A,I0)') values(6)*60000 + values(7)*1000 + &
            values(8), '_', values(5)
        dir = '/tmp/ffc_'//tag//'_'//trim(stamp)
        call execute_command_line('rm -rf '//dir//'; mkdir -p '//dir)
    end subroutine make_scratch_dir

    subroutine remove_scratch_dir(dir)
        character(len=*), intent(in) :: dir

        call execute_command_line('rm -rf '//dir)
    end subroutine remove_scratch_dir

    subroutine show_log(dir)
        character(len=*), intent(in) :: dir

        call execute_command_line('cat '//dir//'/log 2>/dev/null')
    end subroutine show_log

    logical function write_source(path, contents) result(ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: unit, io_stat

        ok = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not write ', path
            return
        end if
        write (unit, '(A)', iostat=io_stat) contents
        close (unit)
        ok = io_stat == 0
    end function write_source

end program test_session_submodule_compiler
