program test_session_read_fmod_compiler
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe, &
        lower_program_to_liric_object
    use ffc_module_artefact, only: module_info_t, write_fmod
    implicit none

    logical :: all_passed

    print *, '=== module .fmod read-on-USE tests ==='

    all_passed = .true.
    if (.not. test_use_module_from_fmod_constant()) all_passed = .false.
    if (.not. test_use_module_fmod_renames()) all_passed = .false.
    if (.not. test_use_module_fmod_rejects_unknown_only()) all_passed = .false.
    if (.not. test_use_module_fmod_not_found_errors()) all_passed = .false.
    if (.not. test_use_module_variable_from_fmod()) all_passed = .false.
    if (.not. test_fmod_preserves_unsupported_public_procedure()) &
        all_passed = .false.
    if (.not. test_fmod_optional_dummy_separate_compilation()) all_passed = .false.
    if (.not. test_fmod_value_dummy_separate_compilation()) all_passed = .false.
    if (.not. test_fmod_intent_out_rejects_literal()) all_passed = .false.
    if (.not. test_fmod_schema_version_is_checked()) all_passed = .false.
    if (.not. test_use_repeated_rename_is_valid()) all_passed = .false.
    if (.not. test_use_repeated_rename_one_statement()) all_passed = .false.
    if (.not. test_use_two_locals_one_remote()) all_passed = .false.
    if (.not. test_use_conflicting_rename_rejected()) all_passed = .false.
    if (.not. test_same_file_repeated_rename_is_valid()) all_passed = .false.
    if (.not. test_same_file_conflicting_rename_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module .fmod read-on-USE'

contains

    logical function test_use_module_from_fmod_constant() result(ok)
        character(len=*), parameter :: dir = '/tmp/ffc_read_fmod_dir'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: value'//new_line('a')// &
            '  stop value'//new_line('a')// &
            'end program main'
        type(module_info_t) :: info
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

        ok = .false.
        call execute_command_line('mkdir -p '//dir)
        info%name = 'm'
        allocate (info%parameters(1))
        info%parameters(1)%name = 'value'
        info%parameters(1)%kind = 'integer'
        info%parameters(1)%value = '37'
        allocate (info%derived_types(0))
        call write_fmod(dir//'/m.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if

        call compile_with_include(source, '/tmp/ffc_read_fmod_use', dir, &
            error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: lowering through .fmod failed: ', trim(error_msg)
            return
        end if
        call execute_command_line('/tmp/ffc_read_fmod_use', exitstat=exit_stat, &
            cmdstat=cmd_stat)
        call execute_command_line('rm -f /tmp/ffc_read_fmod_use')
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run binary'
            return
        end if
        if (exit_stat /= 37) then
            print *, 'FAIL: expected exit 37 from .fmod constant, got ', exit_stat
            return
        end if
        ok = .true.
    end function test_use_module_from_fmod_constant

    logical function test_use_module_fmod_renames() result(ok)
        ! A separately compiled module must preserve the remote export name for
        ! linking while making each renamed export available under its local
        ! name in the using unit (#328).
        character(len=*), parameter :: m_src = '/tmp/ffc_read_fmod_rename_m.f90'
        character(len=*), parameter :: main_src = &
            '/tmp/ffc_read_fmod_rename_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_read_fmod_rename_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_read_fmod_rename'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_read_fmod_renamed'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: answer = 40'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function add_two(value) result(result_value)'// &
            new_line('a')// &
            '    integer, intent(in) :: value'//new_line('a')// &
            '    result_value = value + 2'//new_line('a')// &
            '  end function add_two'//new_line('a')// &
            'end module ffc_read_fmod_renamed')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_read_fmod_renamed, only: local_answer => answer, '// &
            'plus => add_two'//new_line('a')// &
            '  stop plus(local_answer)'//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//m_obj//' '//main_exe//' '// &
            '/tmp/ffc_read_fmod_renamed.fmod')
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: renamed .fmod compile pipeline failed, code ', &
                exit_stat
            call remove_rename_files(m_src, main_src, m_obj, main_exe)
            return
        end if

        call execute_command_line(main_exe, exitstat=exit_stat, &
            cmdstat=cmd_stat)
        call remove_rename_files(m_src, main_src, m_obj, main_exe)
        if (cmd_stat /= 0) then
            print *, 'FAIL: could not run renamed .fmod binary'
            return
        end if
        if (exit_stat /= 42) then
            print *, 'FAIL: expected exit 42 from renamed .fmod exports, got ', &
                exit_stat
            return
        end if
        ok = .true.
    end function test_use_module_fmod_renames

    logical function test_use_module_fmod_rejects_unknown_only() result(ok)
        character(len=*), parameter :: dir = '/tmp/ffc_read_fmod_unknown_dir'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: missing'//new_line('a')// &
            '  stop missing'//new_line('a')// &
            'end program main'
        type(module_info_t) :: info
        character(len=:), allocatable :: error_msg

        ok = .false.
        call execute_command_line('mkdir -p '//dir)
        info%name = 'm'
        allocate (info%parameters(1))
        info%parameters(1)%name = 'answer'
        info%parameters(1)%kind = 'integer'
        info%parameters(1)%value = '40'
        allocate (info%derived_types(0))
        call write_fmod(dir//'/m.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if

        call compile_with_include(source, '/tmp/ffc_read_fmod_unknown', dir, &
            error_msg)
        call execute_command_line('rm -f /tmp/ffc_read_fmod_unknown '// &
            dir//'/m.fmod')
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unknown .fmod ONLY name was accepted'
            return
        end if
        if (index(error_msg, 'use only: name is not exported by module') == 0 .or. &
            index(error_msg, 'missing') == 0) then
            print *, 'FAIL: wrong unknown .fmod ONLY diagnostic: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_use_module_fmod_rejects_unknown_only

    logical function test_use_module_fmod_not_found_errors() result(ok)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use missing_mod'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'
        character(len=:), allocatable :: error_msg

        ok = .false.
        call compile_with_include(source, '/tmp/ffc_read_fmod_missing', &
            '/tmp/ffc_read_fmod_empty', error_msg)
        call execute_command_line('rm -f /tmp/ffc_read_fmod_missing')
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: expected a module-not-found diagnostic'
            return
        end if
        if (index(error_msg, 'not') == 0) then
            print *, 'FAIL: diagnostic was not about a missing module: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_use_module_fmod_not_found_errors

    logical function test_use_module_variable_from_fmod() result(ok)
        ! A scalar integer module variable described in a .fmod resolves on USE
        ! in a separately compiled program; the program lowers to an object that
        ! references the shared global (#274).
        character(len=*), parameter :: dir = '/tmp/ffc_read_fmod_var_dir'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use state, only: counter'//new_line('a')// &
            '  counter = 11'//new_line('a')// &
            '  stop counter'//new_line('a')// &
            'end program main'
        type(module_info_t) :: info
        character(len=:), allocatable :: error_msg
        character(len=len(dir)) :: paths(1)
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        ok = .false.
        call execute_command_line('mkdir -p '//dir)
        info%name = 'state'
        allocate (info%parameters(0))
        allocate (info%derived_types(0))
        allocate (info%variables(1))
        info%variables(1)%name = 'counter'
        info%variables(1)%kind = 'integer'
        call write_fmod(dir//'/state.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        paths(1) = dir
        call execute_command_line('rm -f /tmp/ffc_read_fmod_var.o')
        call lower_program_to_liric_object(frontend_result%arena, &
            frontend_result%root_index, &
            '/tmp/ffc_read_fmod_var.o', &
            error_msg, paths)
        call execute_command_line('rm -f /tmp/ffc_read_fmod_var.o')
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: USE of .fmod variable did not resolve: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_use_module_variable_from_fmod

    logical function test_fmod_preserves_unsupported_public_procedure() result(ok)
        ! A public procedure remains a valid USE ONLY export even when its
        ! derived-type call ABI is not yet supported by the direct backend.
        character(len=*), parameter :: dir = '/tmp/ffc_fmod584_export'
        character(len=*), parameter :: log_path = dir//'/log'
        integer :: status

        ok = .false.
        status = run_separate_compilation(dir, &
            'module fmod584_export'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: token'//new_line('a')// &
            '    integer :: value'//new_line('a')// &
            '  end type token'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(value)'//new_line('a')// &
            '    type(token), intent(in) :: value'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module fmod584_export', &
            'program main'//new_line('a')// &
            '  use fmod584_export, only: inspect'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main', log_path)
        if (status /= 100) then
            print *, 'FAIL: unsupported public procedure export, status ', status
            call execute_command_line('cat '//log_path)
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function test_fmod_preserves_unsupported_public_procedure

    subroutine write_rename_matrix_fmod(dir, error_msg)
        ! A module exporting two distinct integer constants, so a rename
        ! matrix can bind one local name to the same remote entity twice
        ! (valid) or to two different remote entities (ambiguous).
        character(len=*), intent(in) :: dir
        character(len=:), allocatable, intent(out) :: error_msg
        type(module_info_t) :: info

        call execute_command_line('mkdir -p '//dir)
        info%name = 'm'
        allocate (info%parameters(2))
        info%parameters(1)%name = 'alpha'
        info%parameters(1)%kind = 'integer'
        info%parameters(1)%value = '13'
        info%parameters(2)%name = 'beta'
        info%parameters(2)%kind = 'integer'
        info%parameters(2)%value = '21'
        allocate (info%derived_types(0))
        call write_fmod(dir//'/m.fmod', info, error_msg)
    end subroutine write_rename_matrix_fmod

    logical function run_rename_case(tag, source, expected_exit) result(ok)
        ! Compile and run a rename matrix program, checking its exit status.
        character(len=*), intent(in) :: tag
        character(len=*), intent(in) :: source
        integer, intent(in) :: expected_exit
        character(len=*), parameter :: dir = '/tmp/ffc_read_fmod_matrix_dir'
        character(len=*), parameter :: exe = '/tmp/ffc_read_fmod_matrix'
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

        ok = .false.
        call write_rename_matrix_fmod(dir, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        call compile_with_include(source, exe, dir, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: '//tag//' rejected: ', trim(error_msg)
            call execute_command_line('rm -f '//dir//'/m.fmod')
            return
        end if
        call execute_command_line(exe, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe//' '//dir//'/m.fmod')
        if (cmd_stat /= 0) then
            print *, 'FAIL: '//tag//' binary did not run'
            return
        end if
        if (exit_stat /= expected_exit) then
            print *, 'FAIL: '//tag//' expected exit ', expected_exit, &
                ' got ', exit_stat
            return
        end if
        ok = .true.
    end function run_rename_case

    logical function test_use_repeated_rename_is_valid() result(ok)
        ! Importing the same remote binding under the same local name from two
        ! USE statements is not ambiguous (F2018 19.5.2).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: alias => alpha'//new_line('a')// &
            '  use m, only: alias => alpha'//new_line('a')// &
            '  stop alias'//new_line('a')// &
            'end program main'

        ok = run_rename_case('repeated rename', source, 13)
    end function test_use_repeated_rename_is_valid

    logical function test_use_repeated_rename_one_statement() result(ok)
        ! The same repetition within a single ONLY list is equally valid.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: alias => alpha, alias => alpha'//new_line('a')// &
            '  stop alias'//new_line('a')// &
            'end program main'

        ok = run_rename_case('repeated rename in one statement', source, 13)
    end function test_use_repeated_rename_one_statement

    logical function test_use_two_locals_one_remote() result(ok)
        ! One remote binding may reach two distinct local names.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: first => alpha'//new_line('a')// &
            '  use m, only: second => alpha'//new_line('a')// &
            '  stop first + second'//new_line('a')// &
            'end program main'

        ok = run_rename_case('two locals one remote', source, 26)
    end function test_use_two_locals_one_remote

    logical function test_use_conflicting_rename_rejected() result(ok)
        ! Two distinct remote bindings under one local name stay ambiguous.
        character(len=*), parameter :: dir = '/tmp/ffc_read_fmod_matrix_dir'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: alias => alpha'//new_line('a')// &
            '  use m, only: alias => beta'//new_line('a')// &
            '  stop alias'//new_line('a')// &
            'end program main'
        character(len=:), allocatable :: error_msg

        ok = .false.
        call write_rename_matrix_fmod(dir, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        call compile_with_include(source, '/tmp/ffc_read_fmod_ambiguous', dir, &
            error_msg)
        call execute_command_line('rm -f /tmp/ffc_read_fmod_ambiguous '// &
            dir//'/m.fmod')
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: conflicting USE rename was accepted'
            return
        end if
        if (index(error_msg, 'ambiguous') == 0) then
            print *, 'FAIL: wrong conflicting-rename diagnostic: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_use_conflicting_rename_rejected

    function same_file_rename_source(first, second) result(source)
        ! A module and a using program in one file, so the renames resolve
        ! against in-arena module exports rather than a .fmod.
        character(len=*), intent(in) :: first
        character(len=*), intent(in) :: second
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer, parameter :: alpha = 13'//new_line('a')// &
            '  integer, parameter :: beta = 21'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: '//first//new_line('a')// &
            '  use m, only: '//second//new_line('a')// &
            '  stop alias'//new_line('a')// &
            'end program main'
    end function same_file_rename_source

    logical function test_same_file_repeated_rename_is_valid() result(ok)
        ! Repeating one remote binding under one local name must not be
        ! reported as a duplicate when the module is in the same file.
        character(len=*), parameter :: exe = '/tmp/ffc_same_file_rename'
        character(len=:), allocatable :: error_msg
        integer :: exit_stat, cmd_stat

        ok = .false.
        call compile_with_include( &
            same_file_rename_source('alias => alpha', 'alias => alpha'), &
            exe, '/tmp/ffc_read_fmod_empty', error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: same-file repeated rename rejected: ', &
                trim(error_msg)
            return
        end if
        call execute_command_line(exe, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe)
        if (cmd_stat /= 0) then
            print *, 'FAIL: same-file repeated rename binary did not run'
            return
        end if
        if (exit_stat /= 13) then
            print *, 'FAIL: same-file repeated rename exit ', exit_stat
            return
        end if
        ok = .true.
    end function test_same_file_repeated_rename_is_valid

    logical function test_same_file_conflicting_rename_rejected() result(ok)
        ! Distinct remote bindings under one local name stay ambiguous.
        character(len=:), allocatable :: error_msg

        ok = .false.
        call compile_with_include( &
            same_file_rename_source('alias => alpha', 'alias => beta'), &
            '/tmp/ffc_same_file_ambiguous', '/tmp/ffc_read_fmod_empty', &
            error_msg)
        call execute_command_line('rm -f /tmp/ffc_same_file_ambiguous')
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: same-file conflicting rename was accepted'
            return
        end if
        if (index(error_msg, 'ambiguous') == 0) then
            print *, 'FAIL: wrong same-file conflicting diagnostic: ', &
                trim(error_msg)
            return
        end if
        ok = .true.
    end function test_same_file_conflicting_rename_rejected

    integer function run_separate_compilation(dir, mod_source, prog_source, &
                                              log_path) result(status)
        ! Compile mod_source with -c in one ffc invocation, then prog_source in
        ! a second, independent invocation that can only learn the module's
        ! interface from the .fmod artefact the first invocation wrote. Returns
        ! 0 when both compile and the program runs, 90 when no ffc binary was
        ! found, 91 when the module compilation failed, 92 when the program
        ! compilation failed, and 100 + exit status when the program ran.
        ! Combined compiler output is appended to log_path.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: mod_source
        character(len=*), intent(in) :: prog_source
        character(len=*), intent(in) :: log_path
        integer :: exit_stat, cmd_stat

        status = 90
        call execute_command_line('rm -rf '//dir//'; mkdir -p '//dir)
        if (.not. write_file(dir//'/m.f90', mod_source)) return
        if (.not. write_file(dir//'/p.f90', prog_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c m.f90 -o m.o >>'//log_path//' 2>&1 || exit 91; '// &
            '"$exe" p.f90 m.o -o p >>'//log_path//' 2>&1 || exit 92; '// &
            "./p; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) then
            status = 90
            return
        end if
        status = exit_stat
    end function run_separate_compilation

    logical function test_fmod_optional_dummy_separate_compilation() result(ok)
        ! A module procedure with an OPTIONAL dummy stays callable with the
        ! optional omitted, and with it supplied by keyword, from a separately
        ! compiled program. The using unit can only know the dummy is optional,
        ! and what it is named, from the .fmod artefact (#397).
        character(len=*), parameter :: dir = '/tmp/ffc_fmod397_optional'
        character(len=*), parameter :: log_path = dir//'/log'
        integer :: status

        ok = .false.
        status = run_separate_compilation(dir, &
            'module fmod397_optional'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function combine(a, b) result(r)'//new_line('a')// &
            '    integer, intent(in) :: a'//new_line('a')// &
            '    integer, intent(in), optional :: b'//new_line('a')// &
            '    r = a'//new_line('a')// &
            '    if (present(b)) r = r + b'//new_line('a')// &
            '  end function combine'//new_line('a')// &
            'end module fmod397_optional', &
            'program main'//new_line('a')// &
            '  use fmod397_optional, only: combine'//new_line('a')// &
            '  stop combine(4) + 10 * combine(4, b=3)'//new_line('a')// &
            'end program main', log_path)
        if (status /= 174) then
            print *, 'FAIL: optional dummy through .fmod, status ', status
            call execute_command_line('cat '//log_path)
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function test_fmod_optional_dummy_separate_compilation

    logical function test_fmod_value_dummy_separate_compilation() result(ok)
        ! A VALUE dummy receives a copy: the callee's assignment to it must not
        ! reach the caller's actual. The separately compiled caller can only
        ! learn the attribute from the .fmod artefact, and must agree with the
        ! same-unit compilation of the same program (#397).
        character(len=*), parameter :: dir = '/tmp/ffc_fmod397_value'
        character(len=*), parameter :: log_path = dir//'/log'
        character(len=*), parameter :: mod_source = &
            'module fmod397_value'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function bump(x) result(r)'//new_line('a')// &
            '    integer, value :: x'//new_line('a')// &
            '    x = x + 1'//new_line('a')// &
            '    r = x'//new_line('a')// &
            '  end function bump'//new_line('a')// &
            'end module fmod397_value'
        character(len=*), parameter :: prog_source = &
            'program main'//new_line('a')// &
            '  use fmod397_value, only: bump'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  v = 5'//new_line('a')// &
            '  stop bump(v) + v'//new_line('a')// &
            'end program main'
        character(len=:), allocatable :: error_msg
        integer :: status, exit_stat, cmd_stat

        ok = .false.
        status = run_separate_compilation(dir, mod_source, prog_source, log_path)
        if (status /= 111) then
            print *, 'FAIL: VALUE dummy through .fmod, status ', status
            call execute_command_line('cat '//log_path)
            return
        end if
        ! Same-unit compilation of the same module and program must agree.
        call compile_with_include(mod_source//new_line('a')//prog_source, &
            dir//'/same', dir, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: same-unit VALUE program rejected: ', trim(error_msg)
            return
        end if
        call execute_command_line(dir//'/same', exitstat=exit_stat, &
            cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 11) then
            print *, 'FAIL: same-unit VALUE program exit ', exit_stat
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function test_fmod_value_dummy_separate_compilation

    logical function test_fmod_intent_out_rejects_literal() result(ok)
        ! A literal actual cannot be passed to an INTENT(OUT) dummy. The
        ! separately compiled caller knows the dummy's intent only from the
        ! .fmod artefact, so the diagnostic proves the intent round-tripped.
        character(len=*), parameter :: dir = '/tmp/ffc_fmod397_intent'
        character(len=*), parameter :: log_path = dir//'/log'
        integer :: status, grep_stat, cmd_stat

        ok = .false.
        status = run_separate_compilation(dir, &
            'module fmod397_intent'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine put(target_value)'//new_line('a')// &
            '    integer, intent(out) :: target_value'//new_line('a')// &
            '    target_value = 3'//new_line('a')// &
            '  end subroutine put'//new_line('a')// &
            'end module fmod397_intent', &
            'program main'//new_line('a')// &
            '  use fmod397_intent, only: put'//new_line('a')// &
            '  call put(7)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main', log_path)
        if (status /= 92) then
            print *, 'FAIL: literal to INTENT(OUT) dummy was accepted, status ', &
                status
            call execute_command_line('cat '//log_path)
            return
        end if
        call execute_command_line('grep -qi "intent(out)" '//log_path, &
            exitstat=grep_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. grep_stat /= 0) then
            print *, 'FAIL: diagnostic did not mention INTENT(OUT)'
            call execute_command_line('cat '//log_path)
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function test_fmod_intent_out_rejects_literal

    logical function test_fmod_schema_version_is_checked() result(ok)
        ! A .fmod whose schema version this ffc does not implement is rejected
        ! with an artefact-version diagnostic instead of being misread. An
        ! artefact with no schema version at all (written before the field
        ! existed) is stale and rejected the same way (#397).
        character(len=*), parameter :: dir = '/tmp/ffc_fmod397_schema'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use m, only: value'//new_line('a')// &
            '  stop value'//new_line('a')// &
            'end program main'
        type(module_info_t) :: info
        character(len=:), allocatable :: error_msg

        ok = .false.
        call execute_command_line('rm -rf '//dir//'; mkdir -p '//dir)
        info%name = 'm'
        allocate (info%parameters(1))
        info%parameters(1)%name = 'value'
        info%parameters(1)%kind = 'integer'
        info%parameters(1)%value = '37'
        allocate (info%derived_types(0))
        call write_fmod(dir//'/m.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        ! The artefact this ffc just wrote carries its own schema version and
        ! must be accepted.
        call compile_with_include(source, dir//'/current', dir, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: current-schema .fmod rejected: ', trim(error_msg)
            return
        end if

        call execute_command_line("sed -i 's/^fmod_schema = .*/"// &
            "fmod_schema = 99999/' "//dir//'/m.fmod')
        call compile_with_include(source, dir//'/future', dir, error_msg)
        if (index(error_msg, 'schema version') == 0) then
            print *, 'FAIL: future .fmod schema was not rejected: ', &
                trim(error_msg)
            return
        end if

        call execute_command_line("sed -i '/^fmod_schema = /d' "// &
            dir//'/m.fmod')
        call compile_with_include(source, dir//'/stale', dir, error_msg)
        if (index(error_msg, 'schema version') == 0) then
            print *, 'FAIL: unversioned .fmod was not rejected: ', &
                trim(error_msg)
            return
        end if
        call execute_command_line('rm -rf '//dir)
        ok = .true.
    end function test_fmod_schema_version_is_checked

    subroutine compile_with_include(source, exe_path, include_dir, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=*), intent(in) :: include_dir
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=len(include_dir)) :: paths(1)

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                trim(frontend_result%diagnostic_text)
            return
        end if
        paths(1) = include_dir
        call execute_command_line('rm -f '//exe_path)
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe_path, &
            error_msg, paths)
    end subroutine compile_with_include

    logical function write_file(path, contents) result(ok)
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
    end function write_file

    subroutine remove_rename_files(m_src, main_src, m_obj, main_exe)
        character(len=*), intent(in) :: m_src, main_src, m_obj, main_exe

        call execute_command_line('rm -f '//m_src//' '//main_src//' '// &
            m_obj//' '//main_exe//' /tmp/ffc_read_fmod_renamed.fmod')
    end subroutine remove_rename_files

end program test_session_read_fmod_compiler
