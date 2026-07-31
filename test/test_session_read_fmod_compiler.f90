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
