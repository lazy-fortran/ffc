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

end program test_session_read_fmod_compiler
