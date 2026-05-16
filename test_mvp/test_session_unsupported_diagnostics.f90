program test_session_unsupported_diagnostics
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session unsupported diagnostic test ==='

    all_passed = .true.
    if (.not. test_array_declaration_diagnostic()) all_passed = .false.
    if (.not. test_character_declaration_diagnostic()) all_passed = .false.
    if (.not. test_module_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: unsupported direct-session features emit diagnostics'

contains

    logical function test_array_declaration_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: values(3)'//new_line('a')// &
            'end program main'

        test_array_declaration_diagnostic = expect_error_contains( &
            source, 'unsupported array declaration', &
            '/tmp/ffc_session_array_diagnostic_test')
    end function test_array_declaration_diagnostic

    logical function test_character_declaration_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5) :: name'//new_line('a')// &
            'end program main'

        test_character_declaration_diagnostic = expect_error_contains( &
            source, 'unsupported character variable declaration', &
            '/tmp/ffc_session_character_diagnostic_test')
    end function test_character_declaration_diagnostic

    logical function test_module_diagnostic()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'end module m'

        test_module_diagnostic = expect_error_contains( &
            source, 'unsupported module program unit', &
            '/tmp/ffc_session_module_diagnostic_test')
    end function test_module_diagnostic

    logical function expect_error_contains(source, expected, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        expect_error_contains = .false.
        call compile_and_lower(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)

        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported source lowered without error'
            return
        end if
        if (index(error_msg, expected) <= 0) then
            print *, 'FAIL: expected diagnostic substring ', expected
            print *, '  got ', trim(error_msg)
            return
        end if
        if (index(error_msg, 'line ') <= 0 .or. &
            index(error_msg, 'column ') <= 0) then
            print *, 'FAIL: expected line/column diagnostic, got ', &
                trim(error_msg)
            return
        end if

        expect_error_contains = .true.
    end function expect_error_contains

    subroutine compile_and_lower(source, exe_path, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable, intent(out) :: error_msg
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD

        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            error_msg = 'FortFront rejected source: '// &
                        trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
                                        frontend_result%root_index, exe_path, &
                                        error_msg)
    end subroutine compile_and_lower
end program test_session_unsupported_diagnostics
