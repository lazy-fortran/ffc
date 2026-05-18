program test_session_module_exports_derived_type_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    print *, '=== direct session module exports derived type compiler test ==='

    if (.not. test_module_with_single_derived_type()) stop 1
    if (.not. test_module_with_multiple_derived_types()) stop 1
    if (.not. test_empty_module_compiles()) stop 1
    if (.not. test_module_derived_type_with_component_assignment()) stop 1

    print *, 'PASS: module exports derived types lower through direct LIRIC'

contains

    logical function test_module_with_single_derived_type()
        character(len=*), parameter :: source = &
           'module point_mod'//new_line('a')// &
           '  type :: point_t'//new_line('a')// &
           '    integer :: x'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module point_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_module_with_single_derived_type = expect_compile(source, &
            '/tmp/ffc_module_exports_single_test')
    end function test_module_with_single_derived_type

    logical function test_module_with_multiple_derived_types()
        character(len=*), parameter :: source = &
           'module shapes_mod'//new_line('a')// &
           '  type :: circle_t'//new_line('a')// &
           '    integer :: radius'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: rectangle_t'//new_line('a')// &
           '    integer :: width'//new_line('a')// &
           '    integer :: height'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module shapes_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_module_with_multiple_derived_types = expect_compile(source, &
            '/tmp/ffc_module_exports_multi_test')
    end function test_module_with_multiple_derived_types

    logical function test_empty_module_compiles()
        character(len=*), parameter :: source = &
           'module empty_mod'//new_line('a')// &
           'end module empty_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_empty_module_compiles = expect_compile(source, &
            '/tmp/ffc_module_exports_empty_test')
    end function test_empty_module_compiles

    logical function test_module_derived_type_with_component_assignment()
        character(len=*), parameter :: source = &
           'module vec_mod'//new_line('a')// &
           '  type :: vec3_t'//new_line('a')// &
           '    integer :: x, y, z'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module vec_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use vec_mod'//new_line('a')// &
           '  type(vec3_t) :: v'//new_line('a')// &
           '  v%x = 1'//new_line('a')// &
           '  v%y = 2'//new_line('a')// &
           '  v%z = 3'//new_line('a')// &
           '  print *, v%x + v%y + v%z'//new_line('a')// &
           'end program main'

        test_module_derived_type_with_component_assignment = expect_output( &
           source, '6', '/tmp/ffc_module_exports_component_test')
    end function test_module_derived_type_with_component_assignment

    logical function expect_compile(source, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg

        expect_compile = .false.
        call compile_and_lower(source, exe_path, error_msg)
        call execute_command_line('rm -f '//exe_path)

        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: lowering failed: ', trim(error_msg)
            return
        end if

        expect_compile = .true.
    end function expect_compile

    logical function expect_output(source, expected, exe_path)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: out_path
        character(len=64) :: output_line
        integer :: exit_stat
        integer :: io_stat
        integer :: unit

        expect_output = .false.
        out_path = exe_path//'.out'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: lowering failed: ', trim(error_msg)
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        call execute_command_line(exe_path//' > '//out_path, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: executable exit status ', exit_stat
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if

        open (newunit=unit, file=out_path, status='old', action='read', &
              iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not open captured output'
            call execute_command_line('rm -f '//exe_path//' '//out_path)
            return
        end if
        read (unit, '(A)', iostat=io_stat) output_line
        close (unit)
        call execute_command_line('rm -f '//exe_path//' '//out_path)

        if (io_stat /= 0 .or. trim(adjustl(output_line)) /= expected) then
            print *, 'FAIL: expected output "', expected, '" got "', &
                trim(output_line), '"'
            return
        end if

        expect_output = .true.
    end function expect_output

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

end program test_session_module_exports_derived_type_compiler
