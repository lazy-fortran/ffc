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
    if (.not. test_module_with_many_derived_types()) stop 1
    if (.not. test_two_modules_with_derived_types()) stop 1
    if (.not. test_module_with_only_integer_params()) stop 1
    if (.not. test_derived_type_with_single_component()) stop 1
    if (.not. test_derived_type_with_many_components()) stop 1

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

    logical function test_module_with_many_derived_types()
        character(len=*), parameter :: source = &
           'module big_mod'//new_line('a')// &
           '  type :: a1_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a2_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a3_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a4_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a5_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a6_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a7_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a8_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a9_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a10_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a11_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a12_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a13_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a14_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a15_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           '  type :: a16_t'//new_line('a')// &
           '    integer :: v'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module big_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_module_with_many_derived_types = expect_compile(source, &
            '/tmp/ffc_module_exports_many_test')
    end function test_module_with_many_derived_types

    logical function test_two_modules_with_derived_types()
        character(len=*), parameter :: source = &
           'module mod_a'//new_line('a')// &
           '  type :: type_a_t'//new_line('a')// &
           '    integer :: a'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module mod_a'//new_line('a')// &
           'module mod_b'//new_line('a')// &
           '  type :: type_b_t'//new_line('a')// &
           '    integer :: b'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module mod_b'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_two_modules_with_derived_types = expect_compile(source, &
            '/tmp/ffc_module_exports_two_mods_test')
    end function test_two_modules_with_derived_types

    logical function test_module_with_only_integer_params()
        character(len=*), parameter :: source = &
           'module params_mod'//new_line('a')// &
           '  integer, parameter :: p = 42'//new_line('a')// &
           'end module params_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  stop 0'//new_line('a')// &
           'end program main'

        test_module_with_only_integer_params = expect_compile(source, &
            '/tmp/ffc_module_exports_params_test')
    end function test_module_with_only_integer_params

    logical function test_derived_type_with_single_component()
        character(len=*), parameter :: source = &
           'module single_comp_mod'//new_line('a')// &
           '  type :: single_t'//new_line('a')// &
           '    integer :: only'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module single_comp_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use single_comp_mod'//new_line('a')// &
           '  type(single_t) :: s'//new_line('a')// &
           '  s%only = 99'//new_line('a')// &
           '  print *, s%only'//new_line('a')// &
           'end program main'

        test_derived_type_with_single_component = expect_output( &
           source, '99', '/tmp/ffc_module_exports_single_comp_test')
    end function test_derived_type_with_single_component

    logical function test_derived_type_with_many_components()
        character(len=*), parameter :: source = &
           'module many_comp_mod'//new_line('a')// &
           '  type :: many_t'//new_line('a')// &
           '    integer :: c1, c2, c3, c4, c5'//new_line('a')// &
           '    integer :: c6, c7, c8, c9, c10'//new_line('a')// &
           '  end type'//new_line('a')// &
           'end module many_comp_mod'//new_line('a')// &
           'program main'//new_line('a')// &
           '  use many_comp_mod'//new_line('a')// &
           '  type(many_t) :: m'//new_line('a')// &
           '  m%c1 = 1'//new_line('a')// &
           '  m%c2 = 2'//new_line('a')// &
           '  m%c3 = 3'//new_line('a')// &
           '  m%c4 = 4'//new_line('a')// &
           '  m%c5 = 5'//new_line('a')// &
           '  m%c6 = 6'//new_line('a')// &
           '  m%c7 = 7'//new_line('a')// &
           '  m%c8 = 8'//new_line('a')// &
           '  m%c9 = 9'//new_line('a')// &
           '  m%c10 = 10'//new_line('a')// &
           '  print *, m%c1+m%c2+m%c3+m%c4+m%c5+m%c6+m%c7+m%c8+m%c9+m%c10'//new_line('a')// &
           'end program main'

        test_derived_type_with_many_components = expect_output( &
           source, '55', '/tmp/ffc_module_exports_many_comp_test')
    end function test_derived_type_with_many_components

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
