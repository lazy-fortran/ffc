program test_session_derived_type_compiler
    use fortfront, only: compiler_frontend_options_t, &
                         compiler_frontend_result_t, &
                         compile_frontend_from_string, INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session derived type compiler test ==='

    all_passed = .true.
    if (.not. test_component_slots()) all_passed = .false.
    if (.not. test_two_variables_do_not_alias()) all_passed = .false.
    if (.not. test_component_stop_code()) all_passed = .false.
    if (.not. test_real_component_diagnostic()) all_passed = .false.
    if (.not. test_nested_component_diagnostic()) all_passed = .false.
    if (.not. test_constructor_diagnostic()) all_passed = .false.
    if (.not. test_inheritance_diagnostic()) all_passed = .false.
    if (.not. test_type_bound_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived types lower through direct LIRIC'

contains

    logical function test_component_slots()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: cell_t'//new_line('a')// &
                                       '    integer :: lo'//new_line('a')// &
                                       '    integer :: mid'//new_line('a')// &
                                       '    integer :: hi'//new_line('a')// &
                                       '  end type cell_t'//new_line('a')// &
                                       '  type(cell_t) :: node'//new_line('a')// &
                                       '  node%lo = 4'//new_line('a')// &
                                       '  node%mid = 6'//new_line('a')// &
                                       '  node%hi = node%lo + node%mid'//new_line('a')// &
                                       '  print *, node%lo + node%mid + node%hi'// &
                                       new_line('a')// &
                                       'end program main'

        test_component_slots = expect_output(source, '20', &
                                             '/tmp/ffc_session_derived_slots_test')
    end function test_component_slots

    logical function test_two_variables_do_not_alias()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: pair_t'//new_line('a')// &
                                       '    integer :: left'//new_line('a')// &
                                       '    integer :: right'//new_line('a')// &
                                       '  end type pair_t'//new_line('a')// &
                                       '  type(pair_t) :: first'//new_line('a')// &
                                       '  type(pair_t) :: second'//new_line('a')// &
                                       '  first%left = 3'//new_line('a')// &
                                       '  first%right = 8'//new_line('a')// &
                                       '  second%left = 11'//new_line('a')// &
                                       '  second%right = 13'//new_line('a')// &
                                       '  print *, first%left + second%right'// &
                                       new_line('a')// &
                                       'end program main'

        test_two_variables_do_not_alias = expect_output( &
                                          source, '16', &
                                          '/tmp/ffc_session_derived_alias_test')
    end function test_two_variables_do_not_alias

    logical function test_component_stop_code()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: status_t'//new_line('a')// &
                                       '    integer :: code'//new_line('a')// &
                                       '  end type status_t'//new_line('a')// &
                                       '  type(status_t) :: result'//new_line('a')// &
                                       '  result%code = 7'//new_line('a')// &
                                       '  stop result%code'//new_line('a')// &
                                       'end program main'

        test_component_stop_code = expect_exit_status( &
                                   source, 7, &
                                   '/tmp/ffc_session_derived_stop_test')
    end function test_component_stop_code

    logical function test_real_component_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: sample_t'//new_line('a')// &
                                       '    real :: value'//new_line('a')// &
                                       '  end type sample_t'//new_line('a')// &
                                       'end program main'

        test_real_component_diagnostic = expect_error_contains( &
                                         source, 'unsupported derived type component', &
                                         '/tmp/ffc_session_derived_real_test')
    end function test_real_component_diagnostic

    logical function test_nested_component_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: inner_t'//new_line('a')// &
                                       '    integer :: value'//new_line('a')// &
                                       '  end type inner_t'//new_line('a')// &
                                       '  type :: outer_t'//new_line('a')// &
                                       '    type(inner_t) :: inner'//new_line('a')// &
                                       '  end type outer_t'//new_line('a')// &
                                       'end program main'

        test_nested_component_diagnostic = expect_error_contains( &
                                           source, 'unsupported nested derived type', &
                                           '/tmp/ffc_session_derived_nested_test')
    end function test_nested_component_diagnostic

    logical function test_constructor_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: item_t'//new_line('a')// &
                                       '    integer :: count'//new_line('a')// &
                                       '  end type item_t'//new_line('a')// &
                                       '  type(item_t) :: item'//new_line('a')// &
                                       '  item = item_t(4)'//new_line('a')// &
                                       'end program main'

        test_constructor_diagnostic = expect_error_contains( &
                                      source, 'unsupported derived type constructor', &
                                      '/tmp/ffc_session_derived_constructor_test')
    end function test_constructor_diagnostic

    logical function test_inheritance_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: base_t'//new_line('a')// &
                                       '    integer :: code'//new_line('a')// &
                                       '  end type base_t'//new_line('a')// &
                                       '  type, extends(base_t) :: child_t'// &
                                       new_line('a')// &
                                       '    integer :: extra'//new_line('a')// &
                                       '  end type child_t'//new_line('a')// &
                                       'end program main'

        test_inheritance_diagnostic = expect_error_contains( &
                                      source, 'unsupported derived type inheritance', &
                                      '/tmp/ffc_session_derived_extends_test')
    end function test_inheritance_diagnostic

    logical function test_type_bound_diagnostic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  type :: bound_t'//new_line('a')// &
                                       '    integer :: code'//new_line('a')// &
                                       '  contains'//new_line('a')// &
                                       '    procedure :: show'//new_line('a')// &
                                       '  end type bound_t'//new_line('a')// &
                                       'contains'//new_line('a')// &
                                       '  subroutine show(self)'//new_line('a')// &
                                       '    type(bound_t) :: self'//new_line('a')// &
                                       '  end subroutine show'//new_line('a')// &
                                       'end program main'

        test_type_bound_diagnostic = expect_error_contains( &
                                     source, 'unsupported type-bound procedure', &
                                     '/tmp/ffc_session_derived_bound_test')
    end function test_type_bound_diagnostic

    logical function expect_exit_status(source, expected, stem)
        character(len=*), intent(in) :: source
        integer, intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: exe_path
        integer :: cmd_stat
        integer :: exit_stat

        expect_exit_status = .false.
        exe_path = stem
        call execute_command_line('rm -f '//exe_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
            return
        end if

        call execute_command_line(exe_path, exitstat=exit_stat, cmdstat=cmd_stat)
        call execute_command_line('rm -f '//exe_path)
        if (cmd_stat /= 0) then
            print *, 'FAIL: executable did not run'
            return
        end if
        if (exit_stat /= expected) then
            print *, 'FAIL: expected exit status ', expected, ', got ', exit_stat
            return
        end if

        expect_exit_status = .true.
    end function expect_exit_status

    logical function expect_output(source, expected, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: expected
        character(len=*), intent(in) :: stem
        character(len=64) :: output_line
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: exe_path
        character(len=:), allocatable :: out_path
        integer :: exit_stat
        integer :: io_stat
        integer :: unit

        expect_output = .false.
        exe_path = stem
        out_path = stem//'.out'
        call execute_command_line('rm -f '//exe_path//' '//out_path)
        call compile_and_lower(source, exe_path, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: direct LIRIC lowering failed: ', trim(error_msg)
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
            print *, 'FAIL: expected output ', expected, ', got ', &
                trim(output_line)
            return
        end if

        expect_output = .true.
    end function expect_output

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
end program test_session_derived_type_compiler
