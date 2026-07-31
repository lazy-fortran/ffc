program test_session_scalar_procedure_call_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status, &
                                expect_no_error
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar procedure call compiler test ==='

    all_passed = .true.
    if (.not. test_complex_scalar_argument()) all_passed = .false.
    if (.not. test_character_scalar_argument_and_result()) all_passed = .false.
    if (.not. test_derived_scalar_argument_and_result()) all_passed = .false.
    if (.not. test_top_level_scalar_calls()) all_passed = .false.
    if (.not. test_wrong_arity_diagnostic()) all_passed = .false.
    if (.not. test_incompatible_type_diagnostic()) all_passed = .false.
    if (.not. test_host_interface_is_external()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar same-unit procedure calls preserve resolved signatures'

contains

    logical function test_complex_scalar_argument()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  complex :: value'//new_line('a')// &
            '  value = cmplx(1.0, 2.0)'//new_line('a')// &
            '  call shift(value)'//new_line('a')// &
            '  stop int(real(value) + aimag(value))'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine shift(x)'//new_line('a')// &
            '    complex, intent(inout) :: x'//new_line('a')// &
            '    x = x + cmplx(2.0, 3.0)'//new_line('a')// &
            '  end subroutine shift'//new_line('a')// &
            'end program main'

        test_complex_scalar_argument = expect_exit_status( &
            source, 8, '/tmp/ffc_session_scalar_proc_complex')
    end function test_complex_scalar_argument

    logical function test_character_scalar_argument_and_result()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  character(len=:), allocatable :: value'//new_line('a')// &
            '  value = reverse_text("abc")'//new_line('a')// &
            '  call add_mark(value)'//new_line('a')// &
            '  if (value /= "abc!") stop 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function reverse_text(x) result(result_text)'//new_line('a')// &
            '    character(len=*), intent(in) :: x'//new_line('a')// &
            '    character(len=:), allocatable :: result_text'//new_line('a')// &
            '    result_text = x'//new_line('a')// &
            '  end function reverse_text'//new_line('a')// &
            '  subroutine add_mark(x)'//new_line('a')// &
            '    character(len=:), allocatable, intent(inout) :: x'// &
            new_line('a')// &
            '    x = x // "!"'//new_line('a')// &
            '  end subroutine add_mark'//new_line('a')// &
            'end program main'

        test_character_scalar_argument_and_result = expect_exit_status( &
            source, 0, '/tmp/ffc_session_scalar_proc_character')
    end function test_character_scalar_argument_and_result

    logical function test_derived_scalar_argument_and_result()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: pair_t'//new_line('a')// &
            '    integer :: left, right'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(pair_t) :: value, result_value'//new_line('a')// &
            '  value%left = 2'//new_line('a')// &
            '  value%right = 3'//new_line('a')// &
            '  result_value = add_pair(value)'//new_line('a')// &
            '  stop result_value%left + result_value%right'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function add_pair(x) result(y)'//new_line('a')// &
            '    type(pair_t), intent(in) :: x'//new_line('a')// &
            '    type(pair_t) :: y'//new_line('a')// &
            '    y%left = x%left + 10'//new_line('a')// &
            '    y%right = x%right + 20'//new_line('a')// &
            '  end function add_pair'//new_line('a')// &
            'end program main'

        test_derived_scalar_argument_and_result = expect_exit_status( &
            source, 35, '/tmp/ffc_session_scalar_proc_derived')
    end function test_derived_scalar_argument_and_result

    logical function test_top_level_scalar_calls()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer function top_add(x)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '    end function top_add'//new_line('a')// &
            '    subroutine top_set(x)'//new_line('a')// &
            '      integer, intent(inout) :: x'//new_line('a')// &
            '    end subroutine top_set'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer :: value'//new_line('a')// &
            '  value = top_add(4)'//new_line('a')// &
            '  call top_set(value)'//new_line('a')// &
            '  stop value'//new_line('a')// &
            'end program main'//new_line('a')// &
            'integer function top_add(x)'//new_line('a')// &
            '  integer, intent(in) :: x'//new_line('a')// &
            '  top_add = x + 5'//new_line('a')// &
            'end function top_add'//new_line('a')// &
            'subroutine top_set(x)'//new_line('a')// &
            '  integer, intent(inout) :: x'//new_line('a')// &
            '  x = x + 1'//new_line('a')// &
            'end subroutine top_set'

        test_top_level_scalar_calls = expect_exit_status( &
            source, 10, '/tmp/ffc_session_scalar_proc_top_level')
    end function test_top_level_scalar_calls

    logical function test_wrong_arity_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call set_value(1, 2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine set_value(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine set_value'//new_line('a')// &
            'end program main'

        test_wrong_arity_diagnostic = expect_error_contains( &
            source, 'More actual than formal arguments', &
            '/tmp/ffc_session_scalar_proc_arity')
    end function test_wrong_arity_diagnostic

    logical function test_incompatible_type_diagnostic()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: value'//new_line('a')// &
            '  value = 1.0'//new_line('a')// &
            '  call set_value(value)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine set_value(x)'//new_line('a')// &
            '    integer, intent(in) :: x'//new_line('a')// &
            '    print *, x'//new_line('a')// &
            '  end subroutine set_value'//new_line('a')// &
            'end program main'

        test_incompatible_type_diagnostic = expect_error_contains( &
            source, 'mismatched', '/tmp/ffc_session_scalar_proc_type')
    end function test_incompatible_type_diagnostic

    logical function test_host_interface_is_external()
        ! An interface block at host level declares an external procedure whose
        ! body lives in another translation unit (#416), so it lowers to a call
        ! that the linker resolves rather than a "body unavailable" rejection.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    integer function missing_body(x)'//new_line('a')// &
            '      integer, intent(in) :: x'//new_line('a')// &
            '    end function missing_body'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  stop missing_body(1)'//new_line('a')// &
            'end program main'

        test_host_interface_is_external = expect_no_error( &
            source, '/tmp/ffc_session_scalar_proc_missing')
    end function test_host_interface_is_external

end program test_session_scalar_procedure_call_compiler
