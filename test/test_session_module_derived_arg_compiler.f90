program test_session_module_derived_arg_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    print *, '=== direct session module derived-type argument compiler test ==='

    if (.not. test_subroutine_derived_args()) stop 1
    if (.not. test_function_derived_result()) stop 1
    if (.not. test_derived_arg_output()) stop 1

    print *, 'PASS: module derived-type arguments lower through direct LIRIC'

contains

    ! A module subroutine taking module-defined derived dummies (intent in/out)
    ! and writing component results back through the out dummy.
    logical function test_subroutine_derived_args()
        character(len=*), parameter :: source = &
            'module vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: vector'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type vector'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine addvec(a, b, c)'//new_line('a')// &
            '    type(vector), intent(in) :: a'//new_line('a')// &
            '    type(vector), intent(in) :: b'//new_line('a')// &
            '    type(vector), intent(out) :: c'//new_line('a')// &
            '    c%x = a%x + b%x'//new_line('a')// &
            '    c%y = a%y + b%y'//new_line('a')// &
            '  end subroutine addvec'//new_line('a')// &
            'end module vecmod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(vector) :: p, q, r'//new_line('a')// &
            '  p%x = 1'//new_line('a')// &
            '  p%y = 2'//new_line('a')// &
            '  q%x = 10'//new_line('a')// &
            '  q%y = 20'//new_line('a')// &
            '  call addvec(p, q, r)'//new_line('a')// &
            '  stop r%x + r%y'//new_line('a')// &
            'end program main'

        ! 11 + 22 = 33.
        test_subroutine_derived_args = expect_exit_status( &
            source, 33, '/tmp/ffc_module_derived_arg_sub_test')
    end function test_subroutine_derived_args

    ! A module function returning a module-defined derived type, called and its
    ! components read back in the program.
    logical function test_function_derived_result()
        character(len=*), parameter :: source = &
            'module vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: vector'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type vector'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function addvec(a, b) result(c)'//new_line('a')// &
            '    type(vector), intent(in) :: a'//new_line('a')// &
            '    type(vector), intent(in) :: b'//new_line('a')// &
            '    type(vector) :: c'//new_line('a')// &
            '    c%x = a%x + b%x'//new_line('a')// &
            '    c%y = a%y + b%y'//new_line('a')// &
            '  end function addvec'//new_line('a')// &
            'end module vecmod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(vector) :: p, q, r'//new_line('a')// &
            '  p%x = 3'//new_line('a')// &
            '  p%y = 4'//new_line('a')// &
            '  q%x = 30'//new_line('a')// &
            '  q%y = 40'//new_line('a')// &
            '  r = addvec(p, q)'//new_line('a')// &
            '  stop r%x + r%y'//new_line('a')// &
            'end program main'

        ! 33 + 44 = 77.
        test_function_derived_result = expect_exit_status( &
            source, 77, '/tmp/ffc_module_derived_arg_fn_test')
    end function test_function_derived_result

    ! Confirm the stdout of a module subroutine that prints a derived result.
    logical function test_derived_arg_output()
        character(len=*), parameter :: source = &
            'module vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: vector'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type vector'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine addvec(a, b, c)'//new_line('a')// &
            '    type(vector), intent(in) :: a'//new_line('a')// &
            '    type(vector), intent(in) :: b'//new_line('a')// &
            '    type(vector), intent(out) :: c'//new_line('a')// &
            '    c%x = a%x + b%x'//new_line('a')// &
            '    c%y = a%y + b%y'//new_line('a')// &
            '  end subroutine addvec'//new_line('a')// &
            'end module vecmod'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use vecmod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(vector) :: p, q, r'//new_line('a')// &
            '  p%x = 1'//new_line('a')// &
            '  p%y = 2'//new_line('a')// &
            '  q%x = 10'//new_line('a')// &
            '  q%y = 20'//new_line('a')// &
            '  call addvec(p, q, r)'//new_line('a')// &
            '  print *, r%x, r%y'//new_line('a')// &
            'end program main'

        test_derived_arg_output = expect_output( &
            source, '          11          22'//new_line('a'), &
            '/tmp/ffc_module_derived_arg_out_test')
    end function test_derived_arg_output

end program test_session_module_derived_arg_compiler
