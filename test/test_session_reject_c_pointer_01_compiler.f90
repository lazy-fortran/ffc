program test_session_reject_c_pointer_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== C_F_POINTER SHAPE requirement rejection test ==='

    all_passed = .true.
    if (.not. test_array_fptr_without_shape_rejected()) all_passed = .false.
    if (.not. test_deferred_shape_fptr_without_shape_rejected()) &
        all_passed = .false.
    if (.not. test_scalar_fptr_without_shape_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array C_F_POINTER targets require a SHAPE argument'

contains

    logical function test_array_fptr_without_shape_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, dimension(1:2), target :: my_array'//new_line('a')// &
            '  integer, dimension(:), pointer :: my_array_ptr'//new_line('a')// &
            '  type(c_ptr) :: cptr'//new_line('a')// &
            '  my_array = 1'//new_line('a')// &
            '  cptr = c_loc(my_array)'//new_line('a')// &
            '  my_array_ptr => my_array'//new_line('a')// &
            '  call c_f_pointer(cptr, my_array_ptr)'//new_line('a')// &
            'end program main'

        test_array_fptr_without_shape_rejected = expect_error_contains( &
            source, 'Expected SHAPE argument to C_F_POINTER with array FPTR', &
            '/tmp/ffc_session_reject_c_pointer_01_array')
    end function test_array_fptr_without_shape_rejected

    logical function test_deferred_shape_fptr_without_shape_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, dimension(3), target :: my_array'//new_line('a')// &
            '  integer, pointer :: my_array_ptr(:)'//new_line('a')// &
            '  type(c_ptr) :: cptr'//new_line('a')// &
            '  my_array = 2'//new_line('a')// &
            '  cptr = c_loc(my_array)'//new_line('a')// &
            '  call c_f_pointer(cptr, my_array_ptr)'//new_line('a')// &
            'end program main'

        test_deferred_shape_fptr_without_shape_rejected = &
            expect_error_contains(source, &
            'Expected SHAPE argument to C_F_POINTER with array FPTR', &
            '/tmp/ffc_session_reject_c_pointer_01_deferred')
    end function test_deferred_shape_fptr_without_shape_rejected

    logical function test_scalar_fptr_without_shape_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use, intrinsic :: iso_c_binding'//new_line('a')// &
            '  integer, target :: my_value'//new_line('a')// &
            '  integer, pointer :: my_ptr'//new_line('a')// &
            '  type(c_ptr) :: cptr'//new_line('a')// &
            '  my_value = 7'//new_line('a')// &
            '  cptr = c_loc(my_value)'//new_line('a')// &
            '  call c_f_pointer(cptr, my_ptr)'//new_line('a')// &
            '  stop my_ptr'//new_line('a')// &
            'end program main'

        test_scalar_fptr_without_shape_accepted = expect_exit_status( &
            source, 7, '/tmp/ffc_session_c_pointer_01_scalar_ok')
    end function test_scalar_fptr_without_shape_accepted

end program test_session_reject_c_pointer_01_compiler
