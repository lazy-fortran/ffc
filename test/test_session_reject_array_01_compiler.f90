program test_session_reject_array_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== invalid array indexing and shape rejection compiler test ==='

    all_passed = .true.
    if (.not. test_zero_stride_literal_rejected()) all_passed = .false.
    if (.not. test_zero_stride_named_constant_rejected()) all_passed = .false.
    if (.not. test_nonzero_stride_accepted()) all_passed = .false.
    if (.not. test_array_forall_mask_rejected()) all_passed = .false.
    if (.not. test_scalar_forall_mask_accepted()) all_passed = .false.
    if (.not. test_zero_sized_format_rejected()) all_passed = .false.
    if (.not. test_sized_format_accepted()) all_passed = .false.
    if (.not. test_pointer_init_to_function_rejected()) all_passed = .false.
    if (.not. test_pointer_init_to_target_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid strides, FORALL masks, zero-sized formats and '// &
        'pointer initialization targets rejected'

contains

    logical function test_zero_stride_literal_rejected()
        ! F2018 9.5.3.3: a section subscript stride must not be zero.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: a(10) = 0'//new_line('a')// &
            '  integer, parameter :: b(10) = a(1:10:0)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_zero_stride_literal_rejected = expect_error_contains( &
            source, 'Illegal stride of zero', &
            '/tmp/ffc_session_zero_stride_reject')
    end function test_zero_stride_literal_rejected

    logical function test_zero_stride_named_constant_rejected()
        ! The same rule holds when the stride folds to zero via a constant.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: s = 0'//new_line('a')// &
            '  integer :: a(10)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  print *, a(1:10:s)'//new_line('a')// &
            'end program main'

        test_zero_stride_named_constant_rejected = expect_error_contains( &
            source, 'Illegal stride of zero', &
            '/tmp/ffc_session_zero_stride_const_reject')
    end function test_zero_stride_named_constant_rejected

    logical function test_nonzero_stride_accepted()
        ! The corrected neighbour: a stride of one still compiles and runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(10), b(10)'//new_line('a')// &
            '  a = 3'//new_line('a')// &
            '  b = a(1:10:1)'//new_line('a')// &
            '  if (b(4) /= 3) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_nonzero_stride_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_stride_one_accept')
    end function test_nonzero_stride_accepted

    logical function test_array_forall_mask_rejected()
        ! F2018 11.1.7.4: the FORALL mask-expr is a scalar logical expression.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: valid(4) = (/ .true., .true., .false., .true. /)'// &
            new_line('a')// &
            '  real :: vec(4)'//new_line('a')// &
            '  integer :: j'//new_line('a')// &
            '  forall (j = 1:4, valid)'//new_line('a')// &
            '     vec(j) = real(j)'//new_line('a')// &
            '  end forall'//new_line('a')// &
            'end program main'

        test_array_forall_mask_rejected = expect_error_contains( &
            source, 'scalar LOGICAL', '/tmp/ffc_session_forall_mask_reject')
    end function test_array_forall_mask_rejected

    logical function test_scalar_forall_mask_accepted()
        ! The corrected neighbour: a scalar mask over the FORALL index.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: vec(4)'//new_line('a')// &
            '  integer :: j'//new_line('a')// &
            '  vec = 0.0'//new_line('a')// &
            '  forall (j = 1:4, j > 2)'//new_line('a')// &
            '     vec(j) = real(j)'//new_line('a')// &
            '  end forall'//new_line('a')// &
            '  if (vec(3) /= 3.0) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_scalar_forall_mask_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_forall_mask_accept')
    end function test_scalar_forall_mask_accepted

    logical function test_zero_sized_format_rejected()
        ! A character format specifier must carry a format; a zero-sized array
        ! carries none.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(3), parameter :: a(0) = [character(3)::]'// &
            new_line('a')// &
            '  print a'//new_line('a')// &
            'end program main'

        test_zero_sized_format_rejected = expect_error_contains( &
            source, 'zero-sized array', '/tmp/ffc_session_zero_format_reject')
    end function test_zero_sized_format_rejected

    logical function test_sized_format_accepted()
        ! The corrected neighbour: a one-element format array is usable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            "  character(6), parameter :: a(1) = ['(a)   ']"//new_line('a')// &
            "  print a, 'ok'"//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_sized_format_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_sized_format_accept')
    end function test_sized_format_accepted

    logical function test_pointer_init_to_function_rejected()
        ! F2018 C1010: a pointer initialization target must be a named entity
        ! with the TARGET attribute, never an intrinsic result.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: y(:) => shape(1)'//new_line('a')// &
            'end program main'

        test_pointer_init_to_function_rejected = expect_error_contains( &
            source, 'TARGET attribute', '/tmp/ffc_session_ptr_init_reject')
    end function test_pointer_init_to_function_rejected

    logical function test_pointer_init_to_target_accepted()
        ! The corrected neighbour: a pointer initialized to NULL() compiles.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: y(:) => null()'//new_line('a')// &
            '  if (associated(y)) stop 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_pointer_init_to_target_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_session_ptr_init_accept')
    end function test_pointer_init_to_target_accepted

end program test_session_reject_array_01_compiler
