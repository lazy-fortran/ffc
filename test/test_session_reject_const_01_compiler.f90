program test_session_reject_const_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== constant initialization expression rejection test ==='

    all_passed = .true.
    if (.not. test_variable_in_initializer_rejected()) all_passed = .false.
    if (.not. test_function_in_initializer_rejected()) all_passed = .false.
    if (.not. test_named_constant_inquiry_accepted()) all_passed = .false.
    if (.not. test_assumed_shape_inquiry_rejected()) all_passed = .false.
    if (.not. test_fixed_shape_inquiry_accepted()) all_passed = .false.
    if (.not. test_implied_do_variable_rejected()) all_passed = .false.
    if (.not. test_implied_do_index_accepted()) all_passed = .false.
    if (.not. test_integer_bound_overflow_rejected()) all_passed = .false.
    if (.not. test_in_range_bound_accepted()) all_passed = .false.
    if (.not. test_real_conversion_overflow_rejected()) all_passed = .false.
    if (.not. test_in_range_real_conversion_accepted()) all_passed = .false.
    if (.not. test_variable_asynchronous_rejected()) all_passed = .false.
    if (.not. test_constant_asynchronous_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: initialization expressions must reduce to constants'

contains

    ! Rule 1: an initializer may not read a variable.
    logical function test_variable_in_initializer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2) :: xx = "aa"'//new_line('a')// &
            '  integer :: iloc = index(xx, "bb")'//new_line('a')// &
            '  print *, iloc'//new_line('a')// &
            'end program main'

        test_variable_in_initializer_rejected = expect_error_contains( &
            source, 'does not reduce to a constant expression', &
            '/tmp/ffc_reject_const_01_variable')
    end function test_variable_in_initializer_rejected

    logical function test_function_in_initializer_rejected()
        ! Exercise the validator's explicit-procedure lookup across descendant
        ! implementation units. A user function is never an initialization
        ! expression, even when its body returns a literal.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: value = seed()'//new_line('a')// &
            '  print *, value'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function seed()'//new_line('a')// &
            '    seed = 3'//new_line('a')// &
            '  end function seed'//new_line('a')// &
            'end program main'

        test_function_in_initializer_rejected = expect_error_contains( &
            source, "function reference 'seed' is not a constant expression", &
            '/tmp/ffc_reject_const_01_function')
    end function test_function_in_initializer_rejected

    logical function test_named_constant_inquiry_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=2), parameter :: xx = "aa"'//new_line('a')// &
            '  integer, parameter :: n = len(xx)'//new_line('a')// &
            '  stop n'//new_line('a')// &
            'end program main'

        test_named_constant_inquiry_accepted = expect_exit_status( &
            source, 2, '/tmp/ffc_reject_const_01_named')
    end function test_named_constant_inquiry_accepted

    ! Rule 2: the shape of an assumed-shape dummy is not a constant.
    logical function test_assumed_shape_inquiry_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: j(3)'//new_line('a')// &
            '  j = 1'//new_line('a')// &
            '  call doubling(j)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine doubling(n)'//new_line('a')// &
            '    integer, intent(in) :: n(:)'//new_line('a')// &
            '    integer :: m = size(n)'//new_line('a')// &
            '    print *, m'//new_line('a')// &
            '  end subroutine doubling'//new_line('a')// &
            'end program main'

        test_assumed_shape_inquiry_rejected = expect_error_contains( &
            source, 'assumed-shape array', &
            '/tmp/ffc_reject_const_01_assumed')
    end function test_assumed_shape_inquiry_rejected

    logical function test_fixed_shape_inquiry_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: fixed(4)'//new_line('a')// &
            '  integer :: m = size(fixed)'//new_line('a')// &
            '  fixed = 1'//new_line('a')// &
            '  stop m'//new_line('a')// &
            'end program main'

        test_fixed_shape_inquiry_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_reject_const_01_fixed')
    end function test_fixed_shape_inquiry_accepted

    ! Rule 3: an array-constructor implied-do may only use its own index.
    logical function test_implied_do_variable_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  integer :: v(3) = [(j, i=1,3)]'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            'end program main'

        test_implied_do_variable_rejected = expect_error_contains( &
            source, 'does not reduce to a constant expression', &
            '/tmp/ffc_reject_const_01_impliedvar')
    end function test_implied_do_variable_rejected

    logical function test_implied_do_index_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: v(3) = [(i, i=1,3)]'//new_line('a')// &
            '  stop v(3)'//new_line('a')// &
            'end program main'

        test_implied_do_index_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_reject_const_01_impliedidx')
    end function test_implied_do_index_accepted

    ! Rule 4: a folded integer constant must stay inside its kind range.
    logical function test_integer_bound_overflow_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: b(huge(1_8) + 1_8) = 0'//new_line('a')// &
            '  print *, b(1)'//new_line('a')// &
            'end program main'

        test_integer_bound_overflow_rejected = expect_error_contains( &
            source, 'arithmetic overflow in constant expression', &
            '/tmp/ffc_reject_const_01_intovf')
    end function test_integer_bound_overflow_rejected

    logical function test_in_range_bound_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: b(huge(1_8) - huge(1_8) + 4) = 0'//new_line('a')// &
            '  stop size(b)'//new_line('a')// &
            'end program main'

        test_in_range_bound_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_reject_const_01_intok')
    end function test_in_range_bound_accepted

    ! Rule 5: a constant REAL() conversion must fit the requested kind.
    logical function test_real_conversion_overflow_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, real(huge(1.0_8), 4)'//new_line('a')// &
            'end program main'

        test_real_conversion_overflow_rejected = expect_error_contains( &
            source, 'arithmetic overflow in constant expression', &
            '/tmp/ffc_reject_const_01_realovf')
    end function test_real_conversion_overflow_rejected

    logical function test_in_range_real_conversion_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, real(huge(1.0_4), 4)'//new_line('a')// &
            'end program main'

        test_in_range_real_conversion_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_const_01_realok')
    end function test_in_range_real_conversion_accepted

    ! Rule 6: ASYNCHRONOUS= on a transfer statement is an init expression.
    logical function test_variable_asynchronous_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(2) :: no'//new_line('a')// &
            '  no = "no"'//new_line('a')// &
            '  write(*,*,asynchronous=no) 7'//new_line('a')// &
            'end program main'

        test_variable_asynchronous_rejected = expect_error_contains( &
            source, 'does not reduce to a constant expression', &
            '/tmp/ffc_reject_const_01_async')
    end function test_variable_asynchronous_rejected

    logical function test_constant_asynchronous_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  write(*,*,asynchronous="no") 7'//new_line('a')// &
            'end program main'

        test_constant_asynchronous_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_const_01_asyncok')
    end function test_constant_asynchronous_accepted

end program test_session_reject_const_01_compiler
