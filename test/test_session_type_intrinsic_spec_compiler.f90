program test_session_type_intrinsic_spec_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session TYPE(intrinsic-type-spec) compiler test ==='

    all_passed = .true.
    if (.not. test_type_integer_scalar()) all_passed = .false.
    if (.not. test_type_real_scalar()) all_passed = .false.
    if (.not. test_type_logical_scalar()) all_passed = .false.
    if (.not. test_type_character_scalar()) all_passed = .false.
    if (.not. test_type_real_kind_selector()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: TYPE(intrinsic-type-spec) lowers through direct LIRIC'

contains

    logical function test_type_integer_scalar()
        ! F2008 TYPE(integer) names the intrinsic integer type.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type(integer) :: a'//new_line('a')// &
            '  a = 25'//new_line('a')// &
            '  stop a'//new_line('a')// &
            'end program main'

        test_type_integer_scalar = expect_exit_status( &
            source, 25, '/tmp/ffc_session_type_integer_test')
    end function test_type_integer_scalar

    logical function test_type_real_scalar()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type(real) :: b'//new_line('a')// &
            '  b = 3.5'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_type_real_scalar = expect_output( &
            source, '   3.50000000    '//new_line('a'), &
            '/tmp/ffc_session_type_real_test')
    end function test_type_real_scalar

    logical function test_type_logical_scalar()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type(logical) :: flag'//new_line('a')// &
            '  flag = .true.'//new_line('a')// &
            '  print *, flag'//new_line('a')// &
            'end program main'

        test_type_logical_scalar = expect_output( &
            source, ' T'//new_line('a'), &
            '/tmp/ffc_session_type_logical_test')
    end function test_type_logical_scalar

    logical function test_type_character_scalar()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type(character(len=4)) :: s'//new_line('a')// &
            '  s = "abcd"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_type_character_scalar = expect_output( &
            source, ' abcd'//new_line('a'), &
            '/tmp/ffc_session_type_character_test')
    end function test_type_character_scalar

    logical function test_type_real_kind_selector()
        ! TYPE(real(kind=4)) carries the intrinsic kind selector through.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type(real(kind=4)) :: x'//new_line('a')// &
            '  x = 1.25'//new_line('a')// &
            '  print *, x'//new_line('a')// &
            'end program main'

        test_type_real_kind_selector = expect_output( &
            source, '   1.25000000    '//new_line('a'), &
            '/tmp/ffc_session_type_real_kind_test')
    end function test_type_real_kind_selector

end program test_session_type_intrinsic_spec_compiler
