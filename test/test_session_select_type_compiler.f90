program test_session_select_type_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== class(*) / select type compiler test ==='

    all_passed = .true.
    if (.not. test_class_star_dummy_compiles_without_use()) all_passed = .false.
    if (.not. test_select_type_single_arm_matches()) all_passed = .false.
    if (.not. test_select_type_single_arm_does_not_match()) all_passed = .false.
    if (.not. test_select_type_two_arms_first_matches()) all_passed = .false.
    if (.not. test_select_type_two_arms_second_matches()) all_passed = .false.
    if (.not. test_select_type_class_default_matches_neither()) all_passed = .false.
    if (.not. test_host_select_type_class_is()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: class(*) and select type lower through direct LIRIC'

contains

    logical function test_class_star_dummy_compiles_without_use()
        ! #141: a class(*) dummy is callable with an integer scalar.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call probe(3)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine probe(x)'//new_line('a')// &
            '    class(*), intent(in) :: x'//new_line('a')// &
            '  end subroutine probe'//new_line('a')// &
            'end program main'

        test_class_star_dummy_compiles_without_use = expect_exit_status( &
            source, 0, '/tmp/ffc_session_class_star_test')
    end function test_class_star_dummy_compiles_without_use

    logical function test_select_type_single_arm_matches()
        ! #142: integer through class(*); type is (integer) arm runs.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call probe(8)'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine probe(arg)'//new_line('a')// &
            '    class(*), intent(in) :: arg'//new_line('a')// &
            '    select type (x => arg)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      stop x'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine probe'//new_line('a')// &
            'end program main'

        test_select_type_single_arm_matches = expect_exit_status( &
            source, 8, '/tmp/ffc_session_st_single_test')
    end function test_select_type_single_arm_matches

    logical function test_select_type_single_arm_does_not_match()
        ! #142: a real does not match the integer arm; control falls past.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 42'//new_line('a')// &
            '  call probe(2.5d0, r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine probe(arg, out)'//new_line('a')// &
            '    class(*), intent(in) :: arg'//new_line('a')// &
            '    integer, intent(inout) :: out'//new_line('a')// &
            '    select type (x => arg)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      out = x'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine probe'//new_line('a')// &
            'end program main'

        test_select_type_single_arm_does_not_match = expect_exit_status( &
            source, 42, '/tmp/ffc_session_st_nomatch_test')
    end function test_select_type_single_arm_does_not_match

    logical function test_select_type_two_arms_first_matches()
        test_select_type_two_arms_first_matches = expect_exit_status( &
            two_arm_source('1'), 1, '/tmp/ffc_session_st_first_test')
    end function test_select_type_two_arms_first_matches

    logical function test_select_type_two_arms_second_matches()
        test_select_type_two_arms_second_matches = expect_exit_status( &
            two_arm_source('2.5d0'), 2, '/tmp/ffc_session_st_second_test')
    end function test_select_type_two_arms_second_matches

    logical function test_select_type_class_default_matches_neither()
        test_select_type_class_default_matches_neither = expect_exit_status( &
            two_arm_source('.true.'), 9, '/tmp/ffc_session_st_default_test')
    end function test_select_type_class_default_matches_neither

    logical function test_host_select_type_class_is()
        ! A contained procedure must see the host class descriptor after the
        ! host program allocates it; CLASS IS then selects the declared type.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  class(t), allocatable :: x'//new_line('a')// &
            '  allocate (t :: x)'//new_line('a')// &
            '  call s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s'//new_line('a')// &
            '    select type (x)'//new_line('a')// &
            '    class is (t)'//new_line('a')// &
            "      print *, 'class is t'"//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end program main'

        test_host_select_type_class_is = expect_output( &
            source, ' class is t'//new_line('a'), &
            '/tmp/ffc_session_host_class_select')
    end function test_host_select_type_class_is

    function two_arm_source(actual) result(source)
        ! integer arm -> 1, real arm -> 2, class default -> 9.
        character(len=*), intent(in) :: actual
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 0'//new_line('a')// &
            '  call probe('//actual//', r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine probe(arg, out)'//new_line('a')// &
            '    class(*), intent(in) :: arg'//new_line('a')// &
            '    integer, intent(inout) :: out'//new_line('a')// &
            '    select type (x => arg)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      out = 1'//new_line('a')// &
            '    type is (real)'//new_line('a')// &
            '      out = 2'//new_line('a')// &
            '    class default'//new_line('a')// &
            '      out = 9'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine probe'//new_line('a')// &
            'end program main'
    end function two_arm_source

end program test_session_select_type_compiler
