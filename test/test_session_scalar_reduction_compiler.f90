program test_session_scalar_reduction_compiler
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session scalar reduction compiler test ==='

    all_passed = .true.
    if (.not. test_all_bare_logical_array()) all_passed = .false.
    if (.not. test_any_bare_logical_array()) all_passed = .false.
    if (.not. test_all_logical_constructor()) all_passed = .false.
    if (.not. test_neqv_mask()) all_passed = .false.
    if (.not. test_eqv_mask()) all_passed = .false.
    if (.not. test_section_comparison()) all_passed = .false.
    if (.not. test_count_section()) all_passed = .false.
    if (.not. test_all_in_if_condition()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: scalar-result whole-array reductions lower correctly'

contains

    logical function test_all_bare_logical_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: l(4) = [.true., .true., .true., .true.]'//new_line('a')// &
            '  print *, all(l)'//new_line('a')// &
            'end program main'

        test_all_bare_logical_array = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_all_bare')
    end function test_all_bare_logical_array

    logical function test_any_bare_logical_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: k(5) = .false.'//new_line('a')// &
            '  print *, any(k)'//new_line('a')// &
            'end program main'

        test_any_bare_logical_array = expect_output( &
            source, ' F'//new_line('a'), '/tmp/ffc_any_bare')
    end function test_any_bare_logical_array

    logical function test_all_logical_constructor()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, all([.true., .false.])'//new_line('a')// &
            'end program main'

        test_all_logical_constructor = expect_output( &
            source, ' F'//new_line('a'), '/tmp/ffc_all_ctor')
    end function test_all_logical_constructor

    logical function test_neqv_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: d(4) = [.false., .true., .true., .false.]'//new_line('a')// &
            '  print *, any(d .neqv. [.false., .true., .true., .false.])'// &
            new_line('a')// &
            'end program main'

        test_neqv_mask = expect_output( &
            source, ' F'//new_line('a'), '/tmp/ffc_neqv')
    end function test_neqv_mask

    logical function test_eqv_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: d(4) = [.false., .true., .true., .false.]'//new_line('a')// &
            '  print *, all(d .eqv. [.false., .true., .true., .false.])'// &
            new_line('a')// &
            'end program main'

        test_eqv_mask = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_eqv')
    end function test_eqv_mask

    logical function test_section_comparison()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4) = [1, 2, 3, 4], b(4) = [1, 9, 3, 9]'// &
            new_line('a')// &
            '  print *, any(a(2:4) /= b(2:4))'//new_line('a')// &
            'end program main'

        test_section_comparison = expect_output( &
            source, ' T'//new_line('a'), '/tmp/ffc_sec_cmp')
    end function test_section_comparison

    logical function test_count_section()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4) = [1, 2, 3, 4], b(4) = [1, 9, 3, 9]'// &
            new_line('a')// &
            '  print *, count(a(1:4) == b(1:4))'//new_line('a')// &
            'end program main'

        test_count_section = expect_output( &
            source, '           2'//new_line('a'), '/tmp/ffc_count_sec')
    end function test_count_section

    logical function test_all_in_if_condition()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  logical :: l(3) = [.true., .true., .true.]'//new_line('a')// &
            '  if (.not. all(l)) error stop 7'//new_line('a')// &
            'end program main'

        test_all_in_if_condition = expect_exit_status( &
            source, 0, '/tmp/ffc_all_if')
    end function test_all_in_if_condition

end program test_session_scalar_reduction_compiler
