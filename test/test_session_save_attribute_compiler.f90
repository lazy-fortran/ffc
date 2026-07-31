program test_session_save_attribute_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session save-attribute compiler test ==='

    all_passed = .true.
    if (.not. test_saved_counter_persists()) all_passed = .false.
    if (.not. test_saved_counter_prints()) all_passed = .false.
    if (.not. test_distinct_procedures_separate_storage()) all_passed = .false.
    if (.not. test_real_saved_accumulates()) all_passed = .false.
    if (.not. test_saved_array_counter_persists()) all_passed = .false.
    if (.not. test_saved_array_zero_start()) all_passed = .false.
    if (.not. test_saved_data_initializer_applies_once()) all_passed = .false.
    if (.not. test_saved_data_array_applies_once()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: save attribute gives scalar locals persistent storage'

contains

    logical function test_saved_counter_persists()
        ! A saved counter incremented on each call reaches 2 on the second call;
        ! without persistence it would re-initialize to 0 and never stop with 2.
        character(len=*), parameter :: source = &
            'subroutine bump()'//new_line('a')// &
            '  integer, save :: c = 0'//new_line('a')// &
            '  c = c + 1'//new_line('a')// &
            '  if (c == 2) stop 2'//new_line('a')// &
            'end subroutine bump'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            'end program main'

        test_saved_counter_persists = expect_exit_status( &
            source, 2, '/tmp/ffc_session_save_persist')
    end function test_saved_counter_persists

    logical function test_saved_counter_prints()
        ! Mirrors corpus issue_1541: print the saved counter on each call.
        character(len=*), parameter :: source = &
            'subroutine counter()'//new_line('a')// &
            '  integer, save :: count = 0'//new_line('a')// &
            '  count = count + 1'//new_line('a')// &
            '  print *, count'//new_line('a')// &
            'end subroutine counter'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call counter()'//new_line('a')// &
            '  call counter()'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           1'//new_line('a')//'           2'//new_line('a')

        test_saved_counter_prints = expect_output( &
            source, expected, '/tmp/ffc_session_save_print')
    end function test_saved_counter_prints

    logical function test_distinct_procedures_separate_storage()
        ! Two procedures each declare a saved `c`; each keeps its own storage.
        ! a: 1 then 2, b: 100 then 101. The program stops with b's final value.
        character(len=*), parameter :: source = &
            'subroutine a()'//new_line('a')// &
            '  integer, save :: c = 0'//new_line('a')// &
            '  c = c + 1'//new_line('a')// &
            'end subroutine a'//new_line('a')// &
            'subroutine b(out)'//new_line('a')// &
            '  integer, intent(out) :: out'//new_line('a')// &
            '  integer, save :: c = 100'//new_line('a')// &
            '  c = c + 1'//new_line('a')// &
            '  out = c'//new_line('a')// &
            'end subroutine b'//new_line('a')// &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  call a()'//new_line('a')// &
            '  call b(r)'//new_line('a')// &
            '  call a()'//new_line('a')// &
            '  call b(r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'end program main'

        test_distinct_procedures_separate_storage = expect_exit_status( &
            source, 102, '/tmp/ffc_session_save_distinct')
    end function test_distinct_procedures_separate_storage

    logical function test_real_saved_accumulates()
        ! A saved real accumulates across calls: 1.5 -> 2.0 -> 2.5.
        character(len=*), parameter :: source = &
            'subroutine acc()'//new_line('a')// &
            '  real, save :: s = 1.5'//new_line('a')// &
            '  s = s + 0.5'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end subroutine acc'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call acc()'//new_line('a')// &
            '  call acc()'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   2.00000000    '//new_line('a')// &
            '   2.50000000    '//new_line('a')

        test_real_saved_accumulates = expect_output( &
            source, expected, '/tmp/ffc_session_save_real')
    end function test_real_saved_accumulates

    logical function test_saved_array_counter_persists()
        ! A saved array keeps every element across calls: element 1 counts the
        ! calls while element 2 accumulates from its initializer.
        character(len=*), parameter :: source = &
            'subroutine bump()'//new_line('a')// &
            '  integer, save :: c(3) = (/ 0, 100, 7 /)'//new_line('a')// &
            '  c(1) = c(1) + 1'//new_line('a')// &
            '  c(2) = c(2) + 10'//new_line('a')// &
            '  print *, c(1), c(2), c(3)'//new_line('a')// &
            'end subroutine bump'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           1         110           7'//new_line('a')// &
            '           2         120           7'//new_line('a')

        test_saved_array_counter_persists = expect_output( &
            source, expected, '/tmp/ffc_session_save_array')
    end function test_saved_array_counter_persists

    logical function test_saved_array_zero_start()
        ! A saved array without an initializer starts zeroed and persists.
        character(len=*), parameter :: source = &
            'subroutine acc(v)'//new_line('a')// &
            '  real, intent(in) :: v'//new_line('a')// &
            '  real, save :: s(2)'//new_line('a')// &
            '  s(1) = s(1) + v'//new_line('a')// &
            '  print *, s(1)'//new_line('a')// &
            'end subroutine acc'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call acc(1.5)'//new_line('a')// &
            '  call acc(2.5)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '   1.50000000    '//new_line('a')// &
            '   4.00000000    '//new_line('a')

        test_saved_array_zero_start = expect_output( &
            source, expected, '/tmp/ffc_session_save_array_zero')
    end function test_saved_array_zero_start

    logical function test_saved_data_initializer_applies_once()
        ! A DATA initializer for a saved local applies once before first use;
        ! re-applying it on each call would print 11 twice.
        character(len=*), parameter :: source = &
            'subroutine bump()'//new_line('a')// &
            '  integer, save :: c'//new_line('a')// &
            '  data c /10/'//new_line('a')// &
            '  c = c + 1'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end subroutine bump'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          11'//new_line('a')//'          12'//new_line('a')

        test_saved_data_initializer_applies_once = expect_output( &
            source, expected, '/tmp/ffc_session_save_data')
    end function test_saved_data_initializer_applies_once

    logical function test_saved_data_array_applies_once()
        ! DATA over a saved array element list applies once; the accumulating
        ! element keeps its value across calls.
        character(len=*), parameter :: source = &
            'subroutine bump()'//new_line('a')// &
            '  integer, save :: c(2)'//new_line('a')// &
            '  data c /5, 6/'//new_line('a')// &
            '  c(1) = c(1) + 1'//new_line('a')// &
            '  print *, c(1), c(2)'//new_line('a')// &
            'end subroutine bump'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           6           6'//new_line('a')// &
            '           7           6'//new_line('a')

        test_saved_data_array_applies_once = expect_output( &
            source, expected, '/tmp/ffc_session_save_data_array')
    end function test_saved_data_array_applies_once

end program test_session_save_attribute_compiler
