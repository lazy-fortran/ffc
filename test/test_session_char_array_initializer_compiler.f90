program test_session_char_array_initializer_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session character array initializer compiler test ==='

    all_passed = .true.
    if (.not. test_constructor_elements_print()) all_passed = .false.
    if (.not. test_padding_and_len_trim()) all_passed = .false.
    if (.not. test_scalar_broadcast_initializer()) all_passed = .false.
    if (.not. test_parameter_array_initializer()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character array initializers lower through direct LIRIC session'

contains

    logical function test_constructor_elements_print()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=4) :: s(4) = ["sngl", "dble", "xten", "quad"]'// &
            new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_constructor_elements_print = expect_output( &
            source, ' sngldblextenquad'//new_line('a'), &
            '/tmp/ffc_char_arr_init_print')
    end function test_constructor_elements_print

    logical function test_padding_and_len_trim()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=6) :: m(2) = ["ab", "cdef"]'//new_line('a')// &
            '  if (len_trim(m(1)) /= 2) error stop'//new_line('a')// &
            '  if (len_trim(m(2)) /= 4) error stop'//new_line('a')// &
            '  if (m(1) /= "ab") error stop'//new_line('a')// &
            '  if (m(2) /= "cdef") error stop'//new_line('a')// &
            'end program main'

        test_padding_and_len_trim = expect_exit_status( &
            source, 0, '/tmp/ffc_char_arr_init_pad')
    end function test_padding_and_len_trim

    logical function test_scalar_broadcast_initializer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: p(3) = "xy"'//new_line('a')// &
            '  if (p(1) /= "xy") error stop'//new_line('a')// &
            '  if (p(3) /= "xy") error stop'//new_line('a')// &
            'end program main'

        test_scalar_broadcast_initializer = expect_exit_status( &
            source, 0, '/tmp/ffc_char_arr_init_bcast')
    end function test_scalar_broadcast_initializer

    logical function test_parameter_array_initializer()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=5), parameter :: b(4) = ["ab", "cd", "ef", "gh"]'// &
            new_line('a')// &
            '  if (len(b) /= 5) error stop'//new_line('a')// &
            '  if (b(2) /= "cd") error stop'//new_line('a')// &
            'end program main'

        test_parameter_array_initializer = expect_exit_status( &
            source, 0, '/tmp/ffc_char_arr_init_param')
    end function test_parameter_array_initializer

end program test_session_char_array_initializer_compiler
