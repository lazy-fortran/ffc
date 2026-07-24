program test_session_runtime_fixed_character_compiler
    ! A character(len=len(dummy)) local is automatic fixed-width storage, not an
    ! allocatable deferred-length value: its width is captured at declaration and
    ! every later assignment must pad or truncate to that same width.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== runtime fixed-character compiler test ==='

    all_passed = .true.
    if (.not. test_runtime_width_pads_and_truncates()) all_passed = .false.
    if (.not. test_runtime_width_initializer_is_fixed()) all_passed = .false.
    if (.not. test_runtime_width_survives_if_merge()) all_passed = .false.
    if (.not. test_runtime_width_function_result_is_fixed()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: runtime fixed-width characters preserve declared length'

contains

    logical function test_runtime_width_pads_and_truncates()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call verify("abcdef")'//new_line('a')// &
            '  call verify("xy")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine verify(source)'//new_line('a')// &
            '    character(len=*), intent(in) :: source'//new_line('a')// &
            '    character(len=len(source)) :: local'//new_line('a')// &
            '    local = "xy"'//new_line('a')// &
            '    print *, "[", local, "]", len(local)'//new_line('a')// &
            '    local = "abcdefghijklmnopqrstuvwxyz"'//new_line('a')// &
            '    print *, "[", local, "]", len(local)'//new_line('a')// &
            '  end subroutine verify'//new_line('a')// &
            'end program main'

        test_runtime_width_pads_and_truncates = expect_output( &
            source, ' [xy    ]           6'//new_line('a')// &
            ' [abcdef]           6'//new_line('a')// &
            ' [xy]           2'//new_line('a')// &
            ' [ab]           2'//new_line('a'), &
            '/tmp/ffc_runtime_fixed_char_assignment')
    end function test_runtime_width_pads_and_truncates

    logical function test_runtime_width_initializer_is_fixed()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call verify("abcdef")'//new_line('a')// &
            '  call verify("xy")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine verify(source)'//new_line('a')// &
            '    character(len=*), intent(in) :: source'//new_line('a')// &
            '    character(len=len(source)) :: local = "xy"'//new_line('a')// &
            '    print *, "[", local, "]", len(local)'//new_line('a')// &
            '  end subroutine verify'//new_line('a')// &
            'end program main'

        test_runtime_width_initializer_is_fixed = expect_output( &
            source, ' [xy    ]           6'//new_line('a')// &
            ' [xy]           2'//new_line('a'), &
            '/tmp/ffc_runtime_fixed_char_initializer')
    end function test_runtime_width_initializer_is_fixed

    logical function test_runtime_width_survives_if_merge()
        ! The descriptor slots are stable through an IF, but the symbol state
        ! still has to retain the runtime-fixed classification after its value
        ! pointer is merged. Otherwise the next assignment incorrectly turns it
        ! into a deferred-length character.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call verify("abcdef", .true.)'//new_line('a')// &
            '  call verify("xy", .false.)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine verify(source, short_value)'//new_line('a')// &
            '    character(len=*), intent(in) :: source'//new_line('a')// &
            '    logical, intent(in) :: short_value'//new_line('a')// &
            '    character(len=len(source)) :: local'//new_line('a')// &
            '    if (short_value) then'//new_line('a')// &
            '      local = "xy"'//new_line('a')// &
            '    else'//new_line('a')// &
            '      local = "abcdefghijklmnopqrstuvwxyz"'//new_line('a')// &
            '    end if'//new_line('a')// &
            '    local = "Q"'//new_line('a')// &
            '    print *, "[", local, "]", len(local)'//new_line('a')// &
            '  end subroutine verify'//new_line('a')// &
            'end program main'

        test_runtime_width_survives_if_merge = expect_output( &
            source, ' [Q     ]           6'//new_line('a')// &
            ' [Q ]           2'//new_line('a'), &
            '/tmp/ffc_runtime_fixed_char_if_merge')
    end function test_runtime_width_survives_if_merge

    logical function test_runtime_width_function_result_is_fixed()
        ! A contained function result uses the same descriptor ABI as a
        ! deferred result, but its character(len=len(dummy)) declaration is an
        ! automatic fixed width and must pad a short assignment before returning.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  print *, "[", greet("abcdef"), "]"'//new_line('a')// &
            '  print *, "[", greet("xy"), "]"'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function greet(name) result(result)'//new_line('a')// &
            '    character(len=*), intent(in) :: name'//new_line('a')// &
            '    character(len=len(name)) :: result'//new_line('a')// &
            '    result = "xy"'//new_line('a')// &
            '  end function greet'//new_line('a')// &
            'end program main'

        test_runtime_width_function_result_is_fixed = expect_output( &
            source, ' [xy    ]'//new_line('a')//' [xy]'//new_line('a'), &
            '/tmp/ffc_runtime_fixed_char_function_result')
    end function test_runtime_width_function_result_is_fixed

end program test_session_runtime_fixed_character_compiler
