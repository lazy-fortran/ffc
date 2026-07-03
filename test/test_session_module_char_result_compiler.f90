program test_session_module_char_result_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session module character result compiler test ==='

    all_passed = .true.
    if (.not. test_module_deferred_char_result()) all_passed = .false.
    if (.not. test_module_runtime_length_char_result()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module character function results lower through direct LIRIC'

contains

    logical function test_module_deferred_char_result()
        ! A module function with a deferred-length character result
        ! (character(len=:), allocatable) returns through the deferred
        ! descriptor ABI and is callable from a program in the same file (W3).
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function greet()'//new_line('a')// &
            '    character(len=:), allocatable :: greet'//new_line('a')// &
            '    greet = "hello"'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end module'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  print *, greet()'//new_line('a')// &
            'end program main'

        test_module_deferred_char_result = expect_output( &
            source, ' hello'//new_line('a'), &
            '/tmp/ffc_session_module_deferred_char_result_test')
    end function test_module_deferred_char_result

    logical function test_module_runtime_length_char_result()
        ! A module function whose character result length is a runtime
        ! expression (character(len=len(arg))) also returns through the
        ! descriptor ABI; the width comes from the assigned value.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  function dup(s)'//new_line('a')// &
            '    character(len=*), intent(in) :: s'//new_line('a')// &
            '    character(len=len(s)) :: dup'//new_line('a')// &
            '    dup = s'//new_line('a')// &
            '  end function'//new_line('a')// &
            'end module'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  print *, dup("abc")'//new_line('a')// &
            'end program main'

        test_module_runtime_length_char_result = expect_output( &
            source, ' abc'//new_line('a'), &
            '/tmp/ffc_session_module_runtime_char_result_test')
    end function test_module_runtime_length_char_result

end program test_session_module_char_result_compiler
