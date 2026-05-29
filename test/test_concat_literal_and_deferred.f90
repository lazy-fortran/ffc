program test_concat_literal_and_deferred
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== literal + deferred variable concat test ==='

    all_passed = .true.
    if (.not. test_var_then_literal()) all_passed = .false.
    if (.not. test_literal_then_var()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: literal/deferred concat'

contains

    logical function test_var_then_literal()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  a = "hi"'//new_line('a')// &
            '  s = a // "!"'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_var_then_literal = expect_output( &
            source, ' hi!'//new_line('a'), &
            '/tmp/ffc_var_then_literal_test')
    end function test_var_then_literal

    logical function test_literal_then_var()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: a'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  a = "world"'//new_line('a')// &
            '  s = "hello " // a'//new_line('a')// &
            '  print *, s'//new_line('a')// &
            'end program main'

        test_literal_then_var = expect_output( &
            source, ' hello world'//new_line('a'), &
            '/tmp/ffc_literal_then_var_test')
    end function test_literal_then_var

end program test_concat_literal_and_deferred
