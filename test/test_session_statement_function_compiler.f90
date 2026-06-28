program test_session_statement_function
    ! Statement-function lowering through direct LIRIC. A statement function
    ! f(x) = expr defined in the specification section is inlined at each call
    ! site: the actual arguments replace the dummy names in the stored body
    ! expression, which is lowered in place (scalar). Covers an integer and a
    ! real statement function called and printed; outputs match gfortran -w.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session statement function compiler test ==='

    all_passed = .true.
    if (.not. test_integer_statement_function()) all_passed = .false.
    if (.not. test_real_statement_function()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: statement functions lower through direct LIRIC session'

contains

    logical function test_integer_statement_function()
        ! sq(i) = i*i + 1 called with 4 yields 17.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, y'//new_line('a')// &
            '  sq(i) = i*i + 1'//new_line('a')// &
            '  y = sq(4)'//new_line('a')// &
            '  print *, y'//new_line('a')// &
            'end program main'

        test_integer_statement_function = expect_output( &
                source, '          17'//new_line('a'), &
                '/tmp/ffc_stmt_fn_int')
        end function test_integer_statement_function

        logical function test_real_statement_function()
            ! area(r) = pi*r*r called with 2.0 yields 12.5663605 (single precision).
            character(len=*), parameter :: source = &
                'program main'//new_line('a')// &
                '  real :: r, a'//new_line('a')// &
                '  real, parameter :: pi = 3.14159'//new_line('a')// &
                '  area(r) = pi * r * r'//new_line('a')// &
                '  a = area(2.0)'//new_line('a')// &
                '  print *, a'//new_line('a')// &
                'end program main'

            test_real_statement_function = expect_output( &
                    source, '   12.5663605    '//new_line('a'), &
                    '/tmp/ffc_stmt_fn_real')
            end function test_real_statement_function

        end program test_session_statement_function
