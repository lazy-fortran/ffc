program test_session_dimension_statement_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    print *, '=== DIMENSION statement separate from type declaration ==='

    ! A bare DIMENSION statement carries the array shape while a separate typed
    ! declaration names the type. FortFront emits the two as distinct nodes;
    ! ffc merges the shape onto the typed declaration instead of failing the
    ! shapeless attribute statement as an unsupported scalar type.
    if (.not. expect_exit_status( &
        'program p'//new_line('a')// &
        '    dimension arr(3)'//new_line('a')// &
        '    double precision arr'//new_line('a')// &
        '    arr(1) = 1.5d0'//new_line('a')// &
        '    arr(2) = 2.5d0'//new_line('a')// &
        '    arr(3) = arr(1) + arr(2)'//new_line('a')// &
        '    if (abs(arr(3) - 4.0d0) > 1.0d-9) error stop'//new_line('a')// &
        'end program', 0, &
        '/tmp/ffc_session_dimension_statement_test')) stop 1

    print *, 'PASS: DIMENSION statement merges with typed declaration'
end program test_session_dimension_statement_compiler
