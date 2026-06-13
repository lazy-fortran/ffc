program test_session_real_literal_print_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session real literal print compiler test ==='

    ! Default real literal is f32: 9 significant digits, %12s    field.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  print *, 2.5'//new_line('a')// &
         'end program main', '   2.50000000    '//new_line('a'), &
         '/tmp/ffc_session_real_literal_print_test')) stop 1

    ! Explicit real(8) literal retains f64: 17 significant digits.
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  print *, 2.5d0'//new_line('a')// &
         'end program main', '   2.5000000000000000     '//new_line('a'), &
         '/tmp/ffc_session_real_literal_f64_print_test')) stop 1

    print *, 'PASS: real literal print lowers through direct LIRIC session'
end program test_session_real_literal_print_compiler
