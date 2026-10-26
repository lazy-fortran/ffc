program test_session_int_real_mixed_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session int/real mixed expression compiler test ==='

    ! real(4): x = 1.0 + i  (i is integer)
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  real :: x'//new_line('a')// &
         '  integer :: i'//new_line('a')// &
         '  i = 3'//new_line('a')// &
         '  x = 1.0 + i'//new_line('a')// &
         '  print *, x'//new_line('a')// &
         'end program main', '   4.00000000    '//new_line('a'), &
         '/tmp/ffc_int_real_f32_add_test')) stop 1

    ! real(4): x = i - 0.5 (i is integer on left)
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  real :: x'//new_line('a')// &
         '  integer :: i'//new_line('a')// &
         '  i = 4'//new_line('a')// &
         '  x = i - 0.5'//new_line('a')// &
         '  print *, x'//new_line('a')// &
         'end program main', '   3.50000000    '//new_line('a'), &
         '/tmp/ffc_int_real_f32_sub_test')) stop 1

    ! real(8): x = 2.0d0 + i (double precision)
    if (.not. expect_output( &
         'program main'//new_line('a')// &
         '  real(8) :: x'//new_line('a')// &
         '  integer :: i'//new_line('a')// &
         '  i = 5'//new_line('a')// &
         '  x = 2.0d0 + i'//new_line('a')// &
         '  print *, x'//new_line('a')// &
         'end program main', '   7.0000000000000000     '//new_line('a'), &
         '/tmp/ffc_int_real_f64_add_test')) stop 1

    print *, 'PASS: implicit int-to-real conversion lowers through direct LIRIC'
end program test_session_int_real_mixed_compiler
