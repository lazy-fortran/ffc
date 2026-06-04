program test_session_whole_array_arithmetic_compiler
    use ffc_test_support, only: expect_output
    implicit none

    print *, '=== direct session whole-array arithmetic compiler test ==='
    if (.not. test_rank2_whole_array_arithmetic()) stop 1
    print *, 'PASS: whole-array copy and elemental arithmetic lower through direct LIRIC'

contains

    logical function test_rank2_whole_array_arithmetic()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: b(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: c(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: d(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: e(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: f(0:1, 2:3)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4]'//new_line('a')// &
                                       '  b = [5, 6, 7, 8]'//new_line('a')// &
                                       '  c = a + b'//new_line('a')// &
                                       '  d = a - b'//new_line('a')// &
                                       '  e = a * b'//new_line('a')// &
                                       '  f = c'//new_line('a')// &
                                       '  print *, c(0, 2) + c(1, 2) + c(0, 3) + c(1, 3) + &'//new_line('a')// &
                                       '            d(0, 2) + d(1, 2) + d(0, 3) + d(1, 3) + &'//new_line('a')// &
                                       '            e(0, 2) + e(1, 2) + e(0, 3) + e(1, 3) + &'//new_line('a')// &
                                       '            f(0, 2) + f(1, 2) + f(0, 3) + f(1, 3)'//new_line('a')// &
                                       'end program main'

        test_rank2_whole_array_arithmetic = expect_output( &
            source, '         126'//new_line('a'), &
            '/tmp/ffc_session_whole_array_arithmetic_test')
    end function test_rank2_whole_array_arithmetic

end program test_session_whole_array_arithmetic_compiler
