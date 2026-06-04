program test_session_array_intrinsics_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session array intrinsic compiler test ==='

    all_passed = .true.
    if (.not. test_rank2_array_intrinsics()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: array intrinsics lower through direct LIRIC'

contains

    logical function test_rank2_array_intrinsics()
        character(len=*), parameter :: source = &
                                       'program main'//new_line('a')// &
                                       '  integer :: a(0:1, 2:3)'//new_line('a')// &
                                       '  integer :: dims(2)'//new_line('a')// &
                                       '  a = [1, 2, 3, 4]'//new_line('a')// &
                                       '  dims = shape(a)'//new_line('a')// &
                                       '  print *, dims(1)'//new_line('a')// &
                                       '  print *, dims(2)'//new_line('a')// &
                                       '  print *, size(a)'//new_line('a')// &
                                       '  print *, sum(a)'//new_line('a')// &
                                       '  print *, maxval(a)'//new_line('a')// &
                                       '  print *, minval(a)'//new_line('a')// &
                                       'end program main'

        test_rank2_array_intrinsics = expect_output( &
            source, '           2'//new_line('a')// &
                    '           2'//new_line('a')// &
                    '           4'//new_line('a')// &
                    '          10'//new_line('a')// &
                    '           4'//new_line('a')// &
                    '           1'//new_line('a'), &
            '/tmp/ffc_session_array_intrinsics_test')
    end function test_rank2_array_intrinsics

end program test_session_array_intrinsics_compiler
