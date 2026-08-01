program test_session_assumed_shape_target
    use ffc_test_support, only: expect_exit_status
    implicit none

    character(len=*), parameter :: target_source = &
        'PROGRAM main'//new_line('a')// &
        'INTEGER :: a(3)'//new_line('a')// &
        'a = [1, 2, 3]'//new_line('a')// &
        'CALL touch(a)'//new_line('a')// &
        'IF (a(1) /= 11 .OR. a(3) /= 3) ERROR STOP 1'//new_line('a')// &
        'CONTAINS'//new_line('a')// &
        'SUBROUTINE touch(x)'//new_line('a')// &
        'INTEGER, INTENT(INOUT), TARGET :: x(:)'//new_line('a')// &
        'x(1) = x(1) + 10'//new_line('a')// &
        'END SUBROUTINE touch'//new_line('a')// &
        'END PROGRAM main'

    character(len=*), parameter :: rank2_source = &
        'PROGRAM main'//new_line('a')// &
        'REAL :: a(2,2), b(2,2)'//new_line('a')// &
        'a(1,1) = 1.0'//new_line('a')// &
        'a(2,1) = 2.0'//new_line('a')// &
        'a(1,2) = 3.0'//new_line('a')// &
        'a(2,2) = 4.0'//new_line('a')// &
        'CALL copy_array(b, a)'//new_line('a')// &
        'IF (b(1,1) /= 1.0 .OR. b(2,1) /= 2.0 .OR. '// &
        'b(1,2) /= 3.0 .OR. b(2,2) /= 4.0) ERROR STOP 1'//new_line('a')// &
        'CONTAINS'//new_line('a')// &
        'SUBROUTINE copy_array(dst, src)'//new_line('a')// &
        'REAL, INTENT(OUT) :: dst(:,:)'//new_line('a')// &
        'REAL, INTENT(IN) :: src(:,:)'//new_line('a')// &
        'dst = src'//new_line('a')// &
        'END SUBROUTINE copy_array'//new_line('a')// &
        'END PROGRAM main'

    if (.not. expect_exit_status(target_source, 0, &
            '/tmp/ffc_session_assumed_shape_target')) stop 1
    if (.not. expect_exit_status(rank2_source, 0, &
            '/tmp/ffc_session_assumed_shape_rank2')) stop 1
    print *, 'PASS: assumed-shape TARGET and rank-2 copies use descriptors'
end program test_session_assumed_shape_target
