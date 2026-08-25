program test_session_forall_compiler
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session FORALL compiler test ==='

    all_passed = .true.
    if (.not. test_indexed_forall()) all_passed = .false.
    if (.not. test_rank2_indexed_forall()) all_passed = .false.
    if (.not. test_block_forall()) all_passed = .false.
    if (.not. test_masked_forall()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: FORALL lowers through direct LIRIC session'

contains

    logical function test_indexed_forall()
        ! Single-statement FORALL over a rank-1 array. sum(a)=3+6+9+12 = 30.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            '  forall (i = 1:4) a(i) = i * 3'//new_line('a')// &
            '  stop sum(a)'//new_line('a')// &
            'end program main'
        test_indexed_forall = expect_exit_status( &
            source, 30, '/tmp/ffc_forall_indexed')
    end function test_indexed_forall

    logical function test_rank2_indexed_forall()
        ! Both indices, column-major selection, mask, and snapshot semantics.
        ! Only column 1 is selected; both rows swap from the old column.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  integer :: a(2,3)'//new_line('a')// &
            '  a(1,1) = 1'//new_line('a')// &
            '  a(2,1) = 2'//new_line('a')// &
            '  a(1,2) = 3'//new_line('a')// &
            '  a(2,2) = 4'//new_line('a')// &
            '  a(1,3) = 5'//new_line('a')// &
            '  a(2,3) = 6'//new_line('a')// &
            '  forall (i = 1:2, j = 1:3, j == 1) '// &
            'a(i,j) = a(3-i,j) + i + j'//new_line('a')// &
            '  stop sum(a)'//new_line('a')// &
            'end program main'
        test_rank2_indexed_forall = expect_exit_status( &
            source, 26, '/tmp/ffc_forall_rank2_indexed')
    end function test_rank2_indexed_forall

    logical function test_block_forall()
        ! Block FORALL form. sum(a) = 1+2+3+4+5 = 15.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  forall (i = 1:5)'//new_line('a')// &
            '    a(i) = i'//new_line('a')// &
            '  end forall'//new_line('a')// &
            '  stop sum(a)'//new_line('a')// &
            'end program main'
        test_block_forall = expect_exit_status( &
            source, 15, '/tmp/ffc_forall_block')
    end function test_block_forall

    logical function test_masked_forall()
        ! Masked FORALL: only i > 2 is assigned; i <= 2 keeps its prior 0.
        ! a starts at 0, then a(3)=3, a(4)=4, a(5)=5; sum = 12.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  forall (i = 1:5, i > 2) a(i) = i'//new_line('a')// &
            '  stop sum(a)'//new_line('a')// &
            'end program main'
        test_masked_forall = expect_exit_status( &
            source, 12, '/tmp/ffc_forall_masked')
    end function test_masked_forall

end program test_session_forall_compiler
