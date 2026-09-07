program test_session_cycle_branch_values_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    all_passed = .true.
    if (.not. test_cycle_values()) all_passed = .false.
    if (.not. test_while_cycle_values()) all_passed = .false.
    if (.not. test_infinite_cycle_values()) all_passed = .false.

    if (.not. test_nested_cycle_values()) all_passed = .false.
    if (.not. test_real_cycle_values()) all_passed = .false.
    if (.not. test_all_branch_cycle_values()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: CYCLE preserves scalar values on each branch'

contains

    logical function test_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i, s, t'//new_line('a')// &
            's=0'//new_line('a')// &
            't=0'//new_line('a')// &
            'do i=1,4'//new_line('a')// &
            ' s=s+i'//new_line('a')// &
            ' if (i==2) then'//new_line('a')// &
            '  s=s+100'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' if (i==3) then'//new_line('a')// &
            '  t=30'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' s=s+10'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,s,t'//new_line('a')// &
            'end program'

        test_cycle_values = expect_output_matches_gfortran( &
            source, 'cycle_values')
    end function test_cycle_values

    logical function test_while_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i,s'//new_line('a')// &
            'i=0'//new_line('a')// &
            's=0'//new_line('a')// &
            'do while(i<4)'//new_line('a')// &
            ' i=i+1'//new_line('a')// &
            ' if (i==2) then'//new_line('a')// &
            '  s=s+100'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' s=s+i'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,s'//new_line('a')// &
            'end program'

        test_while_cycle_values = expect_output_matches_gfortran( &
            source, 'while_cycle_values')
    end function test_while_cycle_values

    logical function test_infinite_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i,s'//new_line('a')// &
            'i=0'//new_line('a')// &
            's=0'//new_line('a')// &
            'do'//new_line('a')// &
            ' i=i+1'//new_line('a')// &
            ' if (i>4) exit'//new_line('a')// &
            ' if (i==2) then'//new_line('a')// &
            '  s=s+100'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' s=s+i'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,s'//new_line('a')// &
            'end program'

        test_infinite_cycle_values = expect_output_matches_gfortran( &
            source, 'infinite_cycle_values')
    end function test_infinite_cycle_values

    logical function test_nested_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i,j,s'//new_line('a')// &
            's=0'//new_line('a')// &
            'do i=1,4'//new_line('a')// &
            ' if (i==1) then'//new_line('a')// &
            '  s=s+10'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' do j=1,2'//new_line('a')// &
            '  s=s+j'//new_line('a')// &
            '  if (j==1) cycle'//new_line('a')// &
            '  s=s+100'//new_line('a')// &
            ' end do'//new_line('a')// &
            ' if (i==3) cycle'//new_line('a')// &
            ' s=s+1000'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,j,s'//new_line('a')// &
            'end program'

        test_nested_cycle_values = expect_output_matches_gfortran( &
            source, 'nested_cycle_values')
    end function test_nested_cycle_values

    logical function test_real_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i'//new_line('a')// &
            'real(8) :: s'//new_line('a')// &
            'logical :: flag'//new_line('a')// &
            's=0.0_8'//new_line('a')// &
            'flag=.false.'//new_line('a')// &
            'do i=1,4'//new_line('a')// &
            ' s=s+dble(i)'//new_line('a')// &
            ' if (i==2) then'//new_line('a')// &
            '  s=s+100.0_8'//new_line('a')// &
            '  flag=.true.'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            ' s=s+10.0_8'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,int(s),flag'//new_line('a')// &
            'end program'

        test_real_cycle_values = expect_output_matches_gfortran( &
            source, 'real_cycle_values')
    end function test_real_cycle_values

    logical function test_all_branch_cycle_values()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: i,s'//new_line('a')// &
            's=0'//new_line('a')// &
            'do i=1,3'//new_line('a')// &
            ' if (i==2) then'//new_line('a')// &
            '  s=s+10'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' else'//new_line('a')// &
            '  s=s+i'//new_line('a')// &
            '  cycle'//new_line('a')// &
            ' end if'//new_line('a')// &
            'end do'//new_line('a')// &
            'print *, i,s'//new_line('a')// &
            'end program'

        test_all_branch_cycle_values = expect_output_matches_gfortran( &
            source, 'all_branch_cycle_values')
    end function test_all_branch_cycle_values

end program test_session_cycle_branch_values_compiler
