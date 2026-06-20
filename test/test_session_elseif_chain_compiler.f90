program test_session_elseif_chain_compiler
    ! #280: ELSE IF arms must be lowered. lower_if previously ignored
    ! elseif_blocks, so the middle arm of an arithmetic IF (its expr==0 branch
    ! desugars to an elseif) was dropped. Cover the cascade directly.
    use ffc_test_support, only: expect_output, expect_exit_status
    implicit none
    logical :: all_passed

    all_passed = .true.
    print *, '=== ELSE IF cascade compiler test ==='

    if (.not. test_middle_arm_selected()) all_passed = .false.
    if (.not. test_arithmetic_if_zero()) all_passed = .false.

    if (all_passed) then
        print *, 'PASS: ELSE IF cascades lower through direct LIRIC session'
    else
        print *, 'FAIL: ELSE IF cascade test failed'
    end if
    if (.not. all_passed) stop 1

contains

    logical function test_middle_arm_selected()
        ! x==2 must select the second elseif arm and merge r at the join.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: x, r'//new_line('a')// &
            '  x = 2'//new_line('a')// &
            '  if (x == 1) then'//new_line('a')// &
            '    r = 10'//new_line('a')// &
            '  else if (x == 2) then'//new_line('a')// &
            '    r = 20'//new_line('a')// &
            '  else if (x == 3) then'//new_line('a')// &
            '    r = 30'//new_line('a')// &
            '  else'//new_line('a')// &
            '    r = 99'//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'
        test_middle_arm_selected = expect_output( &
            source, '          20'//new_line('a'), '/tmp/ffc_elseif_mid_test')
    end function test_middle_arm_selected

    logical function test_arithmetic_if_zero()
        ! Arithmetic IF with a zero operand must take the equal (==0) arm, which
        ! FortFront desugars into an ELSE IF (expr == 0) GOTO branch.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: x'//new_line('a')// &
            '  x = 0'//new_line('a')// &
            '  if (x) 10, 20, 30'//new_line('a')// &
            '10 stop 1'//new_line('a')// &
            '20 stop 2'//new_line('a')// &
            '30 stop 3'//new_line('a')// &
            'end program main'
        ! The equal arm jumps to label 20, which stops with code 2.
        test_arithmetic_if_zero = expect_exit_status( &
            source, 2, '/tmp/ffc_arith_if_zero_test')
    end function test_arithmetic_if_zero

end program test_session_elseif_chain_compiler
