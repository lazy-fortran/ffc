program test_ffc_suite
    ! One case per process preserves independent STOP and exit status.
    use ffc_suite_group_00, only: &
        run_group_00 => run_group
    use ffc_suite_group_01, only: &
        run_group_01 => run_group
    use ffc_suite_group_02, only: &
        run_group_02 => run_group
    use ffc_suite_group_03, only: &
        run_group_03 => run_group
    use ffc_suite_group_04, only: &
        run_group_04 => run_group
    use ffc_suite_group_05, only: &
        run_group_05 => run_group
    use ffc_suite_group_06, only: &
        run_group_06 => run_group
    use ffc_suite_group_07, only: &
        run_group_07 => run_group
    use ffc_suite_group_08, only: &
        run_group_08 => run_group
    use ffc_suite_group_09, only: &
        run_group_09 => run_group
    use ffc_suite_group_10, only: &
        run_group_10 => run_group
    use ffc_suite_group_11, only: &
        run_group_11 => run_group
    use ffc_suite_group_12, only: &
        run_group_12 => run_group
    use ffc_suite_group_13, only: &
        run_group_13 => run_group
    use ffc_suite_group_14, only: &
        run_group_14 => run_group
    use ffc_suite_group_15, only: &
        run_group_15 => run_group
    implicit none
    character(len=256) :: name
    logical :: matched

    if (command_argument_count() /= 1) then
        print *, "usage: test_ffc_suite <test_name>"
        stop 2
    end if
    call get_command_argument(1, name)

    call run_group_00(trim(name), matched)
    if (matched) goto 100
    call run_group_01(trim(name), matched)
    if (matched) goto 100
    call run_group_02(trim(name), matched)
    if (matched) goto 100
    call run_group_03(trim(name), matched)
    if (matched) goto 100
    call run_group_04(trim(name), matched)
    if (matched) goto 100
    call run_group_05(trim(name), matched)
    if (matched) goto 100
    call run_group_06(trim(name), matched)
    if (matched) goto 100
    call run_group_07(trim(name), matched)
    if (matched) goto 100
    call run_group_08(trim(name), matched)
    if (matched) goto 100
    call run_group_09(trim(name), matched)
    if (matched) goto 100
    call run_group_10(trim(name), matched)
    if (matched) goto 100
    call run_group_11(trim(name), matched)
    if (matched) goto 100
    call run_group_12(trim(name), matched)
    if (matched) goto 100
    call run_group_13(trim(name), matched)
    if (matched) goto 100
    call run_group_14(trim(name), matched)
    if (matched) goto 100
    call run_group_15(trim(name), matched)
    if (matched) goto 100
    print *, "ffc_suite: no such case: "//trim(name)
    stop 3
100 continue
end program test_ffc_suite
