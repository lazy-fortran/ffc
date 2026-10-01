program test_ffc_suite
    !! Consolidated test binary (W0.3). `ffc_suite <test_name>`
    !! runs that one case and exits with its status, so the suite
    !! links libffc once instead of once per test. `fo test
    !! <test_name>` routes here for every source marked
    !! `! fo: dispatcher`; the name it reports is unchanged.
    use ffc_suite_cases_00, only: case_test_session_empty_program_compiler
    implicit none
    !! An assumed-length allocatable array is not legal Fortran; the length
    !! has to be explicit.
    character(len=256), allocatable :: argv(:)
    integer :: i

    allocate(character(len=256) :: argv(max(1, command_argument_count())))
    do i = 1, command_argument_count()
        call get_command_argument(i, argv(i))
    end do
    if (command_argument_count() < 1) then
        print *, "usage: ffc_suite <test_name>"
        stop 2
    end if
    argv(1) = adjustl(argv(1))

        if (argv(1) == "test_session_empty_program_compiler") then
            call case_test_session_empty_program_compiler()
            return
        end if
    ! Unknown name: fail loudly, do not report a pass.
    print *, "ffc_suite: no such case: "//trim(argv(1))
    stop 3
end program test_ffc_suite
