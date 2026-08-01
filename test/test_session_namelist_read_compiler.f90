program test_session_namelist_read_compiler
    ! NAMELIST input (#436). READ(unit, nml=group) scans the file unit for the
    ! group header, then assigns each ` NAME = value` pair to the declared
    ! group member of that name. Group and member names match without regard
    ! to case, members may appear in any order, omitted members keep their
    ! current value, and a malformed or absent group reports IOSTAT.
    !
    ! Every expected string below is the byte-exact output of the same program
    ! compiled with gfortran, so the test is an independent oracle.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session namelist read compiler test ==='

    all_passed = .true.
    if (.not. test_read_reordered_partial_group()) all_passed = .false.
    if (.not. test_unknown_group_reports_iostat()) all_passed = .false.
    if (.not. test_unknown_member_reports_iostat()) all_passed = .false.
    if (.not. test_type_mismatch_reports_iostat()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: namelist input lowers through direct LIRIC session'

contains

    subroutine write_fixture(path, lines, n)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: lines(:)
        integer, intent(in) :: n
        integer :: unit, i

        open (newunit=unit, file=path, status='replace', action='write')
        do i = 1, n
            write (unit, '(A)') trim(lines(i))
        end do
        close (unit)
    end subroutine write_fixture

    ! Reordered members, a member omitted from the file, a lower-case group
    ! header against an upper-case NAMELIST declaration, and a character
    ! member. gfortran prints exactly the expected bytes below.
    logical function test_read_reordered_partial_group()
        character(len=64) :: lines(6)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: n_steps'//new_line('a')// &
            '  real :: beta'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  real :: gamma'//new_line('a')// &
            '  integer :: ios'//new_line('a')// &
            '  character(len=8) :: label'//new_line('a')// &
            '  namelist /PARAMS/ n_steps, beta, flag, gamma, label'// &
            new_line('a')// &
            '  n_steps = 1'//new_line('a')// &
            '  beta = 0.0'//new_line('a')// &
            '  flag = .false.'//new_line('a')// &
            '  gamma = 9.5'//new_line('a')// &
            '  label = ''zzz'''//new_line('a')// &
            '  open(unit=11, file=''/tmp/ffc_nml_read_in1.nml'', '// &
            'status=''old'')'//new_line('a')// &
            '  read(11, nml=params, iostat=ios)'//new_line('a')// &
            '  close(11)'//new_line('a')// &
            '  print *, ios, n_steps'//new_line('a')// &
            '  print *, beta, gamma'//new_line('a')// &
            '  print *, label'//new_line('a')// &
            '  if (flag) print *, ''FLAG TRUE'''//new_line('a')// &
            'end program main'

        lines(1) = '&params'
        lines(2) = '  flag = .true.,'
        lines(3) = '  beta = 2.5,'
        lines(4) = '  label = ''abc'','
        lines(5) = '  n_steps = 7'
        lines(6) = '/'
        call write_fixture('/tmp/ffc_nml_read_in1.nml', lines, 6)

        test_read_reordered_partial_group = expect_output(source, &
            '           0           7'//new_line('a')// &
            '   2.50000000       9.50000000    '//new_line('a')// &
            ' abc     '//new_line('a')// &
            ' FLAG TRUE'//new_line('a'), '/tmp/ffc_nml_read_ok')
    end function test_read_reordered_partial_group

    ! The file holds a different group: IOSTAT is nonzero and the members keep
    ! their pre-read values.
    logical function test_unknown_group_reports_iostat()
        character(len=64) :: lines(3)

        lines(1) = '&other'
        lines(2) = '  n_steps = 7'
        lines(3) = '/'
        call write_fixture('/tmp/ffc_nml_read_in2.nml', lines, 3)

        test_unknown_group_reports_iostat = &
            expect_iostat_failure('/tmp/ffc_nml_read_in2.nml', &
                                  '/tmp/ffc_nml_read_bad_group')
    end function test_unknown_group_reports_iostat

    ! A name that is not a member of the group is a namelist input error.
    logical function test_unknown_member_reports_iostat()
        character(len=64) :: lines(3)

        lines(1) = '&params'
        lines(2) = '  oops = 7'
        lines(3) = '/'
        call write_fixture('/tmp/ffc_nml_read_in3.nml', lines, 3)

        test_unknown_member_reports_iostat = &
            expect_iostat_failure('/tmp/ffc_nml_read_in3.nml', &
                                  '/tmp/ffc_nml_read_bad_member')
    end function test_unknown_member_reports_iostat

    ! A value that does not convert to the member's declared type is an error.
    logical function test_type_mismatch_reports_iostat()
        character(len=64) :: lines(3)

        lines(1) = '&params'
        lines(2) = '  n_steps = abc'
        lines(3) = '/'
        call write_fixture('/tmp/ffc_nml_read_in4.nml', lines, 3)

        test_type_mismatch_reports_iostat = &
            expect_iostat_failure('/tmp/ffc_nml_read_in4.nml', &
                                  '/tmp/ffc_nml_read_bad_value')
    end function test_type_mismatch_reports_iostat

    ! Shared negative driver: the read reports a nonzero IOSTAT and leaves the
    ! member at its pre-read value.
    logical function expect_iostat_failure(fixture, exe_path) result(ok)
        character(len=*), intent(in) :: fixture
        character(len=*), intent(in) :: exe_path
        character(len=:), allocatable :: source

        source = &
            'program main'//new_line('a')// &
            '  integer :: n_steps'//new_line('a')// &
            '  integer :: ios'//new_line('a')// &
            '  namelist /params/ n_steps'//new_line('a')// &
            '  n_steps = 1'//new_line('a')// &
            '  ios = 0'//new_line('a')// &
            '  open(unit=11, file='''//fixture//''', status=''old'')'// &
            new_line('a')// &
            '  read(11, nml=params, iostat=ios)'//new_line('a')// &
            '  close(11)'//new_line('a')// &
            '  if (ios /= 0) print *, ''IOSTAT NONZERO'''//new_line('a')// &
            '  print *, n_steps'//new_line('a')// &
            'end program main'

        ok = expect_output(source, &
            ' IOSTAT NONZERO'//new_line('a')// &
            '           1'//new_line('a'), exe_path)
    end function expect_iostat_failure

end program test_session_namelist_read_compiler
