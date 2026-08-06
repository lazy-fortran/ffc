program test_session_nested_do_tail_compiler
    ! Regression for #626: a statement after a nested DO in a contained
    ! procedure must remain in the procedure body and execute exactly once.
    ! The gfortran executable is the independent behavioural oracle.
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer :: n'//new_line('a')// &
        '  n = 3'//new_line('a')// &
        '  call work(n)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine work(k)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    integer :: a(3, 2)'//new_line('a')// &
        '    integer :: i, j'//new_line('a')// &
        '    print *, ''start'''//new_line('a')// &
        '    do j = 1, 2'//new_line('a')// &
        '      print *, ''outer'', j'//new_line('a')// &
        '      do i = 1, 3'//new_line('a')// &
        '        a(i,j) = i*10 + j'//new_line('a')// &
        '      end do'//new_line('a')// &
        '      print *, ''inner done'''//new_line('a')// &
        '    end do'//new_line('a')// &
        '    print *, ''end'', a(1,1)'//new_line('a')// &
        '  end subroutine work'//new_line('a')// &
        'end program main'//new_line('a')

    character(len=:), allocatable :: error_msg, actual, reference
    character(len=*), parameter :: exe = '/var/tmp/ert/ffc_issue626_nested_do'
    character(len=*), parameter :: ref = '/var/tmp/ert/ffc_issue626_nested_do_gfortran'
    character(len=*), parameter :: src = '/var/tmp/ert/ffc_issue626_nested_do.f90'
    character(len=*), parameter :: actual_out = '/var/tmp/ert/ffc_issue626_nested_do.out'
    character(len=*), parameter :: ref_out = '/var/tmp/ert/ffc_issue626_nested_do_gfortran.out'
    integer :: cmd_stat, exit_stat

    call execute_command_line('mkdir -p /var/tmp/ert', cmdstat=cmd_stat)
    if (cmd_stat /= 0) error stop 1
    call write_source(src)
    call execute_command_line('gfortran -w '//src//' -o '//ref, &
        exitstat=exit_stat, cmdstat=cmd_stat)
    if (cmd_stat /= 0 .or. exit_stat /= 0) error stop 1
    call execute_command_line(ref//' > '//ref_out, &
        exitstat=exit_stat, cmdstat=cmd_stat)
    if (cmd_stat /= 0 .or. exit_stat /= 0) error stop 1
    call read_file(ref_out, reference)
    if (.not. allocated(reference)) error stop 1

    call compile_to_exe(source, exe, error_msg)
    if (len_trim(error_msg) > 0) then
        print *, 'FAIL: ffc rejected nested-do tail: ', trim(error_msg)
        error stop 1
    end if
    call execute_command_line(exe//' > '//actual_out, &
        exitstat=exit_stat, cmdstat=cmd_stat)
    if (cmd_stat /= 0 .or. exit_stat /= 0) error stop 1
    call read_file(actual_out, actual)
    if (.not. allocated(actual)) then
        print *, 'FAIL: nested DO tail differs from gfortran'
        print *, '  expected: ', reference
        error stop 1
    end if
    if (actual /= reference) then
        print *, 'FAIL: nested DO tail differs from gfortran'
        print *, '  expected: ', reference
        print *, '  actual:   ', actual
        error stop 1
    end if
    print *, 'PASS: nested DO tail matches gfortran'

contains

    subroutine write_source(path)
        character(len=*), intent(in) :: path
        integer :: unit

        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
    end subroutine write_source

    subroutine read_file(path, contents)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: contents
        integer :: unit, ios, file_size

        open (newunit=unit, file=path, status='old', action='read', &
            access='stream', form='unformatted', iostat=ios)
        if (ios /= 0) return
        inquire (unit=unit, size=file_size)
        allocate (character(len=file_size) :: contents)
        read (unit, iostat=ios) contents
        close (unit)
        if (ios /= 0) deallocate (contents)
    end subroutine read_file

end program test_session_nested_do_tail_compiler
