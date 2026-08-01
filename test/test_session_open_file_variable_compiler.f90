program test_session_open_file_variable_compiler
    ! OPEN(file=<character variable>) must connect the unit to the value the
    ! variable holds, with trailing blanks trimmed, and must leave the
    ! variable itself untouched (ffc#644).
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session OPEN(file=variable) compiler test ==='

    all_passed = .true.
    if (.not. test_open_file_variable_connects_named_file()) all_passed = .false.
    if (.not. test_open_file_variable_not_corrupted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: OPEN(file=variable) uses the variable value'

contains

    logical function test_open_file_variable_connects_named_file()
        ! The connected file is the trimmed variable value, not a file named
        ! after the variable and not one with trailing blanks in its name.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: path = '/tmp/ffc_644_open_var.txt'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=40) :: fname'//new_line('a')// &
            '  logical :: ex'//new_line('a')// &
            '  fname = '//q//path//q//new_line('a')// &
            '  open(unit=21, file=fname, status='//q//'replace'//q//')'// &
            new_line('a')// &
            '  close(21)'//new_line('a')// &
            '  inquire(file='//q//path//q//', exist=ex)'//new_line('a')// &
            '  if (.not. ex) error stop 3'//new_line('a')// &
            'end program main'

        call delete_if_present(path)
        test_open_file_variable_connects_named_file = &
            expect_exit_status(source, 0, '/tmp/ffc_644_open_var_case')
    end function test_open_file_variable_connects_named_file

    logical function test_open_file_variable_not_corrupted()
        ! OPEN must not write into the file= variable's storage: every
        ! trailing pad character stays a blank (the file_open_08 invariant).
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=40) :: fname'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  fname = '//q//'/tmp/ffc_644_pad.txt'//q//new_line('a')// &
            '  open(unit=22, file=fname, status='//q//'replace'//q//')'// &
            new_line('a')// &
            '  close(22)'//new_line('a')// &
            '  do i = 21, 40'//new_line('a')// &
            '    if (iachar(fname(i:i)) /= 32) error stop 4'//new_line('a')// &
            '  end do'//new_line('a')// &
            'end program main'

        test_open_file_variable_not_corrupted = expect_exit_status(source, 0, &
            '/tmp/ffc_644_open_pad_case')
    end function test_open_file_variable_not_corrupted

    subroutine delete_if_present(path)
        character(len=*), intent(in) :: path
        integer :: unit, ios
        logical :: ex

        inquire (file=path, exist=ex)
        if (.not. ex) return
        open (newunit=unit, file=path, status='old', iostat=ios)
        if (ios /= 0) return
        close (unit, status='delete')
    end subroutine delete_if_present

end program test_session_open_file_variable_compiler
