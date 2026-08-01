program test_session_formatted_read_compiler
    ! Formatted (non-list-directed) READ from a file unit: write values with
    ! explicit edit descriptors, rewind, read them back, and print.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session formatted-read compiler test ==='

    all_passed = .true.
    if (.not. test_formatted_int_and_real_read()) all_passed = .false.
    if (.not. test_formatted_logical_and_char_read()) all_passed = .false.
    if (.not. test_malformed_field_status()) all_passed = .false.
    if (.not. test_internal_logical_read()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted file-unit read lowers through direct LIRIC session'

contains

    logical function test_formatted_int_and_real_read()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, v'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  open(newunit=u, status='//achar(39)//'scratch'//achar(39)//')'// &
            new_line('a')// &
            '  write(u, '//achar(39)//'(I5)'//achar(39)//') 42'//new_line('a')// &
            '  write(u, '//achar(39)//'(F8.3)'//achar(39)//') 3.14'// &
            new_line('a')// &
            '  rewind(u)'//new_line('a')// &
            '  read(u, '//achar(39)//'(I5)'//achar(39)//') v'//new_line('a')// &
            '  read(u, '//achar(39)//'(F8.3)'//achar(39)//') r'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'

        test_formatted_int_and_real_read = expect_output(source, &
            repeat(' ', 10)//'42'//new_line('a')// &
            '   3.14000010    '//new_line('a'), &
            '/tmp/ffc_formatted_read')
    end function test_formatted_int_and_real_read

    logical function test_formatted_logical_and_char_read()
        ! L and A edit descriptors must convert like gfortran does.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: data_path = '/tmp/ffc_fmtread_lc.dat'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  character(len=6) :: word'//new_line('a')// &
            '  open(newunit=u, file='//q//data_path//q//')'//new_line('a')// &
            '  read(u, '//q//'(L2)'//q//') flag'//new_line('a')// &
            '  read(u, '//q//'(A6)'//q//') word'//new_line('a')// &
            '  print *, flag'//new_line('a')// &
            '  print *, word'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'
        integer :: data_unit

        ! The file-unit character WRITE path is a separate feature, so the
        ! fixture is written here rather than by the compiled program.
        open (newunit=data_unit, file=data_path, status='replace')
        write (data_unit, '(A)') 'T'
        write (data_unit, '(A)') 'hello'
        close (data_unit)

        test_formatted_logical_and_char_read = expect_output(source, &
            ' T'//new_line('a')//' hello '//new_line('a'), &
            '/tmp/ffc_formatted_read_lc')
    end function test_formatted_logical_and_char_read

    logical function test_malformed_field_status()
        ! A malformed integer field must yield a nonzero iostat and must not
        ! overwrite the destination.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: data_path = '/tmp/ffc_fmtread_bad.dat'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: u, v, ios'//new_line('a')// &
            '  open(newunit=u, file='//q//data_path//q//')'//new_line('a')// &
            '  v = 7'//new_line('a')// &
            '  read(u, *, iostat=ios) v'//new_line('a')// &
            '  if (ios /= 0) then'//new_line('a')// &
            '    print *, '//q//'BADSTATUS'//q//new_line('a')// &
            '  else'//new_line('a')// &
            '    print *, '//q//'OKSTATUS'//q//new_line('a')// &
            '  end if'//new_line('a')// &
            '  print *, v'//new_line('a')// &
            '  close(u)'//new_line('a')// &
            'end program main'
        integer :: data_unit

        open (newunit=data_unit, file=data_path, status='replace')
        write (data_unit, '(A)') 'xyz'
        close (data_unit)

        test_malformed_field_status = expect_output(source, &
            ' BADSTATUS'//new_line('a')//repeat(' ', 11)//'7'//new_line('a'), &
            '/tmp/ffc_formatted_read_bad')
    end function test_malformed_field_status

    logical function test_internal_logical_read()
        ! Internal list-directed read of a logical scalar.
        character(len=*), parameter :: q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=20) :: buf'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  buf = '//q//'.true.'//q//new_line('a')// &
            '  read(buf, *) flag'//new_line('a')// &
            '  print *, flag'//new_line('a')// &
            'end program main'

        test_internal_logical_read = expect_output(source, &
            ' T'//new_line('a'), '/tmp/ffc_formatted_read_int')
    end function test_internal_logical_read

end program test_session_formatted_read_compiler
