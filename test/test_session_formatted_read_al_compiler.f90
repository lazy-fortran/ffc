program test_session_formatted_read_al_compiler
    ! Fixed-width A and L edit descriptors on a formatted file-unit READ
    ! (#434). Values and IOSTAT classifications match gfortran.
    use ffc_test_support, only: expect_output
    implicit none

    character(len=1), parameter :: q = achar(39)
    character(len=*), parameter :: data_path = '/tmp/ffc_formatted_read_al.dat'
    logical :: all_passed

    print *, '=== direct session formatted-read A/L compiler test ==='

    call write_fixture()

    all_passed = .true.
    if (.not. test_read_al_fields()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: formatted A/L file-unit read lowers through direct session'

contains

    subroutine write_fixture()
        ! Records: an A field followed by an L field, then a malformed
        ! logical field. The file ends after three records, so a fourth
        ! READ hits end-of-file.
        integer :: unit

        open (newunit=unit, file=data_path, status='replace', action='write')
        write (unit, '(A)') 'abcdefT '
        write (unit, '(A)') 'xy  F'
        write (unit, '(A)') 'ab ?? '
        close (unit)
    end subroutine write_fixture

    logical function test_read_al_fields()
        character(len=*), parameter :: nl = new_line('a')
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
                 '  integer :: u, ios'//nl// &
                 '  character(len=6) :: a1'//nl// &
                 '  character(len=4) :: a2'//nl// &
                 '  logical :: l1, l2'//nl// &
                 '  open(newunit=u, file='//q//data_path//q// &
                 ', status='//q//'old'//q//')'//nl// &
                 '  read(u, '//q//'(A6,L2)'//q//') a1, l1'//nl// &
                 '  read(u, '//q//'(A2,L3)'//q//') a2, l2'//nl// &
                 '  print *, a1'//nl// &
                 '  print *, l1'//nl// &
                 '  print *, a2'//nl// &
                 '  print *, l2'//nl// &
                 '  read(u, '//q//'(A2,L3)'//q//', iostat=ios) a2, l2'//nl// &
                 '  if (ios /= 0) print *, '//q//'BADL'//q//nl// &
                 '  read(u, '//q//'(A2,L3)'//q//', iostat=ios) a2, l2'//nl// &
                 '  if (ios /= 0) print *, '//q//'EOFR'//q//nl// &
                 '  close(u)'//nl// &
                 'end program main'

        test_read_al_fields = expect_output(source, &
            ' abcdef'//nl// &
            ' T'//nl// &
            ' xy  '//nl// &
            ' F'//nl// &
            ' BADL'//nl// &
            ' EOFR'//nl, &
            '/tmp/ffc_formatted_read_al')
    end function test_read_al_fields

end program test_session_formatted_read_al_compiler
