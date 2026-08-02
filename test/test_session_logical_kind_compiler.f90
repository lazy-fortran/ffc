program test_session_logical_kind_compiler
    use ffc_test_support, only: expect_output
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, parameter :: lp1 = 1, lp8 = 8'//new_line('a')// &
        '  logical(1) :: small'//new_line('a')// &
        '  logical(8) :: wide'//new_line('a')// &
        '  small = .TRUE._lp1'//new_line('a')// &
        '  wide = .false._lp8'//new_line('a')// &
        '  if (small .neqv. .true.) stop 1'//new_line('a')// &
        '  if (wide .neqv. .false.) stop 2'//new_line('a')// &
        '  if (storage_size(small) /= 8) stop 3'//new_line('a')// &
        '  if (storage_size(wide) /= 64) stop 4'//new_line('a')// &
        '  if (kind(.true._1) /= 1) stop 5'//new_line('a')// &
        '  if (kind(.false._lp8) /= 8) stop 6'//new_line('a')// &
        '  wide = .true._1'//new_line('a')// &
        '  if (wide .neqv. .true.) stop 7'//new_line('a')// &
        "  print *, 'PASS: logical literal kinds'"//new_line('a')// &
        'end program main'

    if (.not. expect_output(source, &
            ' PASS: logical literal kinds'//new_line('a'), &
            '/tmp/ffc_session_logical_kind')) stop 1
end program test_session_logical_kind_compiler
