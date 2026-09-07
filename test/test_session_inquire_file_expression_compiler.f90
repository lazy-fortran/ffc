program test_session_inquire_file_expression_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    character(len=4096) :: scratch
    character(len=:), allocatable :: path, source
    integer :: unit, status
    logical :: ok, spelling_exists

    call get_environment_variable('TMPDIR', scratch, status=status)
    if (status /= 0 .or. len_trim(scratch) == 0) scratch = '.'
    path = trim(scratch)//'/ffc_inquire_value,part.dat'
    inquire (file='ffc_inquire_probe', exist=spelling_exists)
    if (spelling_exists) stop 'test requires no file named ffc_inquire_probe'
    open (newunit=unit, file=path, status='replace', access='stream', &
        form='unformatted')
    write (unit) 'payload'
    close (unit)

    source = &
        'program main'//new_line('a')// &
        '  character(len=512) :: ffc_inquire_probe, saved'//new_line('a')// &
        '  integer :: n, ios'//new_line('a')// &
        '  logical :: ex'//new_line('a')// &
        "  ffc_inquire_probe = '"//path//"'"//new_line('a')// &
        '  saved = ffc_inquire_probe'//new_line('a')// &
        '  inquire(file=ffc_inquire_probe, exist=ex, size=n, iostat=ios)'// &
        new_line('a')// &
        '  if (.not. ex .or. n /= 7 .or. ios /= 0) error stop 1'//new_line('a')// &
        '  if (ffc_inquire_probe /= saved) error stop 2'//new_line('a')// &
        "  ffc_inquire_probe = '"//path//".missing'"//new_line('a')// &
        '  inquire(file=ffc_inquire_probe, exist=ex)'//new_line('a')// &
        '  if (ex) error stop 3'//new_line('a')// &
        "  ffc_inquire_probe = '"//path//"'"//new_line('a')// &
        '  inquire(file=ffc_inquire_probe//"", exist=ex, size=n)'// &
        new_line('a')// &
        '  if (.not. ex .or. n /= 7) error stop 4'//new_line('a')// &
        "  inquire(file='"//path//"', exist=ex, size=n)"//new_line('a')// &
        '  if (.not. ex .or. n /= 7) error stop 5'//new_line('a')// &
        '  print *, "ok"'//new_line('a')// &
        'end program main'
    ok = expect_output_matches_gfortran(source, 'inquire_file_expression')
    open (newunit=unit, file=path, status='old')
    close (unit, status='delete')
    if (.not. ok) stop 1
    print *, 'PASS: INQUIRE FILE evaluates and trims character expressions'
end program test_session_inquire_file_expression_compiler
