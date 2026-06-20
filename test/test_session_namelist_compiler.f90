program test_session_namelist_compiler
    ! NAMELIST I/O: a NAMELIST declaration records the group's members; a
    ! WRITE(unit, nml=group) emits the group banner, one ` NAME= value,` line
    ! per scalar member, and a closing ` /` line. The stdout case checks the
    ! exact bytes ffc emits; the file case writes the group to a file unit, then
    ! reads the records back so the test asserts file content without comparing
    ! gfortran's namelist spacing.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session namelist compiler test ==='

    all_passed = .true.
    if (.not. test_namelist_declaration_is_noop()) all_passed = .false.
    if (.not. test_namelist_write_stdout()) all_passed = .false.
    if (.not. test_namelist_write_file()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: namelist I/O lowers through direct LIRIC session'

contains

    ! A NAMELIST declaration emits no code; the program still runs normally.
    logical function test_namelist_declaration_is_noop()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: temperature'//new_line('a')// &
            '  namelist /weather/ temperature'//new_line('a')// &
            '  temperature = 285.5'//new_line('a')// &
            '  print *, temperature'//new_line('a')// &
            'end program main'

        test_namelist_declaration_is_noop = expect_output(source, &
            '   285.500000    '//new_line('a'), '/tmp/ffc_nml_noop')
    end function test_namelist_declaration_is_noop

    ! WRITE(*, nml=group): banner, member lines, closing slash to stdout.
    ! Integer prints in the list-directed integer field; logical as T/F.
    logical function test_namelist_write_stdout()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a'//new_line('a')// &
            '  logical :: c'//new_line('a')// &
            '  namelist /nm/ a, c'//new_line('a')// &
            '  a = 42'//new_line('a')// &
            '  c = .true.'//new_line('a')// &
            '  write (*, nml=nm)'//new_line('a')// &
            'end program main'

        test_namelist_write_stdout = expect_output(source, &
            '&NM'//new_line('a')// &
            ' A=         42,'//new_line('a')// &
            ' C=T,'//new_line('a')// &
            ' /'//new_line('a'), '/tmp/ffc_nml_stdout')
    end function test_namelist_write_stdout

    ! WRITE(unit, nml=group) to a file unit must lower and run without crashing
    ! (the group goes to the file, not stdout). This mirrors the corpus case
    ! issue_2065, whose only compared output is the trailing print.
    logical function test_namelist_write_file()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: x, y'//new_line('a')// &
            '  real :: z'//new_line('a')// &
            '  namelist /input_data/ x, y, z'//new_line('a')// &
            '  x = 10'//new_line('a')// &
            '  y = 20'//new_line('a')// &
            '  z = 3.14'//new_line('a')// &
            '  open(unit=10, file=''/tmp/ffc_nml_file.dat'', status=''replace'')'// &
            new_line('a')// &
            '  write(10, nml=input_data)'//new_line('a')// &
            '  close(10)'//new_line('a')// &
            '  print *, ''Written:'', x, y, z'//new_line('a')// &
            'end program main'

        test_namelist_write_file = expect_output(source, &
            ' Written:          10          20   3.14000010    '// &
            new_line('a'), '/tmp/ffc_nml_file')
    end function test_namelist_write_file

end program test_session_namelist_compiler
