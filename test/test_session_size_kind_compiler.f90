program test_session_size_kind_compiler
    ! Regression for lfortran/integration_tests/arrays_01_size.f90: SIZE's
    ! ARRAY, DIM, and KIND actuals must not be interpreted by raw position.
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), b(2,4), size_a, size_b, size_c'//new_line('a')// &
        '  integer :: size_d, size_e, size_f, n'//new_line('a')// &
        '  size_a = size(a, kind=4)'//new_line('a')// &
        '  size_b = size(b, dim=1, kind=4)'//new_line('a')// &
        '  size_c = size(array=b, kind=4, dim=2)'//new_line('a')// &
        '  size_d = size(b, 2, 4)'//new_line('a')// &
        '  size_e = size(a, 1, kind=4)'//new_line('a')// &
        '  size_f = size(array=a)'//new_line('a')// &
        '  if (size_a /= 3) error stop 1'//new_line('a')// &
        '  if (size_b /= 2) error stop 2'//new_line('a')// &
        '  if (size_c /= 4) error stop 3'//new_line('a')// &
        '  if (size_d /= 4) error stop 4'//new_line('a')// &
        '  if (size_e /= 3) error stop 5'//new_line('a')// &
        '  if (size_f /= 3) error stop 6'//new_line('a')// &
        '  n = 5'//new_line('a')// &
        '  call check_runtime_size(n)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine check_runtime_size(n)'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: c(n), runtime_a, runtime_b'//new_line('a')// &
        '    integer :: runtime_c, runtime_d, runtime_e, runtime_f'//new_line('a')// &
        '    runtime_a = size(c)'//new_line('a')// &
        '    runtime_b = size(c, 1)'//new_line('a')// &
        '    runtime_c = size(c, kind=4)'//new_line('a')// &
        '    runtime_d = size(c, dim=1, kind=4)'//new_line('a')// &
        '    runtime_e = size(array=c, dim=1, kind=4)'//new_line('a')// &
        '    runtime_f = size(c, 1, 4)'//new_line('a')// &
        '    if (runtime_a /= n) error stop 7'//new_line('a')// &
        '    if (runtime_b /= n) error stop 8'//new_line('a')// &
        '    if (runtime_c /= n) error stop 9'//new_line('a')// &
        '    if (runtime_d /= n) error stop 10'//new_line('a')// &
        '    if (runtime_e /= n) error stop 11'//new_line('a')// &
        '    if (runtime_f /= n) error stop 12'//new_line('a')// &
        '  end subroutine check_runtime_size'//new_line('a')// &
        'end program main'

    print *, '=== direct session SIZE KIND compiler test ==='
    if (.not. matches_gfortran(source, '/tmp/ffc_size_kind_test')) stop 1
    print *, 'PASS: SIZE maps ARRAY, DIM, and KIND on constant/runtime paths'

contains

    logical function matches_gfortran(program_source, stem)
        character(len=*), intent(in) :: program_source, stem
        character(len=:), allocatable :: error_msg, source_path, ffc_exe
        character(len=:), allocatable :: gfortran_exe
        integer :: unit, ffc_status, gfortran_status, cmd_status

        matches_gfortran = .false.
        source_path = trim(stem)//'.f90'
        ffc_exe = trim(stem)//'.ffc'
        gfortran_exe = trim(stem)//'.gfortran'

        call compile_to_exe(program_source, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if
        open (newunit=unit, file=source_path, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//source_path//' -o '// &
                                  gfortran_exe, exitstat=gfortran_status, &
                                  cmdstat=cmd_status)
        if (cmd_status /= 0 .or. gfortran_status /= 0) then
            print *, 'FAIL: gfortran rejected the regression source'
            call cleanup(source_path, ffc_exe, gfortran_exe)
            return
        end if
        call execute_command_line(ffc_exe, exitstat=ffc_status, &
                                  cmdstat=cmd_status)
        if (cmd_status /= 0) then
            print *, 'FAIL: ffc regression executable could not run'
            call cleanup(source_path, ffc_exe, gfortran_exe)
            return
        end if
        call execute_command_line(gfortran_exe, exitstat=gfortran_status, &
                                  cmdstat=cmd_status)
        if (cmd_status /= 0) then
            print *, 'FAIL: gfortran regression executable could not run'
            call cleanup(source_path, ffc_exe, gfortran_exe)
            return
        end if
        if (ffc_status /= gfortran_status) then
            print *, 'FAIL: ffc exit status differs from gfortran: ', &
                     ffc_status, ' vs ', gfortran_status
            call cleanup(source_path, ffc_exe, gfortran_exe)
            return
        end if
        call cleanup(source_path, ffc_exe, gfortran_exe)
        matches_gfortran = .true.
    end function matches_gfortran

    subroutine cleanup(source_path, ffc_exe, gfortran_exe)
        character(len=*), intent(in) :: source_path, ffc_exe, gfortran_exe
        call execute_command_line('rm -f '//source_path//' '//ffc_exe//' '// &
                                  gfortran_exe)
    end subroutine cleanup

end program test_session_size_kind_compiler
