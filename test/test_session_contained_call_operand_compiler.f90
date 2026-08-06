program test_session_contained_call_operand_compiler
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    use ffc_test_support, only: compile_to_exe
    implicit none

    character(len=*), parameter :: source = &
        'module marker'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        'end module marker'//new_line('a')// &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: calls'//new_line('a')// &
        '  real(8) :: product, sum, rooted, magnitude'//new_line('a')// &
        '  calls = 0'//new_line('a')// &
        '  product = 2.0*bump()'//new_line('a')// &
        '  sum = bump() + 1.0d0'//new_line('a')// &
        '  rooted = sqrt(bump())'//new_line('a')// &
        '  magnitude = abs(bump())'//new_line('a')// &
        '  if (calls /= 4) stop 81'//new_line('a')// &
        '  if (product /= 3.0d0) stop 82'//new_line('a')// &
        '  if (sum /= 2.5d0) stop 83'//new_line('a')// &
        '  if (rooted <= 1.0d0) stop 84'//new_line('a')// &
        '  if (magnitude /= 1.5d0) stop 85'//new_line('a')// &
        '  stop calls'//new_line('a')// &
        'contains'//new_line('a')// &
        '  real(8) function bump()'//new_line('a')// &
        '    calls = calls + 1'//new_line('a')// &
        '    bump = 1.5d0'//new_line('a')// &
        '  end function bump'//new_line('a')// &
        'end program main'

    print *, '=== contained-call expression operand compiler test ==='
    if (.not. matches_gfortran(source)) stop 1
    print *, 'PASS: contained calls execute in scalar expression operands'

contains

    logical function matches_gfortran(program_source)
        character(len=*), intent(in) :: program_source
        character(len=:), allocatable :: error_msg, root, source_path
        character(len=:), allocatable :: ffc_exe, gfortran_exe
        integer :: unit, ffc_status, gfortran_status, command_status

        matches_gfortran = .false.
        root = make_temp_root('contained_call_operand')
        source_path = root//'/case.f90'
        ffc_exe = root//'/case.ffc'
        gfortran_exe = root//'/case.gfortran'

        call compile_to_exe(program_source, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            call remove_temp_root(root)
            return
        end if

        open (newunit=unit, file=source_path, status='replace', action='write')
        write (unit, '(A)') program_source
        close (unit)
        call execute_command_line('gfortran -w '//source_path//' -o '// &
            gfortran_exe, exitstat=gfortran_status, &
            cmdstat=command_status)
        if (command_status /= 0 .or. gfortran_status /= 0) then
            print *, 'FAIL: gfortran rejected the regression source'
            call remove_temp_root(root)
            return
        end if

        call execute_command_line('timeout 5s '//ffc_exe, exitstat=ffc_status, &
            cmdstat=command_status)
        if (command_status /= 0) then
            print *, 'FAIL: ffc regression executable could not run'
            call remove_temp_root(root)
            return
        end if
        call execute_command_line('timeout 5s '//gfortran_exe, &
            exitstat=gfortran_status, &
            cmdstat=command_status)
        if (command_status /= 0) then
            print *, 'FAIL: gfortran regression executable could not run'
            call remove_temp_root(root)
            return
        end if
        if (ffc_status == 124 .or. gfortran_status == 124) then
            print *, 'FAIL: a regression executable timed out'
            call remove_temp_root(root)
            return
        end if
        if (ffc_status /= gfortran_status) then
            print *, 'FAIL: ffc exit status differs from gfortran: ', &
                ffc_status, ' vs ', gfortran_status
            call remove_temp_root(root)
            return
        end if

        call remove_temp_root(root)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_contained_call_operand_compiler
