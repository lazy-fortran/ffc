program test_session_class_star_assumed_shape_compiler
    ! Differential behavioral oracle for bounded CLASS(*) assumed-shape arrays.
    ! The gfortran executable is the independent semantic reference; this test
    ! does not merely inspect the emitted source or repository state.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    print *, '=== CLASS(*) assumed-shape array test ==='
    if (.not. test_integer_array_select_type()) stop 1
    if (.not. test_real8_array_select_type()) stop 1
    if (.not. test_refusal_contract()) stop 1
    print *, 'PASS: CLASS(*) assumed-shape array narrows with SELECT TYPE'

contains

    logical function test_integer_array_select_type()
        character(len=*), parameter :: source = &
            'module probe_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (integer)'//new_line('a')// &
            '      print *, size(items), items(1), items(2), items(3)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module probe_m'//new_line('a')// &
            'program probe_class_star_array'//new_line('a')// &
            '  use probe_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: values(3)'//new_line('a')// &
            '  values = [4, 7, 9]'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program probe_class_star_array'
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        test_integer_array_select_type = .false.
        base = '/var/tmp/ert/ffc_class_star_assumed_shape'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected CLASS(*) assumed-shape source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected CLASS(*) assumed-shape source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        test_integer_array_select_type = .true.
    end function test_integer_array_select_type

    logical function test_real8_array_select_type()
        character(len=*), parameter :: source = &
            'module probe_real8_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    class(*), intent(in) :: values(:)'//new_line('a')// &
            '    select type (items => values)'//new_line('a')// &
            '    type is (real(8))'//new_line('a')// &
            '      print *, size(items), items(1), items(2), items(3)'// &
            new_line('a')// &
            '    class default'//new_line('a')// &
            '      print *, -1.0d0'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end module probe_real8_m'//new_line('a')// &
            'program probe_class_star_real8'//new_line('a')// &
            '  use probe_real8_m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real(8) :: values(3)'//new_line('a')// &
            '  values = [1.25d0, -2.5d0, 9.0d0]'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program probe_class_star_real8'
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        test_real8_array_select_type = .false.
        base = '/var/tmp/ert/ffc_class_star_assumed_shape_real8'
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected real(8) CLASS(*) assumed-shape source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc real(8) lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected real(8) CLASS(*) assumed-shape source'
            return
        end if
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc real(8) executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran real(8) executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: ffc real(8) output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        test_real8_array_select_type = .true.
    end function test_real8_array_select_type

    logical function test_refusal_contract()
        character(len=*), parameter :: unsupported_kind = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  real :: values(2)'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: section = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer :: values(3)'//new_line('a')// &
            '  call inspect(values(2:3))'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: allocatable = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer, allocatable :: values(:)'//new_line('a')// &
            '  allocate(values(2))'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: ownership = &
            'program p'//new_line('a')// &
            '  interface'//new_line('a')// &
            '    subroutine inspect(values)'//new_line('a')// &
            '      class(*), intent(in) :: values(:)'//new_line('a')// &
            '    end subroutine inspect'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  integer, target :: values(2)'//new_line('a')// &
            '  call inspect(values)'//new_line('a')// &
            'end program p'

        test_refusal_contract = &
            expect_refusal(unsupported_kind, 'real(8)') .and. &
            expect_refusal(section, 'sections') .and. &
            expect_refusal(allocatable, 'allocatable ownership') .and. &
            expect_refusal(ownership, 'pointer/target ownership')
    end function test_refusal_contract

    logical function expect_refusal(source, expected)
        character(len=*), intent(in) :: source, expected
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=128) :: base, src, exe
        integer :: exit_stat, status, unit

        expect_refusal = .false.
        base = '/var/tmp/ert/ffc_class_star_refusal'
        src = trim(base)//'.f90'
        exe = trim(base)//'.ffc'
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w -fsyntax-only '//trim(src), &
            exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: gfortran rejected valid refusal fixture'
            return
        end if
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected valid refusal fixture: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL: unsupported CLASS(*) fixture compiled'
            return
        end if
        if (index(error_msg, expected) == 0) then
            print *, 'FAIL: refusal diagnostic lacks [', trim(expected), ']: ', &
                trim(error_msg)
            return
        end if
        call execute_command_line('rm -f '//trim(src)//' '//trim(exe), &
            exitstat=exit_stat)
        expect_refusal = .true.
    end function expect_refusal

end program test_session_class_star_assumed_shape_compiler
