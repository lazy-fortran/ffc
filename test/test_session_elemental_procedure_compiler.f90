program test_session_elemental_procedure_compiler
    ! Verify that a user-defined ELEMENTAL function with more than one dummy
    ! lowers through the shared array-expression iterator: array-array and
    ! array-scalar actuals produce one scalar call per result element, and the
    ! printed arrays match gfortran byte-for-byte (#405). Also verify the
    ! negative controls: nonconformable actuals and an elemental dummy that is
    ! not INTENT(IN) are rejected with a source diagnostic.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session elemental procedure test ==='

    all_passed = .true.

    ! Integer elemental function with array-array and array-scalar actuals.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), b(3), c(3)'//new_line('a')// &
        '  a = [1, 2, 3]'//new_line('a')// &
        '  b = [10, 20, 30]'//new_line('a')// &
        '  c = iadd(a, b)'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        '  c = iadd(a, 5)'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        '  c = iadd(7, b)'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        'contains'//new_line('a')// &
        '  elemental integer function iadd(u, v)'//new_line('a')// &
        '    integer, intent(in) :: u, v'//new_line('a')// &
        '    iadd = u + v'//new_line('a')// &
        '  end function iadd'//new_line('a')// &
        'end program main', &
        'integer')) all_passed = .false.

    ! Real elemental function, array-array and array-scalar actuals.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real :: x(4), y(4), z(4)'//new_line('a')// &
        '  x = [1.0, 2.0, 3.0, 4.0]'//new_line('a')// &
        '  y = [0.5, 1.5, 2.5, 3.5]'//new_line('a')// &
        '  z = blend(x, y)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        '  z = blend(x, 2.0)'//new_line('a')// &
        '  print *, z'//new_line('a')// &
        'contains'//new_line('a')// &
        '  elemental real function blend(u, v)'//new_line('a')// &
        '    real, intent(in) :: u, v'//new_line('a')// &
        '    blend = u * v + u'//new_line('a')// &
        '  end function blend'//new_line('a')// &
        'end program main', &
        'real')) all_passed = .false.

    ! Three dummies, mixing array and scalar actuals inside an expression.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), b(3), c(3)'//new_line('a')// &
        '  a = [1, 2, 3]'//new_line('a')// &
        '  b = [4, 5, 6]'//new_line('a')// &
        '  c = combine(a, b, 2) + 1'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        'contains'//new_line('a')// &
        '  elemental integer function combine(u, v, w)'//new_line('a')// &
        '    integer, intent(in) :: u, v, w'//new_line('a')// &
        '    combine = u * w + v'//new_line('a')// &
        '  end function combine'//new_line('a')// &
        'end program main', &
        'three')) all_passed = .false.

    ! Negative: nonconformable array actuals must be diagnosed.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), d(4), c(3)'//new_line('a')// &
        '  a = 1'//new_line('a')// &
        '  d = 2'//new_line('a')// &
        '  c = iadd(a, d)'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        'contains'//new_line('a')// &
        '  elemental integer function iadd(u, v)'//new_line('a')// &
        '    integer, intent(in) :: u, v'//new_line('a')// &
        '    iadd = u + v'//new_line('a')// &
        '  end function iadd'//new_line('a')// &
        'end program main', &
        'shape')) all_passed = .false.

    ! Negative: an elemental dummy that is not INTENT(IN) must be diagnosed.
    if (.not. rejects( &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: a(3), b(3), c(3)'//new_line('a')// &
        '  a = 1'//new_line('a')// &
        '  b = 2'//new_line('a')// &
        '  c = ibad(a, b)'//new_line('a')// &
        '  print *, c'//new_line('a')// &
        'contains'//new_line('a')// &
        '  elemental integer function ibad(u, v)'//new_line('a')// &
        '    integer, intent(in) :: u'//new_line('a')// &
        '    integer, intent(inout) :: v'//new_line('a')// &
        '    ibad = u + v'//new_line('a')// &
        '  end function ibad'//new_line('a')// &
        'end program main', &
        'intent')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: multi-argument elemental procedures lower through the '// &
        'array-expression iterator'

contains

    logical function rejects(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg

        rejects = .false.
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            rejects = .true.
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, '/tmp/ffc_elemproc_'//stem, error_msg)
        if (len_trim(error_msg) == 0) then
            print *, 'FAIL[', trim(stem), ']: invalid elemental call was '// &
                'accepted without a diagnostic'
            return
        end if
        rejects = .true.
    end function rejects

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_elemproc_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref// &
            ' '//ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_elemental_procedure_compiler
