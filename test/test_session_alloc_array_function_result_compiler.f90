program test_session_alloc_array_function_result_compiler
    ! Verify contained functions returning an allocatable rank-1 array lower
    ! through the descriptor-sret ABI: the caller passes a zeroed temporary
    ! descriptor as the hidden result pointer, the callee allocates into it, and
    ! the caller moves that descriptor into the destination allocatable. Output
    ! must match gfortran byte-for-byte.
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    implicit none

    logical :: all_passed

    print *, '=== direct session allocatable array function result test ==='

    all_passed = .true.

    ! Integer result allocated by a constructor, assigned then printed.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, allocatable :: r(:)'//new_line('a')// &
        '  r = make()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make() result(out)'//new_line('a')// &
        '    integer, allocatable :: out(:)'//new_line('a')// &
        '    out = [7, 8, 9]'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        'int_ctor')) all_passed = .false.

    ! Real result allocated then scalar-broadcast; size and indexing of the
    ! moved destination read from the propagated compile-time extent.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real, allocatable :: r(:)'//new_line('a')// &
        '  r = mk()'//new_line('a')// &
        '  print *, size(r), r(2)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function mk() result(x)'//new_line('a')// &
        '    real, allocatable :: x(:)'//new_line('a')// &
        '    allocate(x(4))'//new_line('a')// &
        '    x = 2.5'//new_line('a')// &
        '  end function mk'//new_line('a')// &
        'end program main', &
        'real_bcast')) all_passed = .false.

    ! real(8) result via allocate + broadcast, printed whole.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  real(8), allocatable :: r(:)'//new_line('a')// &
        '  r = dg()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function dg() result(d)'//new_line('a')// &
        '    real(kind=8), allocatable :: d(:)'//new_line('a')// &
        '    allocate(d(3))'//new_line('a')// &
        '    d = 2.0_8'//new_line('a')// &
        '  end function dg'//new_line('a')// &
        'end program main', &
        'f64_bcast')) all_passed = .false.

    ! Result reassigned in a loop: each call sees a freshly zeroed descriptor,
    ! so the destination is rebuilt rather than leaking the prior allocation.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, allocatable :: inp(:)'//new_line('a')// &
        '  integer :: k'//new_line('a')// &
        '  do k = 1, 3'//new_line('a')// &
        '    inp = grow()'//new_line('a')// &
        '    print *, size(inp)'//new_line('a')// &
        '  end do'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function grow() result(ret)'//new_line('a')// &
        '    integer, allocatable :: ret(:)'//new_line('a')// &
        '    allocate(ret(2))'//new_line('a')// &
        '    ret = [1, 2]'//new_line('a')// &
        '  end function grow'//new_line('a')// &
        'end program main', &
        'loop_reuse')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: allocatable array function results lower through '// &
        'direct LIRIC session'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_allocarrfn_'//trim(stem)
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

end program test_session_alloc_array_function_result_compiler
