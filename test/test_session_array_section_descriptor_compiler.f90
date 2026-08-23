program test_session_array_section_descriptor_compiler
    ! A rank-1 section actual is a borrowed descriptor view: its base is the
    ! first selected element, its extent is the runtime trip count, and its
    ! byte stride is the source element stride. The reference executable uses
    ! the repository's pinned /usr/bin/gfortran lane and is compared bytewise.
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: expect_error_contains, expect_stderr_and_exit
    use session_program_lowering, only: lower_program_to_liric_exe
    use iso_c_binding, only: c_int
    implicit none

    interface
        function ffc_getpid() bind(C, name='getpid') result(pid)
            import c_int
            integer(c_int) :: pid
        end function ffc_getpid
    end interface

    character(len=:), allocatable :: artifact_root
    character(len=64) :: process_tag
    integer(c_int) :: process_id
    integer :: clock_count

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  integer, parameter :: one = 1'//new_line('a')// &
        '  integer :: a(8), got(4), lo, hi, step, bad_step, unit_sum'//new_line('a')// &
        '  a = [10, 20, 30, 40, 50, 60, 70, 80]'//new_line('a')// &
        '  lo = 8 - command_argument_count()'//new_line('a')// &
        '  hi = 2 + 2 * command_argument_count()'//new_line('a')// &
        '  step = command_argument_count() - 2'//new_line('a')// &
        '  call inspect(a(lo:hi:step), got)'//new_line('a')// &
        '  print *, got'//new_line('a')// &
        '  call inspect_reverse(a(::-1))'//new_line('a')// &
        '  lo = 2 + command_argument_count()'//new_line('a')// &
        '  hi = 4 + command_argument_count()'//new_line('a')// &
        '  call inspect_unit(a(lo:hi:one), unit_sum)'//new_line('a')// &
        '  print *, unit_sum'//new_line('a')// &
        '  bad_step = command_argument_count() + 3'//new_line('a')// &
        '  call inspect_empty(a(8:7:bad_step))'//new_line('a')// &
        '  bad_step = -bad_step'//new_line('a')// &
        '  call inspect_empty(a(2:3:bad_step))'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine inspect(v, got)'//new_line('a')// &
        '    integer, intent(inout) :: v(:)'//new_line('a')// &
        '    integer, intent(out) :: got(4)'//new_line('a')// &
        '    integer :: i'//new_line('a')// &
        '    print *, size(v), v(1), v(2), v(3), v(4)'//new_line('a')// &
        '    v(2) = 99'//new_line('a')// &
        '    do i = 1, size(v)'//new_line('a')// &
        '      got(i) = v(i)'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end subroutine inspect'//new_line('a')// &
        '  subroutine inspect_unit(v, total)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    integer, intent(out) :: total'//new_line('a')// &
        '    integer :: i'//new_line('a')// &
        '    total = 0'//new_line('a')// &
        '    do i = 1, size(v)'//new_line('a')// &
        '      total = total + v(i)'//new_line('a')// &
        '    end do'//new_line('a')// &
        '  end subroutine inspect_unit'//new_line('a')// &
        '  subroutine inspect_reverse(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v), v(1), v(2), v(size(v))'//new_line('a')// &
        '  end subroutine inspect_reverse'//new_line('a')// &
        '  subroutine inspect_empty(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v)'//new_line('a')// &
        '  end subroutine inspect_empty'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: refused_source = &
        'program main'//new_line('a')// &
        '  integer :: a(4)'//new_line('a')// &
        '  a = 1'//new_line('a')// &
        '  call inspect(a(1:3:2))'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine inspect(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:,:)'//new_line('a')// &
        '  end subroutine inspect'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: zero_stride_source = &
        'program main'//new_line('a')// &
        '  integer :: a(4), step'//new_line('a')// &
        '  a = 1'//new_line('a')// &
        '  step = command_argument_count()'//new_line('a')// &
        '  call inspect(a(1:4:step))'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine inspect(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v)'//new_line('a')// &
        '  end subroutine inspect'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: nested_descriptor_source = &
        'program main'//new_line('a')// &
        '  integer :: a(8)'//new_line('a')// &
        '  a = 1'//new_line('a')// &
        '  call outer(a(8:2:-2))'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine outer(v)'//new_line('a')// &
        '    integer, intent(inout) :: v(:)'//new_line('a')// &
        '    call inner(v(:))'//new_line('a')// &
        '  end subroutine outer'//new_line('a')// &
        '  subroutine inner(v)'//new_line('a')// &
        '    integer, intent(inout) :: v(0:)'//new_line('a')// &
        '    v(1) = 9'//new_line('a')// &
        '    print *, size(v), v(0), v(1), v(2)'//new_line('a')// &
        '  end subroutine inner'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: forwarded_descriptor_source = &
        'program main'//new_line('a')// &
        '  integer :: a(6)'//new_line('a')// &
        '  a = [10, 20, 30, 40, 50, 60]'//new_line('a')// &
        '  call outer(a(1:6:2))'//new_line('a')// &
        '  print *, a'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine outer(v)'//new_line('a')// &
        '    integer, intent(inout) :: v(:)'//new_line('a')// &
        '    call inner(v)'//new_line('a')// &
        '  end subroutine outer'//new_line('a')// &
        '  subroutine inner(v)'//new_line('a')// &
        '    integer, intent(inout) :: v(0:)'//new_line('a')// &
        '    v(1) = 77'//new_line('a')// &
        '    print *, size(v), v(0), v(1), v(2)'//new_line('a')// &
        '  end subroutine inner'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: pointer_forward_source = &
        'program main'//new_line('a')// &
        '  integer, target :: a(4)'//new_line('a')// &
        '  integer, pointer :: p(:)'//new_line('a')// &
        '  a = [10, 20, 30, 40]'//new_line('a')// &
        '  p => a'//new_line('a')// &
        '  call inspect(p)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine inspect(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v), v'//new_line('a')// &
        '  end subroutine inspect'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: runtime_explicit_forward_source = &
        'program main'//new_line('a')// &
        '  integer :: a(4)'//new_line('a')// &
        '  a = [10, 20, 30, 40]'//new_line('a')// &
        '  call outer(a, 4)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine outer(v, n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    integer :: v(n)'//new_line('a')// &
        '    call inner(v)'//new_line('a')// &
        '  end subroutine outer'//new_line('a')// &
        '  subroutine inner(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v), v'//new_line('a')// &
        '  end subroutine inner'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: side_effect_bound_source = &
        'program main'//new_line('a')// &
        '  integer :: a(4), calls'//new_line('a')// &
        '  a = [10, 20, 30, 40]'//new_line('a')// &
        '  calls = 0'//new_line('a')// &
        '  call inspect(a(1:4:next_step()))'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function next_step()'//new_line('a')// &
        '    calls = calls + 1'//new_line('a')// &
        '    next_step = 1'//new_line('a')// &
        '  end function next_step'//new_line('a')// &
        '  subroutine inspect(v)'//new_line('a')// &
        '    integer, intent(in) :: v(:)'//new_line('a')// &
        '    print *, size(v), v'//new_line('a')// &
        '  end subroutine inspect'//new_line('a')// &
        'end program main'

    process_id = ffc_getpid()
    call system_clock(clock_count)
    write (process_tag, '(I0,"_",I0)') process_id, clock_count
    artifact_root = '/var/tmp/ert/ffc_array_section_descriptor_'//trim(process_tag)

    print *, '=== array-section descriptor/view compiler test ==='
    if (.not. matches_gfortran()) stop 1
    if (.not. expect_stderr_and_exit(zero_stride_source, &
        'Fortran runtime error: Zero stride is not allowed'//new_line('a'), 2, &
        trim(artifact_root)//'_zero_stride')) then
        call cleanup_artifact(trim(artifact_root)//'_zero_stride')
        stop 1
    end if
    call cleanup_artifact(trim(artifact_root)//'_zero_stride')
    if (.not. gfortran_zero_stride_oracle()) stop 1
    if (.not. gfortran_accepts_nested_section()) stop 1
    if (.not. gfortran_accepts(forwarded_descriptor_source, &
        trim(artifact_root)//'_forwarded.f90', 'forwarded descriptor')) stop 1
    if (.not. matches_descriptor_forwarding(forwarded_descriptor_source, &
        trim(artifact_root)//'_forwarded', 'forwarded descriptor')) stop 1
    if (.not. gfortran_accepts(pointer_forward_source, &
        trim(artifact_root)//'_pointer.f90', 'pointer whole actual')) stop 1
    if (.not. expect_error_contains(pointer_forward_source, &
        'assumed-shape descriptor forwarding is not yet supported', &
        trim(artifact_root)//'_pointer_refused')) then
        call cleanup_artifact(trim(artifact_root)//'_pointer_refused')
        stop 1
    end if
    call cleanup_artifact(trim(artifact_root)//'_pointer_refused')
    if (.not. matches_runtime_explicit_forward()) stop 1
    if (.not. gfortran_accepts(side_effect_bound_source, &
        trim(artifact_root)//'_side_effect.f90', &
        'side-effectful bound')) stop 1
    if (.not. expect_error_contains(side_effect_bound_source, &
        'side-effectful array-section bounds are not yet supported', &
        trim(artifact_root)//'_side_effect_refused')) then
        call cleanup_artifact(trim(artifact_root)//'_side_effect_refused')
        stop 1
    end if
    call cleanup_artifact(trim(artifact_root)//'_side_effect_refused')
    if (.not. matches_descriptor_forwarding(nested_descriptor_source, &
        trim(artifact_root)//'_nested', 'nested descriptor forwarding')) stop 1
    if (.not. gfortran_rejects_rank_mismatch()) stop 1
    if (.not. expect_error_contains(refused_source, &
        'Rank mismatch in argument to inspect', &
        trim(artifact_root)//'_rank_refused')) then
        call cleanup_artifact(trim(artifact_root)//'_rank_refused')
        stop 1
    end if
    call cleanup_artifact(trim(artifact_root)//'_rank_refused')
    print *, 'PASS: rank-1 strided section descriptor view matches gfortran'

contains

    logical function matches_gfortran() result(ok)
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, ffc_exe, gfortran_exe
        character(len=:), allocatable :: ffc_out, gfortran_out
        character(len=:), allocatable :: gfortran_source
        integer :: unit, exit_stat, diff_status, scenario

        ok = .false.
        base = trim(artifact_root)//'_oracle'
        src = trim(base)//'.f90'
        ffc_exe = trim(base)//'.ffc'
        gfortran_exe = trim(base)//'.gfortran'
        ffc_out = trim(base)//'.ffc.out'
        gfortran_out = trim(base)//'.gfortran.out'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected descriptor-view source: ', &
                trim(frontend_result%diagnostic_text)
            call cleanup_artifact(base)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc descriptor-view lowering failed: ', trim(error_msg)
            call cleanup_artifact(base)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        ! GCC 16.1.1 miscomputes omitted bounds with a negative stride. Use
        ! the semantically equivalent explicit-bound spelling for its oracle;
        ! FFC still compiles and executes the omitted-bound source above.
        gfortran_source = replace_text(source, 'a(::-1)', 'a(8:1:-1)')
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') gfortran_source
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w '//src// &
            ' -o '//gfortran_exe, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: pinned gfortran rejected descriptor-view source'
            call cleanup_artifact(base)
            return
        end if
        do scenario = 0, 1
            if (scenario == 0) then
                call execute_command_line(ffc_exe//' > '//ffc_out, &
                    exitstat=exit_stat)
                if (exit_stat /= 0) then
                    print *, 'FAIL: ffc descriptor-view executable failed'
                    call cleanup_artifact(base)
                    return
                end if
                call execute_command_line(gfortran_exe//' > '//gfortran_out, &
                    exitstat=exit_stat)
            else
                call execute_command_line(ffc_exe//' probe > '//ffc_out, &
                    exitstat=exit_stat)
                if (exit_stat /= 0) then
                    print *, 'FAIL: ffc runtime descriptor-view executable failed'
                    call cleanup_artifact(base)
                    return
                end if
                call execute_command_line(gfortran_exe//' probe > '//gfortran_out, &
                    exitstat=exit_stat)
            end if
            if (exit_stat /= 0) then
                print *, 'FAIL: gfortran descriptor-view executable failed'
                call cleanup_artifact(base)
                return
            end if
            call execute_command_line('diff '//ffc_out//' '//gfortran_out// &
                ' > /dev/null 2>&1', exitstat=diff_status)
            if (diff_status /= 0) then
                print *, 'FAIL: descriptor-view output differs from gfortran'
                call execute_command_line('diff '//ffc_out//' '//gfortran_out)
                call cleanup_artifact(base)
                return
            end if
        end do
        call cleanup_artifact(base)
        ok = .true.
    end function matches_gfortran

    logical function matches_descriptor_forwarding(source_text, base, label) result(ok)
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=*), intent(in) :: source_text, base, label
        character(len=:), allocatable :: error_msg, src, ffc_exe, gfortran_exe
        character(len=:), allocatable :: ffc_out, gfortran_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        src = trim(base)//'.f90'
        ffc_exe = trim(base)//'.ffc'
        gfortran_exe = trim(base)//'.gfortran'
        ffc_out = trim(base)//'.ffc.out'
        gfortran_out = trim(base)//'.gfortran.out'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source_text, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected '//trim(label)//': ', &
                trim(frontend_result%diagnostic_text)
            call cleanup_artifact(base)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc rejected '//trim(label)//': ', trim(error_msg)
            call cleanup_artifact(base)
            return
        end if
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source_text
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w '//src// &
            ' -o '//gfortran_exe, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected '//trim(label)
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line(ffc_exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc '//trim(label)//' executable failed'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line(gfortran_exe//' > '//gfortran_out, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran '//trim(label)//' executable failed'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//gfortran_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: '//trim(label)//' differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//gfortran_out)
            call cleanup_artifact(base)
            return
        end if
        call cleanup_artifact(base)
        ok = .true.
    end function matches_descriptor_forwarding

    logical function matches_runtime_explicit_forward() result(ok)
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, ffc_exe, gfortran_exe
        character(len=:), allocatable :: ffc_out, gfortran_out
        integer :: unit, exit_stat, diff_status

        ok = .false.
        base = trim(artifact_root)//'_runtime_forward'
        src = trim(base)//'.f90'
        ffc_exe = trim(base)//'.ffc'
        gfortran_exe = trim(base)//'.gfortran'
        ffc_out = trim(base)//'.ffc.out'
        gfortran_out = trim(base)//'.gfortran.out'
        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(runtime_explicit_forward_source, &
            frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL: FortFront rejected runtime explicit-shape forwarding source: ', &
                trim(frontend_result%diagnostic_text)
            call cleanup_artifact(base)
            return
        end if
        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, ffc_exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc runtime explicit-shape forwarding failed: ', &
                trim(error_msg)
            call cleanup_artifact(base)
            return
        end if
        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') runtime_explicit_forward_source
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w '//src// &
            ' -o '//gfortran_exe, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected runtime explicit-shape forwarding source'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line(ffc_exe//' > '//ffc_out, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: ffc runtime explicit-shape forwarding executable failed'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line(gfortran_exe//' > '//gfortran_out, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran runtime explicit-shape forwarding executable failed'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//gfortran_out// &
            ' > /dev/null 2>&1', exitstat=diff_status)
        if (diff_status /= 0) then
            print *, 'FAIL: runtime explicit-shape forwarding differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//gfortran_out)
            call cleanup_artifact(base)
            return
        end if
        call cleanup_artifact(base)
        ok = .true.
    end function matches_runtime_explicit_forward

    logical function gfortran_rejects_rank_mismatch() result(ok)
        character(len=:), allocatable :: path
        integer :: unit, exit_stat

        ok = .false.
        path = trim(artifact_root)//'_rank_mismatch.f90'
        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') refused_source
        close (unit)
        call execute_command_line( &
            '/usr/bin/gfortran -std=f2018 -fsyntax-only -w '//path, &
            exitstat=exit_stat)
        call cleanup_artifact(path)
        if (exit_stat == 0) then
            print *, 'FAIL: gfortran accepted the rank-mismatch negative'
            return
        end if
        ok = .true.
    end function gfortran_rejects_rank_mismatch

    logical function gfortran_zero_stride_oracle() result(ok)
        character(len=:), allocatable :: base, source_path, exe_path, output_path
        integer :: unit, compile_status, run_status, grep_status

        ok = .false.
        base = trim(artifact_root)//'_zero_oracle'
        source_path = trim(base)//'.f90'
        exe_path = trim(base)//'.exe'
        output_path = trim(base)//'.out'
        open (newunit=unit, file=source_path, status='replace', action='write')
        write (unit, '(A)') zero_stride_source
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -fcheck=all -w '// &
            source_path//' -o '//exe_path, exitstat=compile_status)
        if (compile_status /= 0) then
            print *, 'FAIL: gfortran rejected runtime zero-stride oracle source'
            call cleanup_artifact(base)
            return
        end if
        call execute_command_line(exe_path//' > '//output_path//' 2>&1', &
            exitstat=run_status)
        call execute_command_line('grep -F "Zero stride is not allowed" '// &
            output_path//' > /dev/null 2>&1', exitstat=grep_status)
        call cleanup_artifact(base)
        if (run_status /= 2 .or. grep_status /= 0) then
            print *, 'FAIL: gfortran did not provide a controlled zero-stride error'
            return
        end if
        ok = .true.
    end function gfortran_zero_stride_oracle

    logical function gfortran_accepts_nested_section() result(ok)
        ok = gfortran_accepts(nested_descriptor_source, &
            trim(artifact_root)//'_nested.f90', &
            'nested descriptor-section')
    end function gfortran_accepts_nested_section

    logical function gfortran_accepts(source_text, path, label) result(ok)
        character(len=*), intent(in) :: source_text, path, label
        character(len=:), allocatable :: exe_path, output_path
        integer :: unit, exit_stat, output_status

        ok = .false.
        exe_path = trim(path)//'.exe'
        output_path = trim(path)//'.out'
        open (newunit=unit, file=path, status='replace', action='write')
        write (unit, '(A)') source_text
        close (unit)
        call execute_command_line('/usr/bin/gfortran -std=f2018 -w '//path// &
            ' -o '//exe_path, exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL: gfortran rejected '//trim(label)//' source'
            call cleanup_artifact(path)
            return
        end if
        call execute_command_line(exe_path//' > '//output_path, &
            exitstat=exit_stat)
        call execute_command_line('test -s '//output_path, exitstat=output_status)
        call cleanup_artifact(path)
        if (exit_stat /= 0 .or. output_status /= 0) then
            print *, 'FAIL: gfortran did not execute '//trim(label)//' source'
            return
        end if
        ok = .true.
    end function gfortran_accepts

    subroutine cleanup_artifact(base)
        character(len=*), intent(in) :: base
        call execute_command_line('rm -f -- '//trim(base)//' '// &
            trim(base)//'.f90 '//trim(base)//'.exe '//trim(base)//'.out '// &
            trim(base)//'.ffc '//trim(base)//'.gfortran '// &
            trim(base)//'.ffc.out '//trim(base)//'.gfortran.out '// &
            trim(base)//'.f90.exe '//trim(base)//'.f90.out')
    end subroutine cleanup_artifact

    function replace_text(input, old_text, new_text) result(output)
        character(len=*), intent(in) :: input, old_text, new_text
        character(len=:), allocatable :: output
        integer :: position, suffix_start

        position = index(input, old_text)
        if (position <= 0) then
            output = input
            return
        end if
        suffix_start = position + len(old_text)
        if (suffix_start > len(input)) then
            output = input(:position - 1)//new_text
        else
            output = input(:position - 1)//new_text//input(suffix_start:)
        end if
    end function replace_text

end program test_session_array_section_descriptor_compiler
