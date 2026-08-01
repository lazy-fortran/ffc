program test_session_separate_generic_compiler
    implicit none

    logical :: all_passed
    ! A generic exported by a separately compiled module must resolve in the
    ! using unit by the same rules a same-unit call uses, including the rank of
    ! each specific's dummies. The using unit sees only the .fmod, so the ranks
    ! have to travel in it (#415).
    character(len=*), parameter :: rank_module_source = &
        'module fmod415_generic'//new_line('a')// &
        '    implicit none'//new_line('a')// &
        '    interface total'//new_line('a')// &
        '        module procedure total_scalar'//new_line('a')// &
        '        module procedure total_vec'//new_line('a')// &
        '    end interface total'//new_line('a')// &
        '    interface merged'//new_line('a')// &
        '        module procedure merged_vec'//new_line('a')// &
        '        module procedure merged_mat'//new_line('a')// &
        '    end interface merged'//new_line('a')// &
        '    interface grid_total'//new_line('a')// &
        '        module procedure grid_scalar'//new_line('a')// &
        '        module procedure grid_mat'//new_line('a')// &
        '    end interface grid_total'//new_line('a')// &
        'contains'//new_line('a')// &
        '    integer function total_scalar(x) result(r)'//new_line('a')// &
        '        integer, intent(in) :: x'//new_line('a')// &
        '        r = x'//new_line('a')// &
        '    end function total_scalar'//new_line('a')// &
        '    integer function total_vec(x) result(r)'//new_line('a')// &
        '        integer, intent(in) :: x(3)'//new_line('a')// &
        '        r = x(1) + x(2) + x(3)'//new_line('a')// &
        '    end function total_vec'//new_line('a')// &
        '    integer function grid_scalar(y) result(r)'//new_line('a')// &
        '        integer, intent(in) :: y'//new_line('a')// &
        '        r = 2 * y'//new_line('a')// &
        '    end function grid_scalar'//new_line('a')// &
        '    integer function grid_mat(y) result(r)'//new_line('a')// &
        '        integer, intent(in) :: y(2,2)'//new_line('a')// &
        '        r = y(1,1) + y(2,2)'//new_line('a')// &
        '    end function grid_mat'//new_line('a')// &
        '    integer function merged_vec(z) result(r)'//new_line('a')// &
        '        integer, intent(in) :: z(3)'//new_line('a')// &
        '        r = z(1)'//new_line('a')// &
        '    end function merged_vec'//new_line('a')// &
        '    integer function merged_mat(z) result(r)'//new_line('a')// &
        '        integer, intent(in) :: z(2,2)'//new_line('a')// &
        '        r = z(1,1)'//new_line('a')// &
        '    end function merged_mat'//new_line('a')// &
        'end module fmod415_generic'

    print *, '=== separate-compilation generic interface tests ==='

    all_passed = .true.
    if (.not. test_use_associated_generic_resolves()) all_passed = .false.
    if (.not. test_rank_aware_specifics_resolve()) all_passed = .false.
    if (.not. test_no_matching_rank_is_diagnosed()) all_passed = .false.
    if (.not. test_rank_only_specifics_share_one_generic()) all_passed = .false.
    if (.not. test_assumed_shape_rank_specifics_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: separate-compilation generic interface'

contains

    logical function test_use_associated_generic_resolves() result(ok)
        ! A module exports a named generic interface over an integer-argument and
        ! a real-argument subroutine. A separately compiled program USEs only the
        ! generic name and calls it with each type; the call must resolve to the
        ! matching specific across the .fmod and link against the module object.
        character(len=*), parameter :: m_src = '/tmp/ffc_gen_m.f90'
        character(len=*), parameter :: main_src = '/tmp/ffc_gen_main.f90'
        character(len=*), parameter :: m_obj = '/tmp/ffc_gen_m.o'
        character(len=*), parameter :: main_exe = '/tmp/ffc_gen_main'
        character(len=*), parameter :: out_file = '/tmp/ffc_gen_out.txt'
        integer :: exit_stat, cmd_stat

        ok = .false.
        if (.not. write_file(m_src, &
            'module ffc_gen_mod'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  interface bump'//new_line('a')// &
            '    module procedure bump_i'//new_line('a')// &
            '    module procedure bump_r'//new_line('a')// &
            '  end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump_i(a)'//new_line('a')// &
            '    integer, intent(inout) :: a'//new_line('a')// &
            '    a = a + 1'//new_line('a')// &
            '  end subroutine bump_i'//new_line('a')// &
            '  subroutine bump_r(a)'//new_line('a')// &
            '    real, intent(inout) :: a'//new_line('a')// &
            '    a = a + 1'//new_line('a')// &
            '  end subroutine bump_r'//new_line('a')// &
            'end module ffc_gen_mod')) return
        if (.not. write_file(main_src, &
            'program main'//new_line('a')// &
            '  use ffc_gen_mod, only: bump'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  i = 5'//new_line('a')// &
            '  call bump(i)'//new_line('a')// &
            '  if (i /= 6) error stop'//new_line('a')// &
            '  r = 6.0'//new_line('a')// &
            '  call bump(r)'//new_line('a')// &
            '  if (r /= 7.0) error stop'//new_line('a')// &
            "  print *, 'OK'"//new_line('a')// &
            'end program main')) return

        call execute_command_line('rm -f '//m_obj//' /tmp/ffc_gen_mod.fmod '// &
            main_exe//' '//out_file)
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; '// &
            '"$exe" -c '//m_src//' -o '//m_obj//' || exit 91; '// &
            '"$exe" '//main_src//' '//m_obj//' -o '//main_exe//" || exit 92'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: generic separate-compile pipeline failed, code ', &
                exit_stat
            return
        end if
        call execute_command_line(main_exe//' > '//out_file//' 2>&1', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: linked generic program did not run cleanly, code ', &
                exit_stat
            return
        end if
        if (.not. file_contains(out_file, 'OK')) then
            print *, 'FAIL: generic calls did not resolve to the right specifics'
            return
        end if
        call execute_command_line('rm -f '//m_src//' '//main_src//' '//m_obj// &
            ' /tmp/ffc_gen_mod.fmod '//main_exe//' '//out_file)
        ok = .true.
    end function test_use_associated_generic_resolves

    logical function test_rank_aware_specifics_resolve() result(ok)
        ! Scalar, rank-1, and rank-2 specifics all resolve from the artefact,
        ! and the program's result matches same-unit compilation.
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod415_generic, only: total, grid_total'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '    integer :: m(2,2)'//new_line('a')// &
            '    v(1) = 1'//new_line('a')// &
            '    v(2) = 2'//new_line('a')// &
            '    v(3) = 3'//new_line('a')// &
            '    m(1,1) = 10'//new_line('a')// &
            '    m(2,1) = 0'//new_line('a')// &
            '    m(1,2) = 0'//new_line('a')// &
            '    m(2,2) = 20'//new_line('a')// &
            '    stop total(4) + total(v) + grid_total(1) + grid_total(m)'// &
            new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod415_resolve', dir)
        separate_status = run_separate_compilation(dir, rank_module_source, &
                                                   program_source)
        same_status = run_same_unit_compilation(dir, rank_module_source, &
                                                program_source)
        if (same_status /= 142) then
            print *, 'FAIL: same-unit generic resolution status ', same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= same_status) then
            print *, 'FAIL: separate generic resolution status ', &
                separate_status, ' differs from same-unit ', same_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_rank_aware_specifics_resolve

    logical function test_no_matching_rank_is_diagnosed() result(ok)
        ! A rank-2 actual passed to a generic whose only array specific is
        ! rank-1 matches no specific. The imported generic refuses it the same
        ! way the same-unit generic does, rather than silently binding the
        ! rank-1 specific and reading past the actual.
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod415_generic, only: total'//new_line('a')// &
            '    integer :: m(2,2)'//new_line('a')// &
            '    m(1,1) = 1'//new_line('a')// &
            '    m(2,1) = 1'//new_line('a')// &
            '    m(1,2) = 1'//new_line('a')// &
            '    m(2,2) = 1'//new_line('a')// &
            '    stop total(m)'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod415_nomatch', dir)
        separate_status = run_separate_compilation(dir, rank_module_source, &
                                                   program_source)
        same_status = run_same_unit_compilation(dir, rank_module_source, &
                                                program_source)
        if (same_status /= 92) then
            print *, 'FAIL: same-unit rank mismatch was accepted, status ', &
                same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= 92) then
            print *, 'FAIL: imported rank mismatch was accepted, status ', &
                separate_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_no_matching_rank_is_diagnosed

    logical function test_rank_only_specifics_share_one_generic() result(ok)
        ! Two specifics of the same element kind that differ only in rank are
        ! distinguishable (F2018 C1514), so one generic may hold both. The
        ! source is valid - gfortran -fsyntax-only accepts it - and must
        ! compile both same-unit and through the .fmod (#595).
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod415_generic, only: merged'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '    integer :: m(2,2)'//new_line('a')// &
            '    v(1) = 1'//new_line('a')// &
            '    v(2) = 2'//new_line('a')// &
            '    v(3) = 3'//new_line('a')// &
            '    m(1,1) = 10'//new_line('a')// &
            '    m(2,1) = 0'//new_line('a')// &
            '    m(1,2) = 0'//new_line('a')// &
            '    m(2,2) = 20'//new_line('a')// &
            '    stop merged(v) + merged(m)'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('ffc595_rank_only', dir)
        same_status = run_same_unit_compilation(dir, rank_module_source, &
                                                program_source)
        separate_status = run_separate_compilation(dir, rank_module_source, &
                                                   program_source)
        if (same_status /= 111) then
            print *, 'FAIL: same-unit rank-only generic status ', same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= 111) then
            print *, 'FAIL: use-associated rank-only generic status ', &
                separate_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_rank_only_specifics_share_one_generic

    logical function test_assumed_shape_rank_specifics_accepted() result(ok)
        ! Assumed-shape specifics that differ only in rank are distinguishable
        ! too, and their dummies carry no shape on the parameter node - the
        ! rank has to come from the dummy's own declaration. gfortran
        ! -fsyntax-only accepts this source (#595).
        character(len=:), allocatable :: dir
        integer :: same_status
        character(len=*), parameter :: module_source = &
            'module fmod595_assumed'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    interface pick'//new_line('a')// &
            '        module procedure pick_vec'//new_line('a')// &
            '        module procedure pick_mat'//new_line('a')// &
            '    end interface pick'//new_line('a')// &
            'contains'//new_line('a')// &
            '    integer function pick_vec(z) result(r)'//new_line('a')// &
            '        integer, intent(in) :: z(:)'//new_line('a')// &
            '        r = z(1)'//new_line('a')// &
            '    end function pick_vec'//new_line('a')// &
            '    integer function pick_mat(z) result(r)'//new_line('a')// &
            '        integer, intent(in) :: z(:,:)'//new_line('a')// &
            '        r = z(1,1)'//new_line('a')// &
            '    end function pick_mat'//new_line('a')// &
            'end module fmod595_assumed'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod595_assumed, only: pick'//new_line('a')// &
            '    integer :: v(3)'//new_line('a')// &
            '    integer :: m(2,2)'//new_line('a')// &
            '    v = 1'//new_line('a')// &
            '    m = 2'//new_line('a')// &
            '    stop pick(v) + pick(m)'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('ffc595_assumed', dir)
        same_status = run_same_unit_compilation(dir, module_source, &
                                                program_source)
        if (same_status /= 103) then
            print *, 'FAIL: assumed-shape rank specifics status ', same_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_assumed_shape_rank_specifics_accepted

    integer function run_separate_compilation(dir, mod_source, prog_source) &
            result(status)
        ! Compile the module with -c in one ffc invocation, then the program in
        ! a second, independent invocation that can only learn the generic's
        ! specifics from the .fmod artefact. Returns 90 when no ffc binary was
        ! found, 91/92 when a compilation failed, and 100 + exit status when the
        ! program ran.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: mod_source
        character(len=*), intent(in) :: prog_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/m.f90', mod_source)) return
        if (.not. write_file(dir//'/p.f90', prog_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c m.f90 -o m.o >>log 2>&1 || exit 91; '// &
            '"$exe" p.f90 m.o -o p >>log 2>&1 || exit 92; '// &
            "./p; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_separate_compilation

    integer function run_same_unit_compilation(dir, mod_source, prog_source) &
            result(status)
        ! Compile the same module and program as one unit, so the separate
        ! result can be held against it.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: mod_source
        character(len=*), intent(in) :: prog_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/same.f90', mod_source//new_line('a')// &
                             prog_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" same.f90 -o same >>log 2>&1 || exit 92; '// &
            "./same; exit $((100 + $?))'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_same_unit_compilation

    subroutine make_scratch_dir(tag, dir)
        ! A scratch directory of this run's own, so concurrent builds of other
        ! worktrees never share it (ffc #547).
        character(len=*), intent(in) :: tag
        character(len=:), allocatable, intent(out) :: dir
        character(len=32) :: stamp
        integer :: values(8)

        call date_and_time(values=values)
        write (stamp, '(I0,A,I0)') values(6)*60000 + values(7)*1000 + &
            values(8), '_', values(5)
        dir = '/tmp/ffc_'//tag//'_'//trim(stamp)
        call execute_command_line('rm -rf '//dir//'; mkdir -p '//dir)
    end subroutine make_scratch_dir

    subroutine remove_scratch_dir(dir)
        character(len=*), intent(in) :: dir

        call execute_command_line('rm -rf '//dir)
    end subroutine remove_scratch_dir

    subroutine show_log(dir)
        character(len=*), intent(in) :: dir

        call execute_command_line('cat '//dir//'/log 2>/dev/null')
    end subroutine show_log

    logical function file_contains(path, fragment) result(found)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: fragment
        integer :: unit, io_stat
        character(len=512) :: line

        found = .false.
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            if (index(line, fragment) > 0) then
                found = .true.
                exit
            end if
        end do
        close (unit)
    end function file_contains

    logical function write_file(path, contents) result(ok)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: contents
        integer :: unit, io_stat

        ok = .false.
        open (newunit=unit, file=path, status='replace', action='write', &
            iostat=io_stat)
        if (io_stat /= 0) then
            print *, 'FAIL: could not write ', path
            return
        end if
        write (unit, '(A)', iostat=io_stat) contents
        close (unit)
        ok = io_stat == 0
    end function write_file

end program test_session_separate_generic_compiler
