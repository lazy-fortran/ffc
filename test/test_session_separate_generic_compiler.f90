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
    if (.not. test_transitive_typebound_generic_reexport()) all_passed = .false.
    if (.not. test_unrelated_same_named_types_remain_distinct()) &
        all_passed = .false.
    if (.not. test_long_vtable_identity()) all_passed = .false.

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

    logical function test_transitive_typebound_generic_reexport() result(ok)
        ! A type-bound generic survives two module boundaries: the defining
        ! module is compiled first, a public bridge module re-exports the
        ! derived type, and the user sees the bridge .fmod plus the original
        ! defining .fmod. The two generic specifics must still resolve to the
        ! defining module's procedures, and the re-exported child must retain
        ! compatibility with its imported parent (#447, modules33/modules34).
        character(len=:), allocatable :: dir
        integer :: ffc_status, gfortran_status
        character(len=*), parameter :: base_source = &
            'module fmod447_base'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: parent_t'//new_line('a')// &
            '        integer :: value'//new_line('a')// &
            '    contains'//new_line('a')// &
            '        procedure :: observe'//new_line('a')// &
            '    end type parent_t'//new_line('a')// &
            '    type, extends(parent_t) :: box_t'//new_line('a')// &
            '    contains'//new_line('a')// &
            '        generic :: set => set_int, set_real'//new_line('a')// &
            '        procedure :: set_int'//new_line('a')// &
            '        procedure :: set_real'//new_line('a')// &
            '    end type box_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '    subroutine set_int(self, value)'//new_line('a')// &
            '        class(box_t), intent(inout) :: self'//new_line('a')// &
            '        integer, intent(in) :: value'//new_line('a')// &
            '        self%value = self%value + value'//new_line('a')// &
            '    end subroutine set_int'//new_line('a')// &
            '    subroutine set_real(self, value)'//new_line('a')// &
            '        class(box_t), intent(inout) :: self'//new_line('a')// &
            '        real, intent(in) :: value'//new_line('a')// &
            '        self%value = self%value + int(value)'//new_line('a')// &
            '    end subroutine set_real'//new_line('a')// &
            '    subroutine observe(self, result)'//new_line('a')// &
            '        class(parent_t), intent(in) :: self'//new_line('a')// &
            '        integer, intent(out) :: result'//new_line('a')// &
            '        result = self%value'//new_line('a')// &
            '    end subroutine observe'//new_line('a')// &
            'end module fmod447_base'
        character(len=*), parameter :: bridge_source = &
            'module fmod447_bridge'//new_line('a')// &
            '    use fmod447_base, only: alias_t => box_t, parent_t'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    public :: alias_t, parent_t'//new_line('a')// &
            'end module fmod447_bridge'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod447_bridge, only: alias_t'//new_line('a')// &
            '    use fmod447_base, only: box_t'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    integer :: observed'//new_line('a')// &
            '    type(alias_t) :: box'//new_line('a')// &
            '    type(box_t) :: canonical_box'//new_line('a')// &
            '    box%value = 0'//new_line('a')// &
            '    call box%set(4)'//new_line('a')// &
            '    call box%set(2.5)'//new_line('a')// &
            '    call box%observe(observed)'//new_line('a')// &
            '    if (box%value /= 6 .or. observed /= 6) error stop 1'// &
            new_line('a')// &
            '    canonical_box%value = 0'//new_line('a')// &
            '    call canonical_box%set(6)'//new_line('a')// &
            '    call canonical_box%observe(observed)'//new_line('a')// &
            '    if (observed /= 6) error stop 2'//new_line('a')// &
            '    print *, box%value'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod447_typebound', dir)
        ffc_status = run_transitive_typebound_compilation(dir, base_source, &
            bridge_source, &
            program_source)
        gfortran_status = run_gfortran_transitive_compilation(dir, base_source, &
            bridge_source, &
            program_source)
        if (gfortran_status /= 0) then
            print *, 'FAIL: gfortran oracle failed for transitive type-bound generic'
            call show_log(dir)
            return
        end if
        if (ffc_status /= gfortran_status) then
            print *, 'FAIL: transitive type-bound status ', ffc_status, &
                ' differs from gfortran ', gfortran_status
            call show_log(dir)
            return
        end if
        if (.not. files_equal(dir//'/ffc.out', dir//'/gfortran.out')) then
            print *, 'FAIL: transitive type-bound output differs from gfortran'
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_transitive_typebound_generic_reexport

    logical function test_unrelated_same_named_types_remain_distinct() result(ok)
        ! Two defining modules may legitimately export the same type spelling.
        ! Their local renames must remain distinct in a consumer; otherwise an
        ! imported layout can silently be applied to the wrong object.
        character(len=:), allocatable :: dir
        integer :: ffc_status, gfortran_status
        character(len=*), parameter :: first_source = &
            'module fmod737_first'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: value_t'//new_line('a')// &
            '        integer :: first_value'//new_line('a')// &
            '    end type value_t'//new_line('a')// &
            'end module fmod737_first'
        character(len=*), parameter :: second_source = &
            'module fmod737_second'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: value_t'//new_line('a')// &
            '        integer :: second_value'//new_line('a')// &
            '    end type value_t'//new_line('a')// &
            'end module fmod737_second'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod737_first, only: first_t => value_t'//new_line('a')// &
            '    use fmod737_second, only: second_t => value_t'//new_line('a')// &
            '    type(first_t) :: first'//new_line('a')// &
            '    type(second_t) :: second'//new_line('a')// &
            '    first%first_value = 4'//new_line('a')// &
            '    second%second_value = 7'//new_line('a')// &
            '    print *, first%first_value + second%second_value'// &
            new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod737_same_named', dir)
        ffc_status = run_transitive_typebound_compilation(dir, first_source, &
            second_source, program_source)
        gfortran_status = run_gfortran_transitive_compilation(dir, first_source, &
            second_source, program_source)
        if (gfortran_status /= 0) then
            print *, 'FAIL: gfortran oracle failed for same-named types'
            call show_log(dir)
            return
        end if
        if (ffc_status /= gfortran_status) then
            print *, 'FAIL: same-named type status ', ffc_status, &
                ' differs from gfortran ', gfortran_status
            call show_log(dir)
            return
        end if
        if (.not. files_equal(dir//'/ffc.out', dir//'/gfortran.out')) then
            print *, 'FAIL: same-named type output differs from gfortran'
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_unrelated_same_named_types_remain_distinct

    logical function test_long_vtable_identity() result(ok)
        ! The encoded canonical identity is longer than the historical 128-byte
        ! scratch buffers when valid module/type names contain many underscores.
        ! Compile and run the aliased type through two module boundaries so the
        ! long vtable symbol is exercised by both FFC and gfortran.
        character(len=:), allocatable :: dir
        integer :: ffc_status, gfortran_status
        character(len=*), parameter :: module_source = &
            'module m_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'// &
            new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: t_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'// &
            new_line('a')// &
            '        integer :: value'//new_line('a')// &
            '    contains'//new_line('a')// &
            '        procedure :: bump'//new_line('a')// &
            '    end type t_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'// &
            new_line('a')// &
            'contains'//new_line('a')// &
            '    subroutine bump(self, amount)'//new_line('a')// &
            '        class(t_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) :: self'// &
            new_line('a')// &
            '        integer, intent(in) :: amount'//new_line('a')// &
            '        self%value = self%value + amount'//new_line('a')// &
            '    end subroutine bump'//new_line('a')// &
            'end module m_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        character(len=*), parameter :: bridge_source = &
            'module m_long_identity_bridge'//new_line('a')// &
            '    use m_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, only: alias_t => &'// &
            new_line('a')// &
            '        t_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    public :: alias_t'//new_line('a')// &
            'end module m_long_identity_bridge'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use m_long_identity_bridge, only: alias_t'//new_line('a')// &
            '    type(alias_t) :: value'//new_line('a')// &
            '    value%value = 0'//new_line('a')// &
            '    call value%bump(7)'//new_line('a')// &
            '    print *, value%value'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod737_long_identity', dir)
        ffc_status = run_transitive_typebound_compilation(dir, module_source, &
            bridge_source, program_source)
        gfortran_status = run_gfortran_transitive_compilation(dir, module_source, &
            bridge_source, program_source)
        if (gfortran_status /= 0) then
            print *, 'FAIL: gfortran oracle failed for long vtable identity'
            call show_log(dir)
            return
        end if
        if (ffc_status /= gfortran_status) then
            print *, 'FAIL: long vtable identity status ', ffc_status, &
                ' differs from gfortran ', gfortran_status
            call show_log(dir)
            return
        end if
        if (.not. files_equal(dir//'/ffc.out', dir//'/gfortran.out')) then
            print *, 'FAIL: long vtable identity output differs from gfortran'
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_long_vtable_identity

    integer function run_transitive_typebound_compilation(dir, base_source, &
            bridge_source, &
            program_source) result(status)
        character(len=*), intent(in) :: dir, base_source, bridge_source
        character(len=*), intent(in) :: program_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/base.f90', base_source)) return
        if (.not. write_file(dir//'/bridge.f90', bridge_source)) return
        if (.not. write_file(dir//'/program.f90', program_source)) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" -c base.f90 -o base.o >log 2>&1 || exit 91; '// &
            '"$exe" -c bridge.f90 -o bridge.o >>log 2>&1 || exit 92; '// &
            '"$exe" program.f90 bridge.o base.o -o ffc >log 2>&1 || exit 93; '// &
            './ffc >ffc.out 2>&1; exit $?'//"'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_transitive_typebound_compilation

    integer function run_gfortran_transitive_compilation(dir, base_source, &
            bridge_source, &
            program_source) result(status)
        character(len=*), intent(in) :: dir, base_source, bridge_source
        character(len=*), intent(in) :: program_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/gbase.f90', base_source)) return
        if (.not. write_file(dir//'/gbridge.f90', bridge_source)) return
        if (.not. write_file(dir//'/gprogram.f90', program_source)) return
        call execute_command_line( &
            'sh -c ''cd '//dir//' || exit 90; '// &
            'gfortran -std=f2018 -w gbase.f90 gbridge.f90 gprogram.f90 '// &
            '-o gfortran >gfortran.log 2>&1 || exit 91; '// &
            './gfortran >gfortran.out 2>&1; exit $?''', &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        status = exit_stat
    end function run_gfortran_transitive_compilation

    logical function files_equal(left, right) result(equal)
        character(len=*), intent(in) :: left, right
        integer :: exit_stat, cmd_stat

        call execute_command_line('diff -u '//left//' '//right, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        equal = cmd_stat == 0 .and. exit_stat == 0
    end function files_equal

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
