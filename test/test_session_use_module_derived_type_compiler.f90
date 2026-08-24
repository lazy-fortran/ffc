program test_session_use_module_derived_type_compiler
    ! A derived type defined in a separately compiled module must reach a using
    ! unit with the same layout the defining unit compiled: component order and
    ! slot offsets, nested type identity, fixed array shapes, and character
    ! lengths. The using unit sees only the .fmod artefact, so every one of
    ! those facts has to travel in it (#414).
    use ffc_test_support, only: expect_exit_status
    use ffc_module_artefact, only: module_info_t, write_fmod
    implicit none

    logical :: all_passed

    print *, '=== direct session use module derived type compiler test ==='

    all_passed = .true.
    if (.not. expect_exit_status( &
        'module point_mod'//new_line('a')// &
        '  type :: point_t'//new_line('a')// &
        '    integer :: x, y'//new_line('a')// &
        '  end type'//new_line('a')// &
        'end module point_mod'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use point_mod'//new_line('a')// &
        '  type(point_t) :: p'//new_line('a')// &
        '  p%x = 7'//new_line('a')// &
        '  stop p%x'//new_line('a')// &
        'end program main', 7, &
        '/tmp/ffc_session_use_module_derived_type_test')) all_passed = .false.

    if (.not. test_nested_derived_layout_round_trip()) all_passed = .false.
    if (.not. test_character_component_length_round_trip()) all_passed = .false.
    if (.not. test_unknown_nested_type_is_rejected()) all_passed = .false.
    if (.not. test_overlapping_offsets_are_rejected()) all_passed = .false.
    if (.not. test_inconsistent_slot_counts_are_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: USE module derived type lowers through direct LIRIC session'

contains

    logical function test_nested_derived_layout_round_trip() result(ok)
        ! A nested derived component, a fixed-size integer array component, and
        ! a character component all keep their layout across separate
        ! compilation; the result matches same-unit compilation of the same
        ! program.
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: module_source = &
            'module fmod414_layout'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: inner_t'//new_line('a')// &
            '        integer :: a'//new_line('a')// &
            '        integer :: b'//new_line('a')// &
            '    end type inner_t'//new_line('a')// &
            '    type :: outer_t'//new_line('a')// &
            '        type(inner_t) :: core'//new_line('a')// &
            '        integer :: tags(3)'//new_line('a')// &
            '        character(len=4) :: label'//new_line('a')// &
            '    end type outer_t'//new_line('a')// &
            'end module fmod414_layout'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod414_layout, only: outer_t'//new_line('a')// &
            '    type(outer_t) :: v'//new_line('a')// &
            '    v%core%a = 2'//new_line('a')// &
            '    v%core%b = 3'//new_line('a')// &
            '    v%tags(1) = 4'//new_line('a')// &
            '    v%tags(2) = 5'//new_line('a')// &
            '    v%tags(3) = 6'//new_line('a')// &
            "    v%label = 'abcd'"//new_line('a')// &
            '    stop v%core%a + v%core%b + v%tags(3) + len_trim(v%label)'// &
            new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod414_layout', dir)
        separate_status = run_separate_compilation(dir, module_source, &
            program_source)
        same_status = run_same_unit_compilation(dir, module_source, &
            program_source)
        if (same_status /= 115) then
            print *, 'FAIL: same-unit derived layout status ', same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= same_status) then
            print *, 'FAIL: separate compilation status ', separate_status, &
                ' differs from same-unit ', same_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_nested_derived_layout_round_trip

    logical function test_character_component_length_round_trip() result(ok)
        ! The declared length of a character component decides the component's
        ! slot count, so a using unit that misread it would place every later
        ! component wrong.
        character(len=:), allocatable :: dir
        integer :: separate_status, same_status
        character(len=*), parameter :: module_source = &
            'module fmod414_char'//new_line('a')// &
            '    implicit none'//new_line('a')// &
            '    type :: tagged_t'//new_line('a')// &
            '        character(len=9) :: name'//new_line('a')// &
            '        integer :: count'//new_line('a')// &
            '    end type tagged_t'//new_line('a')// &
            'end module fmod414_char'
        character(len=*), parameter :: program_source = &
            'program main'//new_line('a')// &
            '    use fmod414_char, only: tagged_t'//new_line('a')// &
            '    type(tagged_t) :: t'//new_line('a')// &
            "    t%name = 'abcdefghi'"//new_line('a')// &
            '    t%count = 7'//new_line('a')// &
            '    stop len_trim(t%name) + t%count'//new_line('a')// &
            'end program main'

        ok = .false.
        call make_scratch_dir('fmod414_char', dir)
        separate_status = run_separate_compilation(dir, module_source, &
            program_source)
        same_status = run_same_unit_compilation(dir, module_source, &
            program_source)
        if (same_status /= 116) then
            print *, 'FAIL: same-unit character component status ', same_status
            call show_log(dir)
            return
        end if
        if (separate_status /= same_status) then
            print *, 'FAIL: separate character component status ', &
                separate_status, ' differs from same-unit ', same_status
            call show_log(dir)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_character_component_length_round_trip

    logical function test_unknown_nested_type_is_rejected() result(ok)
        ! A component whose nested type the artefact never defines describes no
        ! layout at all; reading it must fail rather than invent one.
        character(len=:), allocatable :: dir, error_msg
        type(module_info_t) :: info

        ok = .false.
        call make_scratch_dir('fmod414_unknown', dir)
        info%name = 'fmod414_unknown'
        allocate (info%parameters(0))
        allocate (info%derived_types(1))
        info%derived_types(1)%name = 'holder_t'
        info%derived_types(1)%canonical_identity = &
            'fmod414_unknown::holder_t'
        allocate (info%derived_types(1)%components(1))
        info%derived_types(1)%components(1)%name = 'core'
        info%derived_types(1)%components(1)%kind = 'derived'
        info%derived_types(1)%components(1)%type_name = 'never_defined_t'
        info%derived_types(1)%components(1)%type_identity = &
            'fmod414_unknown::never_defined_t'
        info%derived_types(1)%components(1)%slot_count = 1
        info%derived_types(1)%components(1)%slot_offset = 0
        call write_fmod(dir//'/fmod414_unknown.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        call compile_using_program(dir, 'fmod414_unknown', 'holder_t', &
            error_msg)
        if (index(error_msg, 'never_defined_t') == 0 .and. &
            index(error_msg, 'unknown component identity') == 0) then
            print *, 'FAIL: unknown nested type was accepted: ', trim(error_msg)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_unknown_nested_type_is_rejected

    logical function test_overlapping_offsets_are_rejected() result(ok)
        ! Component offsets are recorded next to the slot counts that produce
        ! them. An artefact whose offsets do not follow from its own slot counts
        ! is corrupt: two components would share storage.
        character(len=:), allocatable :: dir, error_msg
        type(module_info_t) :: info

        ok = .false.
        call make_scratch_dir('fmod414_overlap', dir)
        info%name = 'fmod414_overlap'
        allocate (info%parameters(0))
        allocate (info%derived_types(1))
        info%derived_types(1)%name = 'pair_t'
        info%derived_types(1)%canonical_identity = &
            'fmod414_overlap::pair_t'
        allocate (info%derived_types(1)%components(2))
        info%derived_types(1)%components(1)%name = 'a'
        info%derived_types(1)%components(1)%kind = 'integer'
        info%derived_types(1)%components(1)%slot_count = 1
        info%derived_types(1)%components(1)%slot_offset = 0
        info%derived_types(1)%components(2)%name = 'b'
        info%derived_types(1)%components(2)%kind = 'integer'
        info%derived_types(1)%components(2)%slot_count = 1
        ! b starts where a does: the two components would alias.
        info%derived_types(1)%components(2)%slot_offset = 0
        call write_fmod(dir//'/fmod414_overlap.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        call compile_using_program(dir, 'fmod414_overlap', 'pair_t', error_msg)
        if (index(error_msg, 'offset') == 0) then
            print *, 'FAIL: overlapping offsets were accepted: ', trim(error_msg)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_overlapping_offsets_are_rejected

    logical function test_inconsistent_slot_counts_are_rejected() result(ok)
        ! slot_count is derived from elem_count and slot_width. A corrupt
        ! artefact must not be allowed to invent a larger or smaller storage
        ! span that shifts every following component.
        character(len=:), allocatable :: dir, error_msg
        type(module_info_t) :: info

        ok = .false.
        call make_scratch_dir('fmod414_slot_count', dir)
        info%name = 'fmod414_slot_count'
        allocate (info%parameters(0))
        allocate (info%derived_types(1))
        info%derived_types(1)%name = 'value_t'
        info%derived_types(1)%canonical_identity = &
            'fmod414_slot_count::value_t'
        allocate (info%derived_types(1)%components(1))
        info%derived_types(1)%components(1)%name = 'value'
        info%derived_types(1)%components(1)%kind = 'integer'
        info%derived_types(1)%components(1)%elem_count = 1
        info%derived_types(1)%components(1)%slot_width = 1
        info%derived_types(1)%components(1)%slot_count = 2
        info%derived_types(1)%components(1)%slot_offset = 0
        call write_fmod(dir//'/fmod414_slot_count.fmod', info, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: write_fmod: ', trim(error_msg)
            return
        end if
        call compile_using_program(dir, 'fmod414_slot_count', 'value_t', &
            error_msg)
        if (index(error_msg, 'inconsistent slot counts') == 0) then
            print *, 'FAIL: inconsistent slot counts were accepted: ', &
                trim(error_msg)
            return
        end if
        call remove_scratch_dir(dir)
        ok = .true.
    end function test_inconsistent_slot_counts_are_rejected

    subroutine compile_using_program(dir, module_name, type_name, error_msg)
        ! Compile a program that USEs type_name from module_name, seeing only
        ! the .fmod in dir. error_msg carries the compiler's diagnostic.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: module_name
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: exit_stat, cmd_stat

        error_msg = ''
        if (.not. write_file(dir//'/p.f90', &
            'program main'//new_line('a')// &
            '    use '//module_name//', only: '//type_name//new_line('a')// &
            '    type('//type_name//') :: v'//new_line('a')// &
            '    stop 0'//new_line('a')// &
            'end program main')) return
        call execute_command_line( &
            "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc 2>/dev/null | "// &
            'head -n 1); test -n "$exe" || exit 90; exe=$PWD/$exe; '// &
            'cd '//dir//' || exit 90; '// &
            '"$exe" p.f90 -o p >log 2>&1'//"'", &
            exitstat=exit_stat, cmdstat=cmd_stat)
        call read_text_file(dir//'/log', error_msg)
        if (exit_stat == 0) error_msg = ''
    end subroutine compile_using_program

    integer function run_separate_compilation(dir, module_source, &
            program_source) result(status)
        ! Compile the module with -c in one ffc invocation, then the program in
        ! a second, independent invocation that can only learn the module's
        ! types from the .fmod artefact. Returns 90 when no ffc binary was
        ! found, 91/92 when a compilation failed, and 100 + exit status when the
        ! program ran.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: module_source
        character(len=*), intent(in) :: program_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/m.f90', module_source)) return
        if (.not. write_file(dir//'/p.f90', program_source)) return
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

    integer function run_same_unit_compilation(dir, module_source, &
            program_source) result(status)
        ! Compile the same module and program together in one unit, so the
        ! separate-compilation result can be held against it.
        character(len=*), intent(in) :: dir
        character(len=*), intent(in) :: module_source
        character(len=*), intent(in) :: program_source
        integer :: exit_stat, cmd_stat

        status = 90
        if (.not. write_file(dir//'/same.f90', module_source//new_line('a')// &
            program_source)) return
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

    subroutine read_text_file(path, text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        character(len=1024) :: line
        integer :: unit, io_stat

        text = ''
        open (newunit=unit, file=path, status='old', action='read', &
            iostat=io_stat)
        if (io_stat /= 0) return
        do
            read (unit, '(A)', iostat=io_stat) line
            if (io_stat /= 0) exit
            text = text//trim(line)//' '
        end do
        close (unit)
    end subroutine read_text_file

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

end program test_session_use_module_derived_type_compiler
