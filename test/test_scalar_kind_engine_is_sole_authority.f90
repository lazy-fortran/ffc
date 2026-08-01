program test_scalar_kind_engine_is_sole_authority
    !! Static check for #447: the ad hoc scalar-kind predicates the engine
    !! replaced must have no definitions and no callers left in src/. An engine
    !! that leaves the per-case code in place has replaced nothing, so this
    !! guards the deletion rather than the addition.
    implicit none
    logical :: ok

    print *, '=== scalar kind engine sole-authority check ==='

    ok = .true.
    if (.not. absent('is_f32_expression')) ok = .false.
    if (.not. absent('is_f64_expression')) ok = .false.
    if (.not. absent('is_f64_intrinsic_expression')) ok = .false.
    if (.not. ok) stop 1

    print *, 'PASS: no retired scalar-kind predicate remains in src/'

contains

    logical function absent(name) result(res)
        !! True when `name` appears nowhere in src/ outside the engine's own
        !! explanatory header comment.
        character(len=*), intent(in) :: name
        integer :: stat, unit, count
        character(len=*), parameter :: report = '/tmp/ffc_scalar_kind_grep.txt'

        call execute_command_line( &
            "grep -rn '"//name//"' src/ "// &
            "| grep -v session_program_lowering_scalar_kind.inc "// &
            "| wc -l > "//report, exitstat=stat)
        if (stat /= 0) then
            print *, 'FAIL: could not scan src/ for ', name
            res = .false.
            return
        end if
        open (newunit=unit, file=report, status='old', action='read')
        read (unit, *) count
        close (unit, status='delete')
        res = count == 0
        if (.not. res) then
            print *, 'FAIL: retired predicate still referenced: ', name, &
                ' occurrences:', count
        end if
    end function absent

end program test_scalar_kind_engine_is_sole_authority
