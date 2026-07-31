module conformance_temp_dir
    ! Per-run scratch directories for the conformance tests.
    !
    ! Conformance tests used to write fixtures and reports to fixed /tmp paths.
    ! Two concurrent runs (different worktrees, or `fo test` running tests in
    ! parallel) then overwrote each other's fixtures, which produces spurious
    ! failures and, worse, lets one run read another run's stale artifacts as
    ! if they were its own.
    !
    ! `mktemp -d` cannot be used directly from Fortran because capturing its
    ! stdout would itself need a uniquely named file. Instead the directory
    ! name is drawn from the clock plus a counter and created with plain
    ! `mkdir`, which fails when the directory already exists. That failure is
    ! the atomic uniqueness guarantee: a name can only be claimed once.
    implicit none
    private
    public :: make_temp_root, remove_temp_root

contains

    function make_temp_root(prefix) result(root)
        character(len=*), intent(in) :: prefix
        character(len=:), allocatable :: root
        character(len=:), allocatable :: base
        character(len=64) :: suffix
        integer :: attempt, exit_stat, clock_count, clock_rate

        base = temp_base()
        do attempt = 1, 64
            call system_clock(clock_count, clock_rate)
            write (suffix, '(I0,A,I0)') abs(clock_count), '_', attempt
            root = base//'/ffc_'//prefix//'_'//trim(suffix)
            ! Plain mkdir (no -p): fails if the name is already taken.
            call execute_command_line('mkdir '//quote(root)//' 2>/dev/null', &
                exitstat=exit_stat)
            if (exit_stat == 0) return
        end do
        error stop 'make_temp_root: could not create a unique temporary directory'
    end function make_temp_root

    subroutine remove_temp_root(root)
        character(len=*), intent(in) :: root

        if (len_trim(root) == 0) return
        call execute_command_line('rm -rf '//quote(trim(root)))
    end subroutine remove_temp_root

    function temp_base() result(base)
        character(len=:), allocatable :: base
        character(len=4096) :: buffer
        integer :: length, status

        call get_environment_variable('TMPDIR', buffer, length, status)
        if (status == 0 .and. length > 0) then
            base = buffer(1:length)
            do while (len(base) > 1)
                if (base(len(base):len(base)) /= '/') exit
                base = base(1:len(base) - 1)
            end do
        else
            base = '/tmp'
        end if
    end function temp_base

    function quote(text) result(quoted)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: quoted

        quoted = "'"//text//"'"
    end function quote

end module conformance_temp_dir
