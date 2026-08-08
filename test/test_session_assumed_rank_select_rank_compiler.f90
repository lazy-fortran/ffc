program test_session_assumed_rank_select_rank_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  real :: a(3)'//new_line('a')// &
        '  a = [1.0, 2.0, 3.0]'//new_line('a')// &
        '  call touch(a)'//new_line('a')// &
        '  print *, nint(sum(a))'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine touch(x)'//new_line('a')// &
        '    real :: x(..)'//new_line('a')// &
        '    select rank (x)'//new_line('a')// &
        '    rank (1)'//new_line('a')// &
        '      x(2) = x(2) + 4.0'//new_line('a')// &
        '    end select'//new_line('a')// &
        '  end subroutine touch'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: expected = '          10'//new_line('a')

    print *, '=== direct session assumed-rank SELECT RANK test ==='
    if (.not. expect_output(source, expected, &
            '/tmp/ffc_session_assumed_rank_select_rank')) then
        print *, 'FAIL: ffc behavioral case'
        stop 1
    end if
    if (.not. test_gfortran_oracle()) then
        print *, 'FAIL: gfortran behavioral oracle'
        stop 1
    end if
    if (.not. test_refusals()) then
        print *, 'FAIL: refusal cases'
        stop 1
    end if
    print *, 'PASS: rank-1 REAL assumed-rank descriptor boundary lowers correctly'

contains

    logical function test_gfortran_oracle()
        character(len=*), parameter :: src_path = &
            '/tmp/ffc_assumed_rank_select_rank_oracle.f90'
        character(len=*), parameter :: exe_path = &
            '/tmp/ffc_assumed_rank_select_rank_oracle'
        character(len=*), parameter :: out_path = &
            '/tmp/ffc_assumed_rank_select_rank_oracle.out'
        character(len=:), allocatable :: actual
        integer :: unit, cmd_stat, exit_stat

        test_gfortran_oracle = .false.
        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(a)') source
        close (unit)
        call execute_command_line('gfortran -std=f2018 '//src_path//' -o '//exe_path, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) return
        call execute_command_line(exe_path//' > '//out_path, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) return
        call read_text(out_path, actual)
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        if (.not. allocated(actual)) return
        test_gfortran_oracle = index(actual, '10') == 1
    end function test_gfortran_oracle

    logical function test_refusals()
        character(len=*), parameter :: rank_default = &
            'program main'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (1)'//new_line('a')// &
            '    rank default'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: rank_star = &
            'program main'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (*)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: rank_two = &
            'program main'//new_line('a')// &
            '  real :: a(2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (1)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: scalar_actual = &
            'program main'//new_line('a')// &
            '  real :: a'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (1)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_refusals = .true.
        if (.not. expect_error_contains(rank_default, 'RANK DEFAULT is refused', &
                '/tmp/ffc_assumed_rank_rank_default_reject')) test_refusals = .false.
        if (.not. expect_error_contains(rank_star, 'RANK (*) is refused', &
                '/tmp/ffc_assumed_rank_rank_star_reject')) test_refusals = .false.
        if (.not. expect_error_contains(rank_two, 'only a rank-1 whole REAL array', &
                '/tmp/ffc_assumed_rank_rank_two_reject')) test_refusals = .false.
        if (.not. expect_error_contains(scalar_actual, 'only a whole array', &
                '/tmp/ffc_assumed_rank_scalar_reject')) test_refusals = .false.
    end function test_refusals

    subroutine read_text(path, text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        character(len=256) :: line
        integer :: unit, stat

        allocate (character(len=0) :: text)
        open (newunit=unit, file=path, status='old', action='read', iostat=stat)
        if (stat /= 0) then
            deallocate (text)
            return
        end if
        do
            read (unit, '(a)', iostat=stat) line
            if (stat /= 0) exit
            text = text//adjustl(trim(line))//new_line('a')
        end do
        close (unit)
    end subroutine read_text

end program test_session_assumed_rank_select_rank_compiler
