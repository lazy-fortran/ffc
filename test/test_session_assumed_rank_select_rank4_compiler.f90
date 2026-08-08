program test_session_assumed_rank_select_rank4_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    character(len=*), parameter :: source = &
        'program main'//new_line('a')// &
        '  real :: a(2,3,2,2)'//new_line('a')// &
        '  a = 0.0'//new_line('a')// &
        '  a(1,1,1,1) = 1.0'//new_line('a')// &
        '  a(2,1,1,1) = 2.0'//new_line('a')// &
        '  a(1,2,1,1) = 3.0'//new_line('a')// &
        '  a(2,3,2,2) = 4.0'//new_line('a')// &
        '  call touch(a)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  subroutine touch(x)'//new_line('a')// &
        '    real :: x(..)'//new_line('a')// &
        '    select rank (x)'//new_line('a')// &
        '    rank (4)'//new_line('a')// &
        '      x(2,1,1,1) = x(2,1,1,1) + 10.0'//new_line('a')// &
        '      x(1,2,1,1) = x(1,2,1,1) + 20.0'//new_line('a')// &
        '      x(2,3,2,2) = x(2,3,2,2) + 30.0'//new_line('a')// &
        '      print *, nint(x(1,1,1,1)), nint(x(2,1,1,1)), '// &
        'nint(x(1,2,1,1)), nint(x(2,3,2,2))'//new_line('a')// &
        '    end select'//new_line('a')// &
        '  end subroutine touch'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: expected = &
        '           1          12          23          34'//new_line('a')

    print *, '=== direct session assumed-rank rank-4 SELECT RANK test ==='
    if (.not. expect_output(source, expected, &
            '/tmp/ffc_session_assumed_rank_select_rank4')) then
        print *, 'FAIL: ffc rank-4 behavioral case'
        stop 1
    end if
    if (.not. test_gfortran_oracle()) then
        print *, 'FAIL: independent gfortran behavioral oracle'
        stop 1
    end if
    if (.not. test_refusals()) then
        print *, 'FAIL: rank-4 refusal cases'
        stop 1
    end if
    print *, 'PASS: rank-4 REAL assumed-rank descriptor arm uses column-major access'

contains

    logical function test_gfortran_oracle()
        character(len=*), parameter :: src_path = &
            '/tmp/ffc_assumed_rank_select_rank4_oracle.f90'
        character(len=*), parameter :: exe_path = &
            '/tmp/ffc_assumed_rank_select_rank4_oracle'
        character(len=*), parameter :: out_path = &
            '/tmp/ffc_assumed_rank_select_rank4_oracle.out'
        character(len=:), allocatable :: actual
        integer :: unit, cmd_stat, exit_stat

        test_gfortran_oracle = .false.
        open (newunit=unit, file=src_path, status='replace', action='write')
        write (unit, '(a)') source
        close (unit)
        call execute_command_line('gfortran -std=f2018 '//src_path//' -o '//exe_path, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) return
        call execute_command_line(exe_path//' | tr -s " " > '//out_path, &
            exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) return
        call read_text(out_path, actual)
        call execute_command_line('rm -f '//src_path//' '//exe_path//' '//out_path)
        if (.not. allocated(actual)) return
        test_gfortran_oracle = index(actual, '1 12 23 34') == 1
    end function test_gfortran_oracle

    logical function test_refusals()
        character(len=*), parameter :: rank_default = &
            'program main'//new_line('a')// &
            '  real :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (4)'//new_line('a')// &
            '    rank default'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: rank_star = &
            'program main'//new_line('a')// &
            '  real :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (*)'//new_line('a')// &
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
            '    rank (4)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: dynamic_shape = &
            'program main'//new_line('a')// &
            '  call driver(2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine driver(n)'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '    real :: a(n,2)'//new_line('a')// &
            '    call work(a)'//new_line('a')// &
            '  end subroutine driver'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (2)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: pointer_actual = &
            'program main'//new_line('a')// &
            '  real, target :: a(2,2)'//new_line('a')// &
            '  real, pointer :: p(:,:)'//new_line('a')// &
            '  p => a'//new_line('a')// &
            '  call work(p)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (2)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: section_alias = &
            'program main'//new_line('a')// &
            '  real :: a(2,2)'//new_line('a')// &
            '  call work(a(:,1))'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (1)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: allocatable_actual = &
            'program main'//new_line('a')// &
            '  real, allocatable :: a(:,:,:,:)'//new_line('a')// &
            '  allocate(a(2,2,2,2))'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (4)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: integer_actual = &
            'program main'//new_line('a')// &
            '  integer :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (4)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: global_actual = &
            'module state'//new_line('a')// &
            '  real :: a(2)'//new_line('a')// &
            'end module state'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use state'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (1)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: multiple_arms = &
            'program main'//new_line('a')// &
            '  real :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (4)'//new_line('a')// &
            '    rank (3)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: missing_arm = &
            'program main'//new_line('a')// &
            '  real :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: unsupported_arm = &
            'program main'//new_line('a')// &
            '  real :: a(2,2,2,2)'//new_line('a')// &
            '  call work(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(x)'//new_line('a')// &
            '    real :: x(..)'//new_line('a')// &
            '    select rank (x)'//new_line('a')// &
            '    rank (5)'//new_line('a')// &
            '    end select'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_refusals = .true.
        if (.not. expect_error_contains(rank_default, 'RANK DEFAULT is refused', &
                '/tmp/ffc_assumed_rank_rank4_default_reject')) test_refusals = .false.
        if (.not. expect_error_contains(rank_star, 'RANK (*) is refused', &
                '/tmp/ffc_assumed_rank_rank4_star_reject')) test_refusals = .false.
        if (.not. expect_error_contains(scalar_actual, 'only a whole array is supported', &
                '/tmp/ffc_assumed_rank_rank4_scalar_reject')) test_refusals = .false.
        if (.not. expect_error_contains(dynamic_shape, 'dynamic shapes are not supported', &
                '/tmp/ffc_assumed_rank_rank4_dynamic_reject')) test_refusals = .false.
        if (.not. expect_error_contains(pointer_actual, 'pointer and alias actuals are refused', &
                '/tmp/ffc_assumed_rank_rank4_pointer_reject')) test_refusals = .false.
        if (.not. expect_error_contains(section_alias, 'only a whole array is supported', &
                '/tmp/ffc_assumed_rank_rank4_alias_reject')) test_refusals = .false.
        if (.not. expect_error_contains(allocatable_actual, 'only a whole array is supported', &
                '/tmp/ffc_assumed_rank_rank4_allocatable_reject')) test_refusals = .false.
        if (.not. expect_error_contains(integer_actual, 'only REAL whole arrays are supported', &
                '/tmp/ffc_assumed_rank_rank4_kind_reject')) test_refusals = .false.
        if (.not. expect_error_contains(global_actual, 'global storage is refused', &
                '/tmp/ffc_assumed_rank_rank4_global_reject')) test_refusals = .false.
        if (.not. expect_error_contains(multiple_arms, 'exactly one RANK', &
                '/tmp/ffc_assumed_rank_rank4_multiple_reject')) test_refusals = .false.
        if (.not. expect_error_contains(missing_arm, 'exactly one RANK', &
                '/tmp/ffc_assumed_rank_rank4_missing_reject')) test_refusals = .false.
        if (.not. expect_error_contains(unsupported_arm, &
                'only one statically valid RANK', &
                '/tmp/ffc_assumed_rank_rank4_rank5_arm_reject')) test_refusals = .false.
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

end program test_session_assumed_rank_select_rank4_compiler
