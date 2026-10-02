! fo: dispatcher
module ffc_case_test_session_transfer_descriptor_compiler
    implicit none
    private
    public :: case_test_session_transfer_descriptor_compiler
    interface
        subroutine case_test_session_transfer_descriptor_compiler()
        end subroutine case_test_session_transfer_descriptor_compiler
    end interface
end module ffc_case_test_session_transfer_descriptor_compiler

subroutine case_test_session_transfer_descriptor_compiler()
    use ffc_test_support, only: compile_to_exe
    implicit none
    save

    character(len=*), parameter :: nl = new_line('a')
    character(len=:), allocatable :: source, work
    character(len=32) :: suffix
    integer :: clock, case_index
    logical :: passed

    call system_clock(clock)
    write (suffix, '(I0)') clock
    work = '/var/tmp/ffc_transfer_descriptor_'//trim(suffix)
    call execute_command_line('mkdir -p '//work)
    passed = .true.
    do case_index = 1, 8
        source = oracle_source(case_index)
        if (.not. matches_gfortran(source, case_index)) passed = .false.
    end do
    if (.not. refuses_short_source(.false.)) passed = .false.
    if (.not. refuses_short_source(.true.)) passed = .false.
    call execute_command_line('rm -rf '//work)
    if (.not. passed) stop 1
    print *, 'PASS: descriptor TRANSFER matches gfortran on 20 runs per shape'

contains

    function oracle_source(case_index) result(source)
        integer, intent(in) :: case_index
        character(len=:), allocatable :: source, source_type, result_type, values
        character(len=:), allocatable :: calls, output

        source = 'program p'//nl//'implicit none'//nl
        if (case_index <= 4) then
            select case (case_index)
            case (1)
                source_type = 'real'
                result_type = 'integer'
                values = '[-8.0,7.0,-6.0,5.0,-4.0,3.0,-2.0,1.0]'
            case (2)
                source_type = 'integer'
                result_type = 'real'
                values = '[1065353216,1073741824,1077936128,1082130432,'// &
                         '1084227584,1086324736,1088421888,1090519040]'
            case (3)
                source_type = 'real(8)'
                result_type = 'integer(8)'
                values = '[-8.0d0,7.0d0,-6.0d0,5.0d0,-4.0d0,3.0d0,-2.0d0,1.0d0]'
            case (4)
                source_type = 'integer(8)'
                result_type = 'real(8)'
                values = '[4607182418800017408_8,4611686018427387904_8,'// &
                         '&'//nl// &
                         '4613937818241073152_8,4616189618054758400_8,&'//nl// &
                         '4617315517961601024_8,4618441417868443648_8,&'//nl// &
                         '4619567317775286272_8,4620693217682128896_8]'
            end select
            calls = 'call check(a)'//nl
            if (case_index /= 4) then
                calls = calls//'lo=-1+command_argument_count(); hi=4; step=2'//nl// &
                        'call check(a(lo:hi:step))'//nl// &
                        'lo=5; hi=-2; step=-2'//nl// &
                        'call check(a(lo:hi:step))'//nl// &
                        'call check(a(3:4)); call forward(a)'//nl
            end if
            output = 'print *, size(x)'//nl// &
                     'print *, t(-1)'//nl//'print *, t(0)'//nl//'print *, s'//nl
            if (case_index == 3) then
                ! Printing I64 locals in contained procedures is outside this slice.
                ! Check every bit through scalar equality, then expose its magnitude.
                output = 'print *, size(x)'//nl// &
                         'print *, s==transfer(x(-4),s)'//nl// &
                         'print *, dble(s)'//nl// &
                         's=t(-1)'//nl//'print *, s==transfer(x(-4),s)'//nl// &
                         's=t(0)'//nl//'print *, s==transfer(x(-3),s)'//nl
            end if
            source = source// &
                     source_type//' :: a(-2:5)'//nl// &
                     'integer :: lo,hi,step'//nl// &
                     'a='//values//nl// &
                     calls// &
                     'contains'//nl// &
                     'subroutine forward(y)'//nl// &
                     source_type//' :: y(:)'//nl// &
                     'call check(y)'//nl// &
                     'end subroutine forward'//nl// &
                     'subroutine check(x)'//nl// &
                     source_type//' :: x(-4:)'//nl// &
                     result_type//' :: t(-1:0),s'//nl// &
                     't=transfer(x,t,1+1)'//nl//'s=transfer(x,s)'//nl// &
                     output// &
                     'end subroutine check'//nl
        else
            select case (case_index)
            case (5)
                source = source// &
                         'call check(2); call check(5)'//nl// &
                         'contains'//nl// &
                         'subroutine check(n)'//nl// &
                         'integer :: n,i,t(2),s'//nl// &
                         'real :: x(n)'//nl// &
                         'do i=1,n'//nl//'x(i)=real(i)'//nl//'end do'//nl// &
                         't=transfer(x,t,2)'//nl//'s=transfer(x,s)'//nl// &
                         'print *, n'//nl//'print *, t(1)'//nl// &
                         'print *, t(2)'//nl//'print *, s'//nl// &
                         'end subroutine check'//nl
            case (6)
                source = source// &
                         'real, allocatable :: x(:)'//nl// &
                         'integer :: n,i,t(2),s'//nl// &
                         'n=4; allocate(x(n))'//nl// &
                         'do i=1,n'//nl//'x(i)=real(i)'//nl//'end do'//nl// &
                         't=transfer(x,t,2)'//nl//'s=transfer(x,s)'//nl// &
                         'print *, size(x)'//nl//'print *, t(1)'//nl// &
                         'print *, t(2)'//nl//'print *, s'//nl// &
                         'deallocate(x)'//nl//'n=2'//nl//'allocate(x(n))'//nl// &
                         'do i=1,n'//nl//'x(i)=-real(i+2)'//nl//'end do'//nl// &
                         't=transfer(x,t)'//nl//'s=transfer(x,s)'//nl// &
                         'print *, size(x)'//nl//'print *, t(1)'//nl// &
                         'print *, t(2)'//nl//'print *, s'//nl
            case (8)
                source = source// &
                         'real, target :: a(2)'//nl// &
                         'real, pointer :: x(:)'//nl//'a=[1.0,2.0]'//nl// &
                         'x=>a(2:1:-1)'//nl//'a=transfer(x,a,2)'//nl// &
                         'print *, a'//nl
            case (7)
                source = source// &
                         'real :: a(2)'//nl// &
                         'a=[2.0,-3.0]; call check(a)'//nl// &
                         'contains'//nl// &
                         'subroutine check(x)'//nl// &
                         'real :: x(:),same(2)'//nl// &
                         'integer :: t(2),mold(1)'//nl// &
                         't=transfer(x,mold)'//nl//'same=transfer(x,same)'//nl// &
                         'print *, t,same'//nl// &
                         'end subroutine check'//nl
            end select
        end if
        source = source//'end program p'//nl
    end function oracle_source

    logical function matches_gfortran(source, case_index) result(matches)
        character(len=*), intent(in) :: source
        integer, intent(in) :: case_index
        character(len=:), allocatable :: stem, error_msg
        character(len=8) :: case_name
        integer :: unit, status, run_index

        matches = .false.
        write (case_name, '(I0)') case_index
        stem = work//'/case_'//trim(case_name)
        open (newunit=unit, file=stem//'.f90', status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call compile_to_exe(source, stem//'.ffc', error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: ffc refused case ', case_index, ': ', error_msg
            return
        end if
        call execute_command_line('gfortran -std=f2018 '//stem//'.f90 -o '// &
                                  stem//'.gfortran', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL: gfortran refused case ', case_index
            return
        end if
        call execute_command_line(stem//'.gfortran > '//stem//'.expected', &
                                  exitstat=status)
        if (status /= 0) return
        do run_index = 1, 20
            call execute_command_line('timeout 5s '//stem//'.ffc > '// &
                                      stem//'.actual', exitstat=status)
            if (status /= 0) then
                print *, 'FAIL: case ', case_index, ' run ', run_index, ' crashed'
                return
            end if
            call execute_command_line('cmp -s '//stem//'.expected '// &
                                      stem//'.actual', exitstat=status)
            if (status /= 0) then
                print *, 'FAIL: case ', case_index, ' run ', run_index, &
                    ' differs from gfortran'
                call execute_command_line('diff -u '//stem//'.expected '// &
                                          stem//'.actual')
                return
            end if
        end do
        call execute_command_line('md5sum '//stem//'.ffc '//stem//'.expected')
        matches = .true.
    end function matches_gfortran

    logical function refuses_short_source(empty_scalar) result(refused)
        logical, intent(in) :: empty_scalar
        character(len=:), allocatable :: source, stem, error_msg, line
        character(len=256) :: output_line
        integer :: unit, status

        source = 'program p'//nl//'real :: a(1)'//nl//'a=1.0'//nl
        stem = work//'/short_source'
        if (empty_scalar) then
            source = source//'call check(a(1:0))'//nl
            line = 's=transfer(x,s)'
            stem = work//'/empty_scalar'
        else
            source = source//'call check(a)'//nl
            line = 't=transfer(x,t,2)'
        end if
        source = source//'contains'//nl//'subroutine check(x)'//nl// &
                 'real :: x(:)'//nl//'integer :: t(2),s'//nl//line//nl// &
                 'print *, 99'//nl//'end subroutine check'//nl//'end program p'//nl
        refused = .false.
        call compile_to_exe(source, stem//'.ffc', error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: source extent guard did not compile: ', error_msg
            return
        end if
        call execute_command_line('timeout 5s '//stem//'.ffc > '//stem// &
                                  '.output 2>&1', exitstat=status)
        if (status /= 2) then
            print *, 'FAIL: unsupported source extent did not diagnose'
            return
        end if
        open (newunit=unit, file=stem//'.output', status='old', action='read')
        read (unit, '(A)', iostat=status) output_line
        close (unit)
        if (status /= 0) return
        refused = index(output_line, 'TRANSFER source extent') > 0
    end function refuses_short_source

end subroutine case_test_session_transfer_descriptor_compiler
