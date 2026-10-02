! fo: dispatcher
module ffc_case_test_session_merge_reduction_oracle_compiler
    implicit none
    private
    public :: case_test_session_merge_reduction_oracle_compiler
    interface
        subroutine case_test_session_merge_reduction_oracle_compiler()
        end subroutine case_test_session_merge_reduction_oracle_compiler
    end interface
end module ffc_case_test_session_merge_reduction_oracle_compiler

subroutine case_test_session_merge_reduction_oracle_compiler()
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
    work = '/var/tmp/ffc_merge_reduction_oracle_'//trim(suffix)
    call execute_command_line('mkdir -p '//work)
    passed = .true.
    do case_index = 1, 10
        source = oracle_source(case_index)
        if (.not. matches_gfortran(source, case_index)) passed = .false.
    end do
    call execute_command_line('rm -rf '//work)
    if (.not. passed) stop 1
    print *, 'PASS: MERGE and ABS reductions match gfortran on 20 runs per shape'

contains

    function oracle_source(case_index) result(source)
        integer, intent(in) :: case_index
        character(len=:), allocatable :: source

        source = 'program p'//nl//'implicit none'//nl
        select case (case_index)
        case (1)
            source = source// &
                     'integer :: i,j'//nl// &
                     'real :: x,y'//nl// &
                     'real(8) :: d,e'//nl// &
                     'logical :: m,t,f'//nl// &
                     'i=7; j=9; x=1.5; y=-2.5; d=1.5d0; e=-2.5d0'//nl// &
                     'm=i>j; t=.true.; f=.false.'//nl// &
                     'print *, merge(i,j,m), merge(i,j,.not.m)'//nl// &
                     'print *, merge(x,y,m), merge(x,y,.not.m)'//nl// &
                     'print *, merge(d,e,m), merge(d,e,.not.m)'//nl// &
                     'print *, merge(t,f,m), merge(t,f,.not.m)'//nl// &
                     'print *, merge(mask=m,fsource=j,tsource=i)'//nl// &
                     'x=merge(i,j,m); d=merge(x,y,m)'//nl// &
                     'print *, x,d'//nl
        case (2)
            source = source// &
                     'integer :: a(5),b(5),c(5)'//nl// &
                     'real :: r(4)'//nl// &
                     'logical :: m(5)'//nl// &
                     'a=[-5,2,-3,4,-1]; b=[10,20,30,40,50]'//nl// &
                     'r=[1.5,-2.5,3.0,-4.0]'//nl// &
                     'm=[.true.,.false.,.true.,.false.,.true.]'//nl// &
                     'print *, merge(1,2,a>2)'//nl// &
                     'print *, merge(a,b,m), merge(a,b,a>2)'//nl// &
                     'print *, merge(a,99,a>2), merge(99,b,m)'//nl// &
                     'print *, merge(a+1,b*2,a>2), merge(r,0.0,r>0)'//nl// &
                     'print *, merge(a>2,b<30,m)'//nl// &
                     'c=merge(mask=a>2,fsource=b,tsource=a)'//nl// &
                     'print *, c'//nl// &
                     'c=merge(a,99,m)'//nl// &
                     'print *, c'//nl// &
                     'c=merge(a(5:1:-1),b,m)'//nl// &
                     'print *, c'//nl
        case (3)
            source = source// &
                     'integer :: a(5)'//nl// &
                     'real :: r(4)'//nl// &
                     'real(8) :: d(4)'//nl// &
                     'a=[-5,2,-3,4,-1]'//nl// &
                     'r=[1.5,-2.5,3.0,-4.0]; d=[1.5d0,-2.5d0,3.0d0,-4.0d0]'//nl// &
                     'print *, sum(abs(a)), product(abs(a))'//nl// &
                     'print *, maxval(abs(a)), minval(abs(a))'//nl// &
                     'print *, sum(abs(a(2:4))), sum(abs(a)+1)'//nl// &
                     'print *, sum(abs(r)), maxval(abs(r)), minval(abs(r))'//nl// &
                     'print *, sum(abs(d)), maxval(abs(d)), minval(abs(d))'//nl
        case (4)
            source = source// &
                     'integer :: a(5),b(3)'//nl// &
                     'a=[-5,2,-3,4,-1]; b=[-10,20,-30]'//nl// &
                     'call check(a); call check(b)'//nl// &
                     'contains'//nl// &
                     'subroutine check(x)'//nl// &
                     'integer :: x(:)'//nl// &
                     'print *, sum(abs(x)), product(abs(x))'//nl// &
                     'print *, maxval(abs(x)), minval(abs(x))'//nl// &
                     'print *, sum(x,mask=x>0), product(x,mask=x>0)'//nl// &
                     'print *, maxval(x,mask=x<0), minval(x,mask=x>0)'//nl// &
                     'print *, sum(x,mask=.true.), product(x,mask=.false.)'//nl// &
                     'print *, maxval(x,mask=x>99), minval(x,mask=x>99)'//nl// &
                     'print *, sum(x,mask=x > 0 .and. x < 4)'//nl// &
                     'end subroutine check'//nl
        case (5)
            source = source// &
                     'integer :: a(2,3),b(3,2)'//nl// &
                     'logical :: m(2,3),n(3,2)'//nl// &
                     'a=reshape([-5,2,-3,4,-1,6],[2,3])'//nl// &
                     'b=reshape([-10,20,-30,40,-50,60],[3,2])'//nl// &
                     'm=a>0; n=b>0'//nl// &
                     'call check(a,m); call check(b,n)'//nl// &
                     'contains'//nl// &
                     'subroutine check(x,mask)'//nl// &
                     'integer :: x(:,:)'//nl// &
                     'logical :: mask(:,:)'//nl// &
                     'print *, sum(x,mask=mask), product(x,mask=mask)'//nl// &
                     'print *, maxval(x,mask=mask), minval(x,mask=.not.mask)'//nl// &
                     'end subroutine check'//nl
        case (6)
            source = source// &
                     'real :: a(4),b(2)'//nl// &
                     'real(8) :: d(4),e(2)'//nl// &
                     'a=[1.5,-2.5,3.0,-4.0]; b=[-10.0,20.0]'//nl// &
                     'd=[1.5d0,-2.5d0,3.0d0,-4.0d0]; e=[-10.0d0,20.0d0]'//nl// &
                     'call check(a,d); call check(b,e)'//nl// &
                     'contains'//nl// &
                     'subroutine check(x,y)'//nl// &
                     'real :: x(:)'//nl// &
                     'real(8) :: y(:)'//nl// &
                     'print *, sum(x,mask=x>0), product(x,mask=x>0)'//nl// &
                     'print *, maxval(x,mask=x<0), minval(x,mask=x>0)'//nl// &
                     'print *, sum(abs(x)), minval(abs(x)), maxval(abs(x))'//nl// &
                     'print *, sum(abs(y)), minval(abs(y)), maxval(abs(y))'//nl// &
                     'print *, sum(y,mask=y>0), product(y,mask=y>0)'//nl// &
                     'print *, maxval(y,mask=y<0), minval(y,mask=y>0)'//nl// &
                     'print *, maxval(x,mask=.false.), minval(y,mask=.false.)'//nl// &
                     'end subroutine check'//nl
        case (7)
            source = source// &
                     'integer, allocatable :: a(:)'//nl// &
                     'logical, allocatable :: m(:)'//nl// &
                     'integer :: n'//nl// &
                     'n=5; allocate(a(n),m(n)); a=[-5,2,-3,4,-1]; m=a>0'//nl// &
                     'print *, sum(a,mask=m), product(a,mask=m)'//nl// &
                     'print *, maxval(a,mask=m), minval(a,mask=.not.m)'//nl// &
                     'print *, sum(a,mask=a>2.5)'//nl// &
                     'deallocate(a,m); n=3; allocate(a(n),m(n))'//nl// &
                     'a=[-10,20,-30]; m=a>0'//nl// &
                     'print *, sum(a,mask=m), product(a,mask=m)'//nl// &
                     'print *, maxval(a,mask=m), minval(a,mask=.not.m)'//nl
        case (8)
            source = source// &
                     'integer :: a(-2:5),lo,hi,step'//nl// &
                     'logical :: m(-2:5)'//nl// &
                     'a=[-8,7,-6,5,-4,3,-2,1]; m=a>0'//nl// &
                     'lo=-1+command_argument_count(); hi=4; step=2'//nl// &
                     'call check(a(lo:hi:step),m(lo:hi:step))'//nl// &
                     'lo=5; hi=-2; step=-2'//nl// &
                     'call check(a(lo:hi:step),m(lo:hi:step))'//nl// &
                     'call check(a(4:-1:-1),m(4:-1:-1))'//nl// &
                     'call check(a(3:2),m(3:2))'//nl// &
                     'contains'//nl// &
                     'subroutine check(x,y)'//nl// &
                     'integer :: x(-4:)'//nl// &
                     'logical :: y(-4:)'//nl// &
                     'print *, size(x), sum(x,mask=y), product(x,mask=y)'//nl// &
                     'print *, maxval(x,mask=y), minval(x,mask=y)'//nl// &
                     'print *, sum(x,mask=x<0), sum(abs(x))'//nl// &
                     'end subroutine check'//nl
        case (9)
            source = source// &
                     'call check(2); call check(3)'//nl// &
                     'contains'//nl// &
                     'subroutine check(n)'//nl// &
                     'integer :: n,i,j,x(n,2)'//nl// &
                     'do j=1,2'//nl// &
                     'do i=1,n'//nl// &
                     'x(i,j)=i+10*j'//nl// &
                     'end do'//nl// &
                     'end do'//nl// &
                     'print *, sum(x,mask=x>12), product(x,mask=x<20)'//nl// &
                     'print *, maxval(x,mask=x<20), minval(x,mask=x>12)'//nl// &
                     'end subroutine check'//nl
        case (10)
            source = source// &
                     'integer :: i,j'//nl// &
                     'i=7; j=9'//nl// &
                     'print *, merge(i,j,.true.)'//nl// &
                     'contains'//nl// &
                     'integer function merge(x,y,z)'//nl// &
                     'integer :: x,y'//nl// &
                     'logical :: z'//nl// &
                     'merge=x+y'//nl// &
                     'end function merge'//nl
        end select
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

end subroutine case_test_session_merge_reduction_oracle_compiler
