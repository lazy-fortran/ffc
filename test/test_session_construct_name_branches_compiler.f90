! fo: dispatcher
module ffc_case_test_session_construct_name_branches_compiler
    implicit none
    private
    public :: case_test_session_construct_name_branches_compiler
    interface
        subroutine case_test_session_construct_name_branches_compiler()
        end subroutine case_test_session_construct_name_branches_compiler
    end interface
end module ffc_case_test_session_construct_name_branches_compiler

subroutine case_test_session_construct_name_branches_compiler()
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    save

    logical :: all_passed
    integer :: kind
    character(len=8), parameter :: loop_kinds(3) = &
        [character(len=8) :: 'counted', 'while', 'infinite']

    all_passed = .true.
    do kind = 1, size(loop_kinds)
        if (.not. test_outer('exit', trim(loop_kinds(kind)))) all_passed = .false.
        if (.not. test_outer('cycle', trim(loop_kinds(kind)))) all_passed = .false.
    end do
    if (.not. test_middle('exit')) all_passed = .false.
    if (.not. test_middle('cycle')) all_passed = .false.
    if (.not. test_wide('exit')) all_passed = .false.
    if (.not. test_wide('cycle')) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: named EXIT/CYCLE preserve target and scalar branch values'
contains
    logical function test_outer(branch, loop_kind)
        character(len=*), intent(in) :: branch, loop_kind
        character(len=:), allocatable :: source, opening, advance
        character(len=:), allocatable :: terminate
        character(len=1), parameter :: nl = new_line('a')

        advance = ' i=i+1'//nl
        terminate = ''
        select case (loop_kind)
        case ('counted')
            opening = 'Outer: do i=1,4'
            advance = ''
        case ('while')
            opening = 'Outer: do while (i<4)'
        case ('infinite')
            opening = 'Outer: do'
            terminate = ' if (i>4) exit Outer'//nl
        end select
        source = 'program main'//nl// &
            'integer :: i,j,s'//nl// &
            'i=0'//nl//'s=0'//nl//opening//nl//advance//terminate// &
            ' s=s+i'//nl// &
            ' do j=1,3'//nl// &
            '  s=s+10'//nl// &
            '  if (j==2) '//branch//' OUTER'//nl// &
            '  s=s+1'//nl// &
            ' end do'//nl// &
            ' s=s+1000'//nl// &
            'end do Outer'//nl// &
            'print *, i,j,s'//nl//'end program'
        test_outer = expect_output_matches_gfortran(source, &
            'named_outer_'//loop_kind//'_'//branch)
    end function test_outer

    logical function test_middle(branch)
        character(len=*), intent(in) :: branch
        character(len=:), allocatable :: source
        character(len=1), parameter :: nl = new_line('a')

        source = 'program main'//nl// &
            'integer :: i,j,k,s'//nl//'s=0'//nl// &
            'a: do i=1,2'//nl// &
            ' b: do j=1,3'//nl// &
            '  do k=1,3'//nl// &
            '   s=s+100*i+10*j+k'//nl// &
            '   if (k==2) '//branch//' b'//nl// &
            '  end do'//nl// &
            '  s=s+1000'//nl// &
            ' end do b'//nl// &
            ' s=s+10000'//nl// &
            'end do a'//nl// &
            'print *, i,j,k,s'//nl//'end program'
        test_middle = expect_output_matches_gfortran(source, &
            'named_middle_'//branch)
    end function test_middle

    logical function test_wide(branch)
        character(len=*), intent(in) :: branch
        character(len=:), allocatable :: source
        character(len=1), parameter :: nl = new_line('a')

        source = 'program main'//nl// &
            'integer :: i,j'//nl// &
            'integer(8) :: s'//nl//'s=5000000000_8'//nl// &
            'outer: do i=1,4'//nl// &
            ' do j=1,3'//nl// &
            '  s=s+1000000000_8'//nl// &
            '  if (j==2) '//branch//' outer'//nl// &
            ' end do'//nl// &
            ' s=s+1_8'//nl// &
            'end do outer'//nl// &
            'print *, i,j,s'//nl//'end program'
        test_wide = expect_output_matches_gfortran(source, 'named_wide_'//branch)
    end function test_wide
end subroutine case_test_session_construct_name_branches_compiler
