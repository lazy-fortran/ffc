! fo: dispatcher
module ffc_case_test_session_assumed_shape_derived_bounds_compiler
    implicit none
    private
    public :: case_test_session_assumed_shape_derived_bounds_compiler
    interface
        subroutine case_test_session_assumed_shape_derived_bounds_compiler()
        end subroutine case_test_session_assumed_shape_derived_bounds_compiler
    end interface
end module ffc_case_test_session_assumed_shape_derived_bounds_compiler

subroutine case_test_session_assumed_shape_derived_bounds_compiler()
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none
    save

    logical :: all_passed

    all_passed = test_rank2_rebound_bounds()
    if (.not. test_rank1_rebound_bounds()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: derived dummy element addressing uses descriptor lower bounds'

contains

    logical function test_rank2_rebound_bounds()
        ! One body sees differently shaped fixed and runtime actuals. Its
        ! non-default bounds belong to the receiving dummy at every call.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: x, y'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type(item_t) :: fixed(2,3), small(1,2)'//new_line('a')// &
            '  type(item_t), allocatable :: dynamic(:,:)'//new_line('a')// &
            '  integer :: i, j, n'//new_line('a')// &
            '  n = 3 + command_argument_count()'//new_line('a')// &
            '  allocate(dynamic(n,1))'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      fixed(i,j)%x = 10*i + j'//new_line('a')// &
            '      fixed(i,j)%y = 100*i + j'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  small(1,1)%x = 41'//new_line('a')// &
            '  small(1,1)%y = 401'//new_line('a')// &
            '  small(1,2)%x = 42'//new_line('a')// &
            '  small(1,2)%y = 402'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    dynamic(i,1)%x = 50 + i'//new_line('a')// &
            '    dynamic(i,1)%y = 500 + i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call outer(fixed)'//new_line('a')// &
            '  call outer(small)'//new_line('a')// &
            '  call outer(dynamic)'//new_line('a')// &
            '  print *, fixed(2,3)%x, fixed(1,3)%x'//new_line('a')// &
            '  print *, small(1,2)%x, small(1,1)%x'//new_line('a')// &
            '  print *, dynamic(n,1)%x, dynamic(1,1)%x'//new_line('a')// &
            '  deallocate(dynamic)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(4:,7:)'//new_line('a')// &
            '    print *, values(4,7)%x'//new_line('a')// &
            '    call inner(values)'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(0:,-1:)'//new_line('a')// &
            '    integer :: last_i, last_j'//new_line('a')// &
            '    last_i = size(values,1) - 1'//new_line('a')// &
            '    last_j = size(values,2) - 2'//new_line('a')// &
            '    print *, size(values,1), size(values,2)'//new_line('a')// &
            '    print *, values(0,-1)%x, values(last_i,last_j)%y'// &
            new_line('a')// &
            '    values(last_i,last_j)%x = values(last_i,last_j)%x + 900'// &
            new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            'end program main'

        test_rank2_rebound_bounds = expect_output_matches_gfortran(source, &
            'derived_rank2_rebound_bounds')
    end function test_rank2_rebound_bounds

    logical function test_rank1_rebound_bounds()
        ! The actual, outer dummy and inner dummy all use different bounds.
        ! Element identity must survive both rebinding steps and writeback.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: x, y'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type(item_t) :: fixed(-3:-1)'//new_line('a')// &
            '  type(item_t), allocatable :: dynamic(:)'//new_line('a')// &
            '  integer :: i, n'//new_line('a')// &
            '  n = 3 + command_argument_count()'//new_line('a')// &
            '  allocate(dynamic(n))'//new_line('a')// &
            '  do i = -3, -1'//new_line('a')// &
            '    fixed(i)%x = i + 13'//new_line('a')// &
            '    fixed(i)%y = i + 103'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  do i = 1, n'//new_line('a')// &
            '    dynamic(i)%x = i + 20'//new_line('a')// &
            '    dynamic(i)%y = i + 200'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call outer(fixed)'//new_line('a')// &
            '  call outer(dynamic)'//new_line('a')// &
            '  print *, fixed(-3)%x, fixed(-2)%x, fixed(-1)%y'//new_line('a')// &
            '  print *, dynamic(1)%x, dynamic(2)%x, dynamic(n)%y'//new_line('a')// &
            '  deallocate(dynamic)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(0:)'//new_line('a')// &
            '    print *, values(0)%x, values(1)%y'//new_line('a')// &
            '    call inner(values)'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(-2:)'//new_line('a')// &
            '    print *, values(-2)%x, values(size(values)-3)%y'//new_line('a')// &
            '    values(-1)%x = 99'//new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            'end program main'

        test_rank1_rebound_bounds = expect_output_matches_gfortran(source, &
            'derived_rank1_rebound_bounds')
    end function test_rank1_rebound_bounds

end subroutine case_test_session_assumed_shape_derived_bounds_compiler
