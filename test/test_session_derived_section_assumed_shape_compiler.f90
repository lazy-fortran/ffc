! fo: dispatcher
module ffc_case_test_session_derived_section_assumed_shape_compiler
    implicit none
    private
    public :: case_test_session_derived_section_assumed_shape_compiler
    interface
        subroutine case_test_session_derived_section_assumed_shape_compiler()
        end subroutine case_test_session_derived_section_assumed_shape_compiler
    end interface
end module ffc_case_test_session_derived_section_assumed_shape_compiler

subroutine case_test_session_derived_section_assumed_shape_compiler()
    use ffc_test_support, only: expect_error_contains, &
        expect_output_matches_gfortran
    implicit none
    save

    logical :: all_passed

    all_passed = test_identity_sections()
    if (.not. test_forwarded_rank_mismatch()) all_passed = .false.
    if (.not. test_partial_section_refused()) all_passed = .false.
    if (.not. test_forwarded_intrinsic_rank_mismatch()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: derived identity sections preserve descriptor shape and storage'

contains

    logical function test_identity_sections()
        ! Different shapes call the same body, then forward a whole section
        ! through its incoming descriptor. Writes reach each original actual.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: x, y'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type(item_t) :: fixed(2,3), small(1,2), line(3)'//new_line('a')// &
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
            '  do i = 1, 3'//new_line('a')// &
            '    line(i)%x = i'//new_line('a')// &
            '    line(i)%y = 10*i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  call outer(fixed(:,:))'//new_line('a')// &
            '  call outer(small(:,:))'//new_line('a')// &
            '  call outer(dynamic(:,:))'//new_line('a')// &
            '  call rank_one(line(:))'//new_line('a')// &
            '  print *, fixed(2,3)%x, fixed(1,3)%x'//new_line('a')// &
            '  print *, small(1,2)%x, small(1,1)%x'//new_line('a')// &
            '  print *, dynamic(n,1)%x, dynamic(1,1)%x'//new_line('a')// &
            '  print *, line(2)%x, line(1)%y, line(3)%y'//new_line('a')// &
            '  deallocate(dynamic)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(:,:)'//new_line('a')// &
            '    call inner(values(:,:))'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(:,:)'//new_line('a')// &
            '    integer :: last_i, last_j'//new_line('a')// &
            '    last_i = size(values,1)'//new_line('a')// &
            '    last_j = size(values,2)'//new_line('a')// &
            '    print *, size(values,1), size(values,2)'//new_line('a')// &
            '    print *, values(1,1)%x, values(last_i,last_j)%y'// &
            new_line('a')// &
            '    values(last_i,last_j)%x = values(last_i,last_j)%x + 900'// &
            new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            '  subroutine rank_one(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(:)'//new_line('a')// &
            '    print *, size(values), values(3)%y'//new_line('a')// &
            '    values(2)%x = 99'//new_line('a')// &
            '  end subroutine rank_one'//new_line('a')// &
            'end program main'

        test_identity_sections = expect_output_matches_gfortran(source, &
            'derived_identity_sections')
    end function test_identity_sections

    logical function test_forwarded_rank_mismatch()
        ! Forwarding an incoming descriptor must validate the receiving rank
        ! before copying its header and dimensions into a new dummy view.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type(item_t) :: a(2,2)'//new_line('a')// &
            '  call outer(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(:,:)'//new_line('a')// &
            '    call inner(values(:,:))'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(values)'//new_line('a')// &
            '    type(item_t), intent(inout) :: values(:)'//new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            'end program main'

        test_forwarded_rank_mismatch = expect_error_contains(source, &
            'assumed-shape derived array dummy of rank 1 received '// &
            'an actual of rank 2', &
            '/var/tmp/ffc_derived_identity_forwarded_rank_mismatch')
    end function test_forwarded_rank_mismatch

    logical function test_forwarded_intrinsic_rank_mismatch()
        ! The same forwarding boundary applies to intrinsic arrays, whose
        ! declaration path has no derived-type shape guard to reject a mismatch.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2,2)'//new_line('a')// &
            '  a = 7'//new_line('a')// &
            '  call outer(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine outer(values)'//new_line('a')// &
            '    integer, intent(in) :: values(:,:)'//new_line('a')// &
            '    call inner(values)'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(values)'//new_line('a')// &
            '    integer, intent(in) :: values(:)'//new_line('a')// &
            '    print *, size(values)'//new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            'end program main'

        test_forwarded_intrinsic_rank_mismatch = expect_error_contains(source, &
            'assumed-shape dummy of rank 1 received an actual of rank 2', &
            '/var/tmp/ffc_derived_identity_intrinsic_rank_mismatch')
    end function test_forwarded_intrinsic_rank_mismatch

    logical function test_partial_section_refused()
        ! A proper subset is not an identity view: accepting it as the whole
        ! array would silently expose elements outside the selected section.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: item_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type item_t'//new_line('a')// &
            '  type(item_t) :: a(2,2)'//new_line('a')// &
            '  call inspect(a(1:1,:))'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine inspect(values)'//new_line('a')// &
            '    type(item_t), intent(in) :: values(:,:)'//new_line('a')// &
            '  end subroutine inspect'//new_line('a')// &
            'end program main'

        test_partial_section_refused = expect_error_contains(source, &
            'array-valued actual has no statically known extent in inspect', &
            '/var/tmp/ffc_derived_identity_partial_section')
    end function test_partial_section_refused

end subroutine case_test_session_derived_section_assumed_shape_compiler
