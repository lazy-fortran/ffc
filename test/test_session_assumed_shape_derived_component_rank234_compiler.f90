program test_session_assumed_shape_derived_component_rank234_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== assumed-shape derived component rank-2/3/4 test ==='
    all_passed = test_rank2_expected()
    if (.not. test_rank3_matches_gfortran()) all_passed = .false.
    if (.not. test_rank4_matches_gfortran()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: assumed-shape derived component actuals rank 2/3/4'

contains

    logical function test_rank2_expected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'type item_t'//new_line('a')// &
            'integer :: value'//new_line('a')// &
            'end type item_t'//new_line('a')// &
            'type(item_t) :: items(2,3)'//new_line('a')// &
            'integer :: i, j'//new_line('a')// &
            'do j = 1, 3'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'items(i,j)%value = 100*i + 10*j'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call inspect(items%value)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine inspect(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:)'//new_line('a')// &
            'print *, size(a,1), size(a,2)'//new_line('a')// &
            'print *, a(1,1), a(2,3), a(2,1)'//new_line('a')// &
            'end subroutine inspect'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2           3'//new_line('a')// &
            '         110         230         210'//new_line('a')

        test_rank2_expected = expect_output(source, expected, &
            '/tmp/ffc_session_assumed_shape_derived_component_rank2')
    end function test_rank2_expected

    logical function test_rank3_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'type item_t'//new_line('a')// &
            'integer :: value'//new_line('a')// &
            'end type item_t'//new_line('a')// &
            'type(item_t) :: items(2,2,2)'//new_line('a')// &
            'integer :: i, j, k'//new_line('a')// &
            'do k = 1, 2'//new_line('a')// &
            'do j = 1, 2'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'items(i,j,k)%value = 100*i + 10*j + k'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call inspect(items%value)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine inspect(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:,:)'//new_line('a')// &
            'print *, a(1,1,1), a(2,2,2), a(2,1,2)'//new_line('a')// &
            'end subroutine inspect'//new_line('a')// &
            'end program main'

        test_rank3_matches_gfortran = expect_output_matches_gfortran(source, &
            'assumed_shape_derived_component_rank3')
    end function test_rank3_matches_gfortran

    logical function test_rank4_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'type item_t'//new_line('a')// &
            'integer :: value'//new_line('a')// &
            'end type item_t'//new_line('a')// &
            'type(item_t) :: items(2,1,2,2)'//new_line('a')// &
            'integer :: i, j, k, l'//new_line('a')// &
            'do l = 1, 2'//new_line('a')// &
            'do k = 1, 2'//new_line('a')// &
            'do j = 1, 1'//new_line('a')// &
            'do i = 1, 2'//new_line('a')// &
            'items(i,j,k,l)%value = 1000*i + 100*j + 10*k + l'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'end do'//new_line('a')// &
            'call inspect(items%value)'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine inspect(a)'//new_line('a')// &
            'integer, intent(in) :: a(:,:,:,:)'//new_line('a')// &
            'print *, a(1,1,1,1), a(2,1,2,2), a(2,1,1,2)'//new_line('a')// &
            'end subroutine inspect'//new_line('a')// &
            'end program main'

        test_rank4_matches_gfortran = expect_output_matches_gfortran(source, &
            'assumed_shape_derived_component_rank4')
    end function test_rank4_matches_gfortran

end program test_session_assumed_shape_derived_component_rank234_compiler
