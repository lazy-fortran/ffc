program test_session_pointer_derived_component_rank234_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    print *, '=== direct session derived component pointer rank-2/3/4 test ==='

    if (.not. expect_output_matches_gfortran( &
        'program main'//new_line('a')// &
        'type item_t'//new_line('a')// &
        'integer :: value'//new_line('a')// &
        'end type item_t'//new_line('a')// &
        'type box_t'//new_line('a')// &
        'integer, allocatable :: values(:,:)'//new_line('a')// &
        'end type box_t'//new_line('a')// &
        'type(box_t), target :: box'//new_line('a')// &
        'integer, pointer :: p(:,:)'//new_line('a')// &
        'integer :: i, j'//new_line('a')// &
        'allocate(box%values(2,3))'//new_line('a')// &
        'do j = 1, 3'//new_line('a')// &
        'do i = 1, 2'//new_line('a')// &
        'box%values(i,j) = 100*i + 10*j'//new_line('a')// &
        'end do'//new_line('a')// &
        'end do'//new_line('a')// &
        'p => box%values'//new_line('a')// &
        'p(2,3) = 999'//new_line('a')// &
        'print *, p(1,1), p(2,3), box%values(2,3)'//new_line('a')// &
        'end program main', 'pointer_derived_component_rank2')) stop 1

    if (.not. expect_output_matches_gfortran( &
        'program main'//new_line('a')// &
        'type item_t'//new_line('a')// &
        'integer :: value'//new_line('a')// &
        'end type item_t'//new_line('a')// &
        'type(item_t), target :: items(2, 2, 2)'//new_line('a')// &
        'integer, pointer :: p(:,:,:)'//new_line('a')// &
        'integer :: i, j, k'//new_line('a')// &
        'do k = 1, 2'//new_line('a')// &
        'do j = 1, 2'//new_line('a')// &
        'do i = 1, 2'//new_line('a')// &
        'items(i,j,k)%value = 100*i + 10*j + k'//new_line('a')// &
        'end do'//new_line('a')// &
        'end do'//new_line('a')// &
        'end do'//new_line('a')// &
        'p => items%value'//new_line('a')// &
        'p(2,2,2) = 888'//new_line('a')// &
        'print *, p(1,1,1), p(2,2,2), items(2,2,2)%value'//new_line('a')// &
        'end program main', 'pointer_derived_component_rank3')) stop 2

    if (.not. expect_output_matches_gfortran( &
        'program main'//new_line('a')// &
        'type item_t'//new_line('a')// &
        'integer :: value'//new_line('a')// &
        'end type item_t'//new_line('a')// &
        'type(item_t), target :: items(2, 1, 2, 2)'//new_line('a')// &
        'integer, pointer :: p(:,:,:,:)'//new_line('a')// &
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
        'p => items%value'//new_line('a')// &
        'p(2,1,2,2) = 777'//new_line('a')// &
        'print *, p(1,1,1,1), p(2,1,2,2), items(2,1,2,2)%value'//new_line('a')// &
        'end program main', 'pointer_derived_component_rank4')) stop 3

    print *, 'PASS: derived array component pointers rank 2/3/4'
end program test_session_pointer_derived_component_rank234_compiler
