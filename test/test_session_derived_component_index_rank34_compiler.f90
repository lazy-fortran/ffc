program test_session_derived_component_index_rank34_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== derived component rank-3/rank-4 index and shape test ==='
    all_passed = test_rank3_index_and_shape() .and. test_rank4_matches_gfortran()
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3/rank-4 derived component index and shape helpers'

contains

    logical function test_rank3_index_and_shape()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  allocate(box%values(2,3,4))'//new_line('a')// &
            '  if (size(box%values) /= 24) error stop 2'//new_line('a')// &
            '  if (size(box%values,1) /= 2 .or. size(box%values,2) /= 3) '// &
            'error stop 3'//new_line('a')// &
            '  if (size(box%values,3) /= 4) error stop 4'//new_line('a')// &
            '  box%values(:,:,:) = 0'//new_line('a')// &
            '  box%values(2,3,4) = 234'//new_line('a')// &
            '  box%values(1,2,3) = 123'//new_line('a')// &
            '  if (box%values(2,3,4) /= 234) error stop 5'//new_line('a')// &
            '  if (box%values(1,2,3) /= 123) error stop 6'//new_line('a')// &
            '  do k = 1, 4'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        if (box%values(i,j,k) /= 0 .and. '// &
            'box%values(i,j,k) /= 123 .and. box%values(i,j,k) /= 234) '// &
            'error stop 7'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, box%values(2,3,4), size(box%values), size(box%values,3)'// &
            new_line('a')// &
            'end program main'

        test_rank3_index_and_shape = expect_output(source, &
            '         234          24           4'//new_line('a'), &
            '/tmp/ffc_derived_component_index_rank3')
    end function test_rank3_index_and_shape

    logical function test_rank4_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  allocate(box%values(2,2,3,4))'//new_line('a')// &
            '  box%values(:,:,:,:) = 17'//new_line('a')// &
            '  box%values(2,2,3,4) = 474'//new_line('a')// &
            '  if (size(box%values) /= 48) error stop 1'//new_line('a')// &
            '  if (size(box%values,1) /= 2 .or. size(box%values,2) /= 2) '// &
            'error stop 2'//new_line('a')// &
            '  if (size(box%values,3) /= 3 .or. size(box%values,4) /= 4) '// &
            'error stop 3'//new_line('a')// &
            '  print *, box%values(2,2,3,4), size(box%values), '// &
            'size(box%values,4)'//new_line('a')// &
            'end program main'

        test_rank4_matches_gfortran = expect_output_matches_gfortran(source, &
            'derived_component_index_rank4')
    end function test_rank4_matches_gfortran

end program test_session_derived_component_index_rank34_compiler
