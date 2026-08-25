program test_session_allocate_mold_source_rank34_compiler
    ! Rank-three and rank-four ALLOCATE(MOLD=)/ALLOCATE(SOURCE=) regression
    ! coverage.  The explicit checks are the independent behavioral oracle;
    ! gfortran comparison also checks the complete observable output.
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session allocate mold/source rank-3/4 test ==='
    all_passed = test_rank3() .and. test_rank4()
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3/rank-4 mold=/source= shape and values'

contains

    logical function test_rank3()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: src(:,:,:), shape_target(:,:,:), copy(:,:,:)'// &
            new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  allocate(src(2,3,2))'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        src(i,j,k) = 100*i + 10*j + k'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  allocate(shape_target, mold=src)'//new_line('a')// &
            '  allocate(copy, source=src)'//new_line('a')// &
            '  if (size(shape_target) /= 12 .or. size(shape_target,1) /= 2) error stop 1'// &
            new_line('a')// &
            '  if (size(shape_target,2) /= 3 .or. size(shape_target,3) /= 2) error stop 2'// &
            new_line('a')// &
            '  if (copy(1,1,1) /= 111 .or. copy(2,3,2) /= 232) error stop 3'// &
            new_line('a')// &
            '  print *, size(shape_target,1), size(shape_target,2), '// &
            'size(shape_target,3)'//new_line('a')// &
            '  print *, copy(1,1,1), copy(2,3,2)'//new_line('a')// &
            'end program main'

        test_rank3 = expect_output_matches_gfortran(source, &
            'alloc_mold_source_rank34_rank3')
    end function test_rank3

    logical function test_rank4()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: src(:,:,:,:), shape_target(:,:,:,:), '// &
            'copy(:,:,:,:)'// &
            new_line('a')// &
            '  integer :: i, j, k, l'//new_line('a')// &
            '  allocate(src(2,2,2,3))'//new_line('a')// &
            '  do l = 1, 3'//new_line('a')// &
            '    do k = 1, 2'//new_line('a')// &
            '      do j = 1, 2'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '          src(i,j,k,l) = 1000*i + 100*j + 10*k + l'// &
            new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  allocate(shape_target, mold=src)'//new_line('a')// &
            '  allocate(copy, source=src)'//new_line('a')// &
            '  if (size(shape_target) /= 24 .or. size(shape_target,1) /= 2) '// &
            'error stop 4'//new_line('a')// &
            '  if (size(shape_target,2) /= 2 .or. size(shape_target,3) /= 2) '// &
            'error stop 5'//new_line('a')// &
            '  if (size(shape_target,4) /= 3) error stop 6'//new_line('a')// &
            '  if (copy(1,1,1,1) /= 1111 .or. copy(2,2,2,3) /= 2223) '// &
            'error stop 7'//new_line('a')// &
            '  print *, size(shape_target,1), size(shape_target,2), '// &
            'size(shape_target,3), size(shape_target,4)'// &
            new_line('a')// &
            '  print *, copy(1,1,1,1), copy(2,2,2,3)'//new_line('a')// &
            'end program main'

        test_rank4 = expect_output_matches_gfortran(source, &
            'alloc_mold_source_rank34_rank4')
    end function test_rank4

end program test_session_allocate_mold_source_rank34_compiler
