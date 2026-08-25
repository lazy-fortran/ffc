program test_session_derived_element_rank234_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session derived element rank-2/3/4 test ==='
    all_passed = test_rank2_expected()
    if (.not. test_rank3_matches_gfortran()) all_passed = .false.
    if (.not. test_rank4_matches_gfortran()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: derived allocatable component element read/write rank 2/3/4'

contains

    logical function test_rank2_expected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j'//new_line('a')// &
            '  allocate(box%values(2,3))'//new_line('a')// &
            '  do j = 1, 3'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      box%values(i,j) = 100*i + 10*j'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  box%values(2,3) = 999'//new_line('a')// &
            '  print *, box%values(1,1), box%values(2,3), '// &
            'box%values(2,1)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '         110         999         210'//new_line('a')

        test_rank2_expected = expect_output(source, expected, &
            '/tmp/ffc_session_derived_element_rank2')
    end function test_rank2_expected

    logical function test_rank3_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  allocate(box%values(2,2,2))'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '    do j = 1, 2'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        box%values(i,j,k) = 100*i + 10*j + k'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  box%values(2,2,2) = 888'//new_line('a')// &
            '  print *, box%values(1,1,1), box%values(2,2,2), '// &
            'box%values(2,1,2)'//new_line('a')// &
            'end program main'

        test_rank3_matches_gfortran = expect_output_matches_gfortran(source, &
            'derived_element_rank3')
    end function test_rank3_matches_gfortran

    logical function test_rank4_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j, k, l'//new_line('a')// &
            '  allocate(box%values(2,1,2,2))'//new_line('a')// &
            '  do l = 1, 2'//new_line('a')// &
            '    do k = 1, 2'//new_line('a')// &
            '      do j = 1, 1'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '          box%values(i,j,k,l) = 1000*i + 100*j + 10*k + l'// &
            new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  box%values(2,1,2,2) = 777'//new_line('a')// &
            '  print *, box%values(1,1,1,1), box%values(2,1,2,2), '// &
            'box%values(2,1,1,2)'//new_line('a')// &
            'end program main'

        test_rank4_matches_gfortran = expect_output_matches_gfortran(source, &
            'derived_element_rank4')
    end function test_rank4_matches_gfortran

end program test_session_derived_element_rank234_compiler
