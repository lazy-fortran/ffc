program test_session_derived_component_section_rank34_compiler
    use ffc_test_support, only: expect_output, expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session derived component section rank-3/4 test ==='
    all_passed = test_rank3_expected() .and. test_rank4_matches_gfortran()
    if (.not. all_passed) stop 1
    print *, 'PASS: rank-3/rank-4 allocatable derived-component sections'

contains

    logical function test_rank3_expected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j, k'//new_line('a')// &
            '  allocate(box%values(2,3,2))'//new_line('a')// &
            '  box%values(:,:,:) = 7'//new_line('a')// &
            '  do k = 1, 2'//new_line('a')// &
            '    do j = 1, 3'//new_line('a')// &
            '      do i = 1, 2'//new_line('a')// &
            '        if (box%values(i,j,k) /= 7) error stop 2'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, box%values(1,1,1), box%values(2,3,2)'// &
            new_line('a')// &
            'end program main'

        test_rank3_expected = expect_output(source, &
            '           7           7'//new_line('a'), &
            '/tmp/ffc_derived_component_section_rank3')
    end function test_rank3_expected

    logical function test_rank4_matches_gfortran()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer, allocatable :: values(:,:,:,:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t) :: box'//new_line('a')// &
            '  integer :: i, j, k, l'//new_line('a')// &
            '  allocate(box%values(2,2,2,3))'//new_line('a')// &
            '  box%values(:,:,:,:) = 11'//new_line('a')// &
            '  do l = 1, 3'//new_line('a')// &
            '    do k = 1, 2'//new_line('a')// &
            '      do j = 1, 2'//new_line('a')// &
            '        do i = 1, 2'//new_line('a')// &
            '          if (box%values(i,j,k,l) /= 11) error stop 1'//new_line('a')// &
            '        end do'//new_line('a')// &
            '      end do'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, box%values(1,1,1,1), box%values(2,2,2,3), '// &
            'box%values(1,1,1,2)'//new_line('a')// &
            'end program main'

        test_rank4_matches_gfortran = expect_output_matches_gfortran(source, &
            'derived_component_section_rank4')
    end function test_rank4_matches_gfortran

end program test_session_derived_component_section_rank34_compiler
