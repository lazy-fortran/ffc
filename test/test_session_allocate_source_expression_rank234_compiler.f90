program test_session_allocate_source_expression_rank234_compiler
    ! ALLOCATE(SOURCE=) from fixed-shape rank-2, rank-3, and rank-4
    ! expressions.  The checks in each generated program are independent of
    ! the compiler-under-test output comparison.
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session allocate source expression rank-2/3/4 test ==='
    all_passed = .true.
    if (.not. test_rank2()) all_passed = .false.
    if (.not. test_rank3()) all_passed = .false.
    if (.not. test_rank4()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: higher-rank source expressions preserve shape and values'

contains

    logical function test_rank2()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: src(2,3)'//new_line('a')// &
            '  integer, allocatable :: copy(:,:)'//new_line('a')// &
            '  src(1,1) = 1'//new_line('a')// &
            '  src(2,3) = 6'//new_line('a')// &
            '  allocate(copy, source=src)'//new_line('a')// &
            '  if (size(copy,1) /= 2 .or. size(copy,2) /= 3) error stop 1'// &
            new_line('a')// &
            '  if (copy(1,1) /= 1 .or. copy(2,3) /= 6) error stop 2'// &
            new_line('a')// &
            '  print *, size(copy,1), size(copy,2)'//new_line('a')// &
            '  print *, copy(1,1), copy(2,3)'//new_line('a')// &
            'end program main'

        test_rank2 = expect_output_matches_gfortran(source, &
            'alloc_source_expr_rank2')
    end function test_rank2

    logical function test_rank3()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: src(2,2,2)'//new_line('a')// &
            '  integer, allocatable :: copy(:,:,:)'//new_line('a')// &
            '  src(1,1,1) = 1'//new_line('a')// &
            '  src(2,2,2) = 8'//new_line('a')// &
            '  allocate(copy, source=src)'//new_line('a')// &
            '  if (size(copy,1) /= 2 .or. size(copy,2) /= 2 .or. '// &
            'size(copy,3) /= 2) error stop 3'//new_line('a')// &
            '  if (copy(1,1,1) /= 1 .or. copy(2,2,2) /= 8) error stop 4'// &
            new_line('a')// &
            '  print *, size(copy,1), size(copy,2), size(copy,3)'// &
            new_line('a')// &
            '  print *, copy(1,1,1), copy(2,2,2)'//new_line('a')// &
            'end program main'

        test_rank3 = expect_output_matches_gfortran(source, &
            'alloc_source_expr_rank3')
    end function test_rank3

    logical function test_rank4()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: src(2,2,2,2)'//new_line('a')// &
            '  integer, allocatable :: copy(:,:,:,:)'//new_line('a')// &
            '  src(1,1,1,1) = 1'//new_line('a')// &
            '  src(2,2,2,2) = 8'//new_line('a')// &
            '  allocate(copy, source=src)'//new_line('a')// &
            '  if (size(copy,1) /= 2 .or. size(copy,2) /= 2 .or. '// &
            'size(copy,3) /= 2 .or. size(copy,4) /= 2) error stop 5'// &
            new_line('a')// &
            '  if (copy(1,1,1,1) /= 1 .or. copy(2,2,2,2) /= 8) '// &
            'error stop 6'//new_line('a')// &
            '  print *, size(copy,1), size(copy,2), size(copy,3), '// &
            'size(copy,4)'//new_line('a')// &
            '  print *, copy(1,1,1,1), copy(2,2,2,2)'//new_line('a')// &
            'end program main'

        test_rank4 = expect_output_matches_gfortran(source, &
            'alloc_source_expr_rank4')
    end function test_rank4

end program test_session_allocate_source_expression_rank234_compiler
