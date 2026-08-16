program test_session_where_compiler
    use ffc_test_support, only: expect_output, expect_error_contains
    implicit none

    logical :: all_passed
    character(len=1), parameter :: nl = new_line('a')

    print *, '=== direct session WHERE compiler test ==='

    all_passed = .true.
    if (.not. test_where_masked_assignment()) all_passed = .false.
    if (.not. test_where_elsewhere()) all_passed = .false.
    if (.not. test_where_overlapping_section()) all_passed = .false.
    if (.not. test_where_shape_mismatch_diagnostic()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: WHERE lowers through direct LIRIC'

contains

    function i12(values) result(line)
        ! Reproduce ffc/gfortran list-directed integer output: each value
        ! right-justified in a width-12 field, terminated by newline.
        integer, intent(in) :: values(:)
        character(len=:), allocatable :: line
        character(len=12) :: field
        integer :: i

        line = ''
        do i = 1, size(values)
            write (field, '(i12)') values(i)
            line = line//field
        end do
        line = line//nl
    end function i12

    logical function test_where_masked_assignment() result(ok)
        ! WHERE (a > 0) b = a * 10 over a rank-1 integer array. The complement
        ! keeps the pre-set zeros.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5), b(5)'//new_line('a')// &
            '  a = (/1, -2, 3, -4, 5/)'//new_line('a')// &
            '  b = 0'//new_line('a')// &
            '  where (a > 0)'//new_line('a')// &
            '     b = a * 10'//new_line('a')// &
            '  end where'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        ok = expect_output(source, i12([10, 0, 30, 0, 50]), &
            '/tmp/ffc_where_masked_test')
    end function test_where_masked_assignment

    logical function test_where_elsewhere() result(ok)
        ! WHERE (a > 2) c = a * 10 ELSEWHERE c = a. The elsewhere branch covers
        ! the complement of the mask over the same target.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), c(4)'//new_line('a')// &
            '  a = (/1, 2, 3, 4/)'//new_line('a')// &
            '  c = 0'//new_line('a')// &
            '  where (a > 2)'//new_line('a')// &
            '     c = a * 10'//new_line('a')// &
            '  elsewhere'//new_line('a')// &
            '     c = a'//new_line('a')// &
            '  end where'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end program main'

        ok = expect_output(source, i12([1, 2, 30, 40]), &
            '/tmp/ffc_where_elsewhere_test')
    end function test_where_elsewhere

    logical function test_where_overlapping_section() result(ok)
        ! WHERE whose right-hand side is a reversed section of the target
        ! itself. Fortran 2018 clause 10.2.3.1 evaluates the mask and the
        ! right-hand side before any element of the target is assigned, so
        ! every element must be taken from the pre-assignment array. Expected
        ! values are the gfortran output for this program.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), m(4)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  m = [1, 0, 1, 1]'//new_line('a')// &
            '  where (m == 1) a = a(4:1:-1)'//new_line('a')// &
            '  print *, a'//new_line('a')// &
            'end program main'

        ok = expect_output(source, i12([4, 2, 2, 1]), &
            '/tmp/ffc_where_overlapping_section_test')
    end function test_where_overlapping_section

    logical function test_where_shape_mismatch_diagnostic() result(ok)
        ! A WHERE mask must conform to the target array: it is evaluated at
        ! the same linear elements. A mask of a different extent is a
        ! conformability error and must be rejected at lowering time with a
        ! diagnostic, not silently read out of bounds.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), m(3)'//new_line('a')// &
            '  m = 1'//new_line('a')// &
            '  where (m == 1) a = 5'//new_line('a')// &
            'end program main'

        ok = expect_error_contains(source, 'conforming shapes', &
            '/tmp/ffc_where_shape_mismatch_test')
    end function test_where_shape_mismatch_diagnostic

end program test_session_where_compiler

