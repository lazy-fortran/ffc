program test_session_array_section_compiler
    use ffc_test_support, only: expect_exit_status, expect_output, &
                                expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    print *, '=== direct session array section compiler test ==='

    all_passed = .true.
    if (.not. test_print_section()) all_passed = .false.
    if (.not. test_copy_section()) all_passed = .false.
    if (.not. test_whole_array_copy()) all_passed = .false.
    if (.not. test_elementwise_sections()) all_passed = .false.
    if (.not. test_sum_section()) all_passed = .false.
    if (.not. test_section_after_string()) all_passed = .false.
    if (.not. test_empty_section()) all_passed = .false.
    if (.not. test_runtime_scalar_section_dispatch()) all_passed = .false.
    if (.not. test_runtime_scalar_section_ranks()) all_passed = .false.
    if (.not. test_fixed_rank3_assignment_matches_gfortran()) all_passed = .false.

    if (.not. all_passed) stop 1

    print *, 'PASS: array sections lower through direct LIRIC session'

contains

    logical function test_print_section()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1:4)'//new_line('a')// &
            '  integer :: b(0:1, 2:3)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  b = [5, 6, 7, 8]'//new_line('a')// &
            '  print *, a(2:3)'//new_line('a')// &
            '  print *, b(0:1, 3:3)'//new_line('a')// &
            'end program main'

        test_print_section = expect_output( &
            source, '           2           3'//new_line('a')// &
            '           7           8'//new_line('a'), &
            '/tmp/ffc_session_array_section_print_test')
    end function test_print_section

    logical function test_copy_section()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(1:4)'//new_line('a')// &
            '  integer :: c(2)'//new_line('a')// &
            '  integer :: b(0:1, 2:3)'//new_line('a')// &
            '  integer :: d(0:1, 3:3)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  b = [5, 6, 7, 8]'//new_line('a')// &
            '  c = a(2:3)'//new_line('a')// &
            '  d = b(0:1, 3:3)'//new_line('a')// &
            '  print *, c(1) + c(2)'//new_line('a')// &
            '  print *, d(0, 3) + d(1, 3)'//new_line('a')// &
            'end program main'

        test_copy_section = expect_output( &
            source, '           5'//new_line('a')// &
            '          15'//new_line('a'), &
            '/tmp/ffc_session_array_section_copy_test')
    end function test_copy_section

    logical function test_whole_array_copy()
        ! b = a(2:4): whole-array assignment from a rank-1 integer section.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), b(3)'//new_line('a')// &
            '  a = [1, 2, 3, 4]'//new_line('a')// &
            '  b = a(2:4)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_whole_array_copy = expect_output( &
            source, '           2           3           4'//new_line('a'), &
            '/tmp/ffc_session_array_section_whole_test')
    end function test_whole_array_copy

    logical function test_elementwise_sections()
        ! c = a(1:3) + d(2:4): elementwise op between two conformable sections.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(4), d(4), c(3)'//new_line('a')// &
            '  a = [10, 20, 30, 40]'//new_line('a')// &
            '  d = [1, 2, 3, 4]'//new_line('a')// &
            '  c = a(1:3) + d(2:4)'//new_line('a')// &
            '  print *, c'//new_line('a')// &
            'end program main'

        test_elementwise_sections = expect_output( &
            source, '          12          23          34'//new_line('a'), &
            '/tmp/ffc_session_array_section_elem_test')
    end function test_elementwise_sections

    logical function test_sum_section()
        ! sum(a(lo:hi)) reduces over the section extent only.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(6)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5, 6]'//new_line('a')// &
            '  print *, sum(a(2:5))'//new_line('a')// &
            'end program main'

        test_sum_section = expect_output( &
            source, '          14'//new_line('a'), &
            '/tmp/ffc_session_array_section_sum_test')
    end function test_sum_section

    logical function test_section_after_string()
        ! 'tag', a(lo:hi): a section among other list-directed print items.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  a = [1, 2, 3, 4, 5]'//new_line('a')// &
            "  print *, 'vals:', a(2:4)"//new_line('a')// &
            'end program main'

        test_section_after_string = expect_output( &
            source, ' vals:           2           3           4'//new_line('a'), &
            '/tmp/ffc_session_array_section_after_string_test')
    end function test_section_after_string

    logical function test_empty_section()
        ! A positive-stride section with an upper bound below its lower bound
        ! has zero elements and is valid; it must still compile and print only
        ! the record terminator.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(10)'//new_line('a')// &
            '  print *, a(15:14)'//new_line('a')// &
            'end program main'

        test_empty_section = expect_output( &
            source, new_line('a'), '/tmp/ffc_session_empty_array_section_test')
    end function test_empty_section

    logical function test_runtime_scalar_section_dispatch()
        ! FortFront may retain a runtime-bounded section as a call_or_subscript
        ! node. Keep a scalar element write in front of it to prove that the
        ! scalar-subscript path still wins for a(i).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call work(5)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(l)'//new_line('a')// &
            '    integer, intent(in) :: l'//new_line('a')// &
            '    real :: fvec(l)'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '    fvec = 1.0'//new_line('a')// &
            '    i = 1'//new_line('a')// &
            '    fvec(i) = 2.0'//new_line('a')// &
            '    fvec(2:l) = 0.0'//new_line('a')// &
            '    if (fvec(1) /= 2.0) error stop 1'//new_line('a')// &
            '    if (fvec(2) /= 0.0 .or. fvec(l) /= 0.0) error stop 2'// &
            new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_runtime_scalar_section_dispatch = expect_exit_status( &
            source, 0, '/tmp/ffc_session_runtime_scalar_section_dispatch_test')
    end function test_runtime_scalar_section_dispatch

    logical function test_runtime_scalar_section_ranks()
        ! Exercise scalar RHS broadcast through runtime sections of ranks one
        ! through four.  The printed sums are the independent oracle: they
        ! distinguish each selected section from an accidental whole-array
        ! write and also verify scalar subscripts remain fixed coordinates.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call work(4, 3, 2, 2)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine work(l, m, n, o)'//new_line('a')// &
            '    integer, intent(in) :: l, m, n, o'//new_line('a')// &
            '    integer :: i, j, k'//new_line('a')// &
            '    integer :: fvec(l), fvec2d(l, m), fvec3d(l, m, n)'// &
            new_line('a')// &
            '    integer :: fvec4d(l, m, n, o)'//new_line('a')// &
            '    i = 2'//new_line('a')// &
            '    j = 1'//new_line('a')// &
            '    k = 2'//new_line('a')// &
            '    fvec = 9'//new_line('a')// &
            '    fvec(1) = 7'//new_line('a')// &
            '    fvec(2:l) = 0'//new_line('a')// &
            '    fvec2d = 1'//new_line('a')// &
            '    fvec2d(:, k) = 2'//new_line('a')// &
            '    fvec3d = 1'//new_line('a')// &
            '    fvec3d(:, k, :i) = 3'//new_line('a')// &
            '    fvec4d = 1'//new_line('a')// &
            '    fvec4d(:, k, :i, j) = 4'//new_line('a')// &
            '    print *, fvec(1) + fvec(2) + fvec(l)'//new_line('a')// &
            '    print *, fvec2d(1, 1) + fvec2d(1, k) + fvec2d(l, m)'// &
            new_line('a')// &
            '    print *, fvec3d(1, 1, 1) + fvec3d(1, k, 1)'// &
            ' + fvec3d(1, k, i) + fvec3d(l, m, n)'//new_line('a')// &
            '    print *, fvec4d(1, 1, 1, 1) + fvec4d(1, k, 1, j)'// &
            ' + fvec4d(1, k, i, j) + fvec4d(1, k, 1, o)'//new_line('a')// &
            '  end subroutine work'//new_line('a')// &
            'end program main'

        test_runtime_scalar_section_ranks = expect_output( &
            source, '           7'//new_line('a')// &
            '           4'//new_line('a')// &
            '           8'//new_line('a')// &
            '          10'//new_line('a'), &
            '/tmp/ffc_session_runtime_scalar_section_ranks_test')
    end function test_runtime_scalar_section_ranks

    logical function test_fixed_rank3_assignment_matches_gfortran()
        ! Compare fixed-size rank-3 scalar broadcast and conformable section copy
        ! against gfortran's independent behavioral oracle.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(2, 2, 2), b(2, 2, 2)'//new_line('a')// &
            '  a = 0'//new_line('a')// &
            '  a(:, :, :) = 7'//new_line('a')// &
            '  b = 0'//new_line('a')// &
            '  b(:, :, :) = a(:, :, :)'//new_line('a')// &
            '  print *, sum(b)'//new_line('a')// &
            'end program main'

        test_fixed_rank3_assignment_matches_gfortran = &
            expect_output_matches_gfortran(source, 'array_section_fixed_rank3')
    end function test_fixed_rank3_assignment_matches_gfortran
end program test_session_array_section_compiler
