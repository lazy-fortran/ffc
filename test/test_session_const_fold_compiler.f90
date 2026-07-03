program test_session_const_fold
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session compile-time constant folding test ==='

    all_passed = .true.
    if (.not. test_iso_c_binding_kind_value()) all_passed = .false.
    if (.not. test_suffixed_literal_kind()) all_passed = .false.
    if (.not. test_min_max_fold()) all_passed = .false.
    if (.not. test_int_fold()) all_passed = .false.
    if (.not. test_huge_fold()) all_passed = .false.
    if (.not. test_precision_range_bit_size_fold()) all_passed = .false.
    if (.not. test_selected_char_kind_fold()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: named parameters and kind/intrinsic queries fold at ' &
        //'compile time'

contains

    ! An ISO_C_BINDING kind name used as a bare value folds to the kind
    ! number gfortran reports for it (c_bool = 1, not the storage width 4
    ! ffc otherwise uses for integer(c_bool) declarations).
    logical function test_iso_c_binding_kind_value()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  use iso_c_binding, only: c_bool'//new_line('a')// &
            '  integer, parameter :: lp = c_bool'//new_line('a')// &
            '  stop lp'//new_line('a')// &
            'end program main'

        test_iso_c_binding_kind_value = expect_exit_status( &
            source, 1, '/tmp/ffc_session_const_fold_c_bool')
    end function test_iso_c_binding_kind_value

    ! kind() of an integer literal with a numeric kind suffix reports that
    ! kind, not the default kind 4.
    logical function test_suffixed_literal_kind()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: k8 = kind(0_8)'//new_line('a')// &
            '  integer, parameter :: k4 = kind(0)'//new_line('a')// &
            '  print *, k8, k4'//new_line('a')// &
            'end program main'

        test_suffixed_literal_kind = expect_output( &
            source, '           8           4'//new_line('a'), &
            '/tmp/ffc_session_const_fold_kind_suffix')
    end function test_suffixed_literal_kind

    ! min/max fold over two or more compile-time integer arguments.
    logical function test_min_max_fold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: lo = min(3, 1, 2)'//new_line('a')// &
            '  integer, parameter :: hi = max(3, 1, 2)'//new_line('a')// &
            '  print *, lo, hi'//new_line('a')// &
            'end program main'

        test_min_max_fold = expect_output( &
            source, '           1           3'//new_line('a'), &
            '/tmp/ffc_session_const_fold_min_max')
    end function test_min_max_fold

    ! int() of an already-integer compile-time expression is a pass-through.
    logical function test_int_fold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: x2 = 7'//new_line('a')// &
            '  integer, parameter :: y2 = int(x2)'//new_line('a')// &
            '  print *, y2'//new_line('a')// &
            'end program main'

        test_int_fold = expect_output( &
            source, '           7'//new_line('a'), &
            '/tmp/ffc_session_const_fold_int')
    end function test_int_fold

    ! huge() folds to the largest representable value of its argument's
    ! kind (kinds 1, 2, and 4 all fit the default-integer parameter that
    ! carries the fold; ffc's integer(8) parameters route through a
    ! separate runtime i64 expression lowerer this compile-time folder
    ! does not reach).
    logical function test_huge_fold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: i1 = huge(0)'//new_line('a')// &
            '  integer, parameter :: ib1 = huge(0_1)'//new_line('a')// &
            '  integer, parameter :: ib2 = huge(0_2)'//new_line('a')// &
            '  print *, i1'//new_line('a')// &
            '  print *, ib1, ib2'//new_line('a')// &
            'end program main'

        test_huge_fold = expect_output( &
            source, &
            '  2147483647'//new_line('a')// &
            '         127       32767'//new_line('a'), &
            '/tmp/ffc_session_const_fold_huge')
    end function test_huge_fold

    ! precision()/range()/bit_size() fold to gfortran's kind-derived values.
    logical function test_precision_range_bit_size_fold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: p = precision(1.0d0)'//new_line('a')// &
            '  integer, parameter :: r = range(0_8)'//new_line('a')// &
            '  integer, parameter :: b = bit_size(1_4)'//new_line('a')// &
            '  print *, p, r, b'//new_line('a')// &
            'end program main'

        test_precision_range_bit_size_fold = expect_output( &
            source, '          15          18          32'//new_line('a'), &
            '/tmp/ffc_session_const_fold_prb')
    end function test_precision_range_bit_size_fold

    ! selected_char_kind() folds to the kind number of a recognized charset
    ! name, or -1 for one it does not recognize.
    logical function test_selected_char_kind_fold()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, parameter :: ascii = '// &
            'selected_char_kind("ascii")'//new_line('a')// &
            '  integer, parameter :: bogus = '// &
            'selected_char_kind("bogus")'//new_line('a')// &
            '  print *, ascii, bogus'//new_line('a')// &
            'end program main'

        test_selected_char_kind_fold = expect_output( &
            source, '           1          -1'//new_line('a'), &
            '/tmp/ffc_session_const_fold_sck')
    end function test_selected_char_kind_fold

end program test_session_const_fold
