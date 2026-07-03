program test_session_const_fold_intrinsics
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session compile-time integer intrinsic fold '// &
        'compiler test ==='

    all_passed = .true.
    if (.not. test_arith_bound_folds()) all_passed = .false.
    if (.not. test_bit_op_bound_folds()) all_passed = .false.
    if (.not. test_ibits_scalar_kind8_fold()) all_passed = .false.
    if (.not. test_merge_scalar_mask_bound()) all_passed = .false.
    if (.not. test_kind_inquiry_folds()) all_passed = .false.
    if (.not. test_param_array_index_bound()) all_passed = .false.
    if (.not. test_product_dot_product_folds()) all_passed = .false.
    if (.not. test_array_reduction_folds()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: compile-time integer expressions fold mod/modulo/abs/'// &
        'sign/dim, bit intrinsics, merge, kind inquiries, parameter-array '// &
        'indexing, product/dot_product, and sum/maxval/minval'

contains

    logical function test_arith_bound_folds()
        ! mod, modulo, abs, sign, and dim fold as array-bound expressions.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(mod(17, 5))'//new_line('a')// &
            'integer :: b(modulo(-1, 5))'//new_line('a')// &
            'integer :: c(abs(-3))'//new_line('a')// &
            'integer :: d(sign(3, -7) + 8)'//new_line('a')// &
            'integer :: e(dim(9, 4))'//new_line('a')// &
            'print *, size(a), size(b), size(c), size(d), size(e)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2           4           3           5           5'// &
            new_line('a')

        test_arith_bound_folds = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_arith')
    end function test_arith_bound_folds

    logical function test_bit_op_bound_folds()
        ! iand, ior, ieor/xor, not, ishft, ishftc, ibset, and ibclr fold as
        ! array-bound expressions.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer :: a(iand(12, 10))'//new_line('a')// &
            'integer :: b(ior(1, 2))'//new_line('a')// &
            'integer :: c(ieor(6, 3))'//new_line('a')// &
            'integer :: d(xor(6, 2))'//new_line('a')// &
            'integer :: e(not(-9))'//new_line('a')// &
            'integer :: f(ishft(1, 3))'//new_line('a')// &
            'integer :: g(ibset(0, 2))'//new_line('a')// &
            'integer :: h(ibclr(7, 1))'//new_line('a')// &
            'print *, size(a), size(b), size(c), size(d)'//new_line('a')// &
            'print *, size(e), size(f), size(g), size(h)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           8           3           5           4'//new_line('a')// &
            '           8           8           4           5'//new_line('a')

        test_bit_op_bound_folds = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_bits')
    end function test_bit_op_bound_folds

    logical function test_ibits_scalar_kind8_fold()
        ! ibits(i, pos, len) folds a scalar PARAMETER, including an
        ! integer(8) target whose initializer bypasses the runtime i64
        ! expression lowerer entirely.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: i1 = ibits(10, 2, 2)'//new_line('a')// &
            'integer(8), parameter :: i2 = ibits(10_8, 2, 2)'//new_line('a')// &
            'print *, i1, i2'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           2                    2'//new_line('a')

        test_ibits_scalar_kind8_fold = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_ibits')
    end function test_ibits_scalar_kind8_fold

    logical function test_merge_scalar_mask_bound()
        ! merge(tsource, fsource, mask) with a comparison mask folds as an
        ! array-bound expression (the pattern behind a size()-selecting
        ! local automatic array).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: dim = 2'//new_line('a')// &
            'integer :: a(merge(3, 7, 1 < dim))'//new_line('a')// &
            'integer :: b(merge(3, 7, 1 > dim))'//new_line('a')// &
            'print *, size(a), size(b)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3           7'//new_line('a')

        test_merge_scalar_mask_bound = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_merge')
    end function test_merge_scalar_mask_bound

    logical function test_kind_inquiry_folds()
        ! digits, radix, minexponent, maxexponent, and selected_logical_kind
        ! fold as PARAMETER initializers.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'real :: x'//new_line('a')// &
            'integer, parameter :: d = digits(x)'//new_line('a')// &
            'integer, parameter :: r = radix(x)'//new_line('a')// &
            'integer, parameter :: mn = minexponent(x)'//new_line('a')// &
            'integer, parameter :: mx = maxexponent(x)'//new_line('a')// &
            'integer, parameter :: lk = selected_logical_kind(8)'//new_line('a')// &
            'print *, d, r, mn, mx, lk'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          24           2        -125         128           4'// &
            new_line('a')

        test_kind_inquiry_folds = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_kind')
    end function test_kind_inquiry_folds

    logical function test_param_array_index_bound()
        ! An integer PARAMETER array indexed by a compile-time constant
        ! folds as an array-bound expression.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: dims(2) = [3, 4]'//new_line('a')// &
            'real :: a(dims(1)), b(dims(2))'//new_line('a')// &
            'print *, size(a), size(b)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           3           4'//new_line('a')

        test_param_array_index_bound = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_param_arr')
    end function test_param_array_index_bound

    logical function test_product_dot_product_folds()
        ! product() and dot_product() over a compile-time integer array
        ! constructor, and over a named integer PARAMETER array, fold as
        ! array-bound expressions.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: v(3) = [1, 2, 3]'//new_line('a')// &
            'integer :: a(product([2, 3, 4]))'//new_line('a')// &
            'integer :: b(product(v))'//new_line('a')// &
            'integer :: c(dot_product(v, [1, 1, 1]))'//new_line('a')// &
            'print *, size(a), size(b), size(c)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '          24           6           6'//new_line('a')

        test_product_dot_product_folds = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_product')
    end function test_product_dot_product_folds

    logical function test_array_reduction_folds()
        ! sum(), maxval(), and minval() over a compile-time integer array
        ! constructor or named integer PARAMETER array fold as array-bound
        ! expressions.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'integer, parameter :: v(3) = [5, 1, 3]'//new_line('a')// &
            'integer :: a(sum(v))'//new_line('a')// &
            'integer :: b(maxval(v))'//new_line('a')// &
            'integer :: c(minval([4, 9, 2]))'//new_line('a')// &
            'print *, size(a), size(b), size(c)'//new_line('a')// &
            'end program main'
        character(len=*), parameter :: expected = &
            '           9           5           2'//new_line('a')

        test_array_reduction_folds = expect_output(source, expected, &
            '/tmp/ffc_session_const_fold_reduction')
    end function test_array_reduction_folds

end program test_session_const_fold_intrinsics
