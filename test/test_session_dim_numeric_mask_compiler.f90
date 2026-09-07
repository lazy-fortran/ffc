program test_session_dim_numeric_mask_compiler
    use ffc_test_support, only: expect_output_matches_gfortran
    implicit none

    logical :: all_passed

    all_passed = .true.
    if (.not. check_mask('integer', 'a > 0', 'integer_self')) all_passed = .false.
    if (.not. check_mask('real', 'a > 0', 'real_self')) all_passed = .false.
    if (.not. check_mask('real(8)', 'a > 0', 'real8_self')) all_passed = .false.
    if (.not. check_mask('integer', 'b > 0', 'integer_other')) all_passed = .false.
    if (.not. check_mask('real', 'b > 0', 'real_other')) all_passed = .false.
    if (.not. check_mask('real(8)', 'b > 0', 'real8_other')) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: masked dimension reductions retain source and identities'

contains

    logical function check_mask(type_spec, mask, stem) result(ok)
        character(len=*), intent(in) :: type_spec, mask, stem
        character(len=:), allocatable :: source, zero

        ! False entries follow selected entries and the final column is empty.
        ! The separate integer mask also checks differing source/mask kinds.
        zero = '0'
        if (type_spec == 'real') zero = '0.0'
        if (type_spec == 'real(8)') zero = '0.0_8'
        source = &
            'program main'//new_line('a')// &
            '  '//type_spec//' :: a(3,3), s1(3), s2(3), p1(3), p2(3)'// &
            new_line('a')// &
            '  integer :: b(3,3), lo(3), hi(3)'//new_line('a')// &
            '  a = reshape([2,-3,5,-7,11,-13,-17,-19,-23], [3,3])'// &
            new_line('a')// &
            '  b = reshape([1,0,1,0,1,0,0,0,0], [3,3])'//new_line('a')// &
            '  s1 = sum(a, 1, mask='//mask//')'//new_line('a')// &
            '  s2 = sum(a, 2, mask='//mask//')'//new_line('a')// &
            '  p1 = product(a, 1, mask='//mask//')'//new_line('a')// &
            '  p2 = product(a, 2, mask='//mask//')'//new_line('a')// &
            '  if (any(s1 /= [7,11,0])) error stop 1'//new_line('a')// &
            '  if (any(s2 /= [2,11,5])) error stop 2'//new_line('a')// &
            '  if (any(p1 /= [10,11,1])) error stop 3'//new_line('a')// &
            '  if (any(p2 /= [2,11,5])) error stop 4'//new_line('a')// &
            '  lo = minloc(a, 1, mask='//mask//')'//new_line('a')// &
            '  hi = maxloc(a, 1, mask='//mask//')'//new_line('a')// &
            '  if (any(lo /= [1,2,0])) error stop 5'//new_line('a')// &
            '  if (any(hi /= [3,2,0])) error stop 6'//new_line('a')// &
            '  s1 = minval(a, 1, mask='//mask//')'//new_line('a')// &
            '  p1 = maxval(a, 1, mask='//mask//')'//new_line('a')// &
            '  p2 = maxval(a, 2, mask='//mask//')'//new_line('a')// &
            '  if (s1(1) /= 2 .or. s1(2) /= 11) error stop 7'//new_line('a')// &
            '  if (any(p2 /= [2,11,5])) error stop 8'//new_line('a')// &
            '  if (s1(3) /= huge('//zero//')) error stop 9'//new_line('a')
        if (type_spec == 'integer') then
            source = source// &
                '  print *, p1(3)'//new_line('a')// &
                '  a = -huge(0) - 1'//new_line('a')// &
                '  p1 = maxval(a, 1, mask=b >= 0)'//new_line('a')// &
                '  if (any(p1 /= -huge(0) - 1)) error stop 10'//new_line('a')
        else
            source = source// &
                '  if (p1(3) /= -huge('//zero//')) error stop 10'//new_line('a')
        end if
        source = source// &
            '  print *, "ok"'//new_line('a')// &
            'end program main'
        ok = expect_output_matches_gfortran(source, 'dim_numeric_mask_'//stem)
    end function check_mask

end program test_session_dim_numeric_mask_compiler
