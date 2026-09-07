program test_session_integer_call_real_operand_compiler
    use ffc_test_support, only: expect_output_matches_gfortran, expect_error_contains
    use conformance_temp_dir, only: make_temp_root, remove_temp_root
    implicit none

    character(len=*), parameter :: contained_source = &
        'program main'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  integer :: calls'//new_line('a')// &
        '  real :: single_value'//new_line('a')// &
        '  real(8) :: double_value'//new_line('a')// &
        '  calls = 0'//new_line('a')// &
        '  double_value = 2.0d0*floor(-1.5d0)'//new_line('a')// &
        '  single_value = 0.5*shift(6)'//new_line('a')// &
        '  if (calls /= 2) stop 81'//new_line('a')// &
        '  if (double_value /= 14.0d0) stop 82'//new_line('a')// &
        '  if (single_value /= 5.0) stop 83'//new_line('a')// &
        '  print *, calls'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function floor(x)'//new_line('a')// &
        '    real(8), intent(in) :: x'//new_line('a')// &
        '    calls = calls + 1'//new_line('a')// &
        '    floor = int(x) + 8'//new_line('a')// &
        '  end function floor'//new_line('a')// &
        '  integer function shift(n)'//new_line('a')// &
        '    integer, intent(in) :: n'//new_line('a')// &
        '    calls = calls + 1'//new_line('a')// &
        '    shift = n + 4'//new_line('a')// &
        '  end function shift'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: generic_source = &
        'module choices'//new_line('a')// &
        '  interface choose'//new_line('a')// &
        '    module procedure choose, choose8'//new_line('a')// &
        '  end interface'//new_line('a')// &
        'contains'//new_line('a')// &
        '  integer function choose(x)'//new_line('a')// &
        '    real, intent(in) :: x'//new_line('a')// &
        '    choose = int(x) + 10'//new_line('a')// &
        '  end function choose'//new_line('a')// &
        '  integer function choose8(x)'//new_line('a')// &
        '    real(8), intent(in) :: x'//new_line('a')// &
        '    choose8 = int(x) + 20'//new_line('a')// &
        '  end function choose8'//new_line('a')// &
        'end module choices'//new_line('a')// &
        'program main'//new_line('a')// &
        '  use choices'//new_line('a')// &
        '  implicit none'//new_line('a')// &
        '  real(8) :: value'//new_line('a')// &
        '  value = 0.5d0*(choose(2.0d0) + choose(2.0))'//new_line('a')// &
        '  if (value /= 17.0d0) stop 81'//new_line('a')// &
        '  print *, int(value)'//new_line('a')// &
        'end program main'

    character(len=*), parameter :: array_shadow_source = &
        'program main'//new_line('a')// &
        ' implicit none'//new_line('a')// &
        ' real :: result'//new_line('a')// &
        ' call test(result)'//new_line('a')// &
        ' print *, result'//new_line('a')// &
        ' if (result /= 5.0) stop 91'//new_line('a')// &
        'contains'//new_line('a')// &
        ' integer function item(n)'//new_line('a')// &
        '  integer, intent(in) :: n'//new_line('a')// &
        '  item = n + 100'//new_line('a')// &
        ' end function'//new_line('a')// &
        ' subroutine test(result)'//new_line('a')// &
        '  real, intent(out) :: result'//new_line('a')// &
        '  real :: item(2)'//new_line('a')// &
        '  item = [1.25,2.5]'//new_line('a')// &
        '  result = 2.0*item(2)'//new_line('a')// &
        ' end subroutine'//new_line('a')// &
        'end program'
    character(len=:), allocatable :: root

    if (.not. expect_output_matches_gfortran(contained_source, &
        'integer_call_real_operand')) stop 1
    if (.not. expect_output_matches_gfortran(generic_source, &
        'integer_generic_real_operand')) stop 2
    ! This local REAL array shadows a host INTEGER function. The existing
    ! real-array call path refuses it; integer inference must not turn that
    ! refusal into an executable that interprets the REAL storage as INTEGER.
    root = make_temp_root('integer_call_array_shadow')
    if (.not. expect_error_contains(array_shadow_source, &
        'scalar real(4) function call', root//'/case')) then
        call remove_temp_root(root)
        stop 3
    end if
    call remove_temp_root(root)
    print *, 'PASS: integer call operands preserve their signatures in real arithmetic'
end program test_session_integer_call_real_operand_compiler
