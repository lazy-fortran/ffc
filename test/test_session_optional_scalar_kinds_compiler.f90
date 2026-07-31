program test_session_optional_scalar_kinds_compiler
    ! OPTIONAL scalar dummies of every supported kind share one presence ABI:
    ! the dummy's reference pointer is null when absent, and an absent dummy is
    ! never dereferenced -- not even by the procedure prologue that binds it.
    use ffc_test_support, only: expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== direct session optional scalar kinds compiler test ==='

    all_passed = .true.
    if (.not. test_integer_kinds()) all_passed = .false.
    if (.not. test_real_kinds()) all_passed = .false.
    if (.not. test_logical_kind()) all_passed = .false.
    if (.not. test_complex_kind()) all_passed = .false.
    if (.not. test_character_kind()) all_passed = .false.
    if (.not. test_derived_kind()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: optional scalar dummies lower across kinds'

contains

    logical function run_case(body, expected, tag)
        character(len=*), intent(in) :: body
        integer, intent(in) :: expected
        character(len=*), intent(in) :: tag

        run_case = expect_exit_status(body, expected, &
                                      '/tmp/ffc_optional_kind_'//tag)
    end function run_case

    logical function test_integer_kinds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  call sub(total, 7)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    integer, optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      total = total + x'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 20'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_integer_kinds = run_case(source, 27, 'integer')
    end function test_integer_kinds

    logical function test_real_kinds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  double precision :: d'//new_line('a')// &
            '  real :: r'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  r = 2.5'//new_line('a')// &
            '  d = 4.5d0'//new_line('a')// &
            '  call sub(total, r, d)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x, y)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    real, optional, intent(in) :: x'//new_line('a')// &
            '    double precision, optional, intent(in) :: y'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      total = total + int(x * 2.0)'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 1'//new_line('a')// &
            '    end if'//new_line('a')// &
            '    if (present(y)) then'//new_line('a')// &
            '      total = total + int(y * 2.0d0)'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 2'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_real_kinds = run_case(source, 17, 'real')
    end function test_real_kinds

    logical function test_logical_kind()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  logical :: flag'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  flag = .true.'//new_line('a')// &
            '  call sub(total, flag)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    logical, optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      if (x) total = total + 5'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 30'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_logical_kind = run_case(source, 35, 'logical')
    end function test_logical_kind

    logical function test_complex_kind()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  complex :: z'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  z = (3.0, 4.0)'//new_line('a')// &
            '  call sub(total, z)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    complex, optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      total = total + int(real(x)) + int(aimag(x))'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 40'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_complex_kind = run_case(source, 47, 'complex')
    end function test_complex_kind

    logical function test_character_kind()
        ! The prologue of a fixed-length character dummy must not load the
        ! caller's descriptor eagerly: with the dummy absent that pointer is
        ! null, so an eager load faults before present() is ever evaluated.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  character(len=2) :: c'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  c = "ab"'//new_line('a')// &
            '  call sub(total, c)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    character(len=2), optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      if (x == "ab") total = total + 6'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 50'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_character_kind = run_case(source, 56, 'character')
    end function test_character_kind

    logical function test_derived_kind()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type point_t'//new_line('a')// &
            '  type(point_t) :: p'//new_line('a')// &
            '  integer :: total'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  p%a = 8'//new_line('a')// &
            '  call sub(total, p)'//new_line('a')// &
            '  call sub(total)'//new_line('a')// &
            '  stop total'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub(total, x)'//new_line('a')// &
            '    integer, intent(inout) :: total'//new_line('a')// &
            '    type(point_t), optional, intent(in) :: x'//new_line('a')// &
            '    if (present(x)) then'//new_line('a')// &
            '      total = total + x%a'//new_line('a')// &
            '    else'//new_line('a')// &
            '      total = total + 60'//new_line('a')// &
            '    end if'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_derived_kind = run_case(source, 68, 'derived')
    end function test_derived_kind

end program test_session_optional_scalar_kinds_compiler
