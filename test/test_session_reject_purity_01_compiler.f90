program test_session_reject_purity_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_no_error
    implicit none

    logical :: all_passed
    character(len=*), parameter :: nl = new_line('a')

    print *, '=== PURE/ELEMENTAL attribute rejection test ==='

    all_passed = .true.
    if (.not. test_elemental_proc_pointer_rejected()) all_passed = .false.
    if (.not. test_elemental_intrinsic_pointer_rejected()) all_passed = .false.
    if (.not. test_elemental_dummy_procedure_rejected()) all_passed = .false.
    if (.not. test_elemental_dummy_interface_rejected()) all_passed = .false.
    if (.not. test_pure_volatile_rejected()) all_passed = .false.
    if (.not. test_pure_save_rejected()) all_passed = .false.
    if (.not. test_pure_block_save_rejected()) all_passed = .false.
    if (.not. test_plain_proc_pointer_accepted()) all_passed = .false.
    if (.not. test_plain_dummy_procedure_accepted()) all_passed = .false.
    if (.not. test_elemental_external_accepted()) all_passed = .false.
    if (.not. test_impure_volatile_accepted()) all_passed = .false.
    if (.not. test_impure_save_accepted()) all_passed = .false.
    if (.not. test_pure_plain_local_accepted()) all_passed = .false.
    if (.not. test_elemental_pointer_component_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: PURE/ELEMENTAL attribute constraints enforced'

contains

    logical function test_elemental_proc_pointer_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  interface'//nl// &
            '    elemental real function x(y)'//nl// &
            '      real, intent(in) :: y'//nl// &
            '    end function x'//nl// &
            '  end interface'//nl// &
            '  procedure(x), pointer :: xx2'//nl// &
            'end program main'
        test_elemental_proc_pointer_rejected = expect_error_contains(source, &
            'shall not be elemental', '/tmp/ffc_pure578_pp')
    end function test_elemental_proc_pointer_rejected

    logical function test_elemental_intrinsic_pointer_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  intrinsic :: sin'//nl// &
            '  procedure(sin), pointer :: foo'//nl// &
            'end program main'
        test_elemental_intrinsic_pointer_rejected = expect_error_contains( &
            source, 'shall not be elemental', '/tmp/ffc_pure578_ip')
    end function test_elemental_intrinsic_pointer_rejected

    logical function test_elemental_dummy_procedure_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            'contains'//nl// &
            '  subroutine sub5(z)'//nl// &
            '    intrinsic :: sin'//nl// &
            '    procedure(sin) :: z'//nl// &
            '  end subroutine sub5'//nl// &
            'end program main'
        test_elemental_dummy_procedure_rejected = expect_error_contains( &
            source, 'shall not be elemental', '/tmp/ffc_pure578_dp')
    end function test_elemental_dummy_procedure_rejected

    logical function test_elemental_dummy_interface_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            'contains'//nl// &
            '  subroutine sub3(z)'//nl// &
            '    interface'//nl// &
            '      elemental real function z(y)'//nl// &
            '        real, intent(in) :: y'//nl// &
            '      end function z'//nl// &
            '    end interface'//nl// &
            '  end subroutine sub3'//nl// &
            'end program main'
        test_elemental_dummy_interface_rejected = expect_error_contains( &
            source, 'shall not be elemental', '/tmp/ffc_pure578_di')
    end function test_elemental_dummy_interface_rejected

    logical function test_pure_volatile_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  call bah'//nl// &
            'contains'//nl// &
            '  pure subroutine bah'//nl// &
            '    integer, volatile :: m'//nl// &
            '    m = 1'//nl// &
            '  end subroutine bah'//nl// &
            'end program main'
        test_pure_volatile_rejected = expect_error_contains(source, &
            'cannot be specified in a PURE', '/tmp/ffc_pure578_vol')
    end function test_pure_volatile_rejected

    logical function test_pure_save_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  call bah'//nl// &
            'contains'//nl// &
            '  pure subroutine bah'//nl// &
            '    integer, save :: i'//nl// &
            '    i = 1'//nl// &
            '  end subroutine bah'//nl// &
            'end program main'
        test_pure_save_rejected = expect_error_contains(source, &
            'cannot be specified in a PURE', '/tmp/ffc_pure578_save')
    end function test_pure_save_rejected

    logical function test_pure_block_save_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  call bah'//nl// &
            'contains'//nl// &
            '  pure subroutine bah'//nl// &
            '    block'//nl// &
            '      integer, volatile :: j'//nl// &
            '      j = 2'//nl// &
            '    end block'//nl// &
            '  end subroutine bah'//nl// &
            'end program main'
        test_pure_block_save_rejected = expect_error_contains(source, &
            'cannot be specified in a PURE', '/tmp/ffc_pure578_blk')
    end function test_pure_block_save_rejected

    logical function test_plain_proc_pointer_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  interface'//nl// &
            '    real function x(y)'//nl// &
            '      real, intent(in) :: y'//nl// &
            '    end function x'//nl// &
            '  end interface'//nl// &
            '  procedure(x), pointer :: xx2'//nl// &
            '  print *, 1'//nl// &
            'end program main'
        test_plain_proc_pointer_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_pp')
    end function test_plain_proc_pointer_accepted

    logical function test_plain_dummy_procedure_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  print *, 1'//nl// &
            'contains'//nl// &
            '  subroutine sub(z)'//nl// &
            '    interface'//nl// &
            '      real function z(y)'//nl// &
            '        real, intent(in) :: y'//nl// &
            '      end function z'//nl// &
            '    end interface'//nl// &
            '  end subroutine sub'//nl// &
            'end program main'
        test_plain_dummy_procedure_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_dp')
    end function test_plain_dummy_procedure_accepted

    logical function test_elemental_external_accepted()
        character(len=:), allocatable :: source

        ! An elemental interface used for a non-pointer, non-dummy external
        ! procedure declaration stays valid (F2018 C1518 applies only to
        ! procedure pointers and dummy procedures).
        source = 'program main'//nl// &
            '  interface'//nl// &
            '    elemental real function x(y)'//nl// &
            '      real, intent(in) :: y'//nl// &
            '    end function x'//nl// &
            '  end interface'//nl// &
            '  procedure(x) :: xx1'//nl// &
            '  print *, 1'//nl// &
            'end program main'
        test_elemental_external_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_ext')
    end function test_elemental_external_accepted

    logical function test_impure_volatile_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  call bar'//nl// &
            'contains'//nl// &
            '  subroutine bar'//nl// &
            '    integer, volatile :: m'//nl// &
            '    m = 1'//nl// &
            '    print *, m'//nl// &
            '  end subroutine bar'//nl// &
            'end program main'
        test_impure_volatile_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_vol')
    end function test_impure_volatile_accepted

    logical function test_impure_save_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  call bar'//nl// &
            'contains'//nl// &
            '  subroutine bar'//nl// &
            '    integer, save :: i'//nl// &
            '    i = i + 1'//nl// &
            '    print *, i'//nl// &
            '  end subroutine bar'//nl// &
            'end program main'
        test_impure_save_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_save')
    end function test_impure_save_accepted

    logical function test_pure_plain_local_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
            '  print *, twice(2)'//nl// &
            'contains'//nl// &
            '  pure integer function twice(n)'//nl// &
            '    integer, intent(in) :: n'//nl// &
            '    integer :: tmp'//nl// &
            '    tmp = 2 * n'//nl// &
            '    twice = tmp'//nl// &
            '  end function twice'//nl// &
            'end program main'
        test_pure_plain_local_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_pure')
    end function test_pure_plain_local_accepted

    logical function test_elemental_pointer_component_accepted()
        character(len=:), allocatable :: source

        ! Reduced from gfortran.dg/proc_ptr_comp_45.f90: a procedure pointer
        ! *component* may name an elemental interface, so C1518 must not
        ! reach into a derived type definition.
        source = 'module decays'//nl// &
            '  implicit none'//nl// &
            '  interface'//nl// &
            '    real elemental function iface (arg)'//nl// &
            '      real, intent(in) :: arg'//nl// &
            '    end function'//nl// &
            '  end interface'//nl// &
            '  type :: decay_gen_t'//nl// &
            '     procedure(iface), nopass, pointer :: obs1_int'//nl// &
            '  end type'//nl// &
            'end module decays'//nl// &
            'program main'//nl// &
            '  use decays'//nl// &
            '  print *, 1'//nl// &
            'end program main'
        test_elemental_pointer_component_accepted = expect_no_error(source, &
            '/tmp/ffc_pure578_ok_comp')
    end function test_elemental_pointer_component_accepted

end program test_session_reject_purity_01_compiler
