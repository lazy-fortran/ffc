program test_session_reject_pointer_01_compiler
    ! #381: data and procedure pointer target contracts. A pointer only
    ! associates with something that can be a target, and PRESENT only
    ! accepts a whole optional dummy argument. Each invalid form below is
    ! rejected with a source diagnostic; the corrected neighbour compiles
    ! and runs.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== pointer target contract rejection test ==='

    all_passed = .true.
    if (.not. test_present_subobject_rejected()) all_passed = .false.
    if (.not. test_present_element_rejected()) all_passed = .false.
    if (.not. test_present_whole_dummy_accepted()) all_passed = .false.
    if (.not. test_abstract_interface_pointer_rejected()) all_passed = .false.
    if (.not. test_concrete_proc_pointer_accepted()) all_passed = .false.
    if (.not. test_data_object_proc_target_rejected()) all_passed = .false.
    if (.not. test_result_variable_proc_target_rejected()) all_passed = .false.
    if (.not. test_recursive_proc_target_accepted()) all_passed = .false.
    if (.not. test_unknown_proc_target_rejected()) all_passed = .false.
    if (.not. test_intent_in_pointer_actual_rejected()) all_passed = .false.
    if (.not. test_intent_inout_pointer_actual_accepted()) all_passed = .false.
    if (.not. test_cray_pointer_rejected()) all_passed = .false.
    if (.not. test_standard_pointer_accepted()) all_passed = .false.
    if (.not. test_parenthesised_associated_target_rejected()) all_passed = .false.
    if (.not. test_plain_associated_target_accepted()) all_passed = .false.
    if (.not. test_parenthesised_pointer_actual_rejected()) all_passed = .false.
    if (.not. test_plain_pointer_actual_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid pointer target contracts are rejected'

contains

    ! gfortran.dg/present_1.f90: PRESENT(D1%I).
    logical function test_present_subobject_rejected()
        character(len=*), parameter :: source = &
            'module m1'//new_line('a')// &
            '  type t1'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s1(d1)'//new_line('a')// &
            '    type(t1), optional :: d1(4)'//new_line('a')// &
            '    print *, present(d1%i)'//new_line('a')// &
            '  end subroutine s1'//new_line('a')// &
            'end module m1'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m1'//new_line('a')// &
            '  call s1()'//new_line('a')// &
            'end program main'

        test_present_subobject_rejected = expect_error_contains( &
            source, 'must not be a subobject', &
            '/tmp/ffc_reject_pointer01_present_comp')
    end function test_present_subobject_rejected

    ! gfortran.dg/present_1.f90: PRESENT(D1(1)).
    logical function test_present_element_rejected()
        character(len=*), parameter :: source = &
            'module m1'//new_line('a')// &
            '  type t1'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s1(d1)'//new_line('a')// &
            '    type(t1), optional :: d1(4)'//new_line('a')// &
            '    print *, present(d1(1))'//new_line('a')// &
            '  end subroutine s1'//new_line('a')// &
            'end module m1'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m1'//new_line('a')// &
            '  call s1()'//new_line('a')// &
            'end program main'

        test_present_element_rejected = expect_error_contains( &
            source, 'must not be a subobject', &
            '/tmp/ffc_reject_pointer01_present_elem')
    end function test_present_element_rejected

    ! Corrected neighbour: PRESENT of the whole optional dummy.
    logical function test_present_whole_dummy_accepted()
        character(len=*), parameter :: source = &
            'module m1'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s1(d1)'//new_line('a')// &
            '    integer, optional :: d1'//new_line('a')// &
            '    if (present(d1)) then'//new_line('a')// &
            '      stop 3'//new_line('a')// &
            '    end if'//new_line('a')// &
            '    stop 4'//new_line('a')// &
            '  end subroutine s1'//new_line('a')// &
            'end module m1'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m1'//new_line('a')// &
            '  call s1(9)'//new_line('a')// &
            'end program main'

        test_present_whole_dummy_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_reject_pointer01_present_ok')
    end function test_present_whole_dummy_accepted

    ! gfortran.dg/proc_ptr_44.f90: POINTER on an abstract interface name.
    logical function test_abstract_interface_pointer_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  abstract interface'//new_line('a')// &
            '    subroutine abssub1'//new_line('a')// &
            '    end subroutine'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  procedure(abssub1), pointer :: abssub1'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_abstract_interface_pointer_rejected = expect_error_contains( &
            source, 'conflicts with ABSTRACT attribute', &
            '/tmp/ffc_reject_pointer01_abstract')
    end function test_abstract_interface_pointer_rejected

    ! Corrected neighbour: a distinct procedure pointer of that interface.
    logical function test_concrete_proc_pointer_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  abstract interface'//new_line('a')// &
            '    subroutine abssub1'//new_line('a')// &
            '    end subroutine'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  procedure(abssub1), pointer :: p'//new_line('a')// &
            '  p => sub'//new_line('a')// &
            '  call p()'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine sub'//new_line('a')// &
            '    stop 5'//new_line('a')// &
            '  end subroutine sub'//new_line('a')// &
            'end program main'

        test_concrete_proc_pointer_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_pointer01_abstract_ok')
    end function test_concrete_proc_pointer_accepted

    ! gfortran.dg/pr78719_2.f90: proc-pointer target is a data object.
    logical function test_data_object_proc_target_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real :: g'//new_line('a')// &
            '  abstract interface'//new_line('a')// &
            '    subroutine h'//new_line('a')// &
            '    end subroutine'//new_line('a')// &
            '  end interface'//new_line('a')// &
            '  procedure(h), pointer :: s'//new_line('a')// &
            '  s => g'//new_line('a')// &
            '  call s'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine f'//new_line('a')// &
            '  end subroutine f'//new_line('a')// &
            'end program main'

        test_data_object_proc_target_rejected = expect_error_contains( &
            source, 'is a data object, not a procedure', &
            '/tmp/ffc_reject_pointer01_dataobj')
    end function test_data_object_proc_target_rejected

    ! gfortran.dg/proc_ptr_38.f90: target is the function result variable.
    logical function test_result_variable_proc_target_rejected()
        character(len=*), parameter :: source = &
            'integer function foo()'//new_line('a')// &
            '  procedure(), pointer :: i'//new_line('a')// &
            '  i => foo'//new_line('a')// &
            '  foo = 1'//new_line('a')// &
            'end function foo'

        test_result_variable_proc_target_rejected = expect_error_contains( &
            source, 'is invalid as proc-target', &
            '/tmp/ffc_reject_pointer01_result')
    end function test_result_variable_proc_target_rejected

    ! Corrected neighbour: a recursive function may be its own proc-target.
    logical function test_recursive_proc_target_accepted()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  recursive function bar() result(res)'//new_line('a')// &
            '    integer :: res'//new_line('a')// &
            '    procedure(), pointer :: j'//new_line('a')// &
            '    j => bar'//new_line('a')// &
            '    res = 7'//new_line('a')// &
            '  end function bar'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer :: v'//new_line('a')// &
            '  v = bar()'//new_line('a')// &
            '  if (v == 7) stop 7'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_recursive_proc_target_accepted = expect_exit_status( &
            source, 7, '/tmp/ffc_reject_pointer01_recursive_ok')
    end function test_recursive_proc_target_accepted

    ! gfortran.dg/proc_ptr_46.f90: target is not visible as a procedure.
    logical function test_unknown_proc_target_rejected()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s'//new_line('a')// &
            '    procedure(real), pointer :: p'//new_line('a')// &
            '    p => f'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: s'//new_line('a')// &
            '  call s'//new_line('a')// &
            'end program main'

        test_unknown_proc_target_rejected = expect_error_contains( &
            source, 'must be either an intrinsic, host or use associated', &
            '/tmp/ffc_reject_pointer01_unknown')
    end function test_unknown_proc_target_rejected

    ! gfortran.dg/pointer_intent_7.f90: INTENT(IN) pointer actual passed to
    ! an INTENT(INOUT) pointer dummy.
    logical function test_intent_in_pointer_actual_rejected()
        character(len=*), parameter :: source = &
            'module moda'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine foo(b)'//new_line('a')// &
            '    integer, intent(in), pointer :: b'//new_line('a')// &
            '    call bar2p(b)'//new_line('a')// &
            '  end subroutine foo'//new_line('a')// &
            '  subroutine bar2p(n)'//new_line('a')// &
            '    integer, intent(inout), pointer :: n'//new_line('a')// &
            '  end subroutine bar2p'//new_line('a')// &
            'end module moda'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use moda'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_intent_in_pointer_actual_rejected = expect_error_contains( &
            source, 'in pointer association context', &
            '/tmp/ffc_reject_pointer01_intent')
    end function test_intent_in_pointer_actual_rejected

    ! Corrected neighbour: an INTENT(INOUT) pointer actual is allowed.
    logical function test_intent_inout_pointer_actual_accepted()
        character(len=*), parameter :: source = &
            'module moda'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine foo(b)'//new_line('a')// &
            '    integer, intent(inout), pointer :: b'//new_line('a')// &
            '    call bar2p(b)'//new_line('a')// &
            '  end subroutine foo'//new_line('a')// &
            '  subroutine bar2p(n)'//new_line('a')// &
            '    integer, intent(inout), pointer :: n'//new_line('a')// &
            '    n = 5'//new_line('a')// &
            '  end subroutine bar2p'//new_line('a')// &
            'end module moda'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use moda'//new_line('a')// &
            '  integer, target :: t'//new_line('a')// &
            '  integer, pointer :: q'//new_line('a')// &
            '  q => t'//new_line('a')// &
            '  call foo(q)'//new_line('a')// &
            '  if (t == 5) stop 5'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_intent_inout_pointer_actual_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_pointer01_intent_ok')
    end function test_intent_inout_pointer_actual_accepted

    ! gfortran.dg/cray_pointers_3.f90: Cray pointer declaration.
    logical function test_cray_pointer_rejected()
        character(len=*), parameter :: source = &
            'program crayerr'//new_line('a')// &
            '  real dpte1(10)'//new_line('a')// &
            '  pointer (iptr1,dpte1)'//new_line('a')// &
            'end program crayerr'

        test_cray_pointer_rejected = expect_error_contains( &
            source, 'Cray pointer declaration', &
            '/tmp/ffc_reject_pointer01_cray')
    end function test_cray_pointer_rejected

    ! Corrected neighbour: the standard POINTER attribute declaration.
    logical function test_standard_pointer_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real, target :: store'//new_line('a')// &
            '  real, pointer :: dpte1'//new_line('a')// &
            '  store = 2.0'//new_line('a')// &
            '  dpte1 => store'//new_line('a')// &
            '  if (dpte1 > 1.5) stop 2'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_standard_pointer_accepted = expect_exit_status( &
            source, 2, '/tmp/ffc_reject_pointer01_cray_ok')
    end function test_standard_pointer_accepted

    ! gfortran.dg/associated_target_1.f90: ASSOCIATED(X,(Y)).
    logical function test_parenthesised_associated_target_rejected()
        character(len=*), parameter :: source = &
            'program test'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real, pointer :: x'//new_line('a')// &
            '  real, target :: y'//new_line('a')// &
            '  if (associated(x,(y))) print *, "hello"'//new_line('a')// &
            'end program test'

        test_parenthesised_associated_target_rejected = expect_error_contains( &
            source, 'must be a VARIABLE or FUNCTION', &
            '/tmp/ffc_reject_pointer01_assoc')
    end function test_parenthesised_associated_target_rejected

    ! Corrected neighbour: a plain target variable.
    logical function test_plain_associated_target_accepted()
        character(len=*), parameter :: source = &
            'program test'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  real, pointer :: x'//new_line('a')// &
            '  real, target :: y'//new_line('a')// &
            '  y = 1.0'//new_line('a')// &
            '  x => y'//new_line('a')// &
            '  if (associated(x,y)) stop 6'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program test'

        test_plain_associated_target_accepted = expect_exit_status( &
            source, 6, '/tmp/ffc_reject_pointer01_assoc_ok')
    end function test_plain_associated_target_accepted

    ! gfortran.dg/parens_2.f90: parenthesised actual for a POINTER dummy.
    logical function test_parenthesised_pointer_actual_rejected()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s1(i)'//new_line('a')// &
            '    integer, pointer :: i'//new_line('a')// &
            '  end subroutine s1'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer, pointer :: i'//new_line('a')// &
            '  call s1((i))'//new_line('a')// &
            'end program main'

        test_parenthesised_pointer_actual_rejected = expect_error_contains( &
            source, 'must be a pointer or a valid target', &
            '/tmp/ffc_reject_pointer01_parens')
    end function test_parenthesised_pointer_actual_rejected

    ! Corrected neighbour: the pointer itself is a valid actual.
    logical function test_plain_pointer_actual_accepted()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s1(i)'//new_line('a')// &
            '    integer, pointer :: i'//new_line('a')// &
            '    i = 4'//new_line('a')// &
            '  end subroutine s1'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  integer, target :: t'//new_line('a')// &
            '  integer, pointer :: i'//new_line('a')// &
            '  i => t'//new_line('a')// &
            '  call s1(i)'//new_line('a')// &
            '  if (t == 4) stop 4'//new_line('a')// &
            '  stop 1'//new_line('a')// &
            'end program main'

        test_plain_pointer_actual_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_reject_pointer01_parens_ok')
    end function test_plain_pointer_actual_accepted

end program test_session_reject_pointer_01_compiler
