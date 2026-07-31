program test_session_reject_alloc_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== allocation and pointer definition target rejection test ==='

    all_passed = .true.
    if (.not. test_class_entity_rejected()) all_passed = .false.
    if (.not. test_class_component_rejected()) all_passed = .false.
    if (.not. test_class_dummy_accepted()) all_passed = .false.
    if (.not. test_deferred_length_rejected()) all_passed = .false.
    if (.not. test_deferred_length_allocatable_accepted()) all_passed = .false.
    if (.not. test_pointer_explicit_shape_rejected()) all_passed = .false.
    if (.not. test_pointer_deferred_shape_accepted()) all_passed = .false.
    if (.not. test_allocate_intent_in_rejected()) all_passed = .false.
    if (.not. test_deallocate_intent_in_rejected()) all_passed = .false.
    if (.not. test_allocate_intent_inout_accepted()) all_passed = .false.
    if (.not. test_nonallocatable_actual_rejected()) all_passed = .false.
    if (.not. test_nonpointer_actual_rejected()) all_passed = .false.
    if (.not. test_pointer_actual_accepted()) all_passed = .false.
    if (.not. test_constant_actual_rejected()) all_passed = .false.
    if (.not. test_constant_actual_intent_in_accepted()) all_passed = .false.
    if (.not. test_pure_nonlocal_actual_rejected()) all_passed = .false.
    if (.not. test_pure_local_actual_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid allocation and pointer targets are rejected'

contains

    logical function test_class_entity_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  class(t) :: o'//new_line('a')// &
            'end program main'

        test_class_entity_rejected = expect_error_contains(source, &
            'must be dummy, allocatable or pointer', &
            '/tmp/ffc_reject_alloc_01_class_entity')
    end function test_class_entity_rejected

    logical function test_class_component_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type object_t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type object_t'//new_line('a')// &
            '  type container_t'//new_line('a')// &
            '    class(object_t) :: v'//new_line('a')// &
            '  end type container_t'//new_line('a')// &
            'end program main'

        test_class_component_rejected = expect_error_contains(source, &
            'must be dummy, allocatable or pointer', &
            '/tmp/ffc_reject_alloc_01_class_component')
    end function test_class_component_rejected

    logical function test_class_dummy_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type t'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: v'//new_line('a')// &
            '  v%i = 3'//new_line('a')// &
            '  call show(v)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(o)'//new_line('a')// &
            '    class(t), intent(in) :: o'//new_line('a')// &
            '    stop o%i'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end program main'

        test_class_dummy_accepted = expect_exit_status(source, 3, &
            '/tmp/ffc_reject_alloc_01_class_dummy')
    end function test_class_dummy_accepted

    logical function test_deferred_length_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:) :: b'//new_line('a')// &
            'end program main'

        test_deferred_length_rejected = expect_error_contains(source, &
            'must have the POINTER or ALLOCATABLE attribute', &
            '/tmp/ffc_reject_alloc_01_deferred_len')
    end function test_deferred_length_rejected

    logical function test_deferred_length_allocatable_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  s = ''ok'''//new_line('a')// &
            '  stop len(s)'//new_line('a')// &
            'end program main'

        test_deferred_length_allocatable_accepted = expect_exit_status(source, &
            2, '/tmp/ffc_reject_alloc_01_deferred_len_ok')
    end function test_deferred_length_allocatable_accepted

    logical function test_pointer_explicit_shape_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bnds(a)'//new_line('a')// &
            '    integer, pointer, intent(in) :: a(1:2)'//new_line('a')// &
            '  end subroutine bnds'//new_line('a')// &
            'end program main'

        test_pointer_explicit_shape_rejected = expect_error_contains(source, &
            'must have a deferred shape or assumed rank', &
            '/tmp/ffc_reject_alloc_01_ptr_shape')
    end function test_pointer_explicit_shape_rejected

    logical function test_pointer_deferred_shape_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: a(3)'//new_line('a')// &
            '  integer, pointer :: p(:)'//new_line('a')// &
            '  a = 2'//new_line('a')// &
            '  p => a'//new_line('a')// &
            '  stop p(2)'//new_line('a')// &
            'end program main'

        test_pointer_deferred_shape_accepted = expect_exit_status(source, 2, &
            '/tmp/ffc_reject_alloc_01_ptr_shape_ok')
    end function test_pointer_deferred_shape_accepted

    logical function test_allocate_intent_in_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine init2(x)'//new_line('a')// &
            '    integer, allocatable, intent(in) :: x(:)'//new_line('a')// &
            '    allocate(x(3))'//new_line('a')// &
            '  end subroutine init2'//new_line('a')// &
            'end program main'

        test_allocate_intent_in_rejected = expect_error_contains(source, &
            'cannot appear in a variable definition context (ALLOCATE object)', &
            '/tmp/ffc_reject_alloc_01_alloc_intent_in')
    end function test_allocate_intent_in_rejected

    logical function test_deallocate_intent_in_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine kill(x)'//new_line('a')// &
            '    integer, allocatable, intent(in) :: x(:)'//new_line('a')// &
            '    deallocate(x)'//new_line('a')// &
            '  end subroutine kill'//new_line('a')// &
            'end program main'

        test_deallocate_intent_in_rejected = expect_error_contains(source, &
            'cannot appear in a variable definition context '// &
            '(DEALLOCATE object)', &
            '/tmp/ffc_reject_alloc_01_dealloc_intent_in')
    end function test_deallocate_intent_in_rejected

    logical function test_allocate_intent_inout_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: m(:)'//new_line('a')// &
            '  call grow(m)'//new_line('a')// &
            '  stop size(m)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine grow(x)'//new_line('a')// &
            '    integer, allocatable, intent(inout) :: x(:)'//new_line('a')// &
            '    allocate(x(3))'//new_line('a')// &
            '  end subroutine grow'//new_line('a')// &
            'end program main'

        test_allocate_intent_inout_accepted = expect_exit_status(source, 3, &
            '/tmp/ffc_reject_alloc_01_alloc_inout_ok')
    end function test_allocate_intent_inout_accepted

    logical function test_nonallocatable_actual_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            '  a = 1'//new_line('a')// &
            '  call init(a)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine init(x)'//new_line('a')// &
            '    integer, allocatable, intent(out) :: x(:)'//new_line('a')// &
            '  end subroutine init'//new_line('a')// &
            'end program main'

        test_nonallocatable_actual_rejected = expect_error_contains(source, &
            'must be ALLOCATABLE', &
            '/tmp/ffc_reject_alloc_01_actual_alloc')
    end function test_nonallocatable_actual_rejected

    logical function test_nonpointer_actual_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: b'//new_line('a')// &
            '  b = 1'//new_line('a')// &
            '  call foo(b)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine foo(p)'//new_line('a')// &
            '    integer, pointer, intent(in) :: p'//new_line('a')// &
            '  end subroutine foo'//new_line('a')// &
            'end program main'

        test_nonpointer_actual_rejected = expect_error_contains(source, &
            'must be a pointer', &
            '/tmp/ffc_reject_alloc_01_actual_pointer')
    end function test_nonpointer_actual_rejected

    logical function test_pointer_actual_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, target :: a'//new_line('a')// &
            '  integer, pointer :: p'//new_line('a')// &
            '  a = 7'//new_line('a')// &
            '  p => a'//new_line('a')// &
            '  call foo(p)'//new_line('a')// &
            '  stop a'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine foo(q)'//new_line('a')// &
            '    integer, pointer, intent(in) :: q'//new_line('a')// &
            '  end subroutine foo'//new_line('a')// &
            'end program main'

        test_pointer_actual_accepted = expect_exit_status(source, 7, &
            '/tmp/ffc_reject_alloc_01_actual_pointer_ok')
    end function test_pointer_actual_accepted

    logical function test_constant_actual_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call s(''y'')'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    character(8), intent(out) :: x'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end program main'

        test_constant_actual_rejected = expect_error_contains(source, &
            'constant actual argument', &
            '/tmp/ffc_reject_alloc_01_const_actual')
    end function test_constant_actual_rejected

    logical function test_constant_actual_intent_in_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call s(''y'')'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s(x)'//new_line('a')// &
            '    character(len=*), intent(in) :: x'//new_line('a')// &
            '    stop len(x)'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end program main'

        test_constant_actual_intent_in_accepted = expect_exit_status(source, 1, &
            '/tmp/ffc_reject_alloc_01_const_actual_ok')
    end function test_constant_actual_intent_in_accepted

    logical function test_pure_nonlocal_actual_rejected()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, pointer :: x'//new_line('a')// &
            'end module m'//new_line('a')// &
            'pure subroutine foo()'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call bar(x)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure subroutine bar(y)'//new_line('a')// &
            '    integer, pointer, intent(inout) :: y'//new_line('a')// &
            '  end subroutine bar'//new_line('a')// &
            'end subroutine foo'

        test_pure_nonlocal_actual_rejected = expect_error_contains(source, &
            'is not local to this PURE procedure', &
            '/tmp/ffc_reject_alloc_01_pure_nonlocal')
    end function test_pure_nonlocal_actual_rejected

    logical function test_pure_local_actual_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: r'//new_line('a')// &
            '  r = 0'//new_line('a')// &
            '  call bump(r)'//new_line('a')// &
            '  stop r'//new_line('a')// &
            'contains'//new_line('a')// &
            '  pure subroutine bump(y)'//new_line('a')// &
            '    integer, intent(out) :: y'//new_line('a')// &
            '    integer :: z'//new_line('a')// &
            '    z = 9'//new_line('a')// &
            '    call setit(z)'//new_line('a')// &
            '    y = z'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            '  pure subroutine setit(w)'//new_line('a')// &
            '    integer, intent(inout) :: w'//new_line('a')// &
            '  end subroutine setit'//new_line('a')// &
            'end program main'

        test_pure_local_actual_accepted = expect_exit_status(source, 9, &
            '/tmp/ffc_reject_alloc_01_pure_local_ok')
    end function test_pure_local_actual_accepted

end program test_session_reject_alloc_01_compiler
