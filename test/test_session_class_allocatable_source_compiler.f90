program test_session_class_allocatable_source_compiler
    use ffc_test_support, only: expect_output, expect_error_contains, &
        expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== class(t) allocatable SOURCE= tests ==='

    all_passed = .true.
    if (.not. test_allocate_from_base_source()) all_passed = .false.
    if (.not. test_allocate_from_mold()) all_passed = .false.
    if (.not. test_allocate_from_extension_source()) all_passed = .false.
    if (.not. test_extension_storage_is_exact()) all_passed = .false.
    if (.not. test_dynamic_source_propagates_identity()) all_passed = .false.
    if (.not. test_typebound_dispatch_through_allocatable()) all_passed = .false.
    if (.not. test_reallocate_changes_dynamic_type()) all_passed = .false.
    if (.not. test_final_runs_once_on_deallocate()) all_passed = .false.
    if (.not. test_incompatible_source_is_rejected()) all_passed = .false.
    if (.not. test_double_deallocate_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: class(t) allocatable SOURCE='

contains

    logical function test_allocate_from_mold()
        ! MOLD copies the concrete dynamic type and layout, but not the source
        ! value. The destination must still be writable as an independent value.
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  class(t), allocatable :: original, obj'//new_line('a')// &
            '  allocate(t :: original)'//new_line('a')// &
            '  original%x = 123'//new_line('a')// &
            '  allocate(obj, mold=original)'//new_line('a')// &
            '  obj%x = 456'//new_line('a')// &
            '  if (obj%x /= 456) error stop'//new_line('a')// &
            '  if (original%x /= 123) error stop'//new_line('a')// &
            'end program main'

        test_allocate_from_mold = expect_exit_status(source, 0, &
            '/tmp/ffc_class_alloc_mold')
    end function test_allocate_from_mold

    character(len=:) function hierarchy() result(text)
        allocatable :: text
        text = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: speak => base_speak'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: speak => ext_speak'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            '  type :: other_t'//new_line('a')// &
            '    integer :: z'//new_line('a')// &
            '  end type other_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_speak(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, 100'//new_line('a')// &
            '  end subroutine base_speak'//new_line('a')// &
            '  subroutine ext_speak(self)'//new_line('a')// &
            '    class(ext_t), intent(in) :: self'//new_line('a')// &
            '    print *, 200'//new_line('a')// &
            '  end subroutine ext_speak'//new_line('a')
    end function hierarchy

    logical function test_allocate_from_base_source()
        ! A base source gives the allocatable the base dynamic type; the copied
        ! value is readable through the declared type and deallocation ends the
        ! lifetime.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 41'//new_line('a')// &
            '  allocate(p, source=b)'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_allocate_from_base_source = expect_output(source, &
            '          41'//new_line('a'), '/tmp/ffc_class_alloc_base')
    end function test_allocate_from_base_source

    logical function test_allocate_from_extension_source()
        ! An extension source records the extension as the dynamic type, so
        ! SELECT TYPE takes the TYPE IS (ext_t) arm and the extension component
        ! survived the copy.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 5'//new_line('a')// &
            '  e%y = 7'//new_line('a')// &
            '  allocate(p, source=e)'//new_line('a')// &
            '  select type (q => p)'//new_line('a')// &
            '  type is (ext_t)'//new_line('a')// &
            '    print *, q%x * 10 + q%y'//new_line('a')// &
            '  type is (base_t)'//new_line('a')// &
            '    print *, -1'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_allocate_from_extension_source = expect_output(source, &
            '          57'//new_line('a'), '/tmp/ffc_class_alloc_ext')
    end function test_allocate_from_extension_source

    logical function test_extension_storage_is_exact()
        ! Storage is the concrete extension layout, not the declared-type
        ! prefix: writing the extension component through the SELECT TYPE arm
        ! must not disturb the inherited component.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 3'//new_line('a')// &
            '  e%y = 4'//new_line('a')// &
            '  allocate(p, source=e)'//new_line('a')// &
            '  select type (q => p)'//new_line('a')// &
            '  type is (ext_t)'//new_line('a')// &
            '    q%y = 99'//new_line('a')// &
            '    print *, q%x * 1000 + q%y'//new_line('a')// &
            '  end select'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_extension_storage_is_exact = expect_output(source, &
            '        3099'//new_line('a')//'           3'//new_line('a'), &
            '/tmp/ffc_class_alloc_exact')
    end function test_extension_storage_is_exact

    logical function test_dynamic_source_propagates_identity()
        ! The source is itself polymorphic, so the dynamic type and the exact
        ! storage size are only known at run time and must be taken from the
        ! source's descriptor rather than from its declared type.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine clone_and_report(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    class(base_t), allocatable :: p'//new_line('a')// &
            '    allocate(p, source=self)'//new_line('a')// &
            '    select type (q => p)'//new_line('a')// &
            '    type is (ext_t)'//new_line('a')// &
            '      print *, 2000 + q%x'//new_line('a')// &
            '    type is (base_t)'//new_line('a')// &
            '      print *, 1000 + q%x'//new_line('a')// &
            '    end select'//new_line('a')// &
            '    deallocate(p)'//new_line('a')// &
            '  end subroutine clone_and_report'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call clone_and_report(b)'//new_line('a')// &
            '  call clone_and_report(e)'//new_line('a')// &
            'end program main'

        test_dynamic_source_propagates_identity = expect_output(source, &
            '        1001'//new_line('a')//'        2002'//new_line('a'), &
            '/tmp/ffc_class_alloc_dynsrc')
    end function test_dynamic_source_propagates_identity

    logical function test_typebound_dispatch_through_allocatable()
        ! The allocatable's dynamic type drives type-bound dispatch, so the
        ! override of the source's type is reached (#420 vtables consume the
        ! same descriptor).
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 1'//new_line('a')// &
            '  e%y = 2'//new_line('a')// &
            '  allocate(p, source=e)'//new_line('a')// &
            '  call p%speak()'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_typebound_dispatch_through_allocatable = expect_output(source, &
            '         200'//new_line('a'), '/tmp/ffc_class_alloc_dispatch')
    end function test_typebound_dispatch_through_allocatable

    logical function test_reallocate_changes_dynamic_type()
        ! Deallocating and allocating again from a different source replaces
        ! the recorded dynamic type; the second lifetime is independent.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  b%x = 8'//new_line('a')// &
            '  e%x = 9'//new_line('a')// &
            '  e%y = 6'//new_line('a')// &
            '  allocate(p, source=e)'//new_line('a')// &
            '  call p%speak()'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            '  allocate(p, source=b)'//new_line('a')// &
            '  call p%speak()'//new_line('a')// &
            '  print *, p%x'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_reallocate_changes_dynamic_type = expect_output(source, &
            '         200'//new_line('a')//'         100'//new_line('a')// &
            '           8'//new_line('a'), '/tmp/ffc_class_alloc_realloc')
    end function test_reallocate_changes_dynamic_type

    logical function test_final_runs_once_on_deallocate()
        ! Deallocation ends the value's lifetime exactly once, so the type's
        ! FINAL procedure runs once and not again at scope exit.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    final :: base_final'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_final(self)'//new_line('a')// &
            '    type(base_t), intent(inout) :: self'//new_line('a')// &
            '    print *, 777'//new_line('a')// &
            '  end subroutine base_final'//new_line('a')// &
            '  subroutine run()'//new_line('a')// &
            '    class(base_t), allocatable :: p'//new_line('a')// &
            '    type(base_t) :: b'//new_line('a')// &
            '    b%x = 1'//new_line('a')// &
            '    allocate(p, source=b)'//new_line('a')// &
            '    deallocate(p)'//new_line('a')// &
            '    print *, 5'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  call run()'//new_line('a')// &
            '  print *, 6'//new_line('a')// &
            'end program main'

        ! gfortran: 777 when the allocated value is released, 5, then 777 for
        ! the local source at return, then 6. The deallocated value is not
        ! finalized a second time at scope exit.
        test_final_runs_once_on_deallocate = expect_output(source, &
            '         777'//new_line('a')//'           5'//new_line('a')// &
            '         777'//new_line('a')//'           6'//new_line('a'), &
            '/tmp/ffc_class_alloc_final')
    end function test_final_runs_once_on_deallocate

    logical function test_incompatible_source_is_rejected()
        ! F2018 C946: the source must be type compatible with the allocatable.
        ! An unrelated type is neither the declared type nor an extension.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(other_t) :: o'//new_line('a')// &
            '  o%z = 1'//new_line('a')// &
            '  allocate(p, source=o)'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_incompatible_source_is_rejected = expect_error_contains(source, &
            'not type compatible', '/tmp/ffc_class_alloc_badsource')
    end function test_incompatible_source_is_rejected

    logical function test_double_deallocate_is_rejected()
        ! The second deallocation has no allocated value to release, so it is
        ! rejected instead of freeing the same storage twice.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  class(base_t), allocatable :: p'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  allocate(p, source=b)'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            '  deallocate(p)'//new_line('a')// &
            'end program main'

        test_double_deallocate_is_rejected = expect_error_contains(source, &
            'not currently allocated', '/tmp/ffc_class_alloc_doublefree')
    end function test_double_deallocate_is_rejected

end program test_session_class_allocatable_source_compiler
