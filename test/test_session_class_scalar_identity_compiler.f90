program test_session_class_scalar_identity_compiler
    use ffc_test_support, only: expect_output, expect_error_contains, &
        expect_exe_has_symbol
    implicit none

    logical :: all_passed

    print *, '=== class(t) scalar runtime type identity tests ==='

    all_passed = .true.
    if (.not. test_base_actual_through_class_dummy()) all_passed = .false.
    if (.not. test_extension_actual_keeps_declared_prefix()) all_passed = .false.
    if (.not. test_identity_survives_repassing()) all_passed = .false.
    if (.not. test_type_bound_call_through_class_receiver()) all_passed = .false.
    if (.not. test_type_info_ids_exist_for_hierarchy()) all_passed = .false.
    if (.not. test_unrelated_actual_is_rejected()) all_passed = .false.
    if (.not. test_parent_actual_to_extension_dummy_is_rejected()) &
        all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: class(t) scalar runtime type identity'

contains

    character(len=:) function hierarchy() result(text)
        ! A base type and one extension of it, used by every positive fixture.
        allocatable :: text
        text = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type ext_t'//new_line('a')
    end function hierarchy

    logical function test_base_actual_through_class_dummy()
        ! A monomorphic base actual reaches a class(base_t) dummy and the dummy
        ! reads the declared component through the descriptor's data field.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine show(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, self%x'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 7'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'end program main'

        test_base_actual_through_class_dummy = expect_output(source, &
            '           7'//new_line('a'), '/tmp/ffc_class_identity_base')
    end function test_base_actual_through_class_dummy

    logical function test_extension_actual_keeps_declared_prefix()
        ! An extension actual whose dynamic type differs from the dummy's
        ! declared type is accepted; the callee sees the declared-type prefix
        ! and the extension's own component is untouched by the call, so the
        ! descriptor referenced the whole object and never copied a prefix.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine bump(self)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    self%x = self%x + 1'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 11'//new_line('a')// &
            '  e%y = 13'//new_line('a')// &
            '  call bump(e)'//new_line('a')// &
            '  print *, e%x * 100 + e%y'//new_line('a')// &
            'end program main'

        ! 12 from the bumped declared prefix, 13 from the untouched extension
        ! component: 12 * 100 + 13.
        test_extension_actual_keeps_declared_prefix = expect_output(source, &
            '        1213'//new_line('a'), '/tmp/ffc_class_identity_ext')
    end function test_extension_actual_keeps_declared_prefix

    logical function test_identity_survives_repassing()
        ! A class(base_t) dummy handed on to another class(base_t) dummy keeps
        ! referring to the caller's storage: the inner callee's write is visible
        ! to the original object, so no copy was interposed by the descriptor.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine inner(self)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    self%x = self%x + 5'//new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            '  subroutine outer(self)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    call inner(self)'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 9'//new_line('a')// &
            '  call outer(e)'//new_line('a')// &
            '  print *, e%x * 100 + e%y'//new_line('a')// &
            'end program main'

        ! 7 = 2 + 5 written through two hand-offs, 9 untouched: 7 * 100 + 9.
        test_identity_survives_repassing = expect_output(source, &
            '         709'//new_line('a'), '/tmp/ffc_class_identity_repass')
    end function test_identity_survives_repassing

    logical function test_type_bound_call_through_class_receiver()
        ! A type-bound call inserts the receiver at the passed-object position;
        ! that dummy is class(base_t), so it takes the class descriptor too.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: show => base_show'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_show(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, self%x'//new_line('a')// &
            '  end subroutine base_show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 21'//new_line('a')// &
            '  e%y = 22'//new_line('a')// &
            '  call e%show()'//new_line('a')// &
            'end program main'

        test_type_bound_call_through_class_receiver = expect_output(source, &
            '          21'//new_line('a'), '/tmp/ffc_class_identity_tbp')
    end function test_type_bound_call_through_class_receiver

    logical function test_type_info_ids_exist_for_hierarchy()
        ! The identities stored in a class descriptor are the ids of the
        ! per-type info constants; both the base and the extension must have a
        ! distinct one for the two identities to be distinguishable at all.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 1'//new_line('a')// &
            '  e%y = 2'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_type_info_ids_exist_for_hierarchy = .true.
        if (.not. expect_exe_has_symbol(source, &
            '/tmp/ffc_class_identity_info_base.o', '__ffc_type_info_m_c_cbase_ut')) &
            test_type_info_ids_exist_for_hierarchy = .false.
        if (.not. expect_exe_has_symbol(source, &
            '/tmp/ffc_class_identity_info_ext.o', '__ffc_type_info_m_c_cext_ut')) &
            test_type_info_ids_exist_for_hierarchy = .false.
    end function test_type_info_ids_exist_for_hierarchy

    logical function test_unrelated_actual_is_rejected()
        ! An actual outside the dummy's declared type hierarchy is not type
        ! compatible (F2018 7.3.2.3) and is diagnosed, not silently accepted.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type :: other_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  end type other_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine show(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, self%x'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(other_t) :: o'//new_line('a')// &
            '  o%x = 5'//new_line('a')// &
            '  call show(o)'//new_line('a')// &
            'end program main'

        test_unrelated_actual_is_rejected = expect_error_contains(source, &
            'is not type compatible with class(base_t) dummy', &
            '/tmp/ffc_class_identity_unrelated')
    end function test_unrelated_actual_is_rejected

    logical function test_parent_actual_to_extension_dummy_is_rejected()
        ! Type compatibility is directional: a base actual is not compatible
        ! with a class(extension) dummy, only the other way round.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'contains'//new_line('a')// &
            '  subroutine show(self)'//new_line('a')// &
            '    class(ext_t), intent(in) :: self'//new_line('a')// &
            '    print *, self%y'//new_line('a')// &
            '  end subroutine show'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 3'//new_line('a')// &
            '  call show(b)'//new_line('a')// &
            'end program main'

        test_parent_actual_to_extension_dummy_is_rejected = &
            expect_error_contains(source, &
            'is not type compatible with class(ext_t) dummy', &
            '/tmp/ffc_class_identity_parent_to_ext')
    end function test_parent_actual_to_extension_dummy_is_rejected

end program test_session_class_scalar_identity_compiler
