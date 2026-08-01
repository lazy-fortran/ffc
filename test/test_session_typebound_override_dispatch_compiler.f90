program test_session_typebound_override_dispatch_compiler
    use ffc_test_support, only: expect_output, expect_error_contains, &
        expect_exe_has_symbol
    implicit none

    logical :: all_passed

    print *, '=== type-bound override dispatch tests ==='

    all_passed = .true.
    if (.not. test_override_dispatches_by_dynamic_type()) all_passed = .false.
    if (.not. test_inherited_binding_still_reaches_parent()) all_passed = .false.
    if (.not. test_three_level_hierarchy_picks_most_derived()) all_passed = .false.
    if (.not. test_function_binding_dispatches()) all_passed = .false.
    if (.not. test_identity_survives_repassing_before_dispatch()) &
        all_passed = .false.
    if (.not. test_monomorphic_receiver_stays_static()) all_passed = .false.
    if (.not. test_vtable_symbols_are_emitted()) all_passed = .false.
    if (.not. test_duplicate_binding_is_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: type-bound override dispatch'

contains

    character(len=:) function hierarchy() result(text)
        ! base_t with two bindings; ext_t overrides only `speak`.
        allocatable :: text
        text = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: speak => base_speak'//new_line('a')// &
            '    procedure :: tag => base_tag'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: speak => ext_speak'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_speak(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, 100'//new_line('a')// &
            '  end subroutine base_speak'//new_line('a')// &
            '  subroutine ext_speak(self)'//new_line('a')// &
            '    class(ext_t), intent(in) :: self'//new_line('a')// &
            '    print *, 200'//new_line('a')// &
            '  end subroutine ext_speak'//new_line('a')// &
            '  subroutine base_tag(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, 300'//new_line('a')// &
            '  end subroutine base_tag'//new_line('a')
    end function hierarchy

    logical function test_override_dispatches_by_dynamic_type()
        ! One class(base_t) dummy, two dynamic types: each call reaches the
        ! override of the value's dynamic type, not of the declared type.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine run(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    call self%speak()'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call run(b)'//new_line('a')// &
            '  call run(e)'//new_line('a')// &
            'end program main'

        test_override_dispatches_by_dynamic_type = expect_output(source, &
            '         100'//new_line('a')//'         200'//new_line('a'), &
            '/tmp/ffc_tbp_override_dispatch')
    end function test_override_dispatches_by_dynamic_type

    logical function test_inherited_binding_still_reaches_parent()
        ! `tag` is not overridden by ext_t, so both dynamic types dispatch to
        ! the inherited parent implementation through the same slot.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine run(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    call self%tag()'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call run(b)'//new_line('a')// &
            '  call run(e)'//new_line('a')// &
            'end program main'

        test_inherited_binding_still_reaches_parent = expect_output(source, &
            '         300'//new_line('a')//'         300'//new_line('a'), &
            '/tmp/ffc_tbp_inherited_slot')
    end function test_inherited_binding_still_reaches_parent

    logical function test_three_level_hierarchy_picks_most_derived()
        ! a_t -> b_t (no override) -> c_t (override). A b_t value must still
        ! reach a_t's implementation and a c_t value must reach c_t's, so the
        ! slot number is stable across two inheritance steps.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: a_t'//new_line('a')// &
            '    integer :: v'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: who => a_who'//new_line('a')// &
            '  end type a_t'//new_line('a')// &
            '  type, extends(a_t) :: b_t'//new_line('a')// &
            '  end type b_t'//new_line('a')// &
            '  type, extends(b_t) :: c_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: who => c_who'//new_line('a')// &
            '  end type c_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine a_who(self)'//new_line('a')// &
            '    class(a_t), intent(in) :: self'//new_line('a')// &
            '    print *, 11'//new_line('a')// &
            '  end subroutine a_who'//new_line('a')// &
            '  subroutine c_who(self)'//new_line('a')// &
            '    class(c_t), intent(in) :: self'//new_line('a')// &
            '    print *, 33'//new_line('a')// &
            '  end subroutine c_who'//new_line('a')// &
            '  subroutine run(self)'//new_line('a')// &
            '    class(a_t), intent(in) :: self'//new_line('a')// &
            '    call self%who()'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(a_t) :: a'//new_line('a')// &
            '  type(b_t) :: b'//new_line('a')// &
            '  type(c_t) :: c'//new_line('a')// &
            '  a%v = 1'//new_line('a')// &
            '  b%v = 2'//new_line('a')// &
            '  c%v = 3'//new_line('a')// &
            '  call run(a)'//new_line('a')// &
            '  call run(b)'//new_line('a')// &
            '  call run(c)'//new_line('a')// &
            'end program main'

        test_three_level_hierarchy_picks_most_derived = expect_output(source, &
            '          11'//new_line('a')//'          11'//new_line('a')// &
            '          33'//new_line('a'), '/tmp/ffc_tbp_three_level')
    end function test_three_level_hierarchy_picks_most_derived

    logical function test_function_binding_dispatches()
        ! A type-bound function used in an expression dispatches on the
        ! dynamic type too, so dispatch is not limited to subroutine calls.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: code => base_code'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            '  type, extends(base_t) :: ext_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: code => ext_code'//new_line('a')// &
            '  end type ext_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function base_code(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    base_code = 7'//new_line('a')// &
            '  end function base_code'//new_line('a')// &
            '  integer function ext_code(self)'//new_line('a')// &
            '    class(ext_t), intent(in) :: self'//new_line('a')// &
            '    ext_code = 9'//new_line('a')// &
            '  end function ext_code'//new_line('a')// &
            '  subroutine run(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, self%code()'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  call run(b)'//new_line('a')// &
            '  call run(e)'//new_line('a')// &
            'end program main'

        test_function_binding_dispatches = expect_output(source, &
            '           7'//new_line('a')//'           9'//new_line('a'), &
            '/tmp/ffc_tbp_function_dispatch')
    end function test_function_binding_dispatches

    logical function test_identity_survives_repassing_before_dispatch()
        ! The dynamic type reaches the dispatch site through two hand-offs, so
        ! dispatch consults the original value's identity, not the type of the
        ! nearest dummy.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            '  subroutine outer(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    call inner(self)'//new_line('a')// &
            '  end subroutine outer'//new_line('a')// &
            '  subroutine inner(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    call self%speak()'//new_line('a')// &
            '  end subroutine inner'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call outer(e)'//new_line('a')// &
            'end program main'

        test_identity_survives_repassing_before_dispatch = expect_output(source, &
            '         200'//new_line('a'), '/tmp/ffc_tbp_repassed_dispatch')
    end function test_identity_survives_repassing_before_dispatch

    logical function test_monomorphic_receiver_stays_static()
        ! A type(t) receiver has no runtime identity: it keeps direct dispatch
        ! and still calls its own binding.
        character(len=:), allocatable :: source

        source = hierarchy()// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call e%speak()'//new_line('a')// &
            '  call e%tag()'//new_line('a')// &
            'end program main'

        test_monomorphic_receiver_stays_static = expect_output(source, &
            '         200'//new_line('a')//'         300'//new_line('a'), &
            '/tmp/ffc_tbp_monomorphic')
    end function test_monomorphic_receiver_stays_static

    logical function test_vtable_symbols_are_emitted()
        ! One vtable per type carrying bindings, plus the link-unit table the
        ! dynamic type id indexes, are present in the linked executable.
        character(len=:), allocatable :: source
        logical :: ok

        source = hierarchy()// &
            '  subroutine run(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    call self%speak()'//new_line('a')// &
            '  end subroutine run'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(ext_t) :: e'//new_line('a')// &
            '  e%x = 2'//new_line('a')// &
            '  e%y = 3'//new_line('a')// &
            '  call run(e)'//new_line('a')// &
            'end program main'

        ok = expect_exe_has_symbol(source, '/tmp/ffc_tbp_vtable_syms.o', &
            '__ffc_vtable_base_t')
        if (.not. expect_exe_has_symbol(source, '/tmp/ffc_tbp_vtable_syms2.o', &
            '__ffc_vtable_ext_t')) ok = .false.
        if (.not. expect_exe_has_symbol(source, '/tmp/ffc_tbp_vtable_syms3.o', &
            '__ffc_vtable_table')) ok = .false.
        test_vtable_symbols_are_emitted = ok
    end function test_vtable_symbols_are_emitted

    logical function test_duplicate_binding_is_rejected()
        ! Two bindings of the same name in one type have no single slot
        ! occupant, so the vtable cannot be built and the program is rejected.
        character(len=:), allocatable :: source

        source = 'module m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure :: speak => one_speak'//new_line('a')// &
            '    procedure :: speak => two_speak'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine one_speak(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, 1'//new_line('a')// &
            '  end subroutine one_speak'//new_line('a')// &
            '  subroutine two_speak(self)'//new_line('a')// &
            '    class(base_t), intent(in) :: self'//new_line('a')// &
            '    print *, 2'//new_line('a')// &
            '  end subroutine two_speak'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type(base_t) :: b'//new_line('a')// &
            '  b%x = 1'//new_line('a')// &
            '  call b%speak()'//new_line('a')// &
            'end program main'

        test_duplicate_binding_is_rejected = expect_error_contains(source, &
            'declares the type-bound procedure "speak" more than once', &
            '/tmp/ffc_tbp_duplicate_binding')
    end function test_duplicate_binding_is_rejected

end program test_session_typebound_override_dispatch_compiler
