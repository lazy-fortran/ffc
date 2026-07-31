program test_session_reject_namelist_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_no_error
    implicit none

    logical :: all_passed
    character(len=*), parameter :: nl = new_line('a')

    print *, '=== NAMELIST member constraint rejection test ==='

    all_passed = .true.
    if (.not. test_private_member_rejected()) all_passed = .false.
    if (.not. test_intent_in_read_rejected()) all_passed = .false.
    if (.not. test_private_components_rejected()) all_passed = .false.
    if (.not. test_procedure_member_rejected()) all_passed = .false.
    if (.not. test_syntax_error_rejected()) all_passed = .false.
    if (.not. test_allocatable_component_rejected()) all_passed = .false.
    if (.not. test_polymorphic_member_rejected()) all_passed = .false.
    if (.not. test_late_declaration_rejected()) all_passed = .false.
    if (.not. test_public_member_accepted()) all_passed = .false.
    if (.not. test_intent_inout_read_accepted()) all_passed = .false.
    if (.not. test_public_components_accepted()) all_passed = .false.
    if (.not. test_variable_member_accepted()) all_passed = .false.
    if (.not. test_well_formed_list_accepted()) all_passed = .false.
    if (.not. test_plain_component_type_accepted()) all_passed = .false.
    if (.not. test_nonpolymorphic_member_accepted()) all_passed = .false.
    if (.not. test_early_declaration_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: NAMELIST member constraints enforced'

contains

    logical function test_private_member_rejected()
        character(len=:), allocatable :: source

        source = 'module m'//nl// &
                 '  public'//nl// &
                 '  integer, private :: x'//nl// &
                 '  namelist /n/ x'//nl// &
                 'end module m'//nl// &
                 'program main'//nl// &
                 '  use m'//nl// &
                 'end program main'
        test_private_member_rejected = expect_error_contains(source, &
            'cannot be member of PUBLIC namelist', &
            '/tmp/ffc_nml389_private_member')
    end function test_private_member_rejected

    logical function test_intent_in_read_rejected()
        character(len=:), allocatable :: source

        source = 'subroutine s(x)'//nl// &
                 '  integer, intent(in) :: x'//nl// &
                 '  namelist /n/ x'//nl// &
                 '  read(*,n)'//nl// &
                 'end subroutine s'//nl// &
                 'program main'//nl// &
                 'end program main'
        test_intent_in_read_rejected = expect_error_contains(source, &
            'is INTENT(IN) and cannot be read', &
            '/tmp/ffc_nml389_intent_in')
    end function test_intent_in_read_rejected

    logical function test_private_components_rejected()
        character(len=:), allocatable :: source

        source = 'module types'//nl// &
                 '  type :: tp4'//nl// &
                 '    private'//nl// &
                 '    real :: x'//nl// &
                 '  end type tp4'//nl// &
                 'end module types'//nl// &
                 'module nml'//nl// &
                 '  use types'//nl// &
                 '  type(tp4) :: t4'//nl// &
                 '  namelist /b/ t4'//nl// &
                 'end module nml'//nl// &
                 'program main'//nl// &
                 '  use nml'//nl// &
                 'end program main'
        test_private_components_rejected = expect_error_contains(source, &
            'use-associated PRIVATE components', &
            '/tmp/ffc_nml389_private_components')
    end function test_private_components_rejected

    logical function test_procedure_member_rejected()
        character(len=:), allocatable :: source

        source = 'module m1'//nl// &
                 'contains'//nl// &
                 '  integer function g1()'//nl// &
                 '    namelist /nml1/ g2'//nl// &
                 '    g1 = 1'//nl// &
                 '  end function g1'//nl// &
                 '  integer function g2()'//nl// &
                 '    g2 = 1'//nl// &
                 '  end function g2'//nl// &
                 'end module m1'//nl// &
                 'program main'//nl// &
                 '  use m1'//nl// &
                 'end program main'
        test_procedure_member_rejected = expect_error_contains(source, &
            'PROCEDURE attribute conflicts', &
            '/tmp/ffc_nml389_procedure_member')
    end function test_procedure_member_rejected

    logical function test_syntax_error_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
                 '  integer :: bar, baz'//nl// &
                 '  namelist /foo/ bar, baz'//nl// &
                 '  namelist /foo/ baz, ,'//nl// &
                 'end program main'
        test_syntax_error_rejected = expect_error_contains(source, &
            'Syntax error in NAMELIST', &
            '/tmp/ffc_nml389_syntax')
    end function test_syntax_error_rejected

    logical function test_allocatable_component_rejected()
        character(len=:), allocatable :: source

        source = 'module ma'//nl// &
                 '  implicit none'//nl// &
                 '  type :: ta'//nl// &
                 '    integer, allocatable :: array(:)'//nl// &
                 '  end type ta'//nl// &
                 'end module ma'//nl// &
                 'program main'//nl// &
                 '  use ma'//nl// &
                 '  type(ta) :: x'//nl// &
                 '  namelist /nml/ x'//nl// &
                 '  write(*, nml)'//nl// &
                 'end program main'
        test_allocatable_component_rejected = expect_error_contains(source, &
            'has ALLOCATABLE or POINTER components', &
            '/tmp/ffc_nml389_alloc_component')
    end function test_allocatable_component_rejected

    logical function test_polymorphic_member_rejected()
        character(len=:), allocatable :: source

        source = 'module mb'//nl// &
                 '  implicit none'//nl// &
                 '  type :: tb'//nl// &
                 '    integer :: i'//nl// &
                 '  end type tb'//nl// &
                 'end module mb'//nl// &
                 'program main'//nl// &
                 '  use mb'//nl// &
                 '  class(tb), allocatable :: x'//nl// &
                 '  namelist /nml/ x'//nl// &
                 '  read(*, nml)'//nl// &
                 'end program main'
        test_polymorphic_member_rejected = expect_error_contains(source, &
            'is polymorphic and requires a defined input/output procedure', &
            '/tmp/ffc_nml389_polymorphic')
    end function test_polymorphic_member_rejected

    logical function test_late_declaration_rejected()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
                 '  implicit none'//nl// &
                 '  real :: x'//nl// &
                 '  namelist /grp/ x, q'//nl// &
                 '  integer :: q'//nl// &
                 '  x = 1.0'//nl// &
                 '  q = 3'//nl// &
                 'end program main'
        test_late_declaration_rejected = expect_error_contains(source, &
            'must be declared before the namelist group', &
            '/tmp/ffc_nml389_late_decl')
    end function test_late_declaration_rejected

    logical function test_public_member_accepted()
        character(len=:), allocatable :: source

        source = 'module m'//nl// &
                 '  public'//nl// &
                 '  integer :: x'//nl// &
                 '  namelist /n/ x'//nl// &
                 'end module m'//nl// &
                 'program main'//nl// &
                 '  use m'//nl// &
                 '  x = 1'//nl// &
                 'end program main'
        test_public_member_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_public_member')
    end function test_public_member_accepted

    logical function test_intent_inout_read_accepted()
        character(len=:), allocatable :: source

        source = 'subroutine s(x)'//nl// &
                 '  integer, intent(inout) :: x'//nl// &
                 '  namelist /n/ x'//nl// &
                 '  read(*,n)'//nl// &
                 'end subroutine s'//nl// &
                 'program main'//nl// &
                 'end program main'
        test_intent_inout_read_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_intent_inout')
    end function test_intent_inout_read_accepted

    logical function test_public_components_accepted()
        character(len=:), allocatable :: source

        source = 'module types'//nl// &
                 '  type :: tp4'//nl// &
                 '    real :: x'//nl// &
                 '  end type tp4'//nl// &
                 'end module types'//nl// &
                 'module nml'//nl// &
                 '  use types'//nl// &
                 '  type(tp4) :: t4'//nl// &
                 '  namelist /b/ t4'//nl// &
                 'end module nml'//nl// &
                 'program main'//nl// &
                 '  use nml'//nl// &
                 'end program main'
        test_public_components_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_public_components')
    end function test_public_components_accepted

    logical function test_variable_member_accepted()
        character(len=:), allocatable :: source

        source = 'module m1'//nl// &
                 'contains'//nl// &
                 '  integer function g1()'//nl// &
                 '    integer :: g2'//nl// &
                 '    namelist /nml1/ g2'//nl// &
                 '    g2 = 1'//nl// &
                 '    g1 = g2'//nl// &
                 '  end function g1'//nl// &
                 'end module m1'//nl// &
                 'program main'//nl// &
                 '  use m1'//nl// &
                 'end program main'
        test_variable_member_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_variable_member')
    end function test_variable_member_accepted

    logical function test_well_formed_list_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
                 '  integer :: bar, baz'//nl// &
                 '  namelist /foo/ bar, baz'//nl// &
                 '  namelist /foo2/ baz'//nl// &
                 '  bar = 1'//nl// &
                 '  baz = 2'//nl// &
                 'end program main'
        test_well_formed_list_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_well_formed')
    end function test_well_formed_list_accepted

    logical function test_plain_component_type_accepted()
        character(len=:), allocatable :: source

        source = 'module ma'//nl// &
                 '  implicit none'//nl// &
                 '  type :: ta'//nl// &
                 '    integer :: array(4)'//nl// &
                 '  end type ta'//nl// &
                 'end module ma'//nl// &
                 'program main'//nl// &
                 '  use ma'//nl// &
                 '  type(ta) :: x'//nl// &
                 '  namelist /nml/ x'//nl// &
                 '  write(*, nml)'//nl// &
                 'end program main'
        test_plain_component_type_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_plain_component')
    end function test_plain_component_type_accepted

    logical function test_nonpolymorphic_member_accepted()
        character(len=:), allocatable :: source

        source = 'module mb'//nl// &
                 '  implicit none'//nl// &
                 '  type :: tb'//nl// &
                 '    integer :: i'//nl// &
                 '  end type tb'//nl// &
                 'end module mb'//nl// &
                 'program main'//nl// &
                 '  use mb'//nl// &
                 '  type(tb) :: x'//nl// &
                 '  namelist /nml/ x'//nl// &
                 '  read(*, nml)'//nl// &
                 'end program main'
        test_nonpolymorphic_member_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_nonpolymorphic')
    end function test_nonpolymorphic_member_accepted

    logical function test_early_declaration_accepted()
        character(len=:), allocatable :: source

        source = 'program main'//nl// &
                 '  implicit none'//nl// &
                 '  real :: x'//nl// &
                 '  integer :: q'//nl// &
                 '  namelist /grp/ x, q'//nl// &
                 '  x = 1.0'//nl// &
                 '  q = 3'//nl// &
                 'end program main'
        test_early_declaration_accepted = expect_no_error(source, &
            '/tmp/ffc_nml389_early_decl')
    end function test_early_declaration_accepted

end program test_session_reject_namelist_01_compiler
