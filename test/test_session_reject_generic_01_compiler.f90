program test_session_reject_generic_01_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== malformed and ambiguous generic interface rejection ==='

    all_passed = .true.
    if (.not. test_malformed_generic_binding()) all_passed = .false.
    if (.not. test_implicit_interface_ambiguity()) all_passed = .false.
    if (.not. test_module_procedure_target()) all_passed = .false.
    if (.not. test_generic_name_collision()) all_passed = .false.
    if (.not. test_use_shadows_program_unit()) all_passed = .false.
    if (.not. test_ambiguous_use_association()) all_passed = .false.
    if (.not. test_typebound_generic_inheritance()) all_passed = .false.
    if (.not. test_intrinsic_assignment_redefinition()) all_passed = .false.
    if (.not. test_generic_call_without_specific()) all_passed = .false.
    if (.not. test_allocatable_pointer_ambiguity()) all_passed = .false.
    if (.not. test_unlimited_polymorphic_ambiguity()) all_passed = .false.
    if (.not. test_generic_identifier_accepted()) all_passed = .false.
    if (.not. test_distinguishable_implicit_interfaces()) all_passed = .false.
    if (.not. test_module_procedure_accepted()) all_passed = .false.
    if (.not. test_distinct_generic_and_specific()) all_passed = .false.
    if (.not. test_use_without_shadowing()) all_passed = .false.
    if (.not. test_unambiguous_use_association()) all_passed = .false.
    if (.not. test_unrelated_derived_type_specifics()) all_passed = .false.
    if (.not. test_derived_type_assignment_accepted()) all_passed = .false.
    if (.not. test_generic_call_with_specific()) all_passed = .false.
    if (.not. test_kind_distinguishable_specifics()) all_passed = .false.
    if (.not. test_same_rank_array_specifics_ambiguous()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: malformed and ambiguous generic interfaces rejected'

contains

    logical function test_malformed_generic_binding()
        !! gfortran.dg/generic_29.f90: a GENERIC binding needs a generic
        !! specification and a binding name list; "generic ::" names nothing.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   type t'//new_line('a')// &
            '   contains'//new_line('a')// &
            '      generic ::'//new_line('a')// &
            '   end type'//new_line('a')// &
            'end'

        test_malformed_generic_binding = expect_error_contains( &
            source, 'malformed GENERIC binding', &
            '/tmp/ffc_reject_generic_01_r1')
    end function test_malformed_generic_binding

    logical function test_implicit_interface_ambiguity()
        !! gfortran.dg/pr95584.f90: two interface bodies whose dummies are all
        !! implicitly typed the same way cannot resolve the generic.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   interface s'//new_line('a')// &
            '      subroutine g(x, *)'//new_line('a')// &
            '      end'//new_line('a')// &
            '      subroutine h(y, *)'//new_line('a')// &
            '      end'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'end'

        test_implicit_interface_ambiguity = expect_error_contains( &
            source, 'ambiguous interfaces', &
            '/tmp/ffc_reject_generic_01_r2')
    end function test_implicit_interface_ambiguity

    logical function test_module_procedure_target()
        !! gfortran.dg/generic_14.f90: MODULE PROCEDURE may only name a module
        !! procedure, never an EXTERNAL procedure.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   external ext_sub'//new_line('a')// &
            '   interface gen'//new_line('a')// &
            '      module procedure ext_sub'//new_line('a')// &
            '   end interface gen'//new_line('a')// &
            'end module'//new_line('a')// &
            'end'

        test_module_procedure_target = expect_error_contains( &
            source, 'is not a module procedure', &
            '/tmp/ffc_reject_generic_01_r3')
    end function test_module_procedure_target

    logical function test_generic_name_collision()
        !! gfortran.dg/invalid_procedure_name.f90: a contained procedure cannot
        !! take the name of a generic interface of the same scoping unit.
        character(len=*), parameter :: source = &
            'interface i1'//new_line('a')// &
            '   subroutine s1(i)'//new_line('a')// &
            '   end subroutine s1'//new_line('a')// &
            'end interface i1'//new_line('a')// &
            'contains'//new_line('a')// &
            'subroutine i1(i)'//new_line('a')// &
            'end subroutine i1'//new_line('a')// &
            'end'

        test_generic_name_collision = expect_error_contains( &
            source, 'already defined as a generic', &
            '/tmp/ffc_reject_generic_01_r4')
    end function test_generic_name_collision

    logical function test_use_shadows_program_unit()
        !! gfortran.dg/interface_3.f90: a USE must not make accessible a name
        !! that is also the name of the current program unit.
        character(len=*), parameter :: source = &
            'module test_mod'//new_line('a')// &
            '   interface'//new_line('a')// &
            '      subroutine my_sub(a)'//new_line('a')// &
            '         real a'//new_line('a')// &
            '      end subroutine'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'subroutine my_sub(a)'//new_line('a')// &
            '   use test_mod'//new_line('a')// &
            '   real a'//new_line('a')// &
            '   print *, a'//new_line('a')// &
            'end subroutine'

        test_use_shadows_program_unit = expect_error_contains( &
            source, 'name of the current program unit', &
            '/tmp/ffc_reject_generic_01_r5')
    end function test_use_shadows_program_unit

    logical function test_ambiguous_use_association()
        !! gfortran.dg/generic_11.f90: a name use associated from two modules is
        !! ambiguous and must not be referenced.
        character(len=*), parameter :: source = &
            'module m_foo'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine foo'//new_line('a')// &
            '      print *, ''foo'''//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'module m_bar'//new_line('a')// &
            '   interface foo'//new_line('a')// &
            '      module procedure bar'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine bar'//new_line('a')// &
            '      print *, ''bar'''//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'use m_foo'//new_line('a')// &
            'use m_bar'//new_line('a')// &
            'call foo'//new_line('a')// &
            'end'

        test_ambiguous_use_association = expect_error_contains( &
            source, 'ambiguous reference', &
            '/tmp/ffc_reject_generic_01_r6')
    end function test_ambiguous_use_association

    logical function test_typebound_generic_inheritance()
        !! gfortran.dg/typebound_operator_14.f90: an extension type must not bind
        !! a generic operator that is indistinguishable from the inherited one.
        character(len=*), parameter :: source = &
            'module m_sort'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type, abstract :: sort_t'//new_line('a')// &
            '   contains'//new_line('a')// &
            '      generic :: operator(.gt.) => gt_cmp'//new_line('a')// &
            '      procedure :: gt_cmp'//new_line('a')// &
            '   end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '   logical function gt_cmp(a, b)'//new_line('a')// &
            '      class(sort_t), intent(in) :: a, b'//new_line('a')// &
            '      gt_cmp = .true.'//new_line('a')// &
            '   end function'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'module test'//new_line('a')// &
            '   use m_sort'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type, extends(sort_t) :: sort_int_t'//new_line('a')// &
            '      integer :: i'//new_line('a')// &
            '   contains'//new_line('a')// &
            '      generic :: operator(.gt.) => gt_cmp_int'//new_line('a')// &
            '      procedure :: gt_cmp_int'//new_line('a')// &
            '   end type'//new_line('a')// &
            'contains'//new_line('a')// &
            '   logical function gt_cmp_int(a, b) result(cmp)'//new_line('a')// &
            '      class(sort_int_t), intent(in) :: a, b'//new_line('a')// &
            '      cmp = a%i > b%i'//new_line('a')// &
            '   end function'//new_line('a')// &
            'end module'

        test_typebound_generic_inheritance = expect_error_contains( &
            source, 'are ambiguous', &
            '/tmp/ffc_reject_generic_01_r7')
    end function test_typebound_generic_inheritance

    logical function test_intrinsic_assignment_redefinition()
        !! gfortran.dg/redefined_intrinsic_assignment.f90: ASSIGNMENT(=) must not
        !! redefine an assignment that is already defined intrinsically.
        character(len=*), parameter :: source = &
            'module m1'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface assignment(=)'//new_line('a')// &
            '      module procedure t1'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine t1(i, j)'//new_line('a')// &
            '      integer, intent(out) :: i'//new_line('a')// &
            '      integer, intent(in) :: j'//new_line('a')// &
            '      i = -j'//new_line('a')// &
            '   end subroutine t1'//new_line('a')// &
            'end module m1'

        test_intrinsic_assignment_redefinition = expect_error_contains( &
            source, 'redefine an INTRINSIC', &
            '/tmp/ffc_reject_generic_01_r8')
    end function test_intrinsic_assignment_redefinition

    logical function test_generic_call_without_specific()
        !! gfortran.dg/generic_5.f90: a generic reference must match one of its
        !! specific procedures.
        character(len=*), parameter :: source = &
            'module ice_gfortran'//new_line('a')// &
            '   interface ice'//new_line('a')// &
            '      module procedure ice_i'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine ice_i(i)'//new_line('a')// &
            '      integer, intent(in) :: i'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use ice_gfortran'//new_line('a')// &
            '   call ice(23.0)'//new_line('a')// &
            'end program'

        test_generic_call_without_specific = expect_error_contains( &
            source, 'no specific subroutine', &
            '/tmp/ffc_reject_generic_01_r9')
    end function test_generic_call_without_specific

    logical function test_allocatable_pointer_ambiguity()
        !! gfortran.dg/generic_32.f90: ALLOCATABLE and POINTER do not distinguish
        !! two specifics of the same type, kind and rank.
        character(len=*), parameter :: source = &
            'interface gen'//new_line('a')// &
            '   subroutine suba(a)'//new_line('a')// &
            '      real, allocatable :: a(:)'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            '   subroutine subp(p)'//new_line('a')// &
            '      real, pointer, intent(in) :: p(:)'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end interface'//new_line('a')// &
            'end'

        test_allocatable_pointer_ambiguity = expect_error_contains( &
            source, 'ambiguous interfaces', &
            '/tmp/ffc_reject_generic_01_r10')
    end function test_allocatable_pointer_ambiguity

    logical function test_unlimited_polymorphic_ambiguity()
        !! gfortran.dg/generic_34.f90: an unlimited polymorphic dummy is type
        !! compatible with every actual argument, so it distinguishes nothing.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   type :: t'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   interface sub'//new_line('a')// &
            '      module procedure s1'//new_line('a')// &
            '      module procedure s2'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine s1(x)'//new_line('a')// &
            '      type(t) :: x'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            '   subroutine s2(x)'//new_line('a')// &
            '      class(*), allocatable :: x'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'

        test_unlimited_polymorphic_ambiguity = expect_error_contains( &
            source, 'ambiguous interfaces', &
            '/tmp/ffc_reject_generic_01_r11')
    end function test_unlimited_polymorphic_ambiguity

    logical function test_generic_identifier_accepted()
        !! Corrected neighbour of generic_29: a well formed GENERIC binding
        !! passes this rule; what is left is the unrelated unsupported
        !! type-bound procedure diagnostic, never a malformed-GENERIC one.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: t'//new_line('a')// &
            '    integer :: n'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    generic :: g => f1, f2'//new_line('a')// &
            '  end type t'//new_line('a')// &
            '  type(t) :: x'//new_line('a')// &
            '  x%n = 1'//new_line('a')// &
            '  stop x%n'//new_line('a')// &
            'end program main'

        test_generic_identifier_accepted = expect_error_contains( &
            source, 'type-bound procedure', &
            '/tmp/ffc_reject_generic_01_wellformed')
    end function test_generic_identifier_accepted

    logical function test_distinguishable_implicit_interfaces()
        !! Corrected neighbour of pr95584: implicit typing makes x real and n
        !! integer, so the two interface bodies are distinguishable.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '   interface s'//new_line('a')// &
            '      subroutine g(x, *)'//new_line('a')// &
            '      end subroutine'//new_line('a')// &
            '      subroutine h(n, *)'//new_line('a')// &
            '      end subroutine'//new_line('a')// &
            '   end interface'//new_line('a')// &
            '   print *, ''ok'''//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_distinguishable_implicit_interfaces = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_n2')
    end function test_distinguishable_implicit_interfaces

    logical function test_module_procedure_accepted()
        !! Corrected neighbour of generic_14: MODULE PROCEDURE naming a real
        !! module procedure is accepted.
        character(len=*), parameter :: source = &
            'module m3'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface gen'//new_line('a')// &
            '      module procedure ok3'//new_line('a')// &
            '   end interface gen'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine ok3(a)'//new_line('a')// &
            '      integer, intent(in) :: a'//new_line('a')// &
            '      print *, a'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m3'//new_line('a')// &
            '   call gen(5)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_module_procedure_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_n3')
    end function test_module_procedure_accepted

    logical function test_distinct_generic_and_specific()
        !! Corrected neighbour of invalid_procedure_name: the generic and its
        !! specific have different names.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface i1'//new_line('a')// &
            '      module procedure s1'//new_line('a')// &
            '   end interface i1'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine s1(i)'//new_line('a')// &
            '      integer, intent(in) :: i'//new_line('a')// &
            '      print *, i'//new_line('a')// &
            '   end subroutine s1'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m'//new_line('a')// &
            '   call i1(9)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_distinct_generic_and_specific = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_q4')
    end function test_distinct_generic_and_specific

    logical function test_use_without_shadowing()
        !! Corrected neighbour of interface_3: the used module exports no name
        !! that clashes with the enclosing program unit.
        character(len=*), parameter :: source = &
            'module test_mod'//new_line('a')// &
            '   interface'//new_line('a')// &
            '      subroutine other_sub(a)'//new_line('a')// &
            '         real a'//new_line('a')// &
            '      end subroutine'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'module use_mod'//new_line('a')// &
            '   use test_mod'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine my_sub(a)'//new_line('a')// &
            '      real, intent(in) :: a'//new_line('a')// &
            '      print *, a'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use use_mod'//new_line('a')// &
            '   call my_sub(2.5)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_use_without_shadowing = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_q5')
    end function test_use_without_shadowing

    logical function test_unambiguous_use_association()
        !! Corrected neighbour of generic_11: the two modules export different
        !! names, so both references resolve.
        character(len=*), parameter :: source = &
            'module ma'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine foo_a()'//new_line('a')// &
            '      print *, ''a'''//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'module mb'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface foo_b'//new_line('a')// &
            '      module procedure bar_b'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine bar_b()'//new_line('a')// &
            '      print *, ''b'''//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use ma'//new_line('a')// &
            '   use mb'//new_line('a')// &
            '   call foo_a'//new_line('a')// &
            '   call foo_b'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_unambiguous_use_association = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_n5')
    end function test_unambiguous_use_association

    logical function test_unrelated_derived_type_specifics()
        !! Corrected neighbour of typebound_operator_14: specifics taking two
        !! unrelated derived types stay distinguishable.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type :: alpha_t'//new_line('a')// &
            '      integer :: i = 0'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   type :: beta_t'//new_line('a')// &
            '      real :: r = 0.0'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   interface show'//new_line('a')// &
            '      module procedure show_alpha'//new_line('a')// &
            '      module procedure show_beta'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine show_alpha(a)'//new_line('a')// &
            '      type(alpha_t), intent(in) :: a'//new_line('a')// &
            '      print *, a%i'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            '   subroutine show_beta(b)'//new_line('a')// &
            '      type(beta_t), intent(in) :: b'//new_line('a')// &
            '      print *, b%r'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m'//new_line('a')// &
            '   type(alpha_t) :: a'//new_line('a')// &
            '   type(beta_t) :: b'//new_line('a')// &
            '   a%i = 1'//new_line('a')// &
            '   b%r = 2.0'//new_line('a')// &
            '   call show(a)'//new_line('a')// &
            '   call show(b)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_unrelated_derived_type_specifics = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_q7')
    end function test_unrelated_derived_type_specifics

    logical function test_derived_type_assignment_accepted()
        !! Corrected neighbour of redefined_intrinsic_assignment: assigning an
        !! integer to a derived type is not an intrinsic assignment.
        character(len=*), parameter :: source = &
            'module m6'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type :: pair_t'//new_line('a')// &
            '      integer :: a = 0'//new_line('a')// &
            '   end type'//new_line('a')// &
            '   interface assignment(=)'//new_line('a')// &
            '      module procedure pair_from_int'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine pair_from_int(p, i)'//new_line('a')// &
            '      type(pair_t), intent(out) :: p'//new_line('a')// &
            '      integer, intent(in) :: i'//new_line('a')// &
            '      p%a = i'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m6'//new_line('a')// &
            '   type(pair_t) :: q'//new_line('a')// &
            '   q = 4'//new_line('a')// &
            '   print *, q%a'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_derived_type_assignment_accepted = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_n6')
    end function test_derived_type_assignment_accepted

    logical function test_generic_call_with_specific()
        !! Corrected neighbour of generic_5: the actual argument matches the one
        !! specific procedure.
        character(len=*), parameter :: source = &
            'module m7'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface pick'//new_line('a')// &
            '      module procedure pick_i'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine pick_i(i)'//new_line('a')// &
            '      integer, intent(in) :: i'//new_line('a')// &
            '      print *, i'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m7'//new_line('a')// &
            '   call pick(23)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_generic_call_with_specific = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_n7')
    end function test_generic_call_with_specific

    logical function test_kind_distinguishable_specifics()
        !! Corrected neighbour of generic_32 and generic_34: specifics differing
        !! in declared type remain distinguishable.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface gen'//new_line('a')// &
            '      module procedure int_case'//new_line('a')// &
            '      module procedure real_case'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine int_case(a)'//new_line('a')// &
            '      integer, intent(in) :: a'//new_line('a')// &
            '      print *, a'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            '   subroutine real_case(a)'//new_line('a')// &
            '      real, intent(in) :: a'//new_line('a')// &
            '      print *, a'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m'//new_line('a')// &
            '   call gen(3)'//new_line('a')// &
            '   stop 0'//new_line('a')// &
            'end program'

        test_kind_distinguishable_specifics = expect_exit_status( &
            source, 0, '/tmp/ffc_reject_generic_01_q10')
    end function test_kind_distinguishable_specifics

    logical function test_same_rank_array_specifics_ambiguous()
        !! Negative control for #595: rank distinguishes specifics only when it
        !! differs. Two rank-1 integer array specifics stay ambiguous.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   interface gen'//new_line('a')// &
            '      module procedure one_vec'//new_line('a')// &
            '      module procedure other_vec'//new_line('a')// &
            '   end interface'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine one_vec(a)'//new_line('a')// &
            '      integer, intent(in) :: a(:)'//new_line('a')// &
            '      print *, a(1)'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            '   subroutine other_vec(a)'//new_line('a')// &
            '      integer, intent(in) :: a(:)'//new_line('a')// &
            '      print *, a(1)'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   use m'//new_line('a')// &
            '   integer :: v(3)'//new_line('a')// &
            '   v = 1'//new_line('a')// &
            '   call gen(v)'//new_line('a')// &
            'end program'

        test_same_rank_array_specifics_ambiguous = expect_error_contains( &
            source, 'ambiguous interfaces', &
            '/tmp/ffc_reject_generic_01_r595')
    end function test_same_rank_array_specifics_ambiguous

end program test_session_reject_generic_01_compiler
