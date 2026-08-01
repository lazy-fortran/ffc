program test_session_use_association_valid_compiler
    !! Valid USE associations must not be reported as ambiguous (#587).
    use ffc_test_support, only: expect_error_absent
    implicit none

    logical :: all_passed

    print *, '=== valid USE associations accepted ==='

    all_passed = .true.
    if (.not. test_generic_extends_module_procedure()) all_passed = .false.
    if (.not. test_local_interface_is_not_export()) all_passed = .false.
    if (.not. test_typebound_component_reference()) all_passed = .false.
    if (.not. test_renamed_use_of_own_name()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: valid USE associations accepted'

contains

    logical function test_generic_extends_module_procedure()
        !! gfortran.dg/generic_19.f90: mod2 extends mod1's SUB into a generic
        !! that lists SUB itself, so USE of both modules names one generic.
        character(len=*), parameter :: source = &
            'module mod1'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine sub(x)'//new_line('a')// &
            '      real x'//new_line('a')// &
            '      print *, x'//new_line('a')// &
            '   end subroutine sub'//new_line('a')// &
            'end module mod1'//new_line('a')// &
            ''//new_line('a')// &
            'module mod2'//new_line('a')// &
            '   use mod1'//new_line('a')// &
            '   interface sub'//new_line('a')// &
            '      module procedure sub, sub_int'//new_line('a')// &
            '   end interface sub'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine sub_int(i)'//new_line('a')// &
            '      integer i'//new_line('a')// &
            '      print *, i'//new_line('a')// &
            '   end subroutine sub_int'//new_line('a')// &
            'end module mod2'//new_line('a')// &
            ''//new_line('a')// &
            'program prog'//new_line('a')// &
            '   use mod1'//new_line('a')// &
            '   use mod2'//new_line('a')// &
            '   call sub(1)'//new_line('a')// &
            'end program prog'

        test_generic_extends_module_procedure = expect_error_absent( &
            source, 'ambiguous reference', &
            '/tmp/ffc_use_assoc_valid_1')
    end function test_generic_extends_module_procedure

    logical function test_local_interface_is_not_export()
        !! gfortran.dg/generic_13.f90: the interface block inside a contained
        !! procedure is local to it and exports nothing from its module.
        character(len=*), parameter :: source = &
            'module test'//new_line('a')// &
            '   interface xx'//new_line('a')// &
            '      module procedure xx'//new_line('a')// &
            '   end interface'//new_line('a')// &
            '   public :: xx'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine xx(i)'//new_line('a')// &
            '      integer :: i'//new_line('a')// &
            '      i = 7'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module test'//new_line('a')// &
            ''//new_line('a')// &
            'module too'//new_line('a')// &
            'contains'//new_line('a')// &
            '   subroutine sub(i)'//new_line('a')// &
            '      interface'//new_line('a')// &
            '         subroutine xx(i)'//new_line('a')// &
            '            integer :: i'//new_line('a')// &
            '         end subroutine'//new_line('a')// &
            '      end interface'//new_line('a')// &
            '      integer :: i'//new_line('a')// &
            '      i = 1'//new_line('a')// &
            '   end subroutine'//new_line('a')// &
            'end module too'//new_line('a')// &
            ''//new_line('a')// &
            'program tt'//new_line('a')// &
            '   use test'//new_line('a')// &
            '   use too'//new_line('a')// &
            '   integer :: i'//new_line('a')// &
            '   call xx(i)'//new_line('a')// &
            'end program tt'

        test_local_interface_is_not_export = expect_error_absent( &
            source, 'ambiguous reference', &
            '/tmp/ffc_use_assoc_valid_2')
    end function test_local_interface_is_not_export

    logical function test_typebound_component_reference()
        !! gfortran.dg/use_26.f90: a %sizereturn() type-bound reference names a
        !! binding, not the use associated module procedure of that name.
        character(len=*), parameter :: source = &
            'module a'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type :: a_type'//new_line('a')// &
            '      integer :: isize = 1'//new_line('a')// &
            '   contains'//new_line('a')// &
            '      procedure :: sizereturn'//new_line('a')// &
            '   end type a_type'//new_line('a')// &
            'contains'//new_line('a')// &
            '   function sizereturn(self)'//new_line('a')// &
            '      integer :: sizereturn'//new_line('a')// &
            '      class(a_type) :: self'//new_line('a')// &
            '      sizereturn = self%isize'//new_line('a')// &
            '   end function sizereturn'//new_line('a')// &
            'end module a'//new_line('a')// &
            ''//new_line('a')// &
            'module b'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type :: b_type'//new_line('a')// &
            '      integer :: isize = 2'//new_line('a')// &
            '   contains'//new_line('a')// &
            '      procedure :: sizereturn'//new_line('a')// &
            '   end type b_type'//new_line('a')// &
            'contains'//new_line('a')// &
            '   function sizereturn(self)'//new_line('a')// &
            '      integer :: sizereturn'//new_line('a')// &
            '      class(b_type) :: self'//new_line('a')// &
            '      sizereturn = self%isize'//new_line('a')// &
            '   end function sizereturn'//new_line('a')// &
            'end module b'//new_line('a')// &
            ''//new_line('a')// &
            'program main'//new_line('a')// &
            '   use a'//new_line('a')// &
            '   use b'//new_line('a')// &
            '   implicit none'//new_line('a')// &
            '   type(a_type) :: ai'//new_line('a')// &
            '   type(b_type) :: bi'//new_line('a')// &
            '   print *, ai%sizereturn()'//new_line('a')// &
            '   print *, bi%sizereturn()'//new_line('a')// &
            'end program main'

        test_typebound_component_reference = expect_error_absent( &
            source, 'ambiguous reference', &
            '/tmp/ffc_use_assoc_valid_3')
    end function test_typebound_component_reference

    logical function test_renamed_use_of_own_name()
        !! gfortran.dg/use_14.f90: the module entity is renamed, so the local
        !! name of the program unit stays free.
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
            '   use test_mod, gugu => my_sub'//new_line('a')// &
            '   real a'//new_line('a')// &
            '   print *, a'//new_line('a')// &
            'end subroutine'//new_line('a')// &
            ''//new_line('a')// &
            'program p'//new_line('a')// &
            '   call my_sub(1.0)'//new_line('a')// &
            'end program p'

        test_renamed_use_of_own_name = expect_error_absent( &
            source, 'name of the current program unit', &
            '/tmp/ffc_use_assoc_valid_4')
    end function test_renamed_use_of_own_name

end program test_session_use_association_valid_compiler
