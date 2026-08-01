program test_session_accept_use_generic_heuristics_compiler
    ! ffc #620: the line-based generic/USE rejection heuristics must not
    ! reject valid programs. The accepted-side cases are reduced from
    ! gfortran.dg/generic_6.f90, use_14.f90 and use_26.f90, all of which
    ! gfortran -fsyntax-only accepts. The rejected-side cases are negative
    ! controls: the heuristics must still catch the real violations.
    use ffc_test_support, only: expect_error_contains, expect_error_lacks, &
                                expect_no_error
    implicit none

    logical :: all_passed
    character(len=*), parameter :: nl = new_line('a')

    print *, '=== USE/generic rejection heuristic accept tests ==='

    all_passed = .true.
    if (.not. test_host_and_local_generic_accepted()) all_passed = .false.
    if (.not. test_renamed_use_accepted()) all_passed = .false.
    if (.not. test_type_bound_ambiguity_accepted()) all_passed = .false.
    if (.not. test_unmatched_generic_still_rejected()) all_passed = .false.
    if (.not. test_use_shadow_still_rejected()) all_passed = .false.
    if (.not. test_ambiguous_use_still_rejected()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: USE/generic heuristics accept valid programs'

contains

    ! gfortran.dg/generic_6.f90: the generic CREATE visible in useCreate is the
    ! union of the host-associated one from A and the locally use-associated
    ! one from B.
    logical function test_host_and_local_generic_accepted()
        character(len=:), allocatable :: source

        source = 'module a_mod'//nl// &
                 '  interface create'//nl// &
                 '    module procedure create1'//nl// &
                 '  end interface'//nl// &
                 'contains'//nl// &
                 '  subroutine create1'//nl// &
                 '    print *, "module A"'//nl// &
                 '  end subroutine'//nl// &
                 'end module a_mod'//nl// &
                 'module b_mod'//nl// &
                 '  interface create'//nl// &
                 '    module procedure create1'//nl// &
                 '  end interface'//nl// &
                 'contains'//nl// &
                 '  subroutine create1(a)'//nl// &
                 '    integer a'//nl// &
                 '    print *, "module B"'//nl// &
                 '  end subroutine'//nl// &
                 'end module b_mod'//nl// &
                 'module c_mod'//nl// &
                 '  use a_mod'//nl// &
                 'contains'//nl// &
                 '  subroutine usecreate'//nl// &
                 '    use b_mod'//nl// &
                 '    call create()'//nl// &
                 '    call create(1)'//nl// &
                 '  end subroutine'//nl// &
                 'end module c_mod'//nl// &
                 'program main'//nl// &
                 '  use c_mod'//nl// &
                 '  call usecreate'//nl// &
                 'end program main'
        ! Lowering a generic dispatched across modules is a separate gap;
        ! what must hold here is that the resolution rule does not fire.
        test_host_and_local_generic_accepted = expect_error_lacks(source, &
            'matches this reference', '/tmp/ffc620_generic_union')
    end function test_host_and_local_generic_accepted

    ! gfortran.dg/use_14.f90: the rename moves the clashing name away.
    logical function test_renamed_use_accepted()
        character(len=:), allocatable :: source

        source = 'module test_mod'//nl// &
                 '  interface'//nl// &
                 '    subroutine my_sub(a)'//nl// &
                 '      real a'//nl// &
                 '    end subroutine'//nl// &
                 '  end interface'//nl// &
                 'end module test_mod'//nl// &
                 'subroutine my_sub(a)'//nl// &
                 '  use test_mod, gugu => my_sub'//nl// &
                 '  real a'//nl// &
                 '  print *, a'//nl// &
                 'end subroutine'//nl// &
                 'program main'//nl// &
                 'end program main'
        test_renamed_use_accepted = expect_no_error(source, &
            '/tmp/ffc620_use_rename')
    end function test_renamed_use_accepted

    ! gfortran.dg/use_26.f90: sizereturn is ambiguous as a name, but every
    ! reference is a type-bound call resolved through the declared type.
    logical function test_type_bound_ambiguity_accepted()
        character(len=:), allocatable :: source

        source = 'module a_mod'//nl// &
                 '  implicit none'//nl// &
                 '  type :: a_type'//nl// &
                 '    integer :: isize = 1'//nl// &
                 '  contains'//nl// &
                 '    procedure :: sizereturn'//nl// &
                 '  end type a_type'//nl// &
                 'contains'//nl// &
                 '  function sizereturn(self)'//nl// &
                 '    integer :: sizereturn'//nl// &
                 '    class(a_type) :: self'//nl// &
                 '    sizereturn = self%isize'//nl// &
                 '  end function sizereturn'//nl// &
                 'end module a_mod'//nl// &
                 'module b_mod'//nl// &
                 '  implicit none'//nl// &
                 '  type :: b_type'//nl// &
                 '    integer :: isize = 2'//nl// &
                 '  contains'//nl// &
                 '    procedure :: sizereturn'//nl// &
                 '  end type b_type'//nl// &
                 'contains'//nl// &
                 '  function sizereturn(self)'//nl// &
                 '    integer :: sizereturn'//nl// &
                 '    class(b_type) :: self'//nl// &
                 '    sizereturn = self%isize'//nl// &
                 '  end function sizereturn'//nl// &
                 'end module b_mod'//nl// &
                 'program main'//nl// &
                 '  use a_mod'//nl// &
                 '  use b_mod'//nl// &
                 '  implicit none'//nl// &
                 '  type(a_type) :: ai'//nl// &
                 '  type(b_type) :: bi'//nl// &
                 '  print *, ai%sizereturn()'//nl// &
                 '  print *, bi%sizereturn()'//nl// &
                 'end program main'
        test_type_bound_ambiguity_accepted = expect_error_lacks(source, &
            'use associated from more than one module', &
            '/tmp/ffc620_type_bound')
    end function test_type_bound_ambiguity_accepted

    logical function test_unmatched_generic_still_rejected()
        character(len=:), allocatable :: source

        source = 'module g_mod'//nl// &
                 '  interface create'//nl// &
                 '    module procedure create1'//nl// &
                 '  end interface'//nl// &
                 'contains'//nl// &
                 '  subroutine create1(a)'//nl// &
                 '    integer a'//nl// &
                 '    print *, a'//nl// &
                 '  end subroutine'//nl// &
                 'end module g_mod'//nl// &
                 'program main'//nl// &
                 '  use g_mod'//nl// &
                 '  call create(.true.)'//nl// &
                 'end program main'
        test_unmatched_generic_still_rejected = expect_error_contains(source, &
            'matches this reference', '/tmp/ffc620_generic_bad')
    end function test_unmatched_generic_still_rejected

    logical function test_use_shadow_still_rejected()
        character(len=:), allocatable :: source

        source = 'module test_mod'//nl// &
                 '  interface'//nl// &
                 '    subroutine my_sub(a)'//nl// &
                 '      real a'//nl// &
                 '    end subroutine'//nl// &
                 '  end interface'//nl// &
                 'end module test_mod'//nl// &
                 'subroutine my_sub(a)'//nl// &
                 '  use test_mod'//nl// &
                 '  real a'//nl// &
                 '  print *, a'//nl// &
                 'end subroutine'//nl// &
                 'program main'//nl// &
                 'end program main'
        test_use_shadow_still_rejected = expect_error_contains(source, &
            'is also the name of the current program unit', &
            '/tmp/ffc620_use_shadow')
    end function test_use_shadow_still_rejected

    logical function test_ambiguous_use_still_rejected()
        character(len=:), allocatable :: source

        source = 'module m1'//nl// &
                 'contains'//nl// &
                 '  subroutine shared_thing()'//nl// &
                 '    print *, 1'//nl// &
                 '  end subroutine shared_thing'//nl// &
                 'end module m1'//nl// &
                 'module m2'//nl// &
                 'contains'//nl// &
                 '  subroutine shared_thing()'//nl// &
                 '    print *, 2'//nl// &
                 '  end subroutine shared_thing'//nl// &
                 'end module m2'//nl// &
                 'program main'//nl// &
                 '  use m1'//nl// &
                 '  use m2'//nl// &
                 '  call shared_thing()'//nl// &
                 'end program main'
        test_ambiguous_use_still_rejected = expect_error_contains(source, &
            'use associated from more than one module', &
            '/tmp/ffc620_ambiguous')
    end function test_ambiguous_use_still_rejected

end program test_session_accept_use_generic_heuristics_compiler
