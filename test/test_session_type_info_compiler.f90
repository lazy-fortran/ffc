program test_session_type_info_compiler
    use ffc_test_support, only: expect_exe_has_symbol
    implicit none

    logical :: all_passed

    print *, '=== derived-type type_info emission tests ==='

    all_passed = .true.
    if (.not. test_each_derived_type_definition_emits_type_info()) &
        all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: derived-type type_info emission'

contains

    logical function test_each_derived_type_definition_emits_type_info()
        ! Two derived type definitions emit two distinct __ffc_type_info_*
        ! constants into the object.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: point_t'//new_line('a')// &
            '    integer :: x'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type point_t'//new_line('a')// &
            '  type :: pair_t'//new_line('a')// &
            '    integer :: a'//new_line('a')// &
            '  end type pair_t'//new_line('a')// &
            '  type(point_t) :: p'//new_line('a')// &
            '  p%x = 1'//new_line('a')// &
            '  stop 0'//new_line('a')// &
            'end program main'

        test_each_derived_type_definition_emits_type_info = .true.
        if (.not. expect_exe_has_symbol(source, &
            '/tmp/ffc_type_info_point.o', '__ffc_type_info_point_t')) &
            test_each_derived_type_definition_emits_type_info = .false.
        if (.not. expect_exe_has_symbol(source, &
            '/tmp/ffc_type_info_pair.o', '__ffc_type_info_pair_t')) &
            test_each_derived_type_definition_emits_type_info = .false.
    end function test_each_derived_type_definition_emits_type_info

end program test_session_type_info_compiler
