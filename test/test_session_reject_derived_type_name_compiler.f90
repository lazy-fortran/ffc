program test_session_reject_derived_type_name_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== intrinsic derived-type name rejection test ==='

    all_passed = .true.
    if (.not. test_intrinsic_type_names_rejected()) all_passed = .false.
    if (.not. test_double_colon_intrinsic_name_rejected()) all_passed = .false.
    if (.not. test_valid_near_miss_type_name_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: intrinsic type names cannot name derived types'

contains

    logical function test_intrinsic_type_names_rejected()
        test_intrinsic_type_names_rejected = .true.
        call expect_intrinsic_name_rejected('integer', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('real', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('complex', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('character', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('logical', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('doubleprecision', &
            test_intrinsic_type_names_rejected)
        call expect_intrinsic_name_rejected('doublecomplex', &
            test_intrinsic_type_names_rejected)
    end function test_intrinsic_type_names_rejected

    logical function test_double_colon_intrinsic_name_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: integer'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type integer'//new_line('a')// &
            'end program main'

        test_double_colon_intrinsic_name_rejected = expect_error_contains( &
            source, 'cannot be the same as an intrinsic type', &
            '/tmp/ffc_session_reject_derived_name_colon')
    end function test_double_colon_intrinsic_name_rejected

    subroutine expect_intrinsic_name_rejected(type_name, ok)
        character(len=*), intent(in) :: type_name
        logical, intent(inout) :: ok
        character(len=:), allocatable :: source, exe_path

        source = derived_type_source(type_name)
        exe_path = '/tmp/ffc_session_reject_derived_name_'//trim(type_name)
        if (.not. expect_error_contains(source, &
            'cannot be the same as an intrinsic type', exe_path)) then
            ok = .false.
        end if
    end subroutine expect_intrinsic_name_rejected

    logical function test_valid_near_miss_type_name_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type integer_t'//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type integer_t'//new_line('a')// &
            '  type(integer_t) :: x'//new_line('a')// &
            '  x%y = 3'//new_line('a')// &
            '  stop x%y'//new_line('a')// &
            'end program main'

        test_valid_near_miss_type_name_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_session_derived_name_near_miss')
    end function test_valid_near_miss_type_name_accepted

    function derived_type_source(type_name) result(source)
        character(len=*), intent(in) :: type_name
        character(len=:), allocatable :: source

        source = 'program main'//new_line('a')// &
            '  type '//trim(type_name)//new_line('a')// &
            '    integer :: y'//new_line('a')// &
            '  end type '//trim(type_name)//new_line('a')// &
            'end program main'
    end function derived_type_source

end program test_session_reject_derived_type_name_compiler
