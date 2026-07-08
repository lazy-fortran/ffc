program test_session_module_array_variable_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session module-array-variable compiler test ==='

    all_passed = .true.
    if (.not. test_fixed_size_element_rw()) all_passed = .false.
    if (.not. test_parameter_sized_size_inquiry()) all_passed = .false.
    if (.not. test_zero_size_module_array()) all_passed = .false.
    if (.not. test_constructor_initializer()) all_passed = .false.
    if (.not. test_host_association_fill()) all_passed = .false.
    if (.not. test_contained_function_shadows_module_array()) &
        all_passed = .false.
    if (.not. test_real_whole_array()) all_passed = .false.
    if (.not. test_only_clause_import()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module-level array variables persist across use'

contains

    logical function test_fixed_size_element_rw()
        ! A fixed literal-sized module array, USE-imported, written and read
        ! element-wise; the sum leaves the process via the exit code.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: a(5)'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  a(1) = 10'//new_line('a')// &
            '  a(5) = 7'//new_line('a')// &
            '  stop a(1) + a(5)'//new_line('a')// &
            'end program main'

        test_fixed_size_element_rw = expect_exit_status( &
            source, 17, '/tmp/ffc_session_modarr_rw')
    end function test_fixed_size_element_rw

    logical function test_parameter_sized_size_inquiry()
        ! A module array whose extent is a named PARAMETER; SIZE must fold to
        ! that compile-time extent (mirrors lfortran declaration_02).
        character(len=*), parameter :: source = &
            'module dims'//new_line('a')// &
            '  integer, parameter :: m = 3'//new_line('a')// &
            'end module dims'//new_line('a')// &
            'module store'//new_line('a')// &
            '  use dims'//new_line('a')// &
            '  integer, dimension(m) :: arr'//new_line('a')// &
            'end module store'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use store'//new_line('a')// &
            '  stop size(arr)'//new_line('a')// &
            'end program main'

        test_parameter_sized_size_inquiry = expect_exit_status( &
            source, 3, '/tmp/ffc_session_modarr_size')
    end function test_parameter_sized_size_inquiry

    logical function test_zero_size_module_array()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: a(0)'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop size(a)'//new_line('a')// &
            'end program main'

        test_zero_size_module_array = expect_exit_status( &
            source, 0, '/tmp/ffc_session_modarr_zero_size')
    end function test_zero_size_module_array

    logical function test_constructor_initializer()
        ! An array-constructor initialiser folds into the global's static bytes;
        ! SUM over the whole array leaves via the exit code.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer, dimension(3) :: a = [4, 5, 6]'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  stop sum(a)'//new_line('a')// &
            'end program main'

        test_constructor_initializer = expect_exit_status( &
            source, 15, '/tmp/ffc_session_modarr_init')
    end function test_constructor_initializer

    logical function test_host_association_fill()
        ! A module procedure host-associates the module array and fills it; the
        ! program reads the shared storage back.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: a(4)'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine fill()'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '    do i = 1, 4'//new_line('a')// &
            '      a(i) = i * i'//new_line('a')// &
            '    end do'//new_line('a')// &
            '  end subroutine fill'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call fill()'//new_line('a')// &
            '  stop a(1) + a(2) + a(3) + a(4)'//new_line('a')// &
            'end program main'

        test_host_association_fill = expect_exit_status( &
            source, 30, '/tmp/ffc_session_modarr_host')
    end function test_host_association_fill

    logical function test_contained_function_shadows_module_array()
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  real :: x(3) = [1.5, 2.5, 3.5]'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine s'//new_line('a')// &
            '    if (x(2) == 2.5) stop 1'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    function x(n, m)'//new_line('a')// &
            '      integer, optional :: m'//new_line('a')// &
            '      if (present(m)) then'//new_line('a')// &
            '        x = real(n)**m'//new_line('a')// &
            '      else'//new_line('a')// &
            '        x = 0.0'//new_line('a')// &
            '      end if'//new_line('a')// &
            '    end function x'//new_line('a')// &
            '  end subroutine s'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  call s'//new_line('a')// &
            'end program main'

        test_contained_function_shadows_module_array = expect_exit_status( &
            source, 0, '/tmp/ffc_session_modarr_fn_shadow')
    end function test_contained_function_shadows_module_array

    logical function test_real_whole_array()
        ! A real(8) module array printed whole-array after assignment.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  real(8) :: v(3)'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m'//new_line('a')// &
            '  v = 2.0d0'//new_line('a')// &
            '  print "(f6.2)", sum(v)'//new_line('a')// &
            'end program main'

        test_real_whole_array = expect_output( &
            source, '  6.00'//new_line('a'), '/tmp/ffc_session_modarr_real')
    end function test_real_whole_array

    logical function test_only_clause_import()
        ! use, only: names the module array explicitly.
        character(len=*), parameter :: source = &
            'module m'//new_line('a')// &
            '  integer :: a(2) = [8, 9]'//new_line('a')// &
            '  integer :: b(2) = [1, 1]'//new_line('a')// &
            'end module m'//new_line('a')// &
            'program main'//new_line('a')// &
            '  use m, only: a'//new_line('a')// &
            '  stop a(1) + a(2)'//new_line('a')// &
            'end program main'

        test_only_clause_import = expect_exit_status( &
            source, 17, '/tmp/ffc_session_modarr_only')
    end function test_only_clause_import

end program test_session_module_array_variable_compiler
