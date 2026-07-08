program test_session_block_data_compiler
    use ffc_test_support, only: expect_exit_status, expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session block-data / common compiler test ==='

    all_passed = .true.
    if (.not. test_integer_common_initialized()) all_passed = .false.
    if (.not. test_mixed_common_initialized()) all_passed = .false.
    if (.not. test_common_assignment_writes_through()) all_passed = .false.
    if (.not. test_common_declared_after_common_statement()) all_passed = .false.
    if (.not. test_common_array_shared_across_units()) all_passed = .false.
    if (.not. test_zero_size_common_array()) all_passed = .false.
    if (.not. test_do_loop_over_common_variable()) all_passed = .false.
    if (.not. test_whole_array_data_statement()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: block data initializes shared common storage'

contains

    logical function test_integer_common_initialized()
        ! Corpus issue_1578: a BLOCK DATA unit's DATA value reaches the program
        ! through the shared COMMON slot, so the program prints 999.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: global_val'//new_line('a')// &
            '  common /shared/ global_val'//new_line('a')// &
            '  print *, global_val'//new_line('a')// &
            'end program p'//new_line('a')// &
            'block data init_data'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: global_val'//new_line('a')// &
            '  common /shared/ global_val'//new_line('a')// &
            '  data global_val /999/'//new_line('a')// &
            'end block data init_data'
        character(len=*), parameter :: expected = &
            '         999'//new_line('a')

        test_integer_common_initialized = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_int')
    end function test_integer_common_initialized

    logical function test_mixed_common_initialized()
        ! Corpus issue_1900: integer and real slots in one COMMON block keep
        ! their declared kinds and their BLOCK DATA initial values.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  real :: x, y'//new_line('a')// &
            '  common /myblock/ a, b, x, y'//new_line('a')// &
            '  print *, a, b'//new_line('a')// &
            '  print *, x, y'//new_line('a')// &
            'end program p'//new_line('a')// &
            'block data init_data'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: a, b'//new_line('a')// &
            '  real :: x, y'//new_line('a')// &
            '  common /myblock/ a, b, x, y'//new_line('a')// &
            '  data a, b / 10, 20 /'//new_line('a')// &
            '  data x, y / 3.5, 7.2 /'//new_line('a')// &
            'end block data init_data'
        character(len=*), parameter :: expected = &
            '          10          20'//new_line('a')// &
            '   3.50000000       7.19999981    '//new_line('a')

        test_mixed_common_initialized = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_mixed')
    end function test_mixed_common_initialized

    logical function test_common_assignment_writes_through()
        ! A COMMON variable bound to its slot global is ordinary storage: an
        ! assignment overwrites the BLOCK DATA initial value before the print.
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  common /c/ n'//new_line('a')// &
            '  n = n + 1'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program p'//new_line('a')// &
            'block data bd'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  common /c/ n'//new_line('a')// &
            '  data n /41/'//new_line('a')// &
            'end block data bd'
        character(len=*), parameter :: expected = &
            '          42'//new_line('a')

        test_common_assignment_writes_through = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_write')
    end function test_common_assignment_writes_through

    logical function test_common_declared_after_common_statement()
        ! Corpus issue_1578: the COMMON statement may precede its members'
        ! own type declaration in source order; binding must not depend on
        ! the declaration having already run (session_program_lowering_
        ! common.inc declare_pending_common_symbol).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  common /shared/ n'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 5'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  print *, n'//new_line('a')// &
            'end program p'//new_line('a')// &
            'subroutine bump()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  common /shared/ m'//new_line('a')// &
            '  integer :: m'//new_line('a')// &
            '  m = m + 1'//new_line('a')// &
            'end subroutine'
        character(len=*), parameter :: expected = &
            '           6'//new_line('a')

        test_common_declared_after_common_statement = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_order')
    end function test_common_declared_after_common_statement

    logical function test_common_array_shared_across_units()
        ! An array COMMON member must alias the same LIRIC global across
        ! units: element GEP addresses off element_address, so
        ! rebind_common_symbol has to publish it there too, not just address
        ! (session_program_lowering_common.inc).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: arr(3)'//new_line('a')// &
            '  common /shared/ arr'//new_line('a')// &
            '  arr(1) = 10'//new_line('a')// &
            '  arr(2) = 20'//new_line('a')// &
            '  arr(3) = 30'//new_line('a')// &
            '  call show()'//new_line('a')// &
            'end program p'//new_line('a')// &
            'subroutine show()'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: v(3)'//new_line('a')// &
            '  common /shared/ v'//new_line('a')// &
            '  print *, v(1), v(2), v(3)'//new_line('a')// &
            'end subroutine'
        character(len=*), parameter :: expected = &
            '          10          20          30'//new_line('a')

        test_common_array_shared_across_units = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_array_shared')
    end function test_common_array_shared_across_units

    logical function test_zero_size_common_array()
        character(len=*), parameter :: source = &
            'subroutine test()'//new_line('a')// &
            '  integer(4) :: n'//new_line('a')// &
            '  real :: arr'//new_line('a')// &
            '  parameter(n = 0)'//new_line('a')// &
            '  common /cname/ arr(n)'//new_line('a')// &
            '  if (n /= 0) error stop'//new_line('a')// &
            'end subroutine test'//new_line('a')// &
            'program main'//new_line('a')// &
            '  call test()'//new_line('a')// &
            'end program main'

        test_zero_size_common_array = expect_exit_status( &
            source, 0, '/tmp/ffc_session_common_zero_size')
    end function test_zero_size_common_array

    logical function test_do_loop_over_common_variable()
        ! A COMMON-bound scalar used as a counted DO loop induction variable
        ! is carried in an SSA register through the loop scaffold; that value
        ! must be published back to the COMMON global at the end of each
        ! iteration and at loop exit, or a later memory-backed read sees
        ! stale storage (session_program_lowering_loops.inc).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  common /shared/ i'//new_line('a')// &
            '  do i = 1, 5'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, i'//new_line('a')// &
            'end program p'
        character(len=*), parameter :: expected = &
            '           6'//new_line('a')

        test_do_loop_over_common_variable = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_do_loop')
    end function test_do_loop_over_common_variable

    logical function test_whole_array_data_statement()
        ! A BLOCK DATA whole-array DATA statement (DATA arr /v1, v2, v3/,
        ! no subscripts) initialises one value per element in declaration
        ! order; the object and value lists need not be the same length
        ! (session_program_lowering_common.inc apply_block_data_statement).
        character(len=*), parameter :: source = &
            'program p'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: arr(3)'//new_line('a')// &
            '  common /shared/ arr'//new_line('a')// &
            '  print *, arr(1), arr(2), arr(3)'//new_line('a')// &
            'end program p'//new_line('a')// &
            'block data bd'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: arr(3)'//new_line('a')// &
            '  common /shared/ arr'//new_line('a')// &
            '  data arr /7, 8, 9/'//new_line('a')// &
            'end block data bd'
        character(len=*), parameter :: expected = &
            '           7           8           9'//new_line('a')

        test_whole_array_data_statement = expect_output( &
            source, expected, '/tmp/ffc_session_block_data_whole_array')
    end function test_whole_array_data_statement

end program test_session_block_data_compiler
