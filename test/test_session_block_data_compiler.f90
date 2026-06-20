program test_session_block_data_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session block-data / common compiler test ==='

    all_passed = .true.
    if (.not. test_integer_common_initialized()) all_passed = .false.
    if (.not. test_mixed_common_initialized()) all_passed = .false.
    if (.not. test_common_assignment_writes_through()) all_passed = .false.

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

end program test_session_block_data_compiler
