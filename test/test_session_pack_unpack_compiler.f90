program test_session_pack_unpack_compiler
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session pack/unpack compiler test ==='

    all_passed = .true.
    if (.not. test_pack_logical_mask()) all_passed = .false.
    if (.not. test_pack_comparison_mask()) all_passed = .false.
    if (.not. test_pack_real_array()) all_passed = .false.
    if (.not. test_unpack_array_field()) all_passed = .false.
    if (.not. test_unpack_scalar_field()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: pack/unpack lower through direct LIRIC session'

contains

    logical function test_pack_logical_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  logical :: m(5) = [.true., .false., .true., .false., .true.]'// &
            new_line('a')// &
            '  integer :: b(3)'//new_line('a')// &
            '  b = pack(a, m)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_pack_logical_mask = expect_output( &
            source, '           1           3           5'//new_line('a'), &
            '/tmp/ffc_pack_logical')
    end function test_pack_logical_mask

    logical function test_pack_comparison_mask()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: a(5) = [1, 2, 3, 4, 5]'//new_line('a')// &
            '  integer :: b(3)'//new_line('a')// &
            '  b = pack(a, a > 2)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_pack_comparison_mask = expect_output( &
            source, '           3           4           5'//new_line('a'), &
            '/tmp/ffc_pack_cmp')
    end function test_pack_comparison_mask

    logical function test_pack_real_array()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real :: a(4) = [1.5, 2.5, 3.5, 4.5]'//new_line('a')// &
            '  logical :: m(4) = [.true., .false., .true., .true.]'// &
            new_line('a')// &
            '  real :: b(3)'//new_line('a')// &
            '  b = pack(a, m)'//new_line('a')// &
            '  print *, b'//new_line('a')// &
            'end program main'

        test_pack_real_array = expect_output( &
            source, &
            '   1.50000000       3.50000000       4.50000000    '// &
            new_line('a'), '/tmp/ffc_pack_real')
    end function test_pack_real_array

    logical function test_unpack_array_field()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v(3) = [10, 20, 30]'//new_line('a')// &
            '  logical :: m(5) = [.true., .false., .true., .false., .true.]'// &
            new_line('a')// &
            '  integer :: field(5) = [0, 0, 0, 0, 0]'//new_line('a')// &
            '  integer :: r(5)'//new_line('a')// &
            '  r = unpack(v, m, field)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'

        test_unpack_array_field = expect_output( &
            source, &
            '          10           0          20           0          30'// &
            new_line('a'), '/tmp/ffc_unpack_array')
    end function test_unpack_array_field

    logical function test_unpack_scalar_field()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: v(2) = [7, 8]'//new_line('a')// &
            '  logical :: m(4) = [.false., .true., .false., .true.]'// &
            new_line('a')// &
            '  integer :: r(4)'//new_line('a')// &
            '  r = unpack(v, m, 9)'//new_line('a')// &
            '  print *, r'//new_line('a')// &
            'end program main'

        test_unpack_scalar_field = expect_output( &
            source, &
            '           9           7           9           8'// &
            new_line('a'), '/tmp/ffc_unpack_scalar')
    end function test_unpack_scalar_field

end program test_session_pack_unpack_compiler
