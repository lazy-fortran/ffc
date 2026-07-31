program test_session_reject_alloc_02_compiler
    ! #382: an ALLOCATE stat= specifier must name a scalar INTEGER variable
    ! and an errmsg= specifier must name a scalar default CHARACTER variable.
    ! Array or wrong-typed status targets are rejected with a source
    ! diagnostic; the corrected scalar neighbours still compile and run.
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    character(len=*), parameter :: STAT_FRAGMENT = &
        'ALLOCATE stat= must be a scalar INTEGER variable'
    character(len=*), parameter :: ERRMSG_FRAGMENT = &
        'ALLOCATE errmsg= must be a scalar default CHARACTER variable'
    logical :: all_passed

    print *, '=== ALLOCATE status specifier rejection test ==='

    all_passed = .true.
    if (.not. test_array_stat_rejected()) all_passed = .false.
    if (.not. test_character_stat_rejected()) all_passed = .false.
    if (.not. test_array_errmsg_rejected()) all_passed = .false.
    if (.not. test_integer_errmsg_rejected()) all_passed = .false.
    if (.not. test_scalar_stat_accepted()) all_passed = .false.
    if (.not. test_scalar_stat_and_errmsg_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: invalid ALLOCATE status specifiers are rejected'

contains

    logical function test_array_stat_rejected()
        ! gfortran.dg/allocate_stat_2.f90: stat = ier with integer ier(4).
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, dimension(4) :: ier = 0'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(16), stat = ier)'//new_line('a')// &
            'end program main'

        test_array_stat_rejected = expect_error_contains( &
            source, STAT_FRAGMENT, '/tmp/ffc_session_reject_alloc02_stat_array')
    end function test_array_stat_rejected

    logical function test_character_stat_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=30) :: ier'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(16), stat = ier)'//new_line('a')// &
            'end program main'

        test_character_stat_rejected = expect_error_contains( &
            source, STAT_FRAGMENT, '/tmp/ffc_session_reject_alloc02_stat_char')
    end function test_character_stat_rejected

    logical function test_array_errmsg_rejected()
        ! gfortran.dg/allocate_stat_2.f90: errmsg = er with er(2) character.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ier'//new_line('a')// &
            '  character(len=30), dimension(2) :: er'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(14), stat=ier, errmsg=er)'//new_line('a')// &
            'end program main'

        test_array_errmsg_rejected = expect_error_contains( &
            source, ERRMSG_FRAGMENT, &
            '/tmp/ffc_session_reject_alloc02_errmsg_array')
    end function test_array_errmsg_rejected

    logical function test_integer_errmsg_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ier'//new_line('a')// &
            '  integer :: er'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(14), stat=ier, errmsg=er)'//new_line('a')// &
            'end program main'

        test_integer_errmsg_rejected = expect_error_contains( &
            source, ERRMSG_FRAGMENT, &
            '/tmp/ffc_session_reject_alloc02_errmsg_int')
    end function test_integer_errmsg_rejected

    logical function test_scalar_stat_accepted()
        ! Corrected neighbour: a scalar INTEGER stat= is set to 0 on success.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ier = 7'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(16), stat = ier)'//new_line('a')// &
            '  stop ier + 3'//new_line('a')// &
            'end program main'

        test_scalar_stat_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_session_alloc02_stat_ok')
    end function test_scalar_stat_accepted

    logical function test_scalar_stat_and_errmsg_accepted()
        ! Corrected neighbour: scalar INTEGER stat= plus scalar default
        ! CHARACTER errmsg=.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: ier = 7'//new_line('a')// &
            '  character(len=30) :: er'//new_line('a')// &
            '  integer, dimension(:), allocatable :: a'//new_line('a')// &
            '  allocate (a(14), stat=ier, errmsg=er)'//new_line('a')// &
            '  stop ier + 5'//new_line('a')// &
            'end program main'

        test_scalar_stat_and_errmsg_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_session_alloc02_stat_errmsg_ok')
    end function test_scalar_stat_and_errmsg_accepted

end program test_session_reject_alloc_02_compiler
