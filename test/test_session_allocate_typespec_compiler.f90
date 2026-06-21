program test_session_allocate_typespec
    ! ALLOCATE with a type-spec: allocate(character(len=<expr>) :: str) and
    ! allocate(<derived> :: c). The character form sizes a deferred-length
    ! character scalar from an integer literal, a variable, or len(other); the
    ! derived form binds a scalar allocatable derived variable.
    use ffc_test_support, only: expect_output
    implicit none

    logical :: all_passed

    print *, '=== direct session allocate type-spec compiler test ==='

    all_passed = .true.
    if (.not. test_char_len_literal()) all_passed = .false.
    if (.not. test_char_len_variable()) all_passed = .false.
    if (.not. test_char_len_of_arg()) all_passed = .false.
    if (.not. test_derived_typespec()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: type-spec allocate lowers through direct LIRIC session'

contains

    logical function test_char_len_variable()
        ! allocate(character(len=n) :: str) then assign a literal and print.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: str'//new_line('a')// &
            '  integer :: n'//new_line('a')// &
            '  n = 15'//new_line('a')// &
            '  allocate(character(len=n) :: str)'//new_line('a')// &
            "  str = 'Hello, World!'"//new_line('a')// &
            "  print *, 'String: ', str"//new_line('a')// &
            "  print *, 'Length: ', len(str)"//new_line('a')// &
            '  deallocate(str)'//new_line('a')// &
            'end program main'

        test_char_len_variable = expect_output(source, &
            ' String: Hello, World!'//new_line('a')// &
            ' Length:           13'//new_line('a'), &
            '/tmp/ffc_alloc_typespec_var')
    end function test_char_len_variable

    logical function test_char_len_literal()
        ! allocate(character(len=8) :: str): an integer-literal length.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=:), allocatable :: str'//new_line('a')// &
            '  allocate(character(len=8) :: str)'//new_line('a')// &
            "  str = 'abcdefgh'"//new_line('a')// &
            '  print *, str'//new_line('a')// &
            'end program main'

        test_char_len_literal = expect_output(source, &
            ' abcdefgh'//new_line('a'), '/tmp/ffc_alloc_typespec_lit')
    end function test_char_len_literal

    logical function test_char_len_of_arg()
        ! allocate(character(len=len(input)) :: local_copy) inside a subroutine.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  call process("Hello")'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine process(input)'//new_line('a')// &
            '    character(len=*), intent(in) :: input'//new_line('a')// &
            '    character(len=:), allocatable :: local_copy'//new_line('a')// &
            '    allocate(character(len=len(input)) :: local_copy)'// &
            new_line('a')// &
            '    local_copy = input'//new_line('a')// &
            '    print *, trim(local_copy)'//new_line('a')// &
            '  end subroutine process'//new_line('a')// &
            'end program main'

        test_char_len_of_arg = expect_output(source, &
            ' Hello'//new_line('a'), '/tmp/ffc_alloc_typespec_lenarg')
    end function test_char_len_of_arg

    logical function test_derived_typespec()
        ! allocate(circle :: c) for a scalar allocatable derived variable.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: circle'//new_line('a')// &
            '    real :: radius'//new_line('a')// &
            '  end type circle'//new_line('a')// &
            '  type(circle), allocatable :: c'//new_line('a')// &
            '  allocate(circle :: c)'//new_line('a')// &
            '  c%radius = 5.0'//new_line('a')// &
            '  print *, c%radius'//new_line('a')// &
            'end program main'

        test_derived_typespec = expect_output(source, &
            '   5.00000000    '//new_line('a'), '/tmp/ffc_alloc_typespec_derived')
    end function test_derived_typespec

end program test_session_allocate_typespec
