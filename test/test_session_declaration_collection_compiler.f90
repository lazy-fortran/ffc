program test_session_declaration_collection_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none
    logical :: all_passed

    print *, '=== direct session declaration collection compiler test ==='
    all_passed = .true.
    if (.not. test_host_array_binding_is_reused()) all_passed = .false.
    if (.not. test_nested_declarations_are_reused()) all_passed = .false.
    if (.not. test_missing_binding_is_diagnosed()) all_passed = .false.
    if (.not. test_conflicting_declaration_is_diagnosed()) all_passed = .false.
    if (.not. all_passed) stop 1
    print *, 'PASS: resolved declarations are collected by binding identity'

contains

    logical function test_host_array_binding_is_reused()
        ! The internal procedure is lowered before the host body. Its host array
        ! must therefore use the declaration's one binding-backed global; the
        ! pre-change lowerer materialized a scalar host seed and then rejected
        ! the host declaration as a duplicate array.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: values(2)'//new_line('a')// &
            '  values = [3, 4]'//new_line('a')// &
            '  call bump()'//new_line('a')// &
            '  if (values(1) /= 4 .or. values(2) /= 5) stop 71'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine bump()'//new_line('a')// &
            '    values = values + 1'//new_line('a')// &
            '  end subroutine bump'//new_line('a')// &
            'end program main'

        test_host_array_binding_is_reused = expect_exit_status( &
            source, 0, '/tmp/ffc_session_host_array_collection_test')
    end function test_host_array_binding_is_reused

    logical function test_nested_declarations_are_reused()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: input'//new_line('a')// &
            '  integer :: offset'//new_line('a')// &
            '  input = 3'//new_line('a')// &
            '  offset = 4'//new_line('a')// &
            '  if (evaluate(input) /= 16) stop 91'//new_line('a')// &
            'contains'//new_line('a')// &
            '  integer function evaluate(argument) result(answer)'//new_line('a')// &
            '    integer, intent(in) :: argument'//new_line('a')// &
            '    integer :: answer'//new_line('a')// &
            '    integer :: i'//new_line('a')// &
            '    integer :: samples(2)'//new_line('a')// &
            '    answer = offset + argument'//new_line('a')// &
            '    block'//new_line('a')// &
            '      integer :: offset'//new_line('a')// &
            '      offset = 2'//new_line('a')// &
            '      answer = answer + offset'//new_line('a')// &
            '    end block'//new_line('a')// &
            '    do i = 1, 2'//new_line('a')// &
            '      samples(i) = i'//new_line('a')// &
            '      answer = answer + samples(i)'//new_line('a')// &
            '    end do'//new_line('a')// &
            '    answer = answer + offset'//new_line('a')// &
            '  end function evaluate'//new_line('a')// &
            'end program main'

        test_nested_declarations_are_reused = expect_exit_status( &
            source, 0, '/tmp/ffc_session_declaration_collection_test')
    end function test_nested_declarations_are_reused

    logical function test_missing_binding_is_diagnosed()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: value'//new_line('a')// &
            '  value = missing'//new_line('a')// &
            'end program main'

        test_missing_binding_is_diagnosed = expect_error_contains( &
            source, 'integer identifier was not declared: missing', &
            '/tmp/ffc_session_declaration_missing_binding_test')
    end function test_missing_binding_is_diagnosed

    logical function test_conflicting_declaration_is_diagnosed()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  integer :: value'//new_line('a')// &
            '  real :: value'//new_line('a')// &
            '  print *, value'//new_line('a')// &
            'end program main'

        test_conflicting_declaration_is_diagnosed = expect_error_contains( &
            source, 'value', '/tmp/ffc_session_declaration_conflict_test')
    end function test_conflicting_declaration_is_diagnosed

end program test_session_declaration_collection_compiler
