program test_session_scope_binding_negative_control_compiler
    use ffc_test_support, only: expect_cli_error_contains, expect_cli_no_error
    implicit none

    logical :: all_passed

    print *, '=== module binding negative control ==='

    all_passed = .true.
    if (.not. test_invalid_override_is_rejected()) all_passed = .false.
    if (.not. test_matching_override_is_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: module binding negative control'

contains

    logical function test_invalid_override_is_rejected()
        ! F2018 7.5.7.3: an overriding binding must preserve the INTENT of
        ! every dummy other than the passed-object dummy.
        character(len=*), parameter :: source = &
            'module binding_base'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, pass(self) :: step => base_step'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_step(self, value)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    integer, intent(in) :: value'//new_line('a')// &
            '  end subroutine base_step'//new_line('a')// &
            'end module binding_base'//new_line('a')// &
            'module binding_bad'//new_line('a')// &
            '  use binding_base, only: base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, pass(self) :: step => child_step'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine child_step(self, value)'//new_line('a')// &
            '    class(child_t), intent(inout) :: self'//new_line('a')// &
            '    integer, intent(inout) :: value'//new_line('a')// &
            '  end subroutine child_step'//new_line('a')// &
            'end module binding_bad'

        test_invalid_override_is_rejected = expect_cli_error_contains( &
            source, 'INTENT mismatch in argument', &
            '/tmp/ffc_scope_binding_negative_control')
    end function test_invalid_override_is_rejected

    logical function test_matching_override_is_accepted()
        character(len=*), parameter :: source = &
            'module binding_base_valid'//new_line('a')// &
            '  implicit none'//new_line('a')// &
            '  type :: base_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, pass(self) :: step => base_step'//new_line('a')// &
            '  end type base_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine base_step(self, value)'//new_line('a')// &
            '    class(base_t), intent(inout) :: self'//new_line('a')// &
            '    integer, intent(in) :: value'//new_line('a')// &
            '  end subroutine base_step'//new_line('a')// &
            'end module binding_base_valid'//new_line('a')// &
            'module binding_good'//new_line('a')// &
            '  use binding_base_valid, only: base_t'//new_line('a')// &
            '  type, extends(base_t) :: child_t'//new_line('a')// &
            '  contains'//new_line('a')// &
            '    procedure, pass(self) :: step => child_step'//new_line('a')// &
            '  end type child_t'//new_line('a')// &
            'contains'//new_line('a')// &
            '  subroutine child_step(self, value)'//new_line('a')// &
            '    class(child_t), intent(inout) :: self'//new_line('a')// &
            '    integer, intent(in) :: value'//new_line('a')// &
            '  end subroutine child_step'//new_line('a')// &
            'end module binding_good'//new_line('a')// &
            'program binding_control'//new_line('a')// &
            '  use binding_good, only: child_t'//new_line('a')// &
            '  type(child_t) :: value'//new_line('a')// &
            '  call value%step(1)'//new_line('a')// &
            'end program binding_control'

        test_matching_override_is_accepted = expect_cli_no_error( &
            source, '/tmp/ffc_scope_binding_valid_control')
    end function test_matching_override_is_accepted

end program test_session_scope_binding_negative_control_compiler
