program test_session_reject_decl_02_compiler
    use ffc_test_support, only: expect_error_contains, expect_exit_status
    implicit none

    logical :: all_passed

    print *, '=== conflicting declaration and constructor rejection test ==='

    all_passed = .true.
    if (.not. test_procedure_then_type_rejected()) all_passed = .false.
    if (.not. test_type_then_procedure_rejected()) all_passed = .false.
    if (.not. test_typeless_procedure_accepted()) all_passed = .false.
    if (.not. test_char_ctor_length_mismatch_rejected()) all_passed = .false.
    if (.not. test_char_ctor_assignment_mismatch_rejected()) all_passed = .false.
    if (.not. test_char_ctor_type_spec_accepted()) all_passed = .false.
    if (.not. test_char_ctor_equal_lengths_accepted()) all_passed = .false.
    if (.not. test_null_mold_type_mismatch_rejected()) all_passed = .false.
    if (.not. test_null_mold_rank_mismatch_rejected()) all_passed = .false.
    if (.not. test_null_mold_matching_accepted()) all_passed = .false.
    if (.not. test_lazy_char_ctor_accepted()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: conflicting declarations and constructors are rejected'

contains

    ! Rule 1: a PROCEDURE statement with an interface and a type declaration
    ! may not declare the same name (gfortran "already has basic type of").
    logical function test_procedure_then_type_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  procedure(iabs) :: c'//new_line('a')// &
            '  integer :: c'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_procedure_then_type_rejected = expect_error_contains( &
            source, 'declared both by a PROCEDURE statement', &
            '/tmp/ffc_reject_decl_02_proc_first')
    end function test_procedure_then_type_rejected

    logical function test_type_then_procedure_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: d'//new_line('a')// &
            '  procedure(iabs) :: d'//new_line('a')// &
            '  print *, 1'//new_line('a')// &
            'end program main'

        test_type_then_procedure_rejected = expect_error_contains( &
            source, 'declared both by a PROCEDURE statement', &
            '/tmp/ffc_reject_decl_02_type_first')
    end function test_type_then_procedure_rejected

    ! A PROCEDURE statement without an interface leaves the type open, so it
    ! may be combined with a type declaration (gfortran accepts this).
    logical function test_typeless_procedure_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer :: e'//new_line('a')// &
            '  procedure() :: e'//new_line('a')// &
            '  stop 3'//new_line('a')// &
            'end program main'

        test_typeless_procedure_accepted = expect_exit_status( &
            source, 3, '/tmp/ffc_reject_decl_02_typeless')
    end function test_typeless_procedure_accepted

    ! Rule 2: without a type-spec every character ac-value of an array
    ! constructor has the same length (gfortran "Different CHARACTER lengths").
    logical function test_char_ctor_length_mismatch_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8) :: arr(2) = (/ "abc", "foobar" /)'// &
            new_line('a')// &
            '  print *, arr(1)'//new_line('a')// &
            'end program main'

        test_char_ctor_length_mismatch_rejected = expect_error_contains( &
            source, 'different CHARACTER lengths', &
            '/tmp/ffc_reject_decl_02_char_init')
    end function test_char_ctor_length_mismatch_rejected

    logical function test_char_ctor_assignment_mismatch_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=8) :: arr(2)'//new_line('a')// &
            '  arr = (/ "abc", "foobar" /)'//new_line('a')// &
            '  print *, arr(1)'//new_line('a')// &
            'end program main'

        test_char_ctor_assignment_mismatch_rejected = expect_error_contains( &
            source, 'different CHARACTER lengths', &
            '/tmp/ffc_reject_decl_02_char_assign')
    end function test_char_ctor_assignment_mismatch_rejected

    ! A type-spec fixes the element length, so unequal values are padded.
    logical function test_char_ctor_type_spec_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=6) :: arr(2)'//new_line('a')// &
            '  arr = (/ character(len=6) :: "abc", "foobar" /)'// &
            new_line('a')// &
            '  if (arr(2) == "foobar") stop 4'//new_line('a')// &
            'end program main'

        test_char_ctor_type_spec_accepted = expect_exit_status( &
            source, 4, '/tmp/ffc_reject_decl_02_char_spec')
    end function test_char_ctor_type_spec_accepted

    logical function test_char_ctor_equal_lengths_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  character(len=3) :: arr(2)'//new_line('a')// &
            '  arr = (/ "abc", "def" /)'//new_line('a')// &
            '  if (arr(2) == "def") stop 5'//new_line('a')// &
            'end program main'

        test_char_ctor_equal_lengths_accepted = expect_exit_status( &
            source, 5, '/tmp/ffc_reject_decl_02_char_equal')
    end function test_char_ctor_equal_lengths_accepted

    ! Rule 3: NULL(MOLD) takes the type and rank of MOLD, and those must
    ! match the pointer (gfortran "Different types in pointer assignment").
    logical function test_null_mold_type_mismatch_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, pointer :: i => null()'//new_line('a')// &
            '  real, pointer :: x => null()'//new_line('a')// &
            '  x => null(i)'//new_line('a')// &
            '  print *, associated(x)'//new_line('a')// &
            'end program main'

        test_null_mold_type_mismatch_rejected = expect_error_contains( &
            source, 'different types in pointer assignment', &
            '/tmp/ffc_reject_decl_02_null_type')
    end function test_null_mold_type_mismatch_rejected

    logical function test_null_mold_rank_mismatch_rejected()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, pointer :: x => null()'//new_line('a')// &
            '  real, pointer :: z(:) => null()'//new_line('a')// &
            '  x => null(z)'//new_line('a')// &
            '  print *, associated(x)'//new_line('a')// &
            'end program main'

        test_null_mold_rank_mismatch_rejected = expect_error_contains( &
            source, 'different ranks in pointer assignment', &
            '/tmp/ffc_reject_decl_02_null_rank')
    end function test_null_mold_rank_mismatch_rejected

    logical function test_null_mold_matching_accepted()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  real, pointer :: x => null()'//new_line('a')// &
            '  real, pointer :: w => null()'//new_line('a')// &
            '  x => null(w)'//new_line('a')// &
            '  if (.not. associated(x)) stop 6'//new_line('a')// &
            'end program main'

        test_null_mold_matching_accepted = expect_exit_status( &
            source, 6, '/tmp/ffc_reject_decl_02_null_ok')
    end function test_null_mold_matching_accepted

    ! Lazy Fortran pads character array values to the longest value, so the
    ! standard-mode length rule must not fire on a .lf source (this is the
    ! shape of fortfront/examples/lf/docs_character_arrays.lf).
    logical function test_lazy_char_ctor_accepted()
        character(len=*), parameter :: source = &
            'names = ["alice", "bob", "charlie"]'//new_line('a')// &
            'print *, trim(names(2))'//new_line('a')
        character(len=*), parameter :: stem = '/tmp/ffc_reject_decl_02_lazy'
        character(len=:), allocatable :: command
        integer :: exit_stat, cmd_stat, unit

        test_lazy_char_ctor_accepted = .false.
        open (newunit=unit, file=stem//'.lf', status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        command = "sh -c 'exe=$(ls -t build/*/app/ffc build/fo/bin/ffc "// &
                  "2>/dev/null | head -n 1); test -n ""$exe"" && "// &
                  """$exe"" "//stem//'.lf -o '//stem//".exe'"
        call execute_command_line(command, exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy character array constructor was rejected'
            return
        end if
        call execute_command_line(stem//'.exe', exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
        call execute_command_line('rm -f '//stem//'.lf '//stem//'.exe')
        if (cmd_stat /= 0 .or. exit_stat /= 0) then
            print *, 'FAIL: lazy character array executable did not run'
            return
        end if
        test_lazy_char_ctor_accepted = .true.
    end function test_lazy_char_ctor_accepted

end program test_session_reject_decl_02_compiler
