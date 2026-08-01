program test_session_character_function_result_compiler
    ! Verify character-returning contained functions lower correctly: a
    ! fixed-length result (character(len=N) :: s), a runtime-length result
    ! (character(len=k) where k is a dummy argument), and a character variable
    ! declared with a parameter length (character(max_len)). Each program's
    ! output must match gfortran byte-for-byte (#1614).
    use session_program_lowering, only: lower_program_to_liric_exe
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, &
        compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use ffc_test_support, only: expect_no_leaks
    implicit none

    logical :: all_passed

    print *, '=== direct session character function result compiler test ==='

    all_passed = .true.

    ! Fixed-length character result printed by the caller.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=5) :: r'//new_line('a')// &
        '  r = get_name()'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function get_name() result(s)'//new_line('a')// &
        '    character(len=5) :: s'//new_line('a')// &
        '    s = "Hello"'//new_line('a')// &
        '  end function get_name'//new_line('a')// &
        'end program main', &
        'fixed_result')) all_passed = .false.

    ! Runtime-length character result (length is a dummy argument).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = make(4)'//new_line('a')// &
        '  print *, r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Z", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        'runtime_result')) all_passed = .false.

    ! Character variable declared with a parameter length.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  integer, parameter :: max_len = 6'//new_line('a')// &
        '  character(max_len) :: name'//new_line('a')// &
        '  name = "Test"'//new_line('a')// &
        '  print *, name'//new_line('a')// &
        'end program main', &
        'param_length')) all_passed = .false.

    ! Result length taken from a dummy via len(): character(len=len(name)).
    ! The actual width comes from the assigned value, so the deferred ABI
    ! resolves it without evaluating the declared length expression (#1407).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, greet("Ada")'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)) :: s'//new_line('a')// &
        '    s = "" // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'len_of_dummy')) all_passed = .false.

    ! Result length is an expression over a dummy: character(len=len(name)+7).
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  print *, greet("Ada")'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)+7) :: s'//new_line('a')// &
        '    s = "Hello, " // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'len_expr_of_dummy')) all_passed = .false.

    ! A runtime-length result assigned to a shorter fixed-length destination
    ! truncates to the destination's declared width, and to a longer one pads
    ! with blanks. The result is a value, not an alias of the callee's buffer.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=5) :: short'//new_line('a')// &
        '  character(len=14) :: wide'//new_line('a')// &
        '  short = greet("Ada")'//new_line('a')// &
        '  wide = greet("Ada")'//new_line('a')// &
        '  print *, "[", short, "]"'//new_line('a')// &
        '  print *, "[", wide, "]"'//new_line('a')// &
        '  print *, greet("Bob")'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)+7) :: s'//new_line('a')// &
        '    s = "Hello, " // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'fixed_dest_truncates')) all_passed = .false.

    ! A nested concatenating result: the inner call's result feeds the outer
    ! concatenation, so the outer result is length 5 with the exact value.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = outer()'//new_line('a')// &
        '  print *, len(r), r'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function inner() result(s)'//new_line('a')// &
        '    character(len=:), allocatable :: s'//new_line('a')// &
        '    s = "ab"'//new_line('a')// &
        '  end function inner'//new_line('a')// &
        '  function outer() result(t)'//new_line('a')// &
        '    character(len=:), allocatable :: t'//new_line('a')// &
        '    t = inner() // "cde"'//new_line('a')// &
        '  end function outer'//new_line('a')// &
        'end program main', &
        'nested_concat_result')) all_passed = .false.

    ! Reassigning a deferred destination from repeated calls transfers
    ! ownership of each result and releases the previous one.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = make(4)'//new_line('a')// &
        '  print *, len(r), r'//new_line('a')// &
        '  r = make(2)'//new_line('a')// &
        '  print *, len(r), r'//new_line('a')// &
        '  deallocate(r)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Z", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        'result_reassignment')) all_passed = .false.

    ! Intrinsic assignment gives the destination a value, not a view: b keeps
    ! its own copy of a's text when a is later reassigned and its former
    ! storage released.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: a, b'//new_line('a')// &
        '  a = make(3)'//new_line('a')// &
        '  b = a'//new_line('a')// &
        '  print *, len(a), a, " ", len(b), b'//new_line('a')// &
        '  a = make(6)'//new_line('a')// &
        '  print *, len(a), a, " ", len(b), b'//new_line('a')// &
        '  deallocate(a)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Q", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        'assignment_copies_not_aliases')) all_passed = .false.

    ! A result used as a concatenation operand on either side.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: t'//new_line('a')// &
        '  t = greet("Ann") // "!"'//new_line('a')// &
        '  print *, len(t), t'//new_line('a')// &
        '  t = "x" // greet("Bo")'//new_line('a')// &
        '  print *, len(t), t'//new_line('a')// &
        '  deallocate(t)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(n) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: n'//new_line('a')// &
        '    character(len=len(n)+3) :: s'//new_line('a')// &
        '    s = "Hi " // n'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        'result_as_concat_operand')) all_passed = .false.

    ! Ownership oracle: a result consumed by a fixed-length destination or by
    ! print is a temporary the caller must release. Nothing survives the
    ! statement, so a clean memcheck report is the whole contract.
    if (.not. expect_no_leaks( &
        'program main'//new_line('a')// &
        '  character(len=5) :: f'//new_line('a')// &
        '  f = greet("Ada")'//new_line('a')// &
        '  print *, "[", f, "]"'//new_line('a')// &
        '  print *, greet("Bob")'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function greet(name) result(s)'//new_line('a')// &
        '    character(len=*), intent(in) :: name'//new_line('a')// &
        '    character(len=len(name)+7) :: s'//new_line('a')// &
        '    s = "Hello, " // name'//new_line('a')// &
        '  end function greet'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_charfn_result_temp_leak')) all_passed = .false.

    ! Ownership oracle: a deferred destination assumes ownership of each
    ! result, releases the previous one, and the explicit deallocate returns
    ! the last one.
    if (.not. expect_no_leaks( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: r'//new_line('a')// &
        '  r = make(4)'//new_line('a')// &
        '  print *, len(r), r'//new_line('a')// &
        '  r = make(2)'//new_line('a')// &
        '  print *, len(r), r'//new_line('a')// &
        '  deallocate(r)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function make(k) result(s)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: s'//new_line('a')// &
        '    s = repeat("Z", k)'//new_line('a')// &
        '  end function make'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_charfn_result_transfer_leak')) all_passed = .false.

    ! A character result consumed through the generic character-expression
    ! path is a temporary belonging to the statement that produced it. These
    ! cases pin its output and, through expect_no_leaks, that it is released
    ! exactly once. They passed while leaking before this was fixed, which is
    ! why the leak check and not the output is what makes them meaningful.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: s'//new_line('a')// &
        '  s = trim(pad(3))'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function pad(k) result(t)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=8) :: t'//new_line('a')// &
        '    t = repeat("z", k)'//new_line('a')// &
        '  end function pad'//new_line('a')// &
        'end program main', &
        'trim_of_result')) all_passed = .false.

    if (.not. expect_no_leaks( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: s'//new_line('a')// &
        '  character(len=8) :: f'//new_line('a')// &
        '  s = trim(pad(3))'//new_line('a')// &
        '  print *, len(s)'//new_line('a')// &
        '  f = trim(pad(4))'//new_line('a')// &
        '  print *, "[", f, "]"'//new_line('a')// &
        '  print *, trim(pad(5))'//new_line('a')// &
        '  print *, len_trim(pad(6))'//new_line('a')// &
        '  if (trim(pad(3)) == "zzz") print *, "eq"'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function pad(k) result(t)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=8) :: t'//new_line('a')// &
        '    t = repeat("z", k)'//new_line('a')// &
        '  end function pad'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_charfn_expr_temp_leak')) all_passed = .false.

    ! Two result temporaries alive at once in a single statement: both are
    ! consumed by the concatenation and both must be released.
    if (.not. matches_gfortran( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: s'//new_line('a')// &
        '  s = tag(2) // tag(3)'//new_line('a')// &
        '  print *, len(s), "[", s, "]"'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function tag(k) result(t)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: t'//new_line('a')// &
        '    t = repeat("z", k)'//new_line('a')// &
        '  end function tag'//new_line('a')// &
        'end program main', &
        'two_result_operands')) all_passed = .false.

    if (.not. expect_no_leaks( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: s'//new_line('a')// &
        '  s = tag(2) // tag(3)'//new_line('a')// &
        '  print *, len(s)'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function tag(k) result(t)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=k) :: t'//new_line('a')// &
        '    t = repeat("z", k)'//new_line('a')// &
        '  end function tag'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_charfn_two_operand_leak')) all_passed = .false.

    ! A result temporary inside a loop must be released each iteration rather
    ! than accumulating, which a single-shot release would miss.
    if (.not. expect_no_leaks( &
        'program main'//new_line('a')// &
        '  character(len=:), allocatable :: s'//new_line('a')// &
        '  integer :: i'//new_line('a')// &
        '  do i = 1, 4'//new_line('a')// &
        '    s = trim(pad(i))'//new_line('a')// &
        '    print *, len(s)'//new_line('a')// &
        '  end do'//new_line('a')// &
        '  deallocate(s)'//new_line('a')// &
        '  stop 0'//new_line('a')// &
        'contains'//new_line('a')// &
        '  function pad(k) result(t)'//new_line('a')// &
        '    integer, intent(in) :: k'//new_line('a')// &
        '    character(len=8) :: t'//new_line('a')// &
        '    t = repeat("z", k)'//new_line('a')// &
        '  end function pad'//new_line('a')// &
        'end program main', &
        '/tmp/ffc_charfn_loop_temp_leak')) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: character function results lower through direct LIRIC session'

contains

    logical function matches_gfortran(source, stem)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/tmp/ffc_charfn_'//trim(stem)
        src = base//'.f90'
        exe = base//'.ffc'
        ref = base//'.gf'
        ffc_out = base//'.ffc.out'
        ref_out = base//'.gf.out'

        options = compiler_frontend_options_t()
        options%run_semantics = .true.
        options%input_mode = INPUT_MODE_STANDARD
        call compile_frontend_from_string(source, frontend_result, options)
        if (.not. frontend_result%success()) then
            print *, 'FAIL[', trim(stem), ']: FortFront rejected source: ', &
                trim(frontend_result%diagnostic_text)
            return
        end if

        call lower_program_to_liric_exe(frontend_result%arena, &
            frontend_result%root_index, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', trim(error_msg)
            return
        end if

        open (newunit=unit, file=src, status='replace', action='write')
        write (unit, '(A)') source
        close (unit)
        call execute_command_line('gfortran -w '//src//' -o '//ref, &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran rejected source'
            return
        end if

        call execute_command_line(exe//' > '//ffc_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line(ref//' > '//ref_out//' 2>&1', exitstat=exit_stat)
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref// &
            ' '//ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_character_function_result_compiler
