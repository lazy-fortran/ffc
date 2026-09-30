program test_session_derived_alloc_array_compiler
    ! Positive behavioural oracle for #643.  gfortran 14.2 produces the
    ! expected output for the same source: an allocatable derived array uses
    ! one canonical descriptor while each element retains its concrete layout.
    use ffc_test_support, only: expect_output, expect_exit_status, &
                                expect_error_contains
    use fortfront_compiler, only: compiler_frontend_options_t, &
        compiler_frontend_result_t, compile_frontend_from_string, &
        INPUT_MODE_STANDARD
    use session_program_lowering, only: lower_program_to_liric_exe
    implicit none

    logical :: all_passed

    print *, '=== direct session derived allocatable array test ==='
    all_passed = test_rank1() .and. test_rank2_bounds() .and. test_reallocate() &
                 .and. test_deep_copy() .and. test_rank3_alloc_access() &
                 .and. test_rank3_rank_intrinsic_unsupported()
    if (.not. all_passed) stop 1
    print *, 'PASS: derived allocatable arrays lower through session'

contains

    logical function test_rank1()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:)'//new_line('a')// &
            '  integer :: i, total'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  do i = 1, 3'//new_line('a')// &
            '    a(i)%id = 10 * i'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  do i = 1, size(a)'//new_line('a')// &
            '    total = total + a(i)%id'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, size(a), total'//new_line('a')// &
            'end program main'

        test_rank1 = expect_output(source, &
            '           3          60'//new_line('a'), &
            '/tmp/ffc_derived_alloc_array_rank1')
    end function test_rank1

    logical function test_rank2_bounds()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:, :)'//new_line('a')// &
            '  integer :: i, j, total'//new_line('a')// &
            '  allocate(a(1:2, 1:4))'//new_line('a')// &
            '  total = 0'//new_line('a')// &
            '  a(1,1)%id = 1'//new_line('a')// &
            '  a(2,1)%id = 2'//new_line('a')// &
            '  a(1,2)%id = 101'//new_line('a')// &
            '  a(2,2)%id = 102'//new_line('a')// &
            '  a(1,3)%id = 201'//new_line('a')// &
            '  a(2,3)%id = 202'//new_line('a')// &
            '  a(1,4)%id = 301'//new_line('a')// &
            '  a(2,4)%id = 302'//new_line('a')// &
            '  total = a(1,1)%id + a(2,1)%id + a(1,2)%id + a(2,2)%id + '// &
            'a(1,3)%id + a(2,3)%id + a(1,4)%id + a(2,4)%id'//new_line('a')// &
            '  print *, lbound(a, 1), ubound(a, 1), lbound(a, 2), '// &
            'ubound(a, 2), size(a), total'//new_line('a')// &
            '  print *, a(1,1)%id, a(2,1)%id, a(1,2)%id, a(2,2)%id'// &
            new_line('a')// &
            'end program main'

        test_rank2_bounds = expect_output(source, &
            '           1           2           1           4           8'// &
            '        1212'//new_line('a')// &
            '           1           2         101         102'//new_line('a'), &
            '/tmp/ffc_derived_alloc_array_rank2')
    end function test_rank2_bounds

    logical function test_reallocate()
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a(1)%id = 7'//new_line('a')// &
            '  a(2)%id = 8'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  if (allocated(a)) error stop 1'//new_line('a')// &
            '  allocate(a(1))'//new_line('a')// &
            '  a(1)%id = 9'//new_line('a')// &
            '  print *, allocated(a), size(a), a(1)%id'//new_line('a')// &
            'end program main'

        test_reallocate = expect_exit_status(source, 0, &
                                              '/tmp/ffc_derived_alloc_array_realloc')
    end function test_reallocate

    logical function test_deep_copy()
        ! Intrinsic assignment of allocatable derived arrays with an allocated
        ! source owns an independent destination allocation. Mutating the
        ! source after `b = a` must not change b; the gfortran oracle also checks
        ! allocatable component ownership.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '    integer, allocatable :: values(:)'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:), b(:)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  allocate(a(1)%values(2))'//new_line('a')// &
            '  allocate(a(2)%values(2))'//new_line('a')// &
            '  a(1)%id = 11'//new_line('a')// &
            '  a(2)%id = 22'//new_line('a')// &
            '  a(1)%values = [101, 102]'//new_line('a')// &
            '  a(2)%values = [201, 202]'//new_line('a')// &
            '  b = a'//new_line('a')// &
            '  a(1)%id = 99'//new_line('a')// &
            '  a(1)%values(1) = 999'//new_line('a')// &
            '  print *, size(b), b(1)%id, b(2)%id, b(1)%values(1), '// &
            'b(1)%values(2), b(2)%values(1), b(2)%values(2), '// &
            'allocated(b)'//new_line('a')// &
            'end program main'

        test_deep_copy = matches_gfortran(source, 'deep_copy')
    end function test_deep_copy

    logical function test_rank3_alloc_access() result(ok)
        ! STALE EXPECTATION CORRECTED, not weakened. This case used to demand
        ! that rank-3 be refused (`direct LIRIC session supports rank-1 and
        ! rank-2 derived`) because the first descriptor slice exposed only ranks
        ! one and two. That was a slice boundary, never a language rule, and it
        ! is now obsolete: rank-3 allocatable derived arrays allocate, store,
        ! read and deallocate correctly, matching gfortran byte-exactly. A test
        ! that requires a valid, working program to be rejected is a false
        ! contract, and the string it expected exists nowhere in src, so the case
        ! had been red since it was written. A positive comparison against the
        ! reference compiler is the stronger assertion, so it replaces the
        ! demand for refusal.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:, :, :)'//new_line('a')// &
            '  allocate (a(2, 2, 2))'//new_line('a')// &
            '  a(1, 1, 1)%id = 5'//new_line('a')// &
            '  a(2, 2, 2)%id = 9'//new_line('a')// &
            '  print *, a(1, 1, 1)%id, a(2, 2, 2)%id'//new_line('a')// &
            '  deallocate (a)'//new_line('a')// &
            '  print *, allocated(a)'//new_line('a')// &
            'end program main'

        ok = matches_gfortran(source, 'rank3_alloc_access')
    end function test_rank3_alloc_access

    logical function test_rank3_rank_intrinsic_unsupported() result(ok)
        ! RANK()/SHAPE() on a rank-3 derived allocatable array are genuinely
        ! unsupported, on BOTH the CLI front end and the in-process lowering the
        ! harness uses. The array itself works - allocate, component store, read
        ! and deallocate match gfortran byte-exactly in test_rank3_alloc_access -
        ! so this is an intrinsic-support gap sitting on top of working descriptor
        ! storage, not a storage gap. Both facts are pinned separately so a
        ! future change cannot trade one for the other unnoticed.
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  type :: box_t'//new_line('a')// &
            '    integer :: id'//new_line('a')// &
            '  end type box_t'//new_line('a')// &
            '  type(box_t), allocatable :: a(:, :, :)'//new_line('a')// &
            '  allocate (a(2, 2, 2))'//new_line('a')// &
            '  print *, rank(a)'//new_line('a')// &
            'end program main'

        ok = expect_error_contains(source, &
            'unsupported scalar function call or array expression', &
            '/tmp/ffc_derived_alloc_array_rank_intrinsic')
    end function test_rank3_rank_intrinsic_unsupported

    logical function matches_gfortran(source, stem)
        ! Compile and run both front ends, then compare their complete output.
        ! gfortran is the independent behavioural oracle for this descriptor
        ! contract; no hard-coded expected output can hide a shape mismatch.
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: stem
        type(compiler_frontend_options_t) :: options
        type(compiler_frontend_result_t) :: frontend_result
        character(len=:), allocatable :: error_msg
        character(len=:), allocatable :: base, src, exe, ref, ffc_out, ref_out
        integer :: unit, exit_stat, status

        matches_gfortran = .false.
        base = '/var/tmp/ert/ffc_derived_alloc_array_'//trim(stem)
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
            print *, 'FAIL[', trim(stem), ']: ffc lowering failed: ', &
                trim(error_msg)
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
        call execute_command_line(exe//' > '//ffc_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc executable failed'
            return
        end if
        call execute_command_line(ref//' > '//ref_out//' 2>&1', &
            exitstat=exit_stat)
        if (exit_stat /= 0) then
            print *, 'FAIL[', trim(stem), ']: gfortran executable failed'
            return
        end if
        call execute_command_line('diff '//ffc_out//' '//ref_out// &
            ' > /dev/null 2>&1', exitstat=status)
        if (status /= 0) then
            print *, 'FAIL[', trim(stem), ']: ffc output differs from gfortran'
            call execute_command_line('diff '//ffc_out//' '//ref_out)
            return
        end if
        call execute_command_line('rm -f '//src//' '//exe//' '//ref//' '// &
            ffc_out//' '//ref_out)
        matches_gfortran = .true.
    end function matches_gfortran

end program test_session_derived_alloc_array_compiler
