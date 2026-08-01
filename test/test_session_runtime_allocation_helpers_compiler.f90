! Issue #428: descriptor storage is allocated, resized, and freed through
! runtime entry points.
!
! Behavioral oracle, two halves.
!
! The runtime half drives the helpers from C and pins the contract the
! compiler now depends on: a negative extent, a size that would overflow, a
! double release, and a release through a borrowed descriptor each return
! their own stable code instead of corrupting the heap or succeeding quietly.
! These are the cases emitted code could not check for itself when it called
! malloc and free directly.
!
! The compiler half is the end-to-end guarantee: an emitted program allocates,
! reallocates and frees integer arrays and deferred-length characters through
! the runtime, values survive the transitions, and the program's symbols name
! the runtime helpers rather than malloc and free. The symbol check fails on
! the pre-#428 compiler.
program test_session_runtime_allocation_helpers_compiler
    use ffc_runtime_link, only: ffc_runtime_link_input
    use ffc_test_support, only: expect_output, compile_to_exe
    implicit none

    character(len=*), parameter :: WORK = '/tmp/ffc_alloc_helpers_428'
    logical :: all_passed

    print *, '=== runtime allocation helper tests (#428) ==='

    call run_quiet('rm -rf '//WORK//' && mkdir -p '//WORK)

    all_passed = .true.
    if (.not. test_runtime_contract()) all_passed = .false.
    if (.not. test_values_survive_transitions()) all_passed = .false.
    if (.not. test_lowering_calls_the_runtime()) all_passed = .false.

    if (.not. all_passed) stop 1
    print *, 'PASS: descriptor storage goes through the runtime'

contains

    subroutine run_quiet(command)
        character(len=*), intent(in) :: command
        integer :: exit_stat, cmd_stat

        call execute_command_line(command, exitstat=exit_stat, &
                                  cmdstat=cmd_stat)
    end subroutine run_quiet

    integer function status_of(command) result(status)
        character(len=*), intent(in) :: command
        character(len=*), parameter :: rc_file = WORK//'/rc'
        integer :: unit, ios, exit_stat, cmd_stat

        status = -1
        call execute_command_line('{ '//command//' ; } > '//WORK// &
                                  '/out 2>&1; echo $? > '//rc_file, &
                                  exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) return
        open (newunit=unit, file=rc_file, status='old', iostat=ios)
        if (ios /= 0) return
        read (unit, *, iostat=ios) status
        close (unit)
        if (ios /= 0) status = -1
    end function status_of

    subroutine read_text_file(path, text, ok)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        logical, intent(out) :: ok
        integer :: unit, ios
        integer(kind=8) :: nbytes

        ok = .false.
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) then
            text = ''
            return
        end if
        inquire (unit=unit, size=nbytes)
        if (nbytes <= 0) then
            close (unit)
            text = ''
            return
        end if
        allocate (character(len=nbytes) :: text)
        read (unit, iostat=ios) text
        close (unit)
        ok = ios == 0
    end subroutine read_text_file

    ! Each failing check exits with its own number, so a failure names the
    ! contract that broke rather than only that something did.
    logical function test_runtime_contract() result(ok)
        character(len=*), parameter :: driver = WORK//'/alloc_driver.c'
        character(len=*), parameter :: exe = WORK//'/alloc_driver'
        character(len=:), allocatable :: link_input, error_msg
        integer :: unit, ios, status

        ok = .false.
        call ffc_runtime_link_input(link_input, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: runtime link input: ', trim(error_msg)
            return
        end if
        open (newunit=unit, file=driver, status='replace', iostat=ios)
        if (ios /= 0) then
            print *, 'FAIL: cannot write the allocation driver'
            return
        end if
        write (unit, '(a)') '#include <stddef.h>'
        write (unit, '(a)') 'void *_ffc_alloc(long long, long long);'
        write (unit, '(a)') 'void *_ffc_calloc(long long, long long);'
        write (unit, '(a)') 'void *_ffc_realloc(void *, long long,'
        write (unit, '(a)') '                   long long);'
        write (unit, '(a)') 'int _ffc_dealloc(void *, int);'
        write (unit, '(a)') 'int _ffc_alloc_status(void);'
        write (unit, '(a)') 'int main(void) {'
        write (unit, '(a)') '    char *p, *q;'
        write (unit, '(a)') '    int i;'
        write (unit, '(a)') '    p = _ffc_alloc(4, 8);'
        write (unit, '(a)') '    if (p == 0) return 1;'
        write (unit, '(a)') '    if (_ffc_alloc_status() != 0) return 2;'
        write (unit, '(a)') '    for (i = 0; i < 32; i++) p[i] = (char) i;'
        write (unit, '(a)') '    /* resize keeps the leading bytes */'
        write (unit, '(a)') '    q = _ffc_realloc(p, 16, 8);'
        write (unit, '(a)') '    if (q == 0) return 3;'
        write (unit, '(a)') '    for (i = 0; i < 32; i++)'
        write (unit, '(a)') '        if (q[i] != (char) i) return 4;'
        write (unit, '(a)') '    /* zeroed allocation */'
        write (unit, '(a)') '    p = _ffc_calloc(4, 8);'
        write (unit, '(a)') '    if (p == 0) return 5;'
        write (unit, '(a)') '    for (i = 0; i < 32; i++)'
        write (unit, '(a)') '        if (p[i] != 0) return 6;'
        write (unit, '(a)') '    if (_ffc_dealloc(p, 1) != 0) return 7;'
        write (unit, '(a)') '    /* a second release is reported */'
        write (unit, '(a)') '    if (_ffc_dealloc(p, 1) != 6004) return 8;'
        write (unit, '(a)') '    if (_ffc_alloc_status() != 6004) return 9;'
        write (unit, '(a)') '    /* a borrowed descriptor never frees */'
        write (unit, '(a)') '    if (_ffc_dealloc(q, 0) != 6005) return 10;'
        write (unit, '(a)') '    if (_ffc_dealloc(q, 1) != 0) return 11;'
        write (unit, '(a)') '    /* releasing nothing succeeds */'
        write (unit, '(a)') '    if (_ffc_dealloc(0, 1) != 0) return 12;'
        write (unit, '(a)') '    /* a negative extent is rejected */'
        write (unit, '(a)') '    if (_ffc_alloc(-1, 4) != 0) return 13;'
        write (unit, '(a)') '    if (_ffc_alloc_status() != 6001) return 14;'
        write (unit, '(a)') '    /* an unrepresentable size is rejected */'
        write (unit, '(a)') '    if (_ffc_alloc(1LL << 62, 64) != 0)'
        write (unit, '(a)') '        return 15;'
        write (unit, '(a)') '    if (_ffc_alloc_status() != 6002) return 16;'
        write (unit, '(a)') '    /* a zero-sized array is still releasable */'
        write (unit, '(a)') '    p = _ffc_alloc(0, 4);'
        write (unit, '(a)') '    if (p == 0) return 17;'
        write (unit, '(a)') '    if (_ffc_dealloc(p, 1) != 0) return 18;'
        write (unit, '(a)') '    return 0;'
        write (unit, '(a)') '}'
        close (unit)

        status = status_of('cc -o '//exe//' '//driver//' '//link_input)
        if (status /= 0) then
            print *, 'FAIL: the allocation driver does not build'
            return
        end if
        status = status_of(exe)
        if (status /= 0) then
            print *, 'FAIL: allocation contract check ', status, ' failed'
            return
        end if
        ok = .true.
    end function test_runtime_contract

    ! Allocate, reallocate and free from Fortran; values survive.
    logical function test_values_survive_transitions() result(ok)
        character(len=*), parameter :: Q = achar(39)
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  character(len=:), allocatable :: s'//new_line('a')// &
            '  integer :: i'//new_line('a')// &
            '  allocate(a(4))'//new_line('a')// &
            '  do i = 1, 4'//new_line('a')// &
            '     a(i) = i * 10'//new_line('a')// &
            '  end do'//new_line('a')// &
            '  print *, a(1), a(4), size(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  allocate(a(2))'//new_line('a')// &
            '  a = [5, 6]'//new_line('a')// &
            '  print *, a(1) + a(2), size(a)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            '  allocate(character(len=3) :: s)'//new_line('a')// &
            '  s = '//Q//'xyz'//Q//new_line('a')// &
            '  print *, s'//new_line('a')// &
            '  deallocate(s)'//new_line('a')// &
            'end program main'

        ok = expect_output(source, &
            '          10          40           4'//new_line('a')// &
            '          11           2'//new_line('a')// &
            ' xyz'//new_line('a'), &
            WORK//'/transitions')
    end function test_values_survive_transitions

    ! Emitted code must reach the runtime helpers, not malloc and free.
    logical function test_lowering_calls_the_runtime() result(ok)
        character(len=*), parameter :: exe = WORK//'/symbols'
        character(len=*), parameter :: nm_out = WORK//'/symbols.nm'
        character(len=*), parameter :: source = &
            'program main'//new_line('a')// &
            '  integer, allocatable :: a(:)'//new_line('a')// &
            '  allocate(a(3))'//new_line('a')// &
            '  a(1) = 1'//new_line('a')// &
            '  print *, a(1)'//new_line('a')// &
            '  deallocate(a)'//new_line('a')// &
            'end program main'
        character(len=:), allocatable :: error_msg, symbols
        logical :: read_ok
        integer :: status

        ok = .false.
        call compile_to_exe(source, exe, error_msg)
        if (len_trim(error_msg) > 0) then
            print *, 'FAIL: compiling the probe program: ', trim(error_msg)
            return
        end if
        status = status_of('nm '//exe//' > '//nm_out)
        if (status /= 0) then
            print *, 'FAIL: cannot list the emitted symbols'
            return
        end if
        call read_text_file(nm_out, symbols, read_ok)
        if (.not. read_ok) then
            print *, 'FAIL: cannot read the emitted symbol list'
            return
        end if
        if (index(symbols, '_ffc_alloc') == 0) then
            print *, 'FAIL: allocation does not call the runtime'
            return
        end if
        if (index(symbols, '_ffc_dealloc') == 0) then
            print *, 'FAIL: deallocation does not call the runtime'
            return
        end if
        ! A leftover direct malloc or free in compiler-emitted code would be
        ! a second convention for the same storage, and the two allocators do
        ! not interoperate: storage from malloc is not known to the runtime,
        ! so releasing it would be reported as a double free. The runtime's
        ! own functions are where malloc and free belong, so the check is on
        ! the calling function, not on the mere presence of the call.
        status = status_of('objdump -d '//exe//' | awk '''// &
            '/^[0-9a-f]+ </ { fn = $2 } '// &
            '/call.*<(malloc|free)@plt>/ { if (fn !~ /ffc_/) found = 1 } '// &
            'END { exit found ? 0 : 1 }'//''' ')
        if (status == 0) then
            print *, 'FAIL: emitted code still calls malloc or free directly'
            return
        end if
        ok = .true.
    end function test_lowering_calls_the_runtime

end program test_session_runtime_allocation_helpers_compiler
