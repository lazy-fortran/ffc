! Issue #374: build backend-qualified LIRIC runtime archives.
!
! Behavioral oracle for the runtime artifact production step. It drives the
! standalone runtime/ CMake project and checks the produced artifacts against
! their documented contract rather than against the patch:
!
!   * one archive per backend, at the documented target-qualified path
!   * each archive is nonempty and carries the LIRIC archive magic and a
!     nonzero format version
!   * each archive records the backend it was built for, so the artifacts are
!     genuinely backend-qualified and not copies of one another
!   * the packaged runtime IR defines the probe symbol
!   * a missing tool dependency fails with a named diagnostic
program test_runtime_archives
    implicit none

    character(len=*), parameter :: build_dir = '/tmp/ffc_runtime_374_build'
    character(len=*), parameter :: artifact_dir = &
        build_dir//'/artifacts/host'
    integer :: failures

    failures = 0

    print *, '=== ffc runtime archive tests (#374) ==='

    call configure_and_build(failures)
    if (failures == 0) then
        call check_archive('isel', 1, failures)
        call check_archive('copy-patch', 2, failures)
        call check_probe_symbol(failures)
        call check_backend_qualification(failures)
    end if
    call check_missing_tool_diagnostic(failures)

    if (failures /= 0) then
        print *, 'FAIL: ', failures, ' runtime archive check(s) failed'
        stop 1
    end if
    print *, 'PASS: runtime archive contract'

contains

    subroutine run(cmd, exit_stat)
        character(len=*), intent(in) :: cmd
        integer, intent(out) :: exit_stat
        integer :: cmd_stat

        call execute_command_line(cmd, exitstat=exit_stat, cmdstat=cmd_stat)
        if (cmd_stat /= 0) exit_stat = -1
    end subroutine run

    subroutine configure_and_build(nfail)
        integer, intent(inout) :: nfail
        integer :: stat

        call run('rm -rf '//build_dir, stat)
        call run('cmake -S runtime -B '//build_dir// &
                 ' > /tmp/ffc_runtime_374_cfg.log 2>&1', stat)
        if (stat /= 0) then
            print *, 'FAIL: runtime cmake configure failed'
            call run('cat /tmp/ffc_runtime_374_cfg.log', stat)
            nfail = nfail + 1
            return
        end if
        call run('cmake --build '//build_dir//' -j 3'// &
                 ' > /tmp/ffc_runtime_374_build.log 2>&1', stat)
        if (stat /= 0) then
            print *, 'FAIL: runtime archive build failed'
            call run('cat /tmp/ffc_runtime_374_build.log', stat)
            nfail = nfail + 1
        end if
    end subroutine configure_and_build

    function archive_path(backend) result(path)
        character(len=*), intent(in) :: backend
        character(len=:), allocatable :: path

        path = artifact_dir//'/ffc-runtime-v2-'//backend//'.lrarch'
    end function archive_path

    ! Reads the LIRIC archive header: 8-byte magic, then little-endian u32
    ! version and u32 backend.
    subroutine read_header(path, ok, magic, version, backend_code)
        character(len=*), intent(in) :: path
        logical, intent(out) :: ok
        character(len=8), intent(out) :: magic
        integer, intent(out) :: version
        integer, intent(out) :: backend_code
        integer :: unit, ios, i
        integer :: bytes(16)
        character(len=1) :: byte

        ok = .false.
        magic = ''
        version = 0
        backend_code = 0
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) return
        do i = 1, 16
            read (unit, iostat=ios) byte
            if (ios /= 0) then
                close (unit)
                return
            end if
            bytes(i) = iachar(byte)
            if (i <= 8) magic(i:i) = byte
        end do
        close (unit)
        version = bytes(9) + bytes(10)*256 + bytes(11)*65536 + &
                  bytes(12)*16777216
        backend_code = bytes(13) + bytes(14)*256 + bytes(15)*65536 + &
                       bytes(16)*16777216
        ok = .true.
    end subroutine read_header

    subroutine check_archive(backend, expected_backend_code, nfail)
        character(len=*), intent(in) :: backend
        integer, intent(in) :: expected_backend_code
        integer, intent(inout) :: nfail
        character(len=:), allocatable :: path
        character(len=8) :: magic
        integer :: version, backend_code, unit, ios
        integer(kind=8) :: fsize
        logical :: ok, exists

        path = archive_path(backend)
        inquire (file=path, exist=exists)
        if (.not. exists) then
            print *, 'FAIL: missing archive ', path
            nfail = nfail + 1
            return
        end if
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) then
            print *, 'FAIL: cannot open archive ', path
            nfail = nfail + 1
            return
        end if
        inquire (unit=unit, size=fsize)
        close (unit)
        if (fsize <= 0) then
            print *, 'FAIL: empty archive ', path
            nfail = nfail + 1
            return
        end if

        call read_header(path, ok, magic, version, backend_code)
        if (.not. ok) then
            print *, 'FAIL: cannot read header of ', path
            nfail = nfail + 1
            return
        end if
        if (magic(1:7) /= 'LRARCH1') then
            print *, 'FAIL: bad magic in ', path, ' got ', magic(1:7)
            nfail = nfail + 1
        end if
        if (version == 0) then
            print *, 'FAIL: zero archive format version in ', path
            nfail = nfail + 1
        end if
        if (backend_code /= expected_backend_code) then
            print *, 'FAIL: ', path, ' records backend ', backend_code, &
                ' expected ', expected_backend_code
            nfail = nfail + 1
        end if
    end subroutine check_archive

    ! The packaged runtime IR must define the probe entry point.
    subroutine check_probe_symbol(nfail)
        integer, intent(inout) :: nfail
        integer :: stat

        call run('grep -q "_ffc_runtime_probe" '// &
                 archive_path('copy-patch'), stat)
        if (stat /= 0) then
            print *, 'FAIL: archive does not carry _ffc_runtime_probe'
            nfail = nfail + 1
        end if
    end subroutine check_probe_symbol

    ! Distinct backends must produce genuinely distinct artifacts.
    subroutine check_backend_qualification(nfail)
        integer, intent(inout) :: nfail
        integer :: stat

        call run('cmp -s '//archive_path('isel')//' '// &
                 archive_path('copy-patch'), stat)
        if (stat == 0) then
            print *, 'FAIL: isel and copy-patch archives are identical'
            nfail = nfail + 1
        end if
    end subroutine check_backend_qualification

    ! A missing archive tool must fail configuration with a named dependency
    ! diagnostic, not silently skip the artifact.
    subroutine check_missing_tool_diagnostic(nfail)
        integer, intent(inout) :: nfail
        integer :: stat

        call run('rm -rf /tmp/ffc_runtime_374_missing', stat)
        call run('cmake -S runtime -B /tmp/ffc_runtime_374_missing'// &
                 ' -DFFC_RUNTIME_ARCHIVE_TOOL=FFC_RUNTIME_ARCHIVE_TOOL-NOTFOUND'// &
                 ' -DCMAKE_PREFIX_PATH=/nonexistent'// &
                 ' -DLIRIC_BUILD_DIR=/nonexistent'// &
                 ' > /tmp/ffc_runtime_374_missing.log 2>&1', stat)
        if (stat == 0) then
            print *, 'FAIL: missing archive tool did not fail configuration'
            nfail = nfail + 1
            return
        end if
        call run('grep -q "ffc runtime dependency missing: '// &
                 'liric_runtime_archive" /tmp/ffc_runtime_374_missing.log', &
                 stat)
        if (stat /= 0) then
            print *, 'FAIL: missing tool lacked a named dependency diagnostic'
            nfail = nfail + 1
        end if
    end subroutine check_missing_tool_diagnostic

end program test_runtime_archives
