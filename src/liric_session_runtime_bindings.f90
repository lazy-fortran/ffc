! Runtime archive selection and installation (issue #376).
!
! Selects the LIRIC runtime archive matching the active backend and target,
! then installs it into a session before any runtime call is lowered. The
! archives themselves are produced by runtime/CMakeLists.txt (#374); see
! docs/RUNTIME_ABI.md for the artifact names and the archive header format.
!
! Selection is strict and unconditional: the caller names the artifact
! directory, and a missing archive, or one built for a different backend or
! target, is an error rather than a fallback. #565 removed the environment
! variable that used to make this opt-in, along with the inline-runtime path
! it fell back to; ffc now links its runtime into every executable it emits
! (see ffc_runtime_link), and this facility serves sessions that must resolve
! runtime calls without a system linker.
module liric_session_runtime_bindings
    use liric_session_common, only: liric_session_t, lr_error_t, LR_OK, &
        status_ok, require_open_session, set_empty
    use, intrinsic :: iso_c_binding, only: c_ptr, c_int, c_size_t, c_loc, &
        c_int8_t
    implicit none
    private

    public :: LR_SESSION_BACKEND_DEFAULT, LR_SESSION_BACKEND_ISEL, &
        LR_SESSION_BACKEND_COPY_PATCH, LR_SESSION_BACKEND_LLVM
    public :: runtime_archive_backend_name, runtime_archive_name, &
        runtime_archive_path, effective_archive_backend
    public :: install_runtime_archive

    ! Frozen LIRIC session backend enumerators (LIRIC #527 ABI freeze).
    integer(c_int), parameter :: LR_SESSION_BACKEND_DEFAULT = 0_c_int
    integer(c_int), parameter :: LR_SESSION_BACKEND_ISEL = 1_c_int
    integer(c_int), parameter :: LR_SESSION_BACKEND_COPY_PATCH = 2_c_int
    integer(c_int), parameter :: LR_SESSION_BACKEND_LLVM = 3_c_int

    ! Artifact naming version; see docs/RUNTIME_ABI.md.
    character(len=*), parameter :: ARTIFACT_VERSION = 'v2'
    character(len=*), parameter :: ARCHIVE_MAGIC = 'LRARCH1'

    interface
        function lr_session_set_runtime_archive(handle, data, len, err) &
            result(status) bind(c, name='lr_session_set_runtime_archive')
            import :: c_ptr, c_int, c_size_t, lr_error_t
            type(c_ptr), value :: handle
            type(c_ptr), value :: data
            integer(c_size_t), value :: len
            type(lr_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_session_set_runtime_archive
    end interface

contains

    ! Backend `default` is an alias for copy-patch and has no artifact of its
    ! own, so it resolves to the copy-patch archive.
    pure function effective_archive_backend(backend) result(resolved)
        integer(c_int), intent(in) :: backend
        integer(c_int) :: resolved

        if (backend == LR_SESSION_BACKEND_DEFAULT) then
            resolved = LR_SESSION_BACKEND_COPY_PATCH
        else
            resolved = backend
        end if
    end function effective_archive_backend

    pure function runtime_archive_backend_name(backend) result(name)
        integer(c_int), intent(in) :: backend
        character(len=:), allocatable :: name
        integer(c_int) :: resolved

        resolved = effective_archive_backend(backend)
        select case (resolved)
        case (LR_SESSION_BACKEND_ISEL)
            name = 'isel'
        case (LR_SESSION_BACKEND_COPY_PATCH)
            name = 'copy-patch'
        case (LR_SESSION_BACKEND_LLVM)
            name = 'llvm'
        case default
            name = ''
        end select
    end function runtime_archive_backend_name

    pure function runtime_archive_name(backend) result(name)
        integer(c_int), intent(in) :: backend
        character(len=:), allocatable :: name
        character(len=:), allocatable :: backend_name

        backend_name = runtime_archive_backend_name(backend)
        if (len(backend_name) == 0) then
            name = ''
        else
            name = 'ffc-runtime-'//ARTIFACT_VERSION//'-'//backend_name// &
                   '.lrarch'
        end if
    end function runtime_archive_name

    ! Archives live in a target-qualified subdirectory of the configured
    ! artifact root, matching the layout runtime/CMakeLists.txt writes.
    pure function runtime_archive_path(directory, target, backend) result(path)
        character(len=*), intent(in) :: directory
        character(len=*), intent(in) :: target
        integer(c_int), intent(in) :: backend
        character(len=:), allocatable :: path
        character(len=:), allocatable :: name

        name = runtime_archive_name(backend)
        if (len(name) == 0) then
            path = ''
        else
            path = trim(directory)//'/'//trim(target)//'/'//name
        end if
    end function runtime_archive_path

    subroutine read_archive_bytes(path, bytes, error_msg)
        character(len=*), intent(in) :: path
        integer(c_int8_t), allocatable, intent(out) :: bytes(:)
        character(len=:), allocatable, intent(out) :: error_msg
        integer :: unit, ios
        integer(kind=8) :: nbytes
        logical :: exists

        call set_empty(error_msg)
        inquire (file=path, exist=exists)
        if (.not. exists) then
            error_msg = 'runtime archive not found: '//path
            return
        end if
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) then
            error_msg = 'cannot open runtime archive: '//path
            return
        end if
        inquire (unit=unit, size=nbytes)
        if (nbytes <= 0) then
            close (unit)
            error_msg = 'runtime archive is empty: '//path
            return
        end if
        allocate (bytes(nbytes))
        read (unit, iostat=ios) bytes
        close (unit)
        if (ios /= 0) then
            deallocate (bytes)
            error_msg = 'cannot read runtime archive: '//path
        end if
    end subroutine read_archive_bytes

    ! Reads the little-endian u32 at the given 1-based byte offset.
    pure function le_u32(bytes, offset) result(value)
        integer(c_int8_t), intent(in) :: bytes(:)
        integer, intent(in) :: offset
        integer :: value
        integer :: i, byte

        value = 0
        do i = 3, 0, -1
            byte = int(bytes(offset + i))
            if (byte < 0) byte = byte + 256
            value = value*256 + byte
        end do
    end function le_u32

    ! Rejects an archive that does not match what was asked for, rather than
    ! accepting a mismatched fallback.
    subroutine verify_archive(bytes, path, backend, error_msg)
        integer(c_int8_t), intent(in) :: bytes(:)
        character(len=*), intent(in) :: path
        integer(c_int), intent(in) :: backend
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=len(ARCHIVE_MAGIC)) :: magic
        integer :: i, byte, recorded
        integer(c_int) :: wanted

        call set_empty(error_msg)
        if (size(bytes) < 16) then
            error_msg = 'runtime archive is truncated: '//path
            return
        end if
        do i = 1, len(ARCHIVE_MAGIC)
            byte = int(bytes(i))
            if (byte < 0) byte = byte + 256
            magic(i:i) = achar(byte)
        end do
        if (magic /= ARCHIVE_MAGIC) then
            error_msg = 'not a LIRIC runtime archive: '//path
            return
        end if
        wanted = effective_archive_backend(backend)
        recorded = le_u32(bytes, 13)
        if (recorded /= int(wanted)) then
            error_msg = 'runtime archive backend mismatch for '//path// &
                ': archive records backend '// &
                trim(backend_label(int(recorded, c_int)))// &
                ' but the session uses backend '// &
                trim(backend_label(wanted))
        end if
    end subroutine verify_archive

    pure function backend_label(backend) result(label)
        integer(c_int), intent(in) :: backend
        character(len=:), allocatable :: label
        character(len=:), allocatable :: name
        character(len=16) :: number

        name = runtime_archive_backend_name(backend)
        write (number, '(i0)') backend
        if (len(name) == 0) then
            label = 'unknown('//trim(number)//')'
        else
            label = name//'('//trim(number)//')'
        end if
    end function backend_label

    ! Installs the archive matching `backend` and `target` into `session`,
    ! reading it from `directory`.
    !
    ! The directory is an explicit argument, with no environment variable
    ! behind it. #565 retired the opt-in form of this call: a caller either
    ! installs an archive or it does not, and every failure to install a
    ! requested archive is an error. Nothing falls back to synthesising the
    ! runtime entry points inline. Executables emitted by ffc get their
    ! runtime by static link instead (see ffc_runtime_link); this facility
    ! remains for sessions that resolve runtime calls without a system linker.
    logical function install_runtime_archive(session, directory, backend, &
                                             target, error_msg) result(ok)
        type(liric_session_t), intent(inout) :: session
        character(len=*), intent(in) :: directory
        integer(c_int), intent(in) :: backend
        character(len=*), intent(in) :: target
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: path
        integer(c_int8_t), allocatable, target :: bytes(:)
        type(lr_error_t) :: error
        integer(c_int) :: status

        ok = .false.
        call set_empty(error_msg)

        if (len_trim(directory) == 0) then
            error_msg = 'no runtime archive directory was given'
            return
        end if

        if (.not. require_open_session(session, error_msg)) return

        path = runtime_archive_path(directory, target, backend)
        if (len(path) == 0) then
            error_msg = 'no runtime archive is defined for backend '// &
                trim(backend_label(backend))
            return
        end if

        call read_archive_bytes(path, bytes, error_msg)
        if (len_trim(error_msg) > 0) return

        call verify_archive(bytes, path, backend, error_msg)
        if (len_trim(error_msg) > 0) return

        error%code = LR_OK
        status = lr_session_set_runtime_archive(session%handle, &
                                                c_loc(bytes(1)), &
                                                int(size(bytes), c_size_t), &
                                                error)
        if (.not. status_ok(status, error, error_msg)) then
            error_msg = 'failed to install runtime archive '//path//': '// &
                error_msg
            return
        end if

        call set_empty(error_msg)
        ok = .true.
    end function install_runtime_archive

end module liric_session_runtime_bindings
