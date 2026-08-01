! Runtime delivery for emitted executables (issue #565).
!
! ffc lowers runtime entry points as calls to external symbols, so every
! executable it emits must carry a definition of those symbols. This module
! provides that definition as a link input.
!
! The runtime source is embedded in the compiler binary (ffc_runtime_source,
! generated from runtime/ffc_runtime.c). It is materialised into a
! content-addressed file under the temporary directory and handed to
! lr_session_emit_exe_objects, which passes it to the system C compiler that
! already performs the link. Consequences, all deliberate:
!
!   - There is no installed runtime artifact to discover, so there is no
!     environment variable to set and no opt-in path. Every executable ffc
!     emits is linked against the runtime, unconditionally.
!   - A compiler/runtime version mismatch is impossible by construction: the
!     runtime that gets linked is the one compiled into this binary.
!   - When the runtime cannot be materialised, lowering fails with a named
!     error. It never silently falls back to synthesising the entry points
!     inline; that parallel lowering path is retired (ROADMAP Chunk 3).
!
! FFC_RUNTIME_SYMBOLS is the authoritative list of entry points the lowerer
! may call. test_runtime_link_compiler compiles the embedded source and fails
! unless every listed symbol is defined by it, so emitting a call to a symbol
! the runtime does not define is caught at test time rather than becoming an
! executable that dies with "undefined symbol" at run time.
module ffc_runtime_link
    use ffc_runtime_source, only: ffc_runtime_source_text
    use, intrinsic :: iso_c_binding, only: c_int, c_char, c_null_char
    implicit none
    private

    public :: FFC_RUNTIME_SYMBOLS, ffc_runtime_link_input

    interface
        function c_getpid() result(pid) bind(c, name='getpid')
            import :: c_int
            integer(c_int) :: pid
        end function c_getpid

        function c_rename(old_path, new_path) result(status) &
            bind(c, name='rename')
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: old_path(*)
            character(kind=c_char), intent(in) :: new_path(*)
            integer(c_int) :: status
        end function c_rename
    end interface

    ! Every runtime entry point the lowerer is allowed to call. Each issue
    ! that moves compiler-emitted code behind the runtime ABI adds its symbols
    ! here and to docs/RUNTIME_ABI.md.
    character(len=*), parameter :: FFC_RUNTIME_SYMBOLS(1) = &
        [character(len=32) :: '_ffc_runtime_probe']

contains

    ! Returns the path of a C source file holding the embedded runtime, ready
    ! to be passed to lr_session_emit_exe_objects as a link input. The name is
    ! derived from the runtime contents, so concurrent compilations share one
    ! file and a stale file from an older compiler is never picked up.
    subroutine ffc_runtime_link_input(path, error_msg)
        character(len=:), allocatable, intent(out) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: text

        error_msg = ''
        call ffc_runtime_source_text(text)
        if (len(text) == 0) then
            error_msg = 'ffc runtime source is empty: the compiler was '// &
                'built without a usable runtime'
            path = ''
            return
        end if
        path = runtime_link_path(text)
        call materialise(path, text, error_msg)
        if (len_trim(error_msg) > 0) path = ''
    end subroutine ffc_runtime_link_input

    function runtime_link_path(text) result(path)
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: path
        character(len=16) :: digest

        write (digest, '(z16.16)') fnv1a(text)
        path = temp_directory()//'/ffc-runtime-'//trim(digest)//'.c'
    end function runtime_link_path

    ! FNV-1a over the runtime text. Only used to name the file; correctness
    ! does not rest on it, because the contents are verified before reuse.
    function fnv1a(text) result(hash)
        character(len=*), intent(in) :: text
        integer(kind=8) :: hash
        integer(kind=8), parameter :: PRIME = 1099511628211_8
        integer :: i

        hash = -3750763034362895579_8
        do i = 1, len(text)
            hash = ieor(hash, int(iachar(text(i:i)), kind=8))
            hash = hash*PRIME
        end do
    end function fnv1a

    function temp_directory() result(directory)
        character(len=:), allocatable :: directory
        integer :: length, status

        call get_environment_variable('TMPDIR', length=length, status=status)
        if (status /= 0 .or. length <= 0) then
            directory = '/tmp'
            return
        end if
        allocate (character(len=length) :: directory)
        call get_environment_variable('TMPDIR', directory)
        if (len_trim(directory) == 0) then
            directory = '/tmp'
        else
            directory = trim(directory)
        end if
    end function temp_directory

    ! Writes text to path unless a file with exactly that content is already
    ! there. The write goes to a private temporary and is renamed into place,
    ! so parallel compilations never observe a partial file.
    subroutine materialise(path, text, error_msg)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: text
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=:), allocatable :: staging
        character(len=32) :: pid_text
        integer :: unit, ios

        error_msg = ''
        if (file_matches(path, text)) return

        write (pid_text, '(i0)') getpid()
        staging = path//'.'//trim(pid_text)//'.tmp'
        open (newunit=unit, file=staging, access='stream', &
              form='unformatted', status='replace', action='write', &
              iostat=ios)
        if (ios /= 0) then
            error_msg = 'cannot write the ffc runtime to '//staging
            return
        end if
        write (unit, iostat=ios) text
        close (unit, iostat=ios)
        if (ios /= 0) then
            error_msg = 'cannot write the ffc runtime to '//staging
            return
        end if
        call rename_into_place(staging, path, error_msg)
    end subroutine materialise

    logical function file_matches(path, text) result(matches)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: text
        character(len=:), allocatable :: existing
        integer :: unit, ios
        integer(kind=8) :: nbytes
        logical :: exists

        matches = .false.
        inquire (file=path, exist=exists)
        if (.not. exists) return
        open (newunit=unit, file=path, access='stream', form='unformatted', &
              status='old', action='read', iostat=ios)
        if (ios /= 0) return
        inquire (unit=unit, size=nbytes)
        if (nbytes /= len(text)) then
            close (unit)
            return
        end if
        allocate (character(len=len(text)) :: existing)
        read (unit, iostat=ios) existing
        close (unit)
        if (ios == 0) matches = existing == text
    end function file_matches

    subroutine rename_into_place(staging, path, error_msg)
        character(len=*), intent(in) :: staging
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable, target :: c_from(:), c_to(:)

        error_msg = ''
        call to_c_string(staging, c_from)
        call to_c_string(path, c_to)
        if (c_rename(c_from, c_to) /= 0_c_int) then
            error_msg = 'cannot install the ffc runtime at '//path
        end if
    end subroutine rename_into_place

    subroutine to_c_string(text, buffer)
        character(len=*), intent(in) :: text
        character(kind=c_char), allocatable, intent(out) :: buffer(:)
        integer :: i

        allocate (buffer(len(text) + 1))
        do i = 1, len(text)
            buffer(i) = text(i:i)
        end do
        buffer(len(text) + 1) = c_null_char
    end subroutine to_c_string

    integer function getpid()
        getpid = int(c_getpid())
    end function getpid

end module ffc_runtime_link
