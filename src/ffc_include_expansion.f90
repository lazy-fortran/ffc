module ffc_include_expansion
    ! Textual expansion of Fortran INCLUDE lines (F2018 clause 6.4.2).
    !
    ! The include line is replaced by the content of the referenced file before
    ! the frontend parses the program, so declarations pulled in by an include
    ! participate in semantic analysis exactly as if they had been written in
    ! place. The including file's own directory is searched first, then the
    ! user's -I directories. Include cycles and missing files are reported with
    ! the include site so the diagnostic still points at real source.
    !
    ! A byte order mark is only meaningful at the very start of a file, so an
    ! included file's leading BOM is decoded away as its contents are spliced
    ! in; leaving it would place a BOM mid-source, where it is not a BOM at all
    ! but an invalid source character. The outermost file's own BOM is left
    ! alone: it still starts the text the frontend decodes. Decoding reuses the
    ! frontend's decoder, so ffc and the frontend recognize exactly the same
    ! set of marks.
    use source_bom, only: decode_source_bom
    implicit none
    private

    public :: expand_source_includes

    integer, parameter :: PATH_LEN = 512
    integer, parameter :: MAX_INCLUDE_DEPTH = 64

contains

    ! Reads path and returns its source with every INCLUDE line expanded.
    ! source stays unallocated when path itself cannot be read; that case is
    ! left to the frontend so its file diagnostic is preserved.
    subroutine expand_source_includes(path, include_paths, source, error_msg)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: include_paths(:)
        character(len=:), allocatable, intent(out) :: source
        character(len=:), allocatable, intent(out) :: error_msg
        character(len=PATH_LEN) :: stack(MAX_INCLUDE_DEPTH)

        error_msg = ''
        call expand_file(path, dirname(path), include_paths, stack, 0, source, &
                         error_msg)
        if (len_trim(error_msg) > 0) then
            if (allocated(source)) deallocate (source)
        end if
    end subroutine expand_source_includes

    recursive subroutine expand_file(path, root_dir, include_paths, stack, &
                                     depth, source, error_msg)
        character(len=*), intent(in) :: path
        character(len=*), intent(in) :: root_dir
        character(len=*), intent(in) :: include_paths(:)
        character(len=PATH_LEN), intent(inout) :: stack(:)
        integer, intent(in) :: depth
        character(len=:), allocatable, intent(out) :: source
        character(len=:), allocatable, intent(inout) :: error_msg
        character(len=:), allocatable :: text
        character(len=:), allocatable :: decoded
        character(len=:), allocatable :: line
        character(len=:), allocatable :: name
        character(len=:), allocatable :: resolved
        character(len=:), allocatable :: nested
        integer :: pos, nl, line_no, i
        logical :: terminated

        call read_whole_file(path, text)
        if (.not. allocated(text)) return
        ! depth 0 is the file being compiled, whose leading BOM belongs to the
        ! source the frontend decodes. Any deeper file is being spliced into
        ! the middle of that source, so its own mark is decoded away here.
        if (depth > 0) then
            call decode_source_bom(text, decoded)
            call move_alloc(decoded, text)
        end if
        if (depth >= MAX_INCLUDE_DEPTH) then
            error_msg = 'include nesting too deep at '//trim(path)
            return
        end if
        stack(depth + 1) = path

        source = ''
        pos = 1
        line_no = 0
        do while (pos <= len(text))
            nl = index(text(pos:), new_line('a'))
            if (nl == 0) then
                line = text(pos:)
                pos = len(text) + 1
                terminated = .false.
            else
                line = text(pos:pos + nl - 2)
                pos = pos + nl
                terminated = .true.
            end if
            line_no = line_no + 1

            if (.not. include_name(line, name)) then
                ! A final fragment with no newline of its own must not gain
                ! one: a wide-encoded file (UTF-16/UTF-32 BOM) ends in the
                ! zero bytes that pad its last line terminator, and adding a
                ! byte there breaks the byte count its decoder requires.
                if (terminated) then
                    source = source//line//new_line('a')
                else
                    source = source//line
                end if
                cycle
            end if

            call resolve_include(name, dirname(path), root_dir, &
                                 include_paths, resolved)
            if (.not. allocated(resolved)) then
                error_msg = "include file not found: '"//name// &
                            "' (included at "//trim(path)//':'// &
                            int_text(line_no)//')'
                return
            end if
            do i = 1, depth + 1
                if (trim(stack(i)) == resolved) then
                    error_msg = "include cycle: '"//resolved// &
                                "' (included at "//trim(path)//':'// &
                                int_text(line_no)//')'
                    return
                end if
            end do

            call expand_file(resolved, root_dir, include_paths, stack, &
                             depth + 1, nested, error_msg)
            if (len_trim(error_msg) > 0) return
            if (.not. allocated(nested)) then
                error_msg = "cannot read include file: '"//resolved// &
                            "' (included at "//trim(path)//':'// &
                            int_text(line_no)//')'
                return
            end if
            source = source//nested
        end do
    end subroutine expand_file

    ! True when line is an INCLUDE line; then name holds the quoted file name.
    logical function include_name(line, name) result(is_include)
        character(len=*), intent(in) :: line
        character(len=:), allocatable, intent(out) :: name
        character(len=:), allocatable :: rest
        character(len=1) :: quote
        integer :: closing

        is_include = .false.
        if (allocated(name)) deallocate (name)
        rest = adjustl(strip_bom(line))
        if (len_trim(rest) < 10) return
        if (lowercase(rest(1:7)) /= 'include') return
        rest = adjustl(rest(8:))
        if (len_trim(rest) < 3) return
        quote = rest(1:1)
        if (quote /= '''' .and. quote /= '"') return
        closing = index(rest(2:), quote)
        if (closing <= 1) return
        name = rest(2:closing)
        if (len_trim(name) == 0) return
        ! Only blanks or a trailing comment may follow the file name.
        rest = adjustl(rest(closing + 2:))
        if (len_trim(rest) > 0) then
            if (rest(1:1) /= '!') return
        end if
        is_include = .true.
    end function include_name

    ! Searches the including file's directory first, then the compiled file's
    ! own directory, then the -I paths.
    subroutine resolve_include(name, source_dir, root_dir, include_paths, &
                               resolved)
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: source_dir
        character(len=*), intent(in) :: root_dir
        character(len=*), intent(in) :: include_paths(:)
        character(len=:), allocatable, intent(out) :: resolved
        character(len=:), allocatable :: candidate
        logical :: found
        integer :: i

        if (allocated(resolved)) deallocate (resolved)
        if (name(1:1) == '/') then
            inquire (file=name, exist=found)
            if (found) resolved = name
            return
        end if

        candidate = source_dir//'/'//name
        inquire (file=candidate, exist=found)
        if (found) then
            resolved = candidate
            return
        end if

        candidate = root_dir//'/'//name
        inquire (file=candidate, exist=found)
        if (found) then
            resolved = candidate
            return
        end if

        do i = 1, size(include_paths)
            if (len_trim(include_paths(i)) == 0) cycle
            candidate = trim(include_paths(i))//'/'//name
            inquire (file=candidate, exist=found)
            if (found) then
                resolved = candidate
                return
            end if
        end do
    end subroutine resolve_include

    subroutine read_whole_file(path, text)
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: text
        integer :: unit, io_stat, bytes

        open (newunit=unit, file=path, status='old', action='read', &
              access='stream', form='unformatted', iostat=io_stat)
        if (io_stat /= 0) return
        inquire (unit=unit, size=bytes)
        if (bytes <= 0) then
            allocate (character(len=0) :: text)
            close (unit)
            return
        end if
        allocate (character(len=bytes) :: text)
        read (unit, iostat=io_stat) text
        close (unit)
        if (io_stat /= 0) deallocate (text)
    end subroutine read_whole_file

    ! Drops a UTF-8 byte order mark and a trailing carriage return.
    function strip_bom(line) result(clean)
        character(len=*), intent(in) :: line
        character(len=:), allocatable :: clean
        integer :: n

        clean = line
        n = len(clean)
        if (n >= 3) then
            if (clean(1:3) == achar(239)//achar(187)//achar(191)) then
                clean = clean(4:)
                n = len(clean)
            end if
        end if
        if (n >= 1) then
            if (clean(n:n) == achar(13)) clean = clean(:n - 1)
        end if
    end function strip_bom

    function lowercase(text) result(lowered)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: lowered
        integer :: i, code

        lowered = text
        do i = 1, len(lowered)
            code = iachar(lowered(i:i))
            if (code >= iachar('A') .and. code <= iachar('Z')) then
                lowered(i:i) = achar(code + 32)
            end if
        end do
    end function lowercase

    function dirname(path) result(dir)
        character(len=*), intent(in) :: path
        character(len=:), allocatable :: dir
        integer :: slash

        slash = index(trim(path), '/', back=.true.)
        if (slash > 0) then
            dir = path(1:slash - 1)
        else
            dir = '.'
        end if
    end function dirname

    function int_text(value) result(text)
        integer, intent(in) :: value
        character(len=:), allocatable :: text
        character(len=32) :: buffer

        write (buffer, '(I0)') value
        text = trim(buffer)
    end function int_text

end module ffc_include_expansion
