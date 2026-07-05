module ffc_cli_options
    implicit none
    private

    public :: cli_options_t, parse_arguments

    integer, parameter, public :: CLI_PATH_LEN = 512

    type :: cli_options_t
        character(len=CLI_PATH_LEN) :: input_file = ''
        character(len=CLI_PATH_LEN) :: output_file = ''
        logical :: emit_object = .false.
        character(len=CLI_PATH_LEN), allocatable :: include_paths(:)
        character(len=CLI_PATH_LEN), allocatable :: link_inputs(:)
        logical :: error = .false.
        character(len=:), allocatable :: error_message
        logical :: show_version = .false.
        logical :: show_help = .false.
        integer :: backend = 0
        integer :: opt_level = 0
    end type cli_options_t

contains

    subroutine parse_arguments(argv, opts)
        character(len=*), intent(in) :: argv(:)
        type(cli_options_t), intent(out) :: opts
        integer :: i, n

        n = size(argv)
        allocate (opts%include_paths(0))
        allocate (opts%link_inputs(0))
        if (n < 1) then
            call set_error(opts, 'missing input file')
            return
        end if

        ! The source file may appear at any positional slot; flags and .o link
        ! inputs are recognised position-independently.
        opts%input_file = ''
        i = 1
        do while (i <= n)
            select case (trim(argv(i)))
            case ('--version')
                opts%show_version = .true.
            case ('--help', '-h')
                opts%show_help = .true.
            case ('-c')
                opts%emit_object = .true.
            case ('-o')
                if (i + 1 > n) then
                    call set_error(opts, 'Missing value for -o')
                    return
                end if
                i = i + 1
                opts%output_file = argv(i)
            case ('-I')
                if (i + 1 > n) then
                    call set_error(opts, 'Missing value for -I')
                    return
                end if
                i = i + 1
                call append_include_path(opts%include_paths, trim(argv(i)))
            case ('--backend')
                if (i + 1 > n) then
                    call set_error(opts, 'Missing value for --backend')
                    return
                end if
                i = i + 1
                select case (trim(argv(i)))
                case ('default')
                    opts%backend = 0
                case ('isel')
                    opts%backend = 1
                case ('copy-patch')
                    opts%backend = 2
                case ('llvm')
                    opts%backend = 3
                case default
                    call set_error(opts, 'Unknown --backend value: '// &
                        trim(argv(i))//' (expected default|isel|copy-patch|llvm)')
                    return
                end select
            case ('-O0')
                opts%opt_level = 0
            case ('-O1')
                opts%opt_level = 1
            case ('-O', '-O2')
                opts%opt_level = 2
            case ('-O3')
                opts%opt_level = 3
            case default
                if (ends_with(trim(argv(i)), '.o')) then
                    call append_include_path(opts%link_inputs, trim(argv(i)))
                else if (len_trim(opts%input_file) == 0) then
                    opts%input_file = argv(i)
                else
                    call set_error(opts, 'Unexpected argument: '//trim(argv(i)))
                    return
                end if
            end select
            i = i + 1
        end do

        if (opts%show_version .or. opts%show_help) return

        if (len_trim(opts%input_file) == 0) then
            call set_error(opts, 'missing input file')
            return
        end if
    end subroutine parse_arguments

    logical function ends_with(text, suffix)
        character(len=*), intent(in) :: text
        character(len=*), intent(in) :: suffix

        ends_with = .false.
        if (len(text) < len(suffix)) return
        ends_with = text(len(text) - len(suffix) + 1:) == suffix
    end function ends_with

    subroutine set_error(opts, msg)
        type(cli_options_t), intent(inout) :: opts
        character(len=*), intent(in) :: msg
        opts%error = .true.
        opts%error_message = msg
    end subroutine set_error

    subroutine append_include_path(paths, dir)
        character(len=CLI_PATH_LEN), allocatable, intent(inout) :: paths(:)
        character(len=*), intent(in) :: dir
        character(len=CLI_PATH_LEN), allocatable :: tmp(:)
        integer :: n

        if (.not. allocated(paths)) then
            allocate (paths(0))
        end if
        n = size(paths)
        allocate (tmp(n + 1))
        if (n > 0) tmp(1:n) = paths(1:n)
        tmp(n + 1) = dir
        call move_alloc(tmp, paths)
    end subroutine append_include_path

end module ffc_cli_options
