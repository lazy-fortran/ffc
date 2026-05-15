module liric_bindings
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_int, &
                                           c_null_char, c_null_ptr, c_ptr, &
                                           c_size_t
    implicit none
    private

    integer(c_int), parameter, public :: LR_POLICY_DIRECT = 0_c_int
    integer(c_int), parameter, public :: LR_POLICY_IR = 1_c_int
    integer(c_int), parameter, public :: LR_BACKEND_ISEL = 0_c_int
    integer(c_int), parameter, public :: LR_BACKEND_COPY_PATCH = 1_c_int
    integer(c_int), parameter, public :: LR_BACKEND_LLVM = 2_c_int
    integer(c_int), parameter, public :: LR_COMPILER_OK = 0_c_int

    type, bind(c), public :: lr_compiler_config_t
        integer(c_int) :: policy = LR_POLICY_DIRECT
        integer(c_int) :: backend = LR_BACKEND_ISEL
        type(c_ptr) :: target = c_null_ptr
    end type lr_compiler_config_t

    type, bind(c), public :: lr_compiler_error_t
        integer(c_int) :: code = LR_COMPILER_OK
        character(kind=c_char) :: msg(256)
    end type lr_compiler_error_t

    type, public :: liric_compiler_t
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: destroy => liric_destroy
        procedure :: feed_ll => liric_feed_ll
        procedure :: emit_exe => liric_emit_exe
        procedure :: emit_object => liric_emit_object
        procedure :: is_open => liric_is_open
    end type liric_compiler_t

    public :: liric_create
    public :: liric_compile_ll_to_exe
    public :: liric_error_message

    interface
        function lr_compiler_create(cfg, err) result(handle) bind(c)
            import :: c_ptr, lr_compiler_config_t, lr_compiler_error_t
            type(lr_compiler_config_t), intent(in) :: cfg
            type(lr_compiler_error_t), intent(inout) :: err
            type(c_ptr) :: handle
        end function lr_compiler_create

        subroutine lr_compiler_destroy(handle) bind(c)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine lr_compiler_destroy

        function lr_compiler_feed_ll(handle, source, source_len, err) &
            result(status) bind(c)
            import :: c_char, c_int, c_ptr, c_size_t, lr_compiler_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: source(*)
            integer(c_size_t), value :: source_len
            type(lr_compiler_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_compiler_feed_ll

        function lr_compiler_emit_object(handle, path, err) result(status) &
            bind(c)
            import :: c_char, c_int, c_ptr, lr_compiler_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: path(*)
            type(lr_compiler_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_compiler_emit_object

        function lr_compiler_emit_exe(handle, path, err) result(status) bind(c)
            import :: c_char, c_int, c_ptr, lr_compiler_error_t
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: path(*)
            type(lr_compiler_error_t), intent(inout) :: err
            integer(c_int) :: status
        end function lr_compiler_emit_exe
    end interface

contains

    subroutine liric_create(compiler, error_msg, config)
        type(liric_compiler_t), intent(out) :: compiler
        character(len=:), allocatable, intent(out) :: error_msg
        type(lr_compiler_config_t), intent(in), optional :: config
        type(lr_compiler_config_t) :: local_config
        type(lr_compiler_error_t) :: error

        call clear_liric_error(error)
        local_config = lr_compiler_config_t()
        if (present(config)) local_config = config

        compiler%handle = lr_compiler_create(local_config, error)
        if (c_associated(compiler%handle)) then
            call set_empty(error_msg)
        else
            error_msg = liric_error_message(error)
        end if
    end subroutine liric_create

    subroutine liric_destroy(this)
        class(liric_compiler_t), intent(inout) :: this

        if (c_associated(this%handle)) then
            call lr_compiler_destroy(this%handle)
            this%handle = c_null_ptr
        end if
    end subroutine liric_destroy

    logical function liric_feed_ll(this, source, error_msg)
        class(liric_compiler_t), intent(inout) :: this
        character(len=*), intent(in) :: source
        character(len=:), allocatable, intent(out) :: error_msg
        character(kind=c_char), allocatable :: c_source(:)
        type(lr_compiler_error_t) :: error
        integer(c_int) :: status

        if (.not. c_associated(this%handle)) then
            error_msg = 'LIRIC compiler handle is not open'
            liric_feed_ll = .false.
            return
        end if

        call clear_liric_error(error)
        call to_c_chars(source, c_source)
        status = lr_compiler_feed_ll(this%handle, c_source, &
                                     int(len(source), c_size_t), error)
        liric_feed_ll = status == LR_COMPILER_OK
        if (liric_feed_ll) then
            call set_empty(error_msg)
        else
            error_msg = liric_error_message(error)
        end if
    end function liric_feed_ll

    logical function liric_emit_exe(this, path, error_msg)
        class(liric_compiler_t), intent(inout) :: this
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg

        liric_emit_exe = emit_to_path(this, path, error_msg, .true.)
    end function liric_emit_exe

    logical function liric_emit_object(this, path, error_msg)
        class(liric_compiler_t), intent(inout) :: this
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg

        liric_emit_object = emit_to_path(this, path, error_msg, .false.)
    end function liric_emit_object

    logical function liric_is_open(this)
        class(liric_compiler_t), intent(in) :: this

        liric_is_open = c_associated(this%handle)
    end function liric_is_open

    logical function liric_compile_ll_to_exe(source, path, error_msg)
        character(len=*), intent(in) :: source
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        type(liric_compiler_t) :: compiler

        call liric_create(compiler, error_msg)
        if (len_trim(error_msg) > 0) then
            liric_compile_ll_to_exe = .false.
            return
        end if

        liric_compile_ll_to_exe = compiler%feed_ll(source, error_msg)
        if (liric_compile_ll_to_exe) then
            liric_compile_ll_to_exe = compiler%emit_exe(path, error_msg)
        end if
        call compiler%destroy()
    end function liric_compile_ll_to_exe

    logical function emit_to_path(this, path, error_msg, executable)
        class(liric_compiler_t), intent(inout) :: this
        character(len=*), intent(in) :: path
        character(len=:), allocatable, intent(out) :: error_msg
        logical, intent(in) :: executable
        character(kind=c_char), allocatable :: c_path(:)
        type(lr_compiler_error_t) :: error
        integer(c_int) :: status

        if (.not. c_associated(this%handle)) then
            error_msg = 'LIRIC compiler handle is not open'
            emit_to_path = .false.
            return
        end if

        call clear_liric_error(error)
        call to_c_chars(path, c_path)
        if (executable) then
            status = lr_compiler_emit_exe(this%handle, c_path, error)
        else
            status = lr_compiler_emit_object(this%handle, c_path, error)
        end if

        emit_to_path = status == LR_COMPILER_OK
        if (emit_to_path) then
            call set_empty(error_msg)
        else
            error_msg = liric_error_message(error)
        end if
    end function emit_to_path

    function liric_error_message(error) result(message)
        type(lr_compiler_error_t), intent(in) :: error
        character(len=:), allocatable :: message
        integer :: i
        integer :: message_len

        message_len = 0
        do i = 1, size(error%msg)
            if (error%msg(i) == c_null_char) exit
            message_len = message_len + 1
        end do

        if (message_len == 0) then
            allocate (character(len=32) :: message)
            write (message, '(A,I0)') 'LIRIC error code ', error%code
            return
        end if

        allocate (character(len=message_len) :: message)
        do i = 1, message_len
            message(i:i) = error%msg(i)
        end do
    end function liric_error_message

    subroutine clear_liric_error(error)
        type(lr_compiler_error_t), intent(out) :: error

        error%code = LR_COMPILER_OK
        error%msg = c_null_char
    end subroutine clear_liric_error

    subroutine to_c_chars(text, chars)
        character(len=*), intent(in) :: text
        character(kind=c_char), allocatable, intent(out) :: chars(:)
        integer :: i

        allocate (chars(len(text) + 1))
        do i = 1, len(text)
            chars(i) = text(i:i)
        end do
        chars(len(text) + 1) = c_null_char
    end subroutine to_c_chars

    subroutine set_empty(value)
        character(len=:), allocatable, intent(out) :: value

        allocate (character(len=0) :: value)
    end subroutine set_empty

end module liric_bindings
