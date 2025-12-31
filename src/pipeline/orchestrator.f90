module pipeline_orchestrator
    use backend_constants
    use backend_interface
    use backend_factory
    use fortfront
    implicit none
    private

    ! Export public procedures
    public :: select_backend_from_options
    public :: is_backend_supported
    public :: create_backend_from_options
    public :: run_backend_pipeline

    ! Pipeline options type
    type, public :: pipeline_options_t
        logical :: compile_mode = .false.
        logical :: json_output = .false.
        logical :: run_mode = .false.
        integer :: backend = BACKEND_FORTRAN
    end type pipeline_options_t

contains

    function select_backend_from_options(options) result(backend)
        type(pipeline_options_t), intent(in) :: options
        integer :: backend

        if (options%compile_mode) then
            backend = BACKEND_MLIR
        else
            backend = BACKEND_FORTRAN
        end if
    end function select_backend_from_options

    function is_backend_supported(backend) result(supported)
        integer, intent(in) :: backend
        logical :: supported

        select case (backend)
        case (BACKEND_FORTRAN, BACKEND_MLIR, BACKEND_LLVM, BACKEND_C)
            supported = .true.
        case default
            supported = .false.
        end select
    end function is_backend_supported

    ! Create backend instance from pipeline options
    subroutine create_backend_from_options(options, backend_instance, error_msg)
        type(pipeline_options_t), intent(in) :: options
        class(backend_t), allocatable, intent(out) :: backend_instance
        character(len=*), intent(out) :: error_msg

        type(backend_options_t) :: backend_options
        character(len=32) :: backend_type

        ! Set backend options from pipeline options
        backend_options%compile_mode = options%compile_mode
        backend_options%link_runtime = options%compile_mode

        ! Select backend type
        backend_type = select_backend_type(backend_options)

        ! Create backend instance
        call create_backend(backend_type, backend_instance, error_msg)
    end subroutine create_backend_from_options

    ! Run backend pipeline on AST
    function run_backend_pipeline(options, arena, prog_index, error_msg) result(output)
        type(pipeline_options_t), intent(in) :: options
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        character(len=*), intent(out) :: error_msg
        character(len=:), allocatable :: output

        class(backend_t), allocatable :: backend
        type(backend_options_t) :: backend_options

        ! Create backend
        call create_backend_from_options(options, backend, error_msg)
        if (.not. allocated(backend)) then
            output = ""
            return
        end if

        ! Set backend options
        backend_options%compile_mode = options%compile_mode
        backend_options%link_runtime = options%compile_mode

        ! Generate code
       call backend%generate_code(arena, prog_index, backend_options, output, error_msg)
    end function run_backend_pipeline

end module pipeline_orchestrator
