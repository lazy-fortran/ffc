module backend_factory
    use backend_interface
    use backend_constants
    use fortran_backend
    use mlir_backend_impl
    use logger, only: log_debug
    implicit none
    private

    ! Public interface for backend creation
    public :: create_backend, select_backend_type
    public :: register_backend, is_backend_registered

    ! Backend registration tracking
    type :: backend_registration_t
        character(len=32) :: name = ""
        logical :: is_available = .false.
    end type backend_registration_t

    ! Registry of available backends
    type(backend_registration_t), dimension(3) :: backend_registry = [ &
                          backend_registration_t(name="fortran", is_available=.true.), &
                             backend_registration_t(name="mlir", is_available=.true.), &
                                backend_registration_t(name="c", is_available=.false.) &
                                                  ]

contains

    ! Create a backend instance based on type name
    subroutine create_backend(backend_type, backend_instance, error_msg)
        character(len=*), intent(in) :: backend_type
        class(backend_t), allocatable, intent(out) :: backend_instance
        character(len=*), intent(out) :: error_msg

        error_msg = ""

        select case (trim(adjustl(backend_type)))
        case ("fortran")
            allocate (fortran_backend_t :: backend_instance)

        case ("mlir")
            allocate (mlir_backend_impl_t :: backend_instance)
            call log_debug("Created MLIR backend")

        case ("c")
            error_msg = "C backend not yet implemented"

        case default
     write (error_msg, '(A,A,A)') "Unsupported backend type: '", trim(backend_type), "'"
        end select
    end subroutine create_backend

    ! Select backend type based on compilation options
    function select_backend_type(options) result(backend_type)
        type(backend_options_t), intent(in) :: options
        character(len=32) :: backend_type

        if (options%compile_mode) then
            ! --compile mode uses MLIR backend
            backend_type = "mlir"
        else
            ! Default to Fortran backend for all existing paths
            backend_type = "fortran"
        end if
    end function select_backend_type

    ! Register a new backend as available
    subroutine register_backend(backend_name, is_available)
        character(len=*), intent(in) :: backend_name
        logical, intent(in) :: is_available
        integer :: i

        do i = 1, size(backend_registry)
            if (backend_registry(i)%name == backend_name) then
                backend_registry(i)%is_available = is_available
                return
            end if
        end do
    end subroutine register_backend

    ! Check if a backend is registered and available
    function is_backend_registered(backend_name) result(is_registered)
        character(len=*), intent(in) :: backend_name
        logical :: is_registered
        integer :: i

        is_registered = .false.
        do i = 1, size(backend_registry)
            if (backend_registry(i)%name == backend_name .and. &
                backend_registry(i)%is_available) then
                is_registered = .true.
                return
            end if
        end do
    end function is_backend_registered

end module backend_factory
