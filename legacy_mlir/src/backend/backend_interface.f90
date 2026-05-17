module backend_interface
    use ast_core, only: ast_arena_t
    implicit none
    private

    ! Export types and procedures
    public :: backend_t, backend_options_t

    ! Backend options type
    type :: backend_options_t
        logical :: optimize = .false.
        logical :: debug_info = .false.
        logical :: compile_mode = .false.
        logical :: generate_llvm = .false.
        logical :: generate_executable = .false.
        logical :: link_runtime = .false.
        logical :: enable_ad = .false.
        logical :: generate_gradients = .false.
        logical :: ad_annotations = .false.
        logical :: validate_gradients = .false.
        logical :: emit_hlfir = .false.
        logical :: emit_fir = .false.
        logical :: emit_llvm = .false.
        character(len=:), allocatable :: target
        character(len=:), allocatable :: output_file
    end type backend_options_t

    ! Abstract backend interface
    type, abstract :: backend_t
    contains
        procedure(generate_code_interface), deferred :: generate_code
        procedure(get_name_interface), deferred :: get_name
        procedure(get_version_interface), deferred :: get_version
    end type backend_t

    ! Abstract interfaces
    abstract interface
        subroutine generate_code_interface(this, arena, prog_index, options, &
                                           output, error_msg)
            import :: backend_t, ast_arena_t, backend_options_t
            class(backend_t), intent(inout) :: this
            type(ast_arena_t), intent(in) :: arena
            integer, intent(in) :: prog_index
            type(backend_options_t), intent(in) :: options
            character(len=:), allocatable, intent(out) :: output
            character(len=*), intent(out) :: error_msg
        end subroutine generate_code_interface

        function get_name_interface(this) result(name)
            import :: backend_t
            class(backend_t), intent(in) :: this
            character(len=:), allocatable :: name
        end function get_name_interface

        function get_version_interface(this) result(version)
            import :: backend_t
            class(backend_t), intent(in) :: this
            character(len=:), allocatable :: version
        end function get_version_interface
    end interface

end module backend_interface
