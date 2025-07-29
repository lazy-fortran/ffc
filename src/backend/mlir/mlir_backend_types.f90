! MLIR Backend Types Module
! This module defines the core types used by the MLIR backend
module mlir_backend_types
    use backend_interface
    use ast_core, only: ast_arena_t
    implicit none

    public :: mlir_backend_t

    type, extends(backend_t) :: mlir_backend_t
        integer :: ssa_counter = 0  ! SSA value counter
        logical :: use_standard_dialects = .false.  ! Use standard MLIR dialects for LLVM lowering
        logical :: enable_ad = .false.  ! Enable automatic differentiation
        logical :: compile_mode = .false.  ! Are we compiling to executable
        logical :: emit_hlfir = .false.  ! Are we emitting HLFIR
        character(len=:), allocatable :: last_ssa_value  ! Track last generated SSA value
        character(len=:), allocatable :: current_module_name  ! Current module being processed
        character(len=:), allocatable :: global_declarations  ! Global string constants
        character(len=:), allocatable :: error_messages  ! Accumulated error messages
        ! Simple symbol table for variable names to memref SSA values
        character(len=64), allocatable :: symbol_names(:)
        character(len=32), allocatable :: symbol_memrefs(:)
        ! Loop context for tracking loop induction variables
        character(len=64) :: current_loop_var = ""  ! Current loop variable name
        character(len=32) :: current_loop_ssa = ""  ! Current loop SSA value
        ! Format string counter for generating unique format constants
        integer :: format_counter = 0
    contains
        procedure :: generate_code
        procedure :: get_name => mlir_get_name
        procedure :: get_version => mlir_get_version
        procedure :: reset_ssa_counter
        procedure :: next_ssa_value
        procedure :: add_symbol
        procedure :: get_symbol_memref
        procedure :: add_error
    end type mlir_backend_t

contains

    ! Stub implementation - the real one should be in a backend extension
    subroutine generate_code(this, arena, prog_index, options, output, error_msg)
        class(mlir_backend_t), intent(inout) :: this
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: prog_index
        type(backend_options_t), intent(in) :: options
        character(len=:), allocatable, intent(out) :: output
        character(len=*), intent(out) :: error_msg
        
        ! This is a stub - real implementation should be provided elsewhere
        error_msg = "MLIR backend not properly initialized"
        output = ""
    end subroutine generate_code

    function mlir_get_name(this) result(name)
        class(mlir_backend_t), intent(in) :: this
        character(len=:), allocatable :: name
        name = "MLIR"
    end function mlir_get_name

    function mlir_get_version(this) result(version)
        class(mlir_backend_t), intent(in) :: this
        character(len=:), allocatable :: version
        version = "1.0"
    end function mlir_get_version

    function next_ssa_value(this) result(ssa_name)
        class(mlir_backend_t), intent(inout) :: this
        character(len=:), allocatable :: ssa_name

        this%ssa_counter = this%ssa_counter + 1
        ssa_name = "%"//trim(adjustl(int_to_str(this%ssa_counter)))
    end function next_ssa_value

    ! Reset SSA counter
    subroutine reset_ssa_counter(this)
        class(mlir_backend_t), intent(inout) :: this
        this%ssa_counter = 0
        this%last_ssa_value = ""
    end subroutine reset_ssa_counter

    ! Add symbol to symbol table
    subroutine add_symbol(this, name, memref)
        class(mlir_backend_t), intent(inout) :: this
        character(len=*), intent(in) :: name, memref
        integer :: n
        character(len=64), allocatable :: new_names(:)
        character(len=32), allocatable :: new_memrefs(:)

        if (.not. allocated(this%symbol_names)) then
            allocate(this%symbol_names(1))
            allocate(this%symbol_memrefs(1))
            this%symbol_names(1) = name
            this%symbol_memrefs(1) = memref
        else
            n = size(this%symbol_names)
            allocate(new_names(n+1))
            allocate(new_memrefs(n+1))
            new_names(1:n) = this%symbol_names
            new_memrefs(1:n) = this%symbol_memrefs
            new_names(n+1) = name
            new_memrefs(n+1) = memref
            deallocate(this%symbol_names)
            deallocate(this%symbol_memrefs)
            this%symbol_names = new_names
            this%symbol_memrefs = new_memrefs
        end if
    end subroutine add_symbol

    ! Get symbol memref from symbol table
    function get_symbol_memref(this, name) result(memref)
        class(mlir_backend_t), intent(in) :: this
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: memref
        integer :: i

        memref = ""
        if (allocated(this%symbol_names)) then
            do i = 1, size(this%symbol_names)
                if (trim(this%symbol_names(i)) == trim(name)) then
                    memref = trim(this%symbol_memrefs(i))
                    exit
                end if
            end do
        end if
    end function get_symbol_memref

    ! Add error message
    subroutine add_error(this, msg)
        class(mlir_backend_t), intent(inout) :: this
        character(len=*), intent(in) :: msg

        if (.not. allocated(this%error_messages)) then
            this%error_messages = msg
        else
            this%error_messages = this%error_messages // new_line('a') // msg
        end if
    end subroutine add_error


    ! Helper function for integer to string conversion
    function int_to_str(i) result(str)
        integer, intent(in) :: i
        character(len=20) :: str
        write(str, '(I0)') i
    end function int_to_str

end module mlir_backend_types