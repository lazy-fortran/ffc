module dialect_registry
    use iso_c_binding
    use mlir_c_core
    use fir_dialect
    use hlfir_dialect
    use standard_dialects
    implicit none
    private

    ! Public types
    public :: dialect_set_t
    
    ! Public functions
    public :: create_dialect_set, register_dialects
    public :: register_all_fortran_dialects, register_all_standard_dialects

    ! Dialect set flags
    type :: dialect_set_t
        logical :: fir = .false.
        logical :: hlfir = .false.
        logical :: func = .false.
        logical :: arith = .false.
        logical :: scf = .false.
        logical :: builtin = .false.
        logical :: llvm = .false.
    contains
        procedure :: add => dialect_set_add
        procedure :: clear => dialect_set_clear
    end type dialect_set_t

contains

    ! Create dialect set with common defaults
    function create_dialect_set(preset) result(dialects)
        character(len=*), intent(in), optional :: preset
        type(dialect_set_t) :: dialects
        
        if (present(preset)) then
            select case (preset)
            case ("fortran")
                ! Fortran compilation dialects
                dialects%fir = .true.
                dialects%hlfir = .true.
                dialects%arith = .true.
                dialects%scf = .true.
                dialects%func = .true.
                dialects%builtin = .true.
                
            case ("optimization")
                ! Optimization dialects
                dialects%fir = .true.
                dialects%arith = .true.
                dialects%scf = .true.
                dialects%func = .true.
                
            case ("lowering")
                ! Lowering to LLVM dialects
                dialects%fir = .true.
                dialects%arith = .true.
                dialects%func = .true.
                dialects%llvm = .true.
                
            case ("minimal")
                ! Minimal set for testing
                dialects%func = .true.
                dialects%arith = .true.
                
            case default
                ! Empty set
            end select
        end if
    end function create_dialect_set

    ! Add dialect to set
    subroutine dialect_set_add(this, dialect_name)
        class(dialect_set_t), intent(inout) :: this
        character(len=*), intent(in) :: dialect_name
        
        select case (dialect_name)
        case ("fir")
            this%fir = .true.
        case ("hlfir")
            this%hlfir = .true.
        case ("func")
            this%func = .true.
        case ("arith")
            this%arith = .true.
        case ("scf")
            this%scf = .true.
        case ("builtin")
            this%builtin = .true.
        case ("llvm")
            this%llvm = .true.
        end select
    end subroutine dialect_set_add

    ! Clear all dialects
    subroutine dialect_set_clear(this)
        class(dialect_set_t), intent(inout) :: this
        
        this%fir = .false.
        this%hlfir = .false.
        this%func = .false.
        this%arith = .false.
        this%scf = .false.
        this%builtin = .false.
        this%llvm = .false.
    end subroutine dialect_set_clear

    ! Register dialects based on set
    subroutine register_dialects(context, dialects)
        type(mlir_context_t), intent(in) :: context
        type(dialect_set_t), intent(in) :: dialects
        
        if (dialects%fir) call register_fir_dialect(context)
        if (dialects%hlfir) call register_hlfir_dialect(context)
        if (dialects%func) call register_func_dialect(context)
        if (dialects%arith) call register_arith_dialect(context)
        if (dialects%scf) call register_scf_dialect(context)
        ! Note: builtin and llvm registration would be added when implemented
    end subroutine register_dialects

    ! Register all Fortran-related dialects
    subroutine register_all_fortran_dialects(context)
        type(mlir_context_t), intent(in) :: context
        type(dialect_set_t) :: dialects
        
        dialects = create_dialect_set("fortran")
        call register_dialects(context, dialects)
    end subroutine register_all_fortran_dialects

    ! Register all standard dialects
    subroutine register_all_standard_dialects(context)
        type(mlir_context_t), intent(in) :: context
        type(dialect_set_t) :: dialects
        
        dialects = create_dialect_set("minimal")
        call register_dialects(context, dialects)
    end subroutine register_all_standard_dialects

end module dialect_registry