module hlfir_replacements
    ! Module to provide HLFIR replacements for standard MLIR operations
    use mlir_utils
    implicit none
    private
    
    public :: get_function_declaration
    public :: get_constant_operation
    public :: get_alloca_operation
    public :: get_store_operation
    public :: get_load_operation
    public :: get_return_operation
    public :: get_ptr_type
    public :: get_if_operation
    public :: get_branch_operation
    
contains
    
    ! Get function declaration - always use FIR
    function get_function_declaration(emit_hlfir, func_name, args, ret_type) result(decl)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: func_name
        character(len=*), intent(in) :: args
        character(len=*), intent(in) :: ret_type
        character(len=:), allocatable :: decl
        
        ! Always use FIR function declarations
        if (len_trim(ret_type) > 0) then
            decl = "fir.func @"//trim(func_name)//"("//trim(args)//") -> "//trim(ret_type)//" {"
        else
            decl = "fir.func @"//trim(func_name)//"("//trim(args)//") {"
        end if
    end function get_function_declaration
    
    ! Get constant operation - always use FIR
    function get_constant_operation(emit_hlfir, ssa_val, value, type_str) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: ssa_val
        character(len=*), intent(in) :: value
        character(len=*), intent(in) :: type_str
        character(len=:), allocatable :: op
        
        ! Always use FIR constants
        op = trim(ssa_val)//" = fir.constant "//trim(value)//" : "//trim(type_str)
    end function get_constant_operation
    
    ! Get alloca operation - always use FIR
    function get_alloca_operation(emit_hlfir, ssa_val, type_str) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: ssa_val
        character(len=*), intent(in) :: type_str
        character(len=:), allocatable :: op
        
        ! Always use FIR allocation
        op = trim(ssa_val)//" = fir.alloca "//trim(type_str)
    end function get_alloca_operation
    
    ! Get store operation - always use FIR
    function get_store_operation(emit_hlfir, value_ssa, target_ssa, type_str) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: value_ssa
        character(len=*), intent(in) :: target_ssa
        character(len=*), intent(in) :: type_str
        character(len=:), allocatable :: op
        
        ! Always use FIR store
        op = "fir.store "//trim(value_ssa)//" to "//trim(target_ssa)//" : !fir.ref<"//trim(type_str)//">"
    end function get_store_operation
    
    ! Get load operation - always use FIR
    function get_load_operation(emit_hlfir, ssa_val, source_ssa, type_str) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: ssa_val
        character(len=*), intent(in) :: source_ssa
        character(len=*), intent(in) :: type_str
        character(len=:), allocatable :: op
        
        ! Always use FIR load
        op = trim(ssa_val)//" = fir.load "//trim(source_ssa)//" : !fir.ref<"//trim(type_str)//">"
    end function get_load_operation
    
    ! Get return operation - always use FIR
    function get_return_operation(emit_hlfir, value) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: value
        character(len=:), allocatable :: op
        
        ! Always use FIR return
        if (len_trim(value) > 0) then
            op = "fir.return "//trim(value)
        else
            op = "fir.return"
        end if
    end function get_return_operation
    
    ! Get pointer type - always use FIR
    function get_ptr_type(emit_hlfir, base_type) result(ptr_type)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: base_type
        character(len=:), allocatable :: ptr_type
        
        ! Always use FIR pointer types
        ptr_type = "!fir.ptr<"//trim(base_type)//">"
    end function get_ptr_type
    
    ! Get if operation - always use FIR
    function get_if_operation(emit_hlfir, condition_ssa) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: condition_ssa
        character(len=:), allocatable :: op
        
        ! Always use FIR if operations
        op = "fir.if "//trim(condition_ssa)//" {"
    end function get_if_operation
    
    ! Get branch operation - always use FIR
    function get_branch_operation(emit_hlfir, target_label) result(op)
        logical, intent(in) :: emit_hlfir
        character(len=*), intent(in) :: target_label
        character(len=:), allocatable :: op
        
        ! Always use FIR branch operations
        op = "fir.br ^"//trim(target_label)
    end function get_branch_operation
    
end module hlfir_replacements