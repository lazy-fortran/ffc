module dialect_base
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_operation_builder
    implicit none
    private

    ! Public types
    public :: dialect_builder_base_t
    
    ! Public functions
    public :: create_memory_operation, create_control_flow_operation
    public :: create_array_expr_type, create_fortran_attributes

    ! Base type for dialect builders
    type :: dialect_builder_base_t
        type(mlir_context_t) :: context
        logical :: initialized = .false.
    contains
        procedure :: init_base => builder_init_base
        procedure :: build_load => base_build_load
        procedure :: build_store => base_build_store
        procedure :: build_declare => base_build_declare
    end type dialect_builder_base_t

contains

    ! Initialize base builder
    subroutine builder_init_base(this, context)
        class(dialect_builder_base_t), intent(out) :: this
        type(mlir_context_t), intent(in) :: context
        
        this%context = context
        this%initialized = .true.
    end subroutine builder_init_base

    ! Generic load builder
    function base_build_load(this, op_name, memref, result_type) result(op)
        class(dialect_builder_base_t), intent(in) :: this
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), intent(in) :: memref
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        if (this%initialized) then
            call builder%init(this%context, op_name)
            call builder%operand(memref)
            call builder%result(result_type)
            op = builder%build()
        else
            op%ptr = c_null_ptr
        end if
    end function base_build_load

    ! Generic store builder
    function base_build_store(this, op_name, value, memref) result(op)
        class(dialect_builder_base_t), intent(in) :: this
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), intent(in) :: value, memref
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        
        if (this%initialized) then
            call builder%init(this%context, op_name)
            call builder%operand(value)
            call builder%operand(memref)
            op = builder%build()
        else
            op%ptr = c_null_ptr
        end if
    end function base_build_store

    ! Generic declare builder
    function base_build_declare(this, op_name, memref, name, result_type, extra_attrs) result(op)
        class(dialect_builder_base_t), intent(in) :: this
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), intent(in) :: memref
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: result_type
        type(mlir_attribute_t), intent(in), optional :: extra_attrs
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        type(mlir_attribute_t) :: name_attr
        
        if (this%initialized) then
            call builder%init(this%context, op_name)
            call builder%operand(memref)
            call builder%result(result_type)
            
            name_attr = create_string_attribute(this%context, name)
            call builder%attr("uniq_name", name_attr)
            
            if (present(extra_attrs)) then
                call builder%attr("extra_attrs", extra_attrs)
            end if
            
            op = builder%build()
        else
            op%ptr = c_null_ptr
        end if
    end function base_build_declare

    ! Create memory operation (load/store)
    function create_memory_operation(context, op_name, operands, result_types) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), dimension(:), intent(in) :: operands
        type(mlir_type_t), dimension(:), intent(in), optional :: result_types
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, op_name)
        
        ! Add operands
        do i = 1, size(operands)
            call builder%operand(operands(i))
        end do
        
        ! Add result types if present
        if (present(result_types)) then
            do i = 1, size(result_types)
                call builder%result(result_types(i))
            end do
        end if
        
        op = builder%build()
    end function create_memory_operation

    ! Create control flow operation (if/do_loop)
    function create_control_flow_operation(context, op_name, condition, regions) result(op)
        type(mlir_context_t), intent(in) :: context
        character(len=*), intent(in) :: op_name
        type(mlir_value_t), dimension(:), intent(in) :: condition
        type(mlir_region_t), dimension(:), intent(in), optional :: regions
        type(mlir_operation_t) :: op
        type(operation_builder_t) :: builder
        integer :: i
        
        call builder%init(context, op_name)
        
        ! Add condition/bounds
        do i = 1, size(condition)
            call builder%operand(condition(i))
        end do
        
        ! Note: Regions not yet supported in builder
        
        op = builder%build()
    end function create_control_flow_operation

    ! Create array expression type (shared between FIR and HLFIR)
    function create_array_expr_type(context, element_type, shape_or_rank, is_rank) result(array_type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        integer, dimension(:), intent(in) :: shape_or_rank
        logical, intent(in) :: is_rank
        type(mlir_type_t) :: array_type
        integer(c_int64_t), dimension(:), allocatable :: shape
        integer :: i, rank
        
        if (is_rank) then
            ! shape_or_rank contains [rank]
            rank = shape_or_rank(1)
            if (rank > 0) then
                allocate(shape(rank))
                do i = 1, rank
                    shape(i) = 10_c_int64_t  ! Default dynamic size
                end do
                array_type = create_array_type(context, element_type, shape)
                deallocate(shape)
            else
                array_type = element_type
            end if
        else
            ! shape_or_rank contains actual shape
            allocate(shape(size(shape_or_rank)))
            do i = 1, size(shape_or_rank)
                shape(i) = int(shape_or_rank(i), c_int64_t)
            end do
            array_type = create_array_type(context, element_type, shape)
            deallocate(shape)
        end if
    end function create_array_expr_type

    ! Create Fortran-specific attributes
    function create_fortran_attributes(context, attrs) result(attr)
        type(mlir_context_t), intent(in) :: context
        logical, dimension(:), intent(in) :: attrs  ! [contiguous, target, optional, ...]
        type(mlir_attribute_t) :: attr
        character(len=256) :: attr_str
        character(len=32), dimension(6) :: attr_names
        integer :: i, n_attrs
        
        ! Define attribute names
        attr_names = ["contiguous ", "target     ", "optional   ", "allocatable", "pointer    ", "parameter  "]
        n_attrs = min(size(attrs), 6)
        
        ! Build attribute string
        attr_str = ""
        do i = 1, n_attrs
            if (i > 1) attr_str = trim(attr_str) // ","
            write(attr_str, '(A,A,A,L1)') trim(attr_str), trim(attr_names(i)), "=", attrs(i)
        end do
        
        attr = create_string_attribute(context, trim(attr_str))
    end function create_fortran_attributes

end module dialect_base