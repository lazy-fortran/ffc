module mlir_c_core
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_bool, c_int, &
        c_intptr_t, c_char, c_size_t, c_associated, c_f_pointer, c_loc
    use, intrinsic :: iso_fortran_env, only: int8
    implicit none
    private

    public :: mlir_string_ref_t
    public :: mlir_logical_result_t
    public :: mlir_context_t
    public :: mlir_dialect_t
    public :: mlir_dialect_registry_t
    public :: mlir_location_t
    public :: mlir_module_t
    public :: mlir_operation_t
    public :: mlir_block_t
    public :: mlir_region_t
    public :: mlir_value_t
    public :: mlir_type_t
    public :: mlir_attribute_t
    public :: mlir_identifier_t

    public :: mlir_string_ref_create
    public :: mlir_context_create
    public :: mlir_context_destroy
    public :: mlir_context_is_null
    public :: mlir_context_set_allow_unregistered_dialects
    public :: mlir_context_get_num_loaded_dialects
    public :: mlir_context_get_or_load_dialect
    public :: mlir_context_load_all_available_dialects

    public :: mlir_dialect_registry_create
    public :: mlir_dialect_registry_destroy
    public :: mlir_context_append_dialect_registry

    public :: mlir_location_unknown_get
    public :: mlir_location_file_line_col_get
    public :: mlir_location_is_null

    public :: mlir_module_create_empty
    public :: mlir_module_destroy
    public :: mlir_module_is_null
    public :: mlir_module_get_body
    public :: mlir_module_get_context
    public :: mlir_module_get_operation

    public :: mlir_operation_verify
    public :: mlir_operation_dump
    public :: mlir_operation_is_null

    public :: mlir_block_is_null
    public :: mlir_block_get_first_operation
    public :: mlir_block_append_owned_operation

    public :: mlir_region_create
    public :: mlir_region_destroy
    public :: mlir_region_is_null
    public :: mlir_region_append_owned_block

    public :: mlir_block_create

    public :: mlir_identifier_get

    type :: mlir_string_ref_t
        type(c_ptr) :: data = c_null_ptr
        integer(c_size_t) :: length = 0
    end type mlir_string_ref_t

    type :: mlir_logical_result_t
        integer(int8) :: value = 0
    end type mlir_logical_result_t

    type :: mlir_context_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_context_t

    type :: mlir_dialect_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_dialect_t

    type :: mlir_dialect_registry_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_dialect_registry_t

    type :: mlir_location_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_location_t

    type :: mlir_module_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_module_t

    type :: mlir_operation_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_operation_t

    type :: mlir_block_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_block_t

    type :: mlir_region_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_region_t

    type :: mlir_value_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_value_t

    type :: mlir_type_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_type_t

    type :: mlir_attribute_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_attribute_t

    type :: mlir_identifier_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_identifier_t

    interface
        function mlirContextCreate() bind(C, name="mlirContextCreate")
            import :: c_ptr
            type(c_ptr) :: mlirContextCreate
        end function mlirContextCreate

        subroutine mlirContextDestroy(context) bind(C, name="mlirContextDestroy")
            import :: c_ptr
            type(c_ptr), value :: context
        end subroutine mlirContextDestroy

        subroutine mlirContextSetAllowUnregisteredDialects(context, allow) &
                bind(C, name="mlirContextSetAllowUnregisteredDialects")
            import :: c_ptr, c_bool
            type(c_ptr), value :: context
            logical(c_bool), value :: allow
        end subroutine mlirContextSetAllowUnregisteredDialects

        function mlirContextGetNumLoadedDialects(context) &
                bind(C, name="mlirContextGetNumLoadedDialects")
            import :: c_ptr, c_intptr_t
            type(c_ptr), value :: context
            integer(c_intptr_t) :: mlirContextGetNumLoadedDialects
        end function mlirContextGetNumLoadedDialects

        function mlirContextGetOrLoadDialect_c(context, name_data, name_len) &
                bind(C, name="mlirContextGetOrLoadDialect")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: context
            type(c_ptr), value :: name_data
            integer(c_size_t), value :: name_len
            type(c_ptr) :: mlirContextGetOrLoadDialect_c
        end function mlirContextGetOrLoadDialect_c

        subroutine mlirContextLoadAllAvailableDialects(context) &
                bind(C, name="mlirContextLoadAllAvailableDialects")
            import :: c_ptr
            type(c_ptr), value :: context
        end subroutine mlirContextLoadAllAvailableDialects

        function mlirDialectRegistryCreate() &
                bind(C, name="mlirDialectRegistryCreate")
            import :: c_ptr
            type(c_ptr) :: mlirDialectRegistryCreate
        end function mlirDialectRegistryCreate

        subroutine mlirDialectRegistryDestroy(registry) &
                bind(C, name="mlirDialectRegistryDestroy")
            import :: c_ptr
            type(c_ptr), value :: registry
        end subroutine mlirDialectRegistryDestroy

        subroutine mlirContextAppendDialectRegistry_c(ctx, registry) &
                bind(C, name="mlirContextAppendDialectRegistry")
            import :: c_ptr
            type(c_ptr), value :: ctx
            type(c_ptr), value :: registry
        end subroutine mlirContextAppendDialectRegistry_c

        function mlirLocationUnknownGet(context) &
                bind(C, name="mlirLocationUnknownGet")
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: mlirLocationUnknownGet
        end function mlirLocationUnknownGet

        function mlirLocationFileLineColGet_c(context, filename_data, &
                filename_len, line, col) &
                bind(C, name="mlirLocationFileLineColGet")
            import :: c_ptr, c_size_t, c_int
            type(c_ptr), value :: context
            type(c_ptr), value :: filename_data
            integer(c_size_t), value :: filename_len
            integer(c_int), value :: line
            integer(c_int), value :: col
            type(c_ptr) :: mlirLocationFileLineColGet_c
        end function mlirLocationFileLineColGet_c

        function mlirModuleCreateEmpty(location) &
                bind(C, name="mlirModuleCreateEmpty")
            import :: c_ptr
            type(c_ptr), value :: location
            type(c_ptr) :: mlirModuleCreateEmpty
        end function mlirModuleCreateEmpty

        subroutine mlirModuleDestroy(module) bind(C, name="mlirModuleDestroy")
            import :: c_ptr
            type(c_ptr), value :: module
        end subroutine mlirModuleDestroy

        function mlirModuleGetBody(module) bind(C, name="mlirModuleGetBody")
            import :: c_ptr
            type(c_ptr), value :: module
            type(c_ptr) :: mlirModuleGetBody
        end function mlirModuleGetBody

        function mlirModuleGetContext(module) &
                bind(C, name="mlirModuleGetContext")
            import :: c_ptr
            type(c_ptr), value :: module
            type(c_ptr) :: mlirModuleGetContext
        end function mlirModuleGetContext

        function mlirModuleGetOperation(module) &
                bind(C, name="mlirModuleGetOperation")
            import :: c_ptr
            type(c_ptr), value :: module
            type(c_ptr) :: mlirModuleGetOperation
        end function mlirModuleGetOperation

        function mlirOperationVerify(op) bind(C, name="mlirOperationVerify")
            import :: c_ptr, c_bool
            type(c_ptr), value :: op
            logical(c_bool) :: mlirOperationVerify
        end function mlirOperationVerify

        subroutine mlirOperationDump(op) bind(C, name="mlirOperationDump")
            import :: c_ptr
            type(c_ptr), value :: op
        end subroutine mlirOperationDump

        function mlirBlockGetFirstOperation(block) &
                bind(C, name="mlirBlockGetFirstOperation")
            import :: c_ptr
            type(c_ptr), value :: block
            type(c_ptr) :: mlirBlockGetFirstOperation
        end function mlirBlockGetFirstOperation

        subroutine mlirBlockAppendOwnedOperation(block, operation) &
                bind(C, name="mlirBlockAppendOwnedOperation")
            import :: c_ptr
            type(c_ptr), value :: block
            type(c_ptr), value :: operation
        end subroutine mlirBlockAppendOwnedOperation

        function mlirRegionCreate() bind(C, name="mlirRegionCreate")
            import :: c_ptr
            type(c_ptr) :: mlirRegionCreate
        end function mlirRegionCreate

        subroutine mlirRegionDestroy(region) bind(C, name="mlirRegionDestroy")
            import :: c_ptr
            type(c_ptr), value :: region
        end subroutine mlirRegionDestroy

        subroutine mlirRegionAppendOwnedBlock(region, block) &
                bind(C, name="mlirRegionAppendOwnedBlock")
            import :: c_ptr
            type(c_ptr), value :: region
            type(c_ptr), value :: block
        end subroutine mlirRegionAppendOwnedBlock

        function mlirBlockCreate_c(nArgs, args, locs) &
                bind(C, name="mlirBlockCreate")
            import :: c_ptr, c_intptr_t
            integer(c_intptr_t), value :: nArgs
            type(c_ptr), value :: args
            type(c_ptr), value :: locs
            type(c_ptr) :: mlirBlockCreate_c
        end function mlirBlockCreate_c

        function mlirIdentifierGet_c(context, str_data, str_len) &
                bind(C, name="mlirIdentifierGet")
            import :: c_ptr, c_size_t
            type(c_ptr), value :: context
            type(c_ptr), value :: str_data
            integer(c_size_t), value :: str_len
            type(c_ptr) :: mlirIdentifierGet_c
        end function mlirIdentifierGet_c
    end interface

contains

    function mlir_string_ref_create(str) result(ref)
        character(len=*), intent(in), target :: str
        type(mlir_string_ref_t) :: ref

        if (len(str) > 0) then
            ref%data = c_loc(str)
            ref%length = int(len(str), c_size_t)
        else
            ref%data = c_null_ptr
            ref%length = 0
        end if
    end function mlir_string_ref_create

    function mlir_context_create() result(ctx)
        type(mlir_context_t) :: ctx
        ctx%ptr = mlirContextCreate()
    end function mlir_context_create

    subroutine mlir_context_destroy(ctx)
        type(mlir_context_t), intent(inout) :: ctx
        if (c_associated(ctx%ptr)) then
            call mlirContextDestroy(ctx%ptr)
            ctx%ptr = c_null_ptr
        end if
    end subroutine mlir_context_destroy

    pure function mlir_context_is_null(ctx) result(is_null)
        type(mlir_context_t), intent(in) :: ctx
        logical :: is_null
        is_null = .not. c_associated(ctx%ptr)
    end function mlir_context_is_null

    subroutine mlir_context_set_allow_unregistered_dialects(ctx, allow)
        type(mlir_context_t), intent(in) :: ctx
        logical, intent(in) :: allow
        call mlirContextSetAllowUnregisteredDialects(ctx%ptr, &
            logical(allow, c_bool))
    end subroutine mlir_context_set_allow_unregistered_dialects

    function mlir_context_get_num_loaded_dialects(ctx) result(num)
        type(mlir_context_t), intent(in) :: ctx
        integer :: num
        num = int(mlirContextGetNumLoadedDialects(ctx%ptr))
    end function mlir_context_get_num_loaded_dialects

    function mlir_context_get_or_load_dialect(ctx, name) result(dialect)
        type(mlir_context_t), intent(in) :: ctx
        character(len=*), intent(in), target :: name
        type(mlir_dialect_t) :: dialect

        dialect%ptr = mlirContextGetOrLoadDialect_c(ctx%ptr, c_loc(name), &
            int(len(name), c_size_t))
    end function mlir_context_get_or_load_dialect

    subroutine mlir_context_load_all_available_dialects(ctx)
        type(mlir_context_t), intent(in) :: ctx
        call mlirContextLoadAllAvailableDialects(ctx%ptr)
    end subroutine mlir_context_load_all_available_dialects

    function mlir_dialect_registry_create() result(registry)
        type(mlir_dialect_registry_t) :: registry
        registry%ptr = mlirDialectRegistryCreate()
    end function mlir_dialect_registry_create

    subroutine mlir_dialect_registry_destroy(registry)
        type(mlir_dialect_registry_t), intent(inout) :: registry
        if (c_associated(registry%ptr)) then
            call mlirDialectRegistryDestroy(registry%ptr)
            registry%ptr = c_null_ptr
        end if
    end subroutine mlir_dialect_registry_destroy

    subroutine mlir_context_append_dialect_registry(ctx, registry)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_dialect_registry_t), intent(in) :: registry
        call mlirContextAppendDialectRegistry_c(ctx%ptr, registry%ptr)
    end subroutine mlir_context_append_dialect_registry

    function mlir_location_unknown_get(ctx) result(loc)
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_location_t) :: loc
        loc%ptr = mlirLocationUnknownGet(ctx%ptr)
    end function mlir_location_unknown_get

    function mlir_location_file_line_col_get(ctx, filename, line, col) &
            result(loc)
        type(mlir_context_t), intent(in) :: ctx
        character(len=*), intent(in), target :: filename
        integer, intent(in) :: line
        integer, intent(in) :: col
        type(mlir_location_t) :: loc

        loc%ptr = mlirLocationFileLineColGet_c(ctx%ptr, c_loc(filename), &
            int(len(filename), c_size_t), int(line, c_int), int(col, c_int))
    end function mlir_location_file_line_col_get

    pure function mlir_location_is_null(loc) result(is_null)
        type(mlir_location_t), intent(in) :: loc
        logical :: is_null
        is_null = .not. c_associated(loc%ptr)
    end function mlir_location_is_null

    function mlir_module_create_empty(loc) result(mod)
        type(mlir_location_t), intent(in) :: loc
        type(mlir_module_t) :: mod
        mod%ptr = mlirModuleCreateEmpty(loc%ptr)
    end function mlir_module_create_empty

    subroutine mlir_module_destroy(mod)
        type(mlir_module_t), intent(inout) :: mod
        if (c_associated(mod%ptr)) then
            call mlirModuleDestroy(mod%ptr)
            mod%ptr = c_null_ptr
        end if
    end subroutine mlir_module_destroy

    pure function mlir_module_is_null(mod) result(is_null)
        type(mlir_module_t), intent(in) :: mod
        logical :: is_null
        is_null = .not. c_associated(mod%ptr)
    end function mlir_module_is_null

    function mlir_module_get_body(mod) result(block)
        type(mlir_module_t), intent(in) :: mod
        type(mlir_block_t) :: block
        block%ptr = mlirModuleGetBody(mod%ptr)
    end function mlir_module_get_body

    function mlir_module_get_context(mod) result(ctx)
        type(mlir_module_t), intent(in) :: mod
        type(mlir_context_t) :: ctx
        ctx%ptr = mlirModuleGetContext(mod%ptr)
    end function mlir_module_get_context

    function mlir_module_get_operation(mod) result(op)
        type(mlir_module_t), intent(in) :: mod
        type(mlir_operation_t) :: op
        op%ptr = mlirModuleGetOperation(mod%ptr)
    end function mlir_module_get_operation

    function mlir_operation_verify(op) result(verified)
        type(mlir_operation_t), intent(in) :: op
        logical :: verified
        verified = mlirOperationVerify(op%ptr)
    end function mlir_operation_verify

    subroutine mlir_operation_dump(op)
        type(mlir_operation_t), intent(in) :: op
        call mlirOperationDump(op%ptr)
    end subroutine mlir_operation_dump

    pure function mlir_operation_is_null(op) result(is_null)
        type(mlir_operation_t), intent(in) :: op
        logical :: is_null
        is_null = .not. c_associated(op%ptr)
    end function mlir_operation_is_null

    pure function mlir_block_is_null(block) result(is_null)
        type(mlir_block_t), intent(in) :: block
        logical :: is_null
        is_null = .not. c_associated(block%ptr)
    end function mlir_block_is_null

    function mlir_block_get_first_operation(block) result(op)
        type(mlir_block_t), intent(in) :: block
        type(mlir_operation_t) :: op
        op%ptr = mlirBlockGetFirstOperation(block%ptr)
    end function mlir_block_get_first_operation

    subroutine mlir_block_append_owned_operation(block, op)
        type(mlir_block_t), intent(in) :: block
        type(mlir_operation_t), intent(in) :: op
        call mlirBlockAppendOwnedOperation(block%ptr, op%ptr)
    end subroutine mlir_block_append_owned_operation

    function mlir_region_create() result(region)
        type(mlir_region_t) :: region
        region%ptr = mlirRegionCreate()
    end function mlir_region_create

    subroutine mlir_region_destroy(region)
        type(mlir_region_t), intent(inout) :: region
        if (c_associated(region%ptr)) then
            call mlirRegionDestroy(region%ptr)
            region%ptr = c_null_ptr
        end if
    end subroutine mlir_region_destroy

    pure function mlir_region_is_null(region) result(is_null)
        type(mlir_region_t), intent(in) :: region
        logical :: is_null
        is_null = .not. c_associated(region%ptr)
    end function mlir_region_is_null

    subroutine mlir_region_append_owned_block(region, block)
        type(mlir_region_t), intent(in) :: region
        type(mlir_block_t), intent(in) :: block
        call mlirRegionAppendOwnedBlock(region%ptr, block%ptr)
    end subroutine mlir_region_append_owned_block

    function mlir_block_create(types, locs) result(block)
        type(mlir_type_t), intent(in), optional, target :: types(:)
        type(mlir_location_t), intent(in), optional, target :: locs(:)
        type(mlir_block_t) :: block
        integer(c_intptr_t) :: n_args

        if (present(types)) then
            n_args = int(size(types), c_intptr_t)
            block%ptr = mlirBlockCreate_c(n_args, c_loc(types(1)), &
                c_loc(locs(1)))
        else
            n_args = 0
            block%ptr = mlirBlockCreate_c(n_args, c_null_ptr, c_null_ptr)
        end if
    end function mlir_block_create

    function mlir_identifier_get(ctx, str) result(ident)
        type(mlir_context_t), intent(in) :: ctx
        character(len=*), intent(in), target :: str
        type(mlir_identifier_t) :: ident

        ident%ptr = mlirIdentifierGet_c(ctx%ptr, c_loc(str), &
            int(len(str), c_size_t))
    end function mlir_identifier_get

end module mlir_c_core
