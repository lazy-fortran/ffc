module mlir_c_dialects
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_associated
    use mlir_c_core, only: mlir_context_t, mlir_dialect_t, &
        mlir_dialect_registry_t
    implicit none
    private

    public :: mlir_dialect_handle_t

    public :: mlir_register_all_dialects
    public :: mlir_register_all_llvm_translations

    public :: mlir_get_dialect_handle_func
    public :: mlir_get_dialect_handle_arith
    public :: mlir_get_dialect_handle_scf
    public :: mlir_get_dialect_handle_llvm
    public :: mlir_get_dialect_handle_cf
    public :: mlir_get_dialect_handle_math
    public :: mlir_get_dialect_handle_memref
    public :: mlir_get_dialect_handle_index
    public :: mlir_get_dialect_handle_vector
    public :: mlir_get_dialect_handle_tensor

    public :: mlir_dialect_handle_register_dialect
    public :: mlir_dialect_handle_load_dialect
    public :: mlir_dialect_handle_insert_dialect

    public :: mlir_dialect_is_null

    type :: mlir_dialect_handle_t
        type(c_ptr) :: ptr = c_null_ptr
    end type mlir_dialect_handle_t

    interface
        subroutine mlirRegisterAllDialects(registry) &
                bind(C, name="mlirRegisterAllDialects")
            import :: c_ptr
            type(c_ptr), value :: registry
        end subroutine mlirRegisterAllDialects

        subroutine mlirRegisterAllLLVMTranslations(context) &
                bind(C, name="mlirRegisterAllLLVMTranslations")
            import :: c_ptr
            type(c_ptr), value :: context
        end subroutine mlirRegisterAllLLVMTranslations

        function mlirGetDialectHandle__func__() &
                bind(C, name="mlirGetDialectHandle__func__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__func__
        end function mlirGetDialectHandle__func__

        function mlirGetDialectHandle__arith__() &
                bind(C, name="mlirGetDialectHandle__arith__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__arith__
        end function mlirGetDialectHandle__arith__

        function mlirGetDialectHandle__scf__() &
                bind(C, name="mlirGetDialectHandle__scf__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__scf__
        end function mlirGetDialectHandle__scf__

        function mlirGetDialectHandle__llvm__() &
                bind(C, name="mlirGetDialectHandle__llvm__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__llvm__
        end function mlirGetDialectHandle__llvm__

        function mlirGetDialectHandle__cf__() &
                bind(C, name="mlirGetDialectHandle__cf__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__cf__
        end function mlirGetDialectHandle__cf__

        function mlirGetDialectHandle__math__() &
                bind(C, name="mlirGetDialectHandle__math__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__math__
        end function mlirGetDialectHandle__math__

        function mlirGetDialectHandle__memref__() &
                bind(C, name="mlirGetDialectHandle__memref__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__memref__
        end function mlirGetDialectHandle__memref__

        function mlirGetDialectHandle__index__() &
                bind(C, name="mlirGetDialectHandle__index__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__index__
        end function mlirGetDialectHandle__index__

        function mlirGetDialectHandle__vector__() &
                bind(C, name="mlirGetDialectHandle__vector__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__vector__
        end function mlirGetDialectHandle__vector__

        function mlirGetDialectHandle__tensor__() &
                bind(C, name="mlirGetDialectHandle__tensor__")
            import :: c_ptr
            type(c_ptr) :: mlirGetDialectHandle__tensor__
        end function mlirGetDialectHandle__tensor__

        subroutine mlirDialectHandleRegisterDialect(handle, context) &
                bind(C, name="mlirDialectHandleRegisterDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: context
        end subroutine mlirDialectHandleRegisterDialect

        function mlirDialectHandleLoadDialect(handle, context) &
                bind(C, name="mlirDialectHandleLoadDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: context
            type(c_ptr) :: mlirDialectHandleLoadDialect
        end function mlirDialectHandleLoadDialect

        subroutine mlirDialectHandleInsertDialect(handle, registry) &
                bind(C, name="mlirDialectHandleInsertDialect")
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: registry
        end subroutine mlirDialectHandleInsertDialect
    end interface

contains

    subroutine mlir_register_all_dialects(registry)
        type(mlir_dialect_registry_t), intent(in) :: registry
        call mlirRegisterAllDialects(registry%ptr)
    end subroutine mlir_register_all_dialects

    subroutine mlir_register_all_llvm_translations(ctx)
        type(mlir_context_t), intent(in) :: ctx
        call mlirRegisterAllLLVMTranslations(ctx%ptr)
    end subroutine mlir_register_all_llvm_translations

    function mlir_get_dialect_handle_func() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__func__()
    end function mlir_get_dialect_handle_func

    function mlir_get_dialect_handle_arith() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__arith__()
    end function mlir_get_dialect_handle_arith

    function mlir_get_dialect_handle_scf() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__scf__()
    end function mlir_get_dialect_handle_scf

    function mlir_get_dialect_handle_llvm() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__llvm__()
    end function mlir_get_dialect_handle_llvm

    function mlir_get_dialect_handle_cf() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__cf__()
    end function mlir_get_dialect_handle_cf

    function mlir_get_dialect_handle_math() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__math__()
    end function mlir_get_dialect_handle_math

    function mlir_get_dialect_handle_memref() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__memref__()
    end function mlir_get_dialect_handle_memref

    function mlir_get_dialect_handle_index() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__index__()
    end function mlir_get_dialect_handle_index

    function mlir_get_dialect_handle_vector() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__vector__()
    end function mlir_get_dialect_handle_vector

    function mlir_get_dialect_handle_tensor() result(handle)
        type(mlir_dialect_handle_t) :: handle
        handle%ptr = mlirGetDialectHandle__tensor__()
    end function mlir_get_dialect_handle_tensor

    subroutine mlir_dialect_handle_register_dialect(handle, ctx)
        type(mlir_dialect_handle_t), intent(in) :: handle
        type(mlir_context_t), intent(in) :: ctx
        call mlirDialectHandleRegisterDialect(handle%ptr, ctx%ptr)
    end subroutine mlir_dialect_handle_register_dialect

    function mlir_dialect_handle_load_dialect(handle, ctx) result(dialect)
        type(mlir_dialect_handle_t), intent(in) :: handle
        type(mlir_context_t), intent(in) :: ctx
        type(mlir_dialect_t) :: dialect
        dialect%ptr = mlirDialectHandleLoadDialect(handle%ptr, ctx%ptr)
    end function mlir_dialect_handle_load_dialect

    subroutine mlir_dialect_handle_insert_dialect(handle, registry)
        type(mlir_dialect_handle_t), intent(in) :: handle
        type(mlir_dialect_registry_t), intent(in) :: registry
        call mlirDialectHandleInsertDialect(handle%ptr, registry%ptr)
    end subroutine mlir_dialect_handle_insert_dialect

    pure function mlir_dialect_is_null(dialect) result(is_null)
        type(mlir_dialect_t), intent(in) :: dialect
        logical :: is_null
        is_null = .not. c_associated(dialect%ptr)
    end function mlir_dialect_is_null

end module mlir_c_dialects
