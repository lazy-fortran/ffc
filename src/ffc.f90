module ffc
    use mlir_c_core
    use mlir_c_types
    use mlir_c_attributes
    use mlir_c_operations
    use mlir_c_dialects
    implicit none
    private
    public :: ffc_version
    character(len=*), parameter :: ffc_version = "0.1.0-mlir"
end module ffc
