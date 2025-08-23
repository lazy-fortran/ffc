module mlir_backend_helpers
    use mlir_backend_types
    use mlir_hlfir_helpers
    implicit none
    private

    public :: format_string_to_array, generate_hlfir_constant, generate_hlfir_load
    public :: generate_hlfir_string_literal, int_to_char

contains

    ! Helper function to format a string as an array of ASCII values
    function format_string_to_array(str) result(array_str)
        character(len=*), intent(in) :: str
        character(len=:), allocatable :: array_str
        integer :: i
        character(len=20) :: num_str
        
        array_str = ""
        do i = 1, len(str)
            if (i > 1) array_str = array_str // ", "
            write(num_str, '(I0)') iachar(str(i:i))
            array_str = array_str // trim(num_str)
        end do
    end function format_string_to_array

    ! Helper function to generate HLFIR constant expressions
    function generate_hlfir_constant(backend, value, mlir_type, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: value, mlir_type, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: ssa_val

        ssa_val = backend%next_ssa_value()
        mlir = generate_hlfir_constant_code(ssa_val, value, mlir_type, indent_str)
        backend%last_ssa_value = ssa_val
    end function generate_hlfir_constant

    ! Helper function to generate HLFIR designate and load operations
    function generate_hlfir_load(backend, memref_ssa, indices, element_type, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: memref_ssa, indices, element_type, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: designate_ssa, load_ssa

        designate_ssa = backend%next_ssa_value()
        load_ssa = backend%next_ssa_value()
        mlir = generate_hlfir_load_code(designate_ssa, load_ssa, memref_ssa, indices, element_type, indent_str)
        backend%last_ssa_value = load_ssa
    end function generate_hlfir_load

    ! Helper function to generate HLFIR string literal expressions
    function generate_hlfir_string_literal(backend, string_value, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        character(len=*), intent(in) :: string_value, indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: ssa_val

        ssa_val = backend%next_ssa_value()
        mlir = generate_hlfir_string_literal_code(ssa_val, string_value, indent_str)
        backend%last_ssa_value = ssa_val
    end function generate_hlfir_string_literal

    function int_to_char(i) result(str)
        integer, intent(in) :: i
        character(len=:), allocatable :: str
        character(len=20) :: temp
        
        write(temp, '(I0)') i
        str = trim(temp)
    end function int_to_char

end module mlir_backend_helpers