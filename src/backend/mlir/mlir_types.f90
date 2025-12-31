module mlir_types
    use backend_interface
    use fortfront
    use mlir_utils
    implicit none
    private

    public :: fortran_to_mlir_type, fortran_array_to_mlir_type
    public :: generate_type_conversion, fortran_to_llvm_type
    public :: fortran_to_fir_type, fortran_to_fir_array_type

contains

 function fortran_to_mlir_type(fortran_type, kind_value, compile_mode) result(mlir_type)
        character(len=*), intent(in) :: fortran_type
        integer, intent(in), optional :: kind_value
        logical, intent(in), optional :: compile_mode
        character(len=:), allocatable :: mlir_type
        integer :: kind_val
        logical :: use_compile_mode

        ! Determine kind value
        if (present(kind_value)) then
            kind_val = kind_value
        else
            kind_val = 4  ! Default kind
        end if

        ! Determine compile mode
        if (present(compile_mode)) then
            use_compile_mode = compile_mode
        else
            use_compile_mode = .false.
        end if

        select case (trim(fortran_type))
        case ("integer")
            select case (kind_val)
            case (1)
                mlir_type = "i8"
            case (2)
                mlir_type = "i16"
            case (4)
                mlir_type = "i32"
            case (8)
                mlir_type = "i64"
            case default
                mlir_type = "i32"
            end select
        case ("real")
            select case (kind_val)
            case (4)
                mlir_type = "f32"
            case (8)
                mlir_type = "f64"
            case default
                mlir_type = "f32"
            end select
        case ("logical")
            mlir_type = "i1"
        case ("character")
            ! Always use HLFIR character types
            mlir_type = "!fir.char<1>"
        case ("complex")
            ! Always use FIR complex types
            select case (kind_val)
            case (4)
                mlir_type = "!fir.complex<4>"
            case (8)
                mlir_type = "!fir.complex<8>"
            case default
                mlir_type = "!fir.complex<4>"
            end select
        case default
            mlir_type = "i32"  ! Default fallback
        end select
    end function fortran_to_mlir_type

    function fortran_array_to_mlir_type(compile_mode, arena, node, indent_str) result(mlir_type)
        logical, intent(in) :: compile_mode
        type(ast_arena_t), intent(in) :: arena
        type(declaration_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir_type
        character(len=:), allocatable :: element_type, dim_spec
        integer :: i

        ! Get base element type
      element_type = fortran_to_mlir_type(node%type_name, node%kind_value, compile_mode)

        ! Build dimension specification
        dim_spec = ""
        if (allocated(node%dimension_indices)) then
            do i = 1, size(node%dimension_indices)
                if (i > 1) dim_spec = dim_spec//"x"
                ! Try to extract constant dimension from AST
                block
                    type(literal_node) :: lit_node
                    integer :: dim_idx
                    character(len=20) :: dim_str

                    dim_idx = node%dimension_indices(i)
                    if (dim_idx > 0 .and. dim_idx <= arena%size) then
                        if (allocated(arena%entries(dim_idx)%node)) then
                            select type (dim_node => arena%entries(dim_idx)%node)
                            type is (literal_node)
                                if (dim_node%literal_kind == LITERAL_INTEGER) then
                                    dim_spec = dim_spec//trim(dim_node%value)
                                else
                                    dim_spec = dim_spec//"?"
                                end if
                            class default
                                dim_spec = dim_spec//"?"
                            end select
                        end if
                    else
                        dim_spec = dim_spec//"?"
                    end if
                end block
            end do
        else
            dim_spec = "?"
        end if

        ! For compile mode, use memref instead of FIR array
        if (compile_mode) then
            mlir_type = "memref<"//dim_spec//"x"//element_type//">"
        else
            mlir_type = "!hlfir.array<"//dim_spec//"x"//element_type//">"
        end if
    end function fortran_array_to_mlir_type

    function generate_type_conversion(from_type, to_type, value_ssa, indent_str) result(mlir)
        character(len=*), intent(in) :: from_type, to_type, value_ssa, indent_str
        character(len=:), allocatable :: mlir

        mlir = ""

        ! Skip conversion if types are the same
        if (from_type == to_type) then
            return
        end if

        ! For now, we'll generate a placeholder - this needs a backend instance to generate SSA values
        ! This should be called from the main backend module

        ! Integer to float conversion
        if (index(from_type, "i") == 1 .and. index(to_type, "f") == 1) then
    mlir = indent_str//"// Type conversion: "//from_type//" to "//to_type//new_line('a')
            ! Float to integer conversion
        else if (index(from_type, "f") == 1 .and. index(to_type, "i") == 1) then
    mlir = indent_str//"// Type conversion: "//from_type//" to "//to_type//new_line('a')
            ! Integer extension/truncation
        else if (index(from_type, "i") == 1 .and. index(to_type, "i") == 1) then
    mlir = indent_str//"// Type conversion: "//from_type//" to "//to_type//new_line('a')
            ! Float extension/truncation
        else if (index(from_type, "f") == 1 .and. index(to_type, "f") == 1) then
    mlir = indent_str//"// Type conversion: "//from_type//" to "//to_type//new_line('a')
        else
            ! Fallback - just comment the conversion
            mlir = indent_str // "// Type conversion needed: " // from_type // " to " // to_type // new_line('a')
        end if
    end function generate_type_conversion

    function fortran_to_llvm_type(fortran_type, kind_value) result(llvm_type)
        character(len=*), intent(in) :: fortran_type
        integer, intent(in), optional :: kind_value
        character(len=:), allocatable :: llvm_type
        
        ! Redirect to FIR types - no LLVM types in pure HLFIR approach
        llvm_type = fortran_to_fir_type(fortran_type, kind_value)
    end function fortran_to_llvm_type

    ! Convert Fortran type to FIR type for HLFIR
    function fortran_to_fir_type(fortran_type, kind_value) result(fir_type)
        character(len=*), intent(in) :: fortran_type
        integer, intent(in), optional :: kind_value
        character(len=:), allocatable :: fir_type
        integer :: kind_val
        
        ! Determine kind value
        if (present(kind_value)) then
            kind_val = kind_value
        else
            kind_val = 4  ! Default kind
        end if
        
        select case (trim(fortran_type))
        case ("integer")
            select case (kind_val)
            case (1)
                fir_type = "!fir.int<1>"
            case (2)
                fir_type = "!fir.int<2>"
            case (4)
                fir_type = "!fir.int<4>"
            case (8)
                fir_type = "!fir.int<8>"
            case default
                fir_type = "!fir.int<4>"
            end select
        case ("real")
            select case (kind_val)
            case (4)
                fir_type = "!fir.real<4>"
            case (8)
                fir_type = "!fir.real<8>"
            case default
                fir_type = "!fir.real<4>"
            end select
        case ("logical")
            fir_type = "!fir.logical<1>"
        case ("character")
            fir_type = "!fir.char<1>"
        case ("complex")
            select case (kind_val)
            case (4)
                fir_type = "!fir.complex<4>"
            case (8)
                fir_type = "!fir.complex<8>"
            case default
                fir_type = "!fir.complex<4>"
            end select
        case default
            fir_type = "!fir.int<4>"  ! Default fallback
        end select
    end function fortran_to_fir_type
    
    ! Convert Fortran array type to FIR array type for HLFIR
    function fortran_to_fir_array_type(fortran_type, kind_value, arena, dimension_indices) result(fir_type)
        character(len=*), intent(in) :: fortran_type
        integer, intent(in), optional :: kind_value
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: dimension_indices(:)
        character(len=:), allocatable :: fir_type
        character(len=:), allocatable :: element_type
        character(len=32) :: dim_str
        integer :: i
        
        ! Get element type
        element_type = fortran_to_fir_type(fortran_type, kind_value)
        
        ! Build array type with dimensions
        fir_type = "!fir.array<"
        do i = 1, size(dimension_indices)
            if (i > 1) fir_type = fir_type//"x"
            fir_type = fir_type//"?"  ! Use unknown extent for now
        end do
        fir_type = fir_type//"x"//element_type//">"
        
    end function fortran_to_fir_array_type

end module mlir_types
