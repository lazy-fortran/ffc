! MLIR Intrinsic Functions Module - Pure HLFIR Implementation
! This module contains all intrinsic function handlers for MLIR code generation using HLFIR
module mlir_intrinsics
    use ast_core
    use mlir_backend_types
    use mlir_hlfir_helpers
    implicit none

    private
    public :: is_array_intrinsic_function, generate_mlir_array_intrinsic
    public :: is_complex_intrinsic_function, generate_mlir_complex_intrinsic
    public :: is_pointer_intrinsic_function, generate_mlir_pointer_intrinsic

contains

    ! Check if function name is an array intrinsic
    function is_array_intrinsic_function(func_name) result(is_intrinsic)
        character(len=*), intent(in) :: func_name
        logical :: is_intrinsic

        select case (trim(func_name))
        case ("size", "shape", "lbound", "ubound", "sum", "product", "maxval", "minval", &
              "count", "any", "all", "transpose", "reshape", "pack", "unpack")
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end function is_array_intrinsic_function

    ! Generate MLIR for array intrinsic functions - Pure HLFIR
    function generate_mlir_array_intrinsic(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: result_ssa, array_arg, dim_arg

        mlir = ""
        result_ssa = backend%next_ssa_value()
        backend%last_ssa_value = result_ssa
        
        select case (trim(node%name))
        case ("size")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                
                if (size(node%arg_indices) > 1) then
                    ! size(array, dim)
                    dim_arg = generate_mlir_expression(backend, arena, node%arg_indices(2), indent_str)
                    mlir = mlir//dim_arg
                    dim_arg = backend%last_ssa_value
                    mlir = mlir//indent_str//result_ssa//" = hlfir.size "//array_arg//" dim "//dim_arg// &
                           " : (!hlfir.expr<?xi32>, !hlfir.expr<i32>) -> !hlfir.expr<i64>"//new_line('a')
                else
                    ! size(array)
                    mlir = mlir//indent_str//result_ssa//" = hlfir.size "//array_arg// &
                           " : (!hlfir.expr<?xi32>) -> !hlfir.expr<i64>"//new_line('a')
                end if
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %err = fir.undefined : i64; "// &
                       "fir.result %err : i64 } : !hlfir.expr<i64>  // ERROR: size() requires array"//new_line('a')
            end if
            
        case ("sum")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                mlir = mlir//indent_str//result_ssa//" = hlfir.sum "//array_arg// &
                       " : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>"//new_line('a')
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %zero = fir.constant 0 : i32; "// &
                       "fir.result %zero : i32 } : !hlfir.expr<i32>  // ERROR: sum() requires array"//new_line('a')
            end if
            
        case ("product")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                mlir = mlir//indent_str//result_ssa//" = hlfir.product "//array_arg// &
                       " : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>"//new_line('a')
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %one = fir.constant 1 : i32; "// &
                       "fir.result %one : i32 } : !hlfir.expr<i32>  // ERROR: product() requires array"//new_line('a')
            end if
            
        case ("maxval")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                mlir = mlir//indent_str//result_ssa//" = hlfir.maxval "//array_arg// &
                       " : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>"//new_line('a')
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %min = fir.constant -2147483648 : i32; "// &
                       "fir.result %min : i32 } : !hlfir.expr<i32>  // ERROR: maxval() requires array"//new_line('a')
            end if
            
        case ("minval")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                mlir = mlir//indent_str//result_ssa//" = hlfir.minval "//array_arg// &
                       " : (!hlfir.expr<?xi32>) -> !hlfir.expr<i32>"//new_line('a')
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %max = fir.constant 2147483647 : i32; "// &
                       "fir.result %max : i32 } : !hlfir.expr<i32>  // ERROR: minval() requires array"//new_line('a')
            end if
            
        case ("transpose")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                mlir = mlir//indent_str//result_ssa//" = hlfir.transpose "//array_arg// &
                       " : (!hlfir.expr<?x?xi32>) -> !hlfir.expr<?x?xi32>"//new_line('a')
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %err = fir.undefined : !fir.array<0x0xi32>; "// &
                       "fir.result %err : !fir.array<0x0xi32> } : !hlfir.expr<!fir.array<0x0xi32>>"// &
                       "  ! ERROR: transpose() requires 2D array"//new_line('a')
            end if
            
        case ("lbound", "ubound")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                array_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//array_arg
                array_arg = backend%last_ssa_value
                
                if (size(node%arg_indices) > 1) then
                    ! bound(array, dim)
                    dim_arg = generate_mlir_expression(backend, arena, node%arg_indices(2), indent_str)
                    mlir = mlir//dim_arg
                    dim_arg = backend%last_ssa_value
                    if (trim(node%name) == "lbound") then
                        mlir = mlir//indent_str//result_ssa//" = hlfir.lbound "//array_arg//" dim "//dim_arg// &
                               " : (!hlfir.expr<?xi32>, !hlfir.expr<i32>) -> !hlfir.expr<i64>"//new_line('a')
                    else
                        mlir = mlir//indent_str//result_ssa//" = hlfir.ubound "//array_arg//" dim "//dim_arg// &
                               " : (!hlfir.expr<?xi32>, !hlfir.expr<i32>) -> !hlfir.expr<i64>"//new_line('a')
                    end if
                else
                    ! bound(array) - returns array of bounds
                    if (trim(node%name) == "lbound") then
                        mlir = mlir//indent_str//result_ssa//" = hlfir.lbound "//array_arg// &
                               " : (!hlfir.expr<?xi32>) -> !hlfir.expr<?xi64>"//new_line('a')
                    else
                        mlir = mlir//indent_str//result_ssa//" = hlfir.ubound "//array_arg// &
                               " : (!hlfir.expr<?xi32>) -> !hlfir.expr<?xi64>"//new_line('a')
                    end if
                end if
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %err = fir.undefined : i64; "// &
                       "fir.result %err : i64 } : !hlfir.expr<i64>  // ERROR: "//trim(node%name)//"() requires array"//new_line('a')
            end if
            
        case default
            ! Unknown array intrinsic - generate error
            mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %err = fir.undefined : i32; "// &
                   "fir.result %err : i32 } : !hlfir.expr<i32>  // ERROR: Unknown array intrinsic: "//trim(node%name)//new_line('a')
        end select
    end function generate_mlir_array_intrinsic

    ! Check if function name is a complex intrinsic
    function is_complex_intrinsic_function(func_name) result(is_intrinsic)
        character(len=*), intent(in) :: func_name
        logical :: is_intrinsic

        select case (trim(func_name))
        case ("real", "aimag", "abs", "conjg", "cmplx")
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end function is_complex_intrinsic_function

    ! Generate MLIR for complex intrinsic functions - Pure HLFIR
    function generate_mlir_complex_intrinsic(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: result_ssa, arg_ssa

        mlir = ""
        result_ssa = backend%next_ssa_value()
        backend%last_ssa_value = result_ssa

        if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
            arg_ssa = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
            mlir = mlir//arg_ssa
            arg_ssa = backend%last_ssa_value

            select case (trim(node%name))
            case ("real")
                mlir = mlir//indent_str//result_ssa//" = hlfir.real "//arg_ssa// &
                       " : (!hlfir.expr<!fir.complex<4>>) -> !hlfir.expr<f32>"//new_line('a')
            case ("aimag")
                mlir = mlir//indent_str//result_ssa//" = hlfir.aimag "//arg_ssa// &
                       " : (!hlfir.expr<!fir.complex<4>>) -> !hlfir.expr<f32>"//new_line('a')
            case ("abs")
                mlir = mlir//indent_str//result_ssa//" = hlfir.abs "//arg_ssa// &
                       " : (!hlfir.expr<!fir.complex<4>>) -> !hlfir.expr<f32>"//new_line('a')
            case ("conjg")
                mlir = mlir//indent_str//result_ssa//" = hlfir.conjg "//arg_ssa// &
                       " : (!hlfir.expr<!fir.complex<4>>) -> !hlfir.expr<!fir.complex<4>>"//new_line('a')
            case ("cmplx")
                ! Handle cmplx(real, imag) - needs two arguments
                if (size(node%arg_indices) > 1) then
                    block
                        character(len=:), allocatable :: imag_ssa, imag_arg
                        imag_arg = generate_mlir_expression(backend, arena, node%arg_indices(2), indent_str)
                        mlir = mlir//imag_arg
                        imag_ssa = backend%last_ssa_value
                        mlir = mlir//indent_str//result_ssa//" = hlfir.cmplx "//arg_ssa//", "//imag_ssa// &
                               " : (!hlfir.expr<f32>, !hlfir.expr<f32>) -> !hlfir.expr<!fir.complex<4>>"//new_line('a')
                    end block
                else
                    ! cmplx(x) - imaginary part is zero
                    mlir = mlir//indent_str//result_ssa//" = hlfir.cmplx "//arg_ssa// &
                           " : (!hlfir.expr<f32>) -> !hlfir.expr<!fir.complex<4>>"//new_line('a')
                end if
            end select
        else
            ! No arguments - error
            mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %err = fir.undefined : f32; "// &
                   "fir.result %err : f32 } : !hlfir.expr<f32>  // ERROR: "//trim(node%name)//"() requires argument"//new_line('a')
        end if
    end function generate_mlir_complex_intrinsic

    ! Check if function name is a pointer intrinsic
    function is_pointer_intrinsic_function(func_name) result(is_intrinsic)
        character(len=*), intent(in) :: func_name
        logical :: is_intrinsic

        select case (trim(func_name))
        case ("associated", "null")
            is_intrinsic = .true.
        case default
            is_intrinsic = .false.
        end select
    end function is_pointer_intrinsic_function

    ! Generate MLIR for pointer intrinsic functions - Pure HLFIR
    function generate_mlir_pointer_intrinsic(backend, arena, node, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        type(call_or_subscript_node), intent(in) :: node
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        character(len=:), allocatable :: result_ssa, ptr_arg

        mlir = ""
        result_ssa = backend%next_ssa_value()
        backend%last_ssa_value = result_ssa

        select case (trim(node%name))
        case ("associated")
            if (allocated(node%arg_indices) .and. size(node%arg_indices) > 0) then
                ptr_arg = generate_mlir_expression(backend, arena, node%arg_indices(1), indent_str)
                mlir = mlir//ptr_arg
                ptr_arg = backend%last_ssa_value
                
                if (size(node%arg_indices) > 1) then
                    ! associated(ptr, target)
                    block
                        character(len=:), allocatable :: target_arg, target_ssa
                        target_arg = generate_mlir_expression(backend, arena, node%arg_indices(2), indent_str)
                        mlir = mlir//target_arg
                        target_ssa = backend%last_ssa_value
                        mlir = mlir//indent_str//result_ssa//" = hlfir.associated "//ptr_arg//", "//target_ssa// &
                               " : (!hlfir.expr<!fir.ptr<i32>>, !hlfir.expr<!fir.ptr<i32>>) -> " // &
                               "!hlfir.expr<!fir.logical<4>>"//new_line('a')
                    end block
                else
                    ! associated(ptr)
                    mlir = mlir//indent_str//result_ssa//" = hlfir.associated "//ptr_arg// &
                           " : (!hlfir.expr<!fir.ptr<i32>>) -> !hlfir.expr<!fir.logical<4>>"//new_line('a')
                end if
            else
                mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %false = fir.constant false : !fir.logical<4>; "// &
                       "fir.result %false : !fir.logical<4> } : !hlfir.expr<!fir.logical<4>>"// &
                       "  ! ERROR: associated() requires pointer"//new_line('a')
            end if
            
        case ("null")
            ! null() - return null pointer
            mlir = mlir//indent_str//result_ssa//" = hlfir.null : !hlfir.expr<!fir.ptr<none>>"//new_line('a')
            
        case default
            ! Unknown pointer intrinsic
            mlir = mlir//indent_str//result_ssa//" = hlfir.expr { %false = fir.constant false : !fir.logical<4>; "// &
                   "fir.result %false : !fir.logical<4> } : !hlfir.expr<!fir.logical<4>>"// &
                   "  ! ERROR: Unknown pointer intrinsic: "//trim(node%name)//new_line('a')
        end select
    end function generate_mlir_pointer_intrinsic

    ! Forward declaration placeholder for generate_mlir_expression
    function generate_mlir_expression(backend, arena, node_index, indent_str) result(mlir)
        class(mlir_backend_t), intent(inout) :: backend
        type(ast_arena_t), intent(in) :: arena
        integer, intent(in) :: node_index
        character(len=*), intent(in) :: indent_str
        character(len=:), allocatable :: mlir
        
        mlir = "! generate_mlir_expression placeholder"//new_line('a')
    end function generate_mlir_expression

end module mlir_intrinsics