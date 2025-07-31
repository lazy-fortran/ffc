module mlir_c_types
    use iso_c_binding
    use mlir_c_core
    implicit none
    private

    ! Public types
    public :: mlir_type_t
    
    ! Public functions
    public :: create_integer_type, create_float_type
    public :: create_array_type, create_reference_type
    public :: get_type_kind, get_integer_width
    public :: is_integer_type, is_float_type, is_array_type
    public :: validate_integer_width, validate_float_width
    public :: types_are_equal

    ! Type wrapper
    type :: mlir_type_t
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => type_is_valid
    end type mlir_type_t

    ! Type kind enumeration
    integer, parameter, public :: TYPE_INTEGER = 1
    integer, parameter, public :: TYPE_FLOAT = 2
    integer, parameter, public :: TYPE_ARRAY = 3
    integer, parameter, public :: TYPE_REFERENCE = 4
    integer, parameter, public :: TYPE_UNKNOWN = 0

    ! C interface declarations
    interface
        ! Integer type creation
        function mlirIntegerTypeGet(context, width) bind(c, name="mlirIntegerTypeGet") result(type)
            import :: c_ptr, c_int
            type(c_ptr), value :: context
            integer(c_int), value :: width
            type(c_ptr) :: type
        end function mlirIntegerTypeGet

        function mlirIntegerTypeSignedGet(context, width) bind(c, name="mlirIntegerTypeSignedGet") result(type)
            import :: c_ptr, c_int
            type(c_ptr), value :: context
            integer(c_int), value :: width
            type(c_ptr) :: type
        end function mlirIntegerTypeSignedGet

        function mlirIntegerTypeUnsignedGet(context, width) bind(c, name="mlirIntegerTypeUnsignedGet") result(type)
            import :: c_ptr, c_int
            type(c_ptr), value :: context
            integer(c_int), value :: width
            type(c_ptr) :: type
        end function mlirIntegerTypeUnsignedGet

        ! Float type creation
        function mlirF32TypeGet(context) bind(c, name="mlirF32TypeGet") result(type)
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: type
        end function mlirF32TypeGet

        function mlirF64TypeGet(context) bind(c, name="mlirF64TypeGet") result(type)
            import :: c_ptr
            type(c_ptr), value :: context
            type(c_ptr) :: type
        end function mlirF64TypeGet

        ! Array type creation - using MemRef for now
        function mlirMemRefTypeGet(element_type, rank, shape, layout, memspace) &
            bind(c, name="mlirMemRefTypeGet") result(type)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: element_type
            integer(c_int64_t), value :: rank
            type(c_ptr), value :: shape  ! pointer to array of int64_t
            type(c_ptr), value :: layout
            type(c_ptr), value :: memspace
            type(c_ptr) :: type
        end function mlirMemRefTypeGet

        ! Type queries
        function mlirTypeIsAInteger(type) bind(c, name="mlirTypeIsAInteger") result(is_int)
            import :: c_ptr, c_bool
            type(c_ptr), value :: type
            logical(c_bool) :: is_int
        end function mlirTypeIsAInteger

        function mlirTypeIsAFloat(type) bind(c, name="mlirTypeIsAFloat") result(is_float)
            import :: c_ptr, c_bool
            type(c_ptr), value :: type
            logical(c_bool) :: is_float
        end function mlirTypeIsAFloat

        function mlirTypeIsAMemRef(type) bind(c, name="mlirTypeIsAMemRef") result(is_memref)
            import :: c_ptr, c_bool
            type(c_ptr), value :: type
            logical(c_bool) :: is_memref
        end function mlirTypeIsAMemRef

        function mlirIntegerTypeGetWidth(type) bind(c, name="mlirIntegerTypeGetWidth") result(width)
            import :: c_ptr, c_int
            type(c_ptr), value :: type
            integer(c_int) :: width
        end function mlirIntegerTypeGetWidth
    end interface

contains

    ! Create integer type
    function create_integer_type(context, width, signed) result(type)
        type(mlir_context_t), intent(in) :: context
        integer, intent(in) :: width
        logical, intent(in), optional :: signed
        type(mlir_type_t) :: type
        logical :: is_signed
        
        is_signed = .true.  ! Default to signed
        if (present(signed)) is_signed = signed
        
        if (is_signed) then
            type%ptr = mlirIntegerTypeSignedGet(context%ptr, int(width, c_int))
        else
            type%ptr = mlirIntegerTypeUnsignedGet(context%ptr, int(width, c_int))
        end if
    end function create_integer_type

    ! Create float type
    function create_float_type(context, width) result(type)
        type(mlir_context_t), intent(in) :: context
        integer, intent(in) :: width
        type(mlir_type_t) :: type
        
        select case(width)
        case(32)
            type%ptr = mlirF32TypeGet(context%ptr)
        case(64)
            type%ptr = mlirF64TypeGet(context%ptr)
        case default
            type%ptr = c_null_ptr  ! Invalid width
        end select
    end function create_float_type

    ! Create array type (simplified - fixed shape only for now)
    function create_array_type(context, element_type, shape) result(type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        integer(c_int64_t), dimension(:), intent(in) :: shape
        type(mlir_type_t) :: type
        integer(c_int64_t), dimension(:), allocatable, target :: shape_copy
        
        ! Make a copy of shape for C interface
        allocate(shape_copy(size(shape)))
        shape_copy = shape
        
        ! Create memref type (simpler than full array type)
        type%ptr = mlirMemRefTypeGet(element_type%ptr, &
                                      int(size(shape), c_int64_t), &
                                      c_loc(shape_copy), &
                                      c_null_ptr, &
                                      c_null_ptr)
        
        deallocate(shape_copy)
    end function create_array_type

    ! Create reference type (using memref with rank 0 as approximation)
    function create_reference_type(context, element_type) result(type)
        type(mlir_context_t), intent(in) :: context
        type(mlir_type_t), intent(in) :: element_type
        type(mlir_type_t) :: type
        integer(c_int64_t), dimension(0) :: empty_shape
        
        ! Create rank-0 memref as reference
        type%ptr = mlirMemRefTypeGet(element_type%ptr, &
                                      0_c_int64_t, &
                                      c_null_ptr, &
                                      c_null_ptr, &
                                      c_null_ptr)
    end function create_reference_type

    ! Get type kind
    function get_type_kind(type) result(kind)
        type(mlir_type_t), intent(in) :: type
        integer :: kind
        
        if (.not. c_associated(type%ptr)) then
            kind = TYPE_UNKNOWN
        else if (mlirTypeIsAInteger(type%ptr)) then
            kind = TYPE_INTEGER
        else if (mlirTypeIsAFloat(type%ptr)) then
            kind = TYPE_FLOAT
        else if (mlirTypeIsAMemRef(type%ptr)) then
            kind = TYPE_ARRAY  ! Using memref for arrays
        else
            kind = TYPE_UNKNOWN
        end if
    end function get_type_kind

    ! Get integer width
    function get_integer_width(type) result(width)
        type(mlir_type_t), intent(in) :: type
        integer :: width
        
        if (c_associated(type%ptr) .and. mlirTypeIsAInteger(type%ptr)) then
            width = mlirIntegerTypeGetWidth(type%ptr)
        else
            width = 0
        end if
    end function get_integer_width

    ! Type query functions
    function is_integer_type(type) result(is_int)
        type(mlir_type_t), intent(in) :: type
        logical :: is_int
        
        is_int = c_associated(type%ptr) .and. mlirTypeIsAInteger(type%ptr)
    end function is_integer_type

    function is_float_type(type) result(is_float)
        type(mlir_type_t), intent(in) :: type
        logical :: is_float
        
        is_float = c_associated(type%ptr) .and. mlirTypeIsAFloat(type%ptr)
    end function is_float_type

    function is_array_type(type) result(is_array)
        type(mlir_type_t), intent(in) :: type
        logical :: is_array
        
        is_array = c_associated(type%ptr) .and. mlirTypeIsAMemRef(type%ptr)
    end function is_array_type

    ! Type validity check
    function type_is_valid(this) result(valid)
        class(mlir_type_t), intent(in) :: this
        logical :: valid
        valid = c_associated(this%ptr)
    end function type_is_valid

    ! Validate integer width
    function validate_integer_width(width) result(valid)
        integer, intent(in) :: width
        logical :: valid
        
        ! Common integer widths
        valid = (width == 1) .or. &    ! i1 (boolean)
                (width == 8) .or. &    ! i8
                (width == 16) .or. &   ! i16
                (width == 32) .or. &   ! i32
                (width == 64) .or. &   ! i64
                (width == 128)         ! i128
    end function validate_integer_width

    ! Validate float width
    function validate_float_width(width) result(valid)
        integer, intent(in) :: width
        logical :: valid
        
        ! Standard float widths
        valid = (width == 32) .or. &   ! f32
                (width == 64)          ! f64
    end function validate_float_width

    ! Check if two types are equal
    function types_are_equal(type1, type2) result(equal)
        type(mlir_type_t), intent(in) :: type1, type2
        logical :: equal
        
        ! Simple pointer comparison for now
        equal = c_associated(type1%ptr, type2%ptr)
    end function types_are_equal

end module mlir_c_types