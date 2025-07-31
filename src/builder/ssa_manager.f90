module ssa_manager
    use iso_c_binding
    use mlir_c_core
    use mlir_c_types
    use mlir_c_operations
    use logger, only: log_debug, log_error
    implicit none
    private

    ! Public types
    public :: ssa_manager_t, ssa_value_info_t

    ! Public functions
    public :: create_ssa_manager, destroy_ssa_manager
    public :: hash_string  ! Export for testing

    ! SSA value information wrapper
    type :: ssa_value_info_t
        type(mlir_value_t) :: value
        type(mlir_type_t) :: value_type
        character(len=:), allocatable :: name
        integer :: id = 0
        type(mlir_operation_t) :: defining_op
        integer, allocatable :: use_ops(:)
        integer :: use_count = 0
    contains
        procedure :: is_valid => ssa_value_is_valid
        procedure :: get_id => ssa_value_get_id
        procedure :: get_name => ssa_value_get_name
        procedure :: get_type => ssa_value_get_type
        procedure :: equals => ssa_value_equals
    end type ssa_value_info_t

    ! Hash table entry for name lookups
    type :: hash_entry_t
        character(len=:), allocatable :: name
        integer :: value_index = 0
        type(hash_entry_t), pointer :: next => null()
    end type hash_entry_t

    ! SSA Manager type
    type :: ssa_manager_t
        type(mlir_context_t), pointer :: context => null()
        type(ssa_value_info_t), allocatable :: values(:)
        integer :: num_values = 0
        integer :: next_id = 1
        ! Hash table for O(1) name lookups
        type(hash_entry_t), pointer :: name_hash(:) => null()
        integer :: hash_size = 31  ! Prime number for better distribution
    contains
        procedure :: is_valid => manager_is_valid
        procedure :: generate_value => manager_generate_value
        procedure :: generate_named_value => manager_generate_named_value
        procedure :: has_value => manager_has_value
        procedure :: get_value_by_name => manager_get_value_by_name
        procedure :: types_equal => manager_types_equal
        procedure :: is_integer_type => manager_is_integer_type
        procedure :: is_float_type => manager_is_float_type
        procedure :: set_defining_op => manager_set_defining_op
        procedure :: get_defining_op => manager_get_defining_op
        procedure :: add_use => manager_add_use
        procedure :: remove_use => manager_remove_use
        procedure :: get_use_count => manager_get_use_count
        procedure :: get_uses => manager_get_uses
        procedure :: is_dead_value => manager_is_dead_value
        ! Debug helpers
        procedure :: dump_values => manager_dump_values
        procedure :: get_memory_usage => manager_get_memory_usage
        procedure :: validate_integrity => manager_validate_integrity
    end type ssa_manager_t

contains

    ! Create SSA manager
    function create_ssa_manager(context) result(manager)
        type(mlir_context_t), intent(in), target :: context
        type(ssa_manager_t) :: manager
        integer :: i
        
        manager%context => context
        allocate(manager%values(10))
        manager%num_values = 0
        manager%next_id = 1
        
        ! Initialize hash table
        allocate(manager%name_hash(manager%hash_size))
        do i = 1, manager%hash_size
            manager%name_hash(i)%next => null()
        end do
    end function create_ssa_manager

    ! Destroy SSA manager
    subroutine destroy_ssa_manager(manager)
        type(ssa_manager_t), intent(inout) :: manager
        integer :: i
        type(hash_entry_t), pointer :: current, next_entry
        
        if (allocated(manager%values)) then
            do i = 1, manager%num_values
                if (allocated(manager%values(i)%name)) then
                    deallocate(manager%values(i)%name)
                end if
                if (allocated(manager%values(i)%use_ops)) then
                    deallocate(manager%values(i)%use_ops)
                end if
            end do
            deallocate(manager%values)
        end if
        
        ! Clean up hash table
        if (associated(manager%name_hash)) then
            do i = 1, manager%hash_size
                current => manager%name_hash(i)%next
                do while (associated(current))
                    next_entry => current%next
                    if (allocated(current%name)) deallocate(current%name)
                    deallocate(current)
                    current => next_entry
                end do
            end do
            deallocate(manager%name_hash)
        end if
        
        manager%context => null()
        manager%num_values = 0
        manager%next_id = 1
    end subroutine destroy_ssa_manager

    ! Check if manager is valid
    function manager_is_valid(this) result(valid)
        class(ssa_manager_t), intent(in) :: this
        logical :: valid
        valid = associated(this%context)
    end function manager_is_valid

    ! Generate anonymous SSA value
    function manager_generate_value(this, value_type) result(val_info)
        class(ssa_manager_t), intent(inout) :: this
        type(mlir_type_t), intent(in) :: value_type
        type(ssa_value_info_t) :: val_info
        character(len=20) :: temp_name
        
        ! Create automatic name
        write(temp_name, '("%", I0)') this%next_id - 1
        val_info = this%generate_named_value(trim(temp_name), value_type)
    end function manager_generate_value

    ! Generate named SSA value
    function manager_generate_named_value(this, name, value_type) result(val_info)
        class(ssa_manager_t), intent(inout) :: this
        character(len=*), intent(in) :: name
        type(mlir_type_t), intent(in) :: value_type
        type(ssa_value_info_t) :: val_info
        type(ssa_value_info_t), allocatable :: temp_values(:)
        
        ! Resize array if needed
        if (this%num_values >= size(this%values)) then
            allocate(temp_values(size(this%values) * 2))
            temp_values(1:this%num_values) = this%values(1:this%num_values)
            call move_alloc(temp_values, this%values)
        end if
        
        ! Create the value info
        val_info%id = this%next_id
        val_info%name = name
        if (name(1:1) /= "%") then
            val_info%name = "%" // name
        end if
        val_info%value_type = value_type
        val_info%value%ptr = transfer(int(this%next_id, c_intptr_t), val_info%value%ptr)
        val_info%use_count = 0
        allocate(val_info%use_ops(5))
        
        ! Store in array
        this%num_values = this%num_values + 1
        this%values(this%num_values) = val_info
        this%next_id = this%next_id + 1
        
        ! Add to hash table for fast lookup
        call add_to_hash_table(this, val_info%name, this%num_values)
    end function manager_generate_named_value

    ! Check if value name exists (optimized with hash table)
    function manager_has_value(this, name) result(exists)
        class(ssa_manager_t), intent(in) :: this
        character(len=*), intent(in) :: name
        logical :: exists
        character(len=:), allocatable :: full_name
        
        full_name = name
        if (name(1:1) /= "%") then
            full_name = "%" // name
        end if
        
        exists = (find_in_hash_table(this, full_name) > 0)
    end function manager_has_value

    ! Get value by name (optimized with hash table)
    function manager_get_value_by_name(this, name) result(val_info)
        class(ssa_manager_t), intent(in) :: this
        character(len=*), intent(in) :: name
        type(ssa_value_info_t) :: val_info
        integer :: value_index
        character(len=:), allocatable :: full_name
        
        full_name = name
        if (name(1:1) /= "%") then
            full_name = "%" // name
        end if
        
        value_index = find_in_hash_table(this, full_name)
        if (value_index > 0) then
            val_info = this%values(value_index)
        else
            ! Return invalid value if not found
            val_info%id = 0
            val_info%value%ptr = c_null_ptr
        end if
    end function manager_get_value_by_name

    ! Check if types are equal (stub implementation)
    function manager_types_equal(this, type1, type2) result(equal)
        class(ssa_manager_t), intent(in) :: this
        type(mlir_type_t), intent(in) :: type1, type2
        logical :: equal
        
        ! Simple pointer comparison for stub
        equal = c_associated(type1%ptr, type2%ptr)
    end function manager_types_equal

    ! Check if value is integer type
    function manager_is_integer_type(this, val_info) result(is_int)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        logical :: is_int
        
        ! Stub: assume it's integer if type ptr is set and name contains "int"
        is_int = c_associated(val_info%value_type%ptr)
        if (is_int .and. allocated(val_info%name)) then
            is_int = index(val_info%name, "int") > 0
        end if
    end function manager_is_integer_type

    ! Check if value is float type
    function manager_is_float_type(this, val_info) result(is_float)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        logical :: is_float
        
        ! Stub: assume it's float if type ptr is set and name contains "float"
        is_float = c_associated(val_info%value_type%ptr)
        if (is_float .and. allocated(val_info%name)) then
            is_float = index(val_info%name, "float") > 0
        end if
    end function manager_is_float_type

    ! Set defining operation for value
    subroutine manager_set_defining_op(this, val_info, op)
        class(ssa_manager_t), intent(inout) :: this
        type(ssa_value_info_t), intent(inout) :: val_info
        type(mlir_operation_t), intent(in) :: op
        integer :: i
        
        ! Find the value in our array and update it
        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                this%values(i)%defining_op = op
                val_info%defining_op = op
                return
            end if
        end do
    end subroutine manager_set_defining_op

    ! Get defining operation
    function manager_get_defining_op(this, val_info) result(op)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        type(mlir_operation_t) :: op
        integer :: i
        
        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                op = this%values(i)%defining_op
                return
            end if
        end do
        
        ! Return invalid op if not found
        op%ptr = c_null_ptr
    end function manager_get_defining_op

    ! Add use of value
    subroutine manager_add_use(this, val_info, op)
        class(ssa_manager_t), intent(inout) :: this
        type(ssa_value_info_t), intent(inout) :: val_info
        type(mlir_operation_t), intent(in) :: op
        integer :: i, j
        integer, allocatable :: temp_uses(:)

        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                ! Resize use array if needed
                if (this%values(i)%use_count >= size(this%values(i)%use_ops)) then
                    allocate(temp_uses(size(this%values(i)%use_ops) * 2))
                    temp_uses(1:this%values(i)%use_count) = &
                        this%values(i)%use_ops(1:this%values(i)%use_count)
                    call move_alloc(temp_uses, this%values(i)%use_ops)
                end if
                
                this%values(i)%use_count = this%values(i)%use_count + 1
                this%values(i)%use_ops(this%values(i)%use_count) = &
                    int(transfer(op%ptr, 0_c_intptr_t))
                
                ! Update the passed value info
                val_info%use_count = this%values(i)%use_count
                return
            end if
        end do
    end subroutine manager_add_use

    ! Remove use of value
    subroutine manager_remove_use(this, val_info, op)
        class(ssa_manager_t), intent(inout) :: this
        type(ssa_value_info_t), intent(inout) :: val_info
        type(mlir_operation_t), intent(in) :: op
        integer :: i, j, op_id
        
        op_id = int(transfer(op%ptr, 0_c_intptr_t))
        
        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                ! Find and remove the use
                do j = 1, this%values(i)%use_count
                    if (this%values(i)%use_ops(j) == op_id) then
                        ! Shift remaining uses down
                        this%values(i)%use_ops(j:this%values(i)%use_count-1) = &
                            this%values(i)%use_ops(j+1:this%values(i)%use_count)
                        this%values(i)%use_count = this%values(i)%use_count - 1
                        val_info%use_count = this%values(i)%use_count
                        return
                    end if
                end do
                return
            end if
        end do
    end subroutine manager_remove_use

    ! Get use count
    function manager_get_use_count(this, val_info) result(count)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        integer :: count
        integer :: i
        
        count = 0
        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                count = this%values(i)%use_count
                return
            end if
        end do
    end function manager_get_use_count

    ! Get uses list
    function manager_get_uses(this, val_info) result(uses)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        integer, allocatable :: uses(:)
        integer :: i
        
        do i = 1, this%num_values
            if (this%values(i)%id == val_info%id) then
                if (this%values(i)%use_count > 0) then
                    allocate(uses(this%values(i)%use_count))
                    uses(:) = this%values(i)%use_ops(1:this%values(i)%use_count)
                else
                    allocate(uses(0))
                end if
                return
            end if
        end do
        
        ! Return empty if not found
        allocate(uses(0))
    end function manager_get_uses

    ! Check if value is dead
    function manager_is_dead_value(this, val_info) result(is_dead)
        class(ssa_manager_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: val_info
        logical :: is_dead
        
        is_dead = (this%get_use_count(val_info) == 0)
    end function manager_is_dead_value

    ! SSA value info methods
    function ssa_value_is_valid(this) result(valid)
        class(ssa_value_info_t), intent(in) :: this
        logical :: valid
        valid = (this%id > 0)
    end function ssa_value_is_valid

    function ssa_value_get_id(this) result(id)
        class(ssa_value_info_t), intent(in) :: this
        integer :: id
        id = this%id
    end function ssa_value_get_id

    function ssa_value_get_name(this) result(name)
        class(ssa_value_info_t), intent(in) :: this
        character(len=:), allocatable :: name
        if (allocated(this%name)) then
            name = this%name
        else
            name = ""
        end if
    end function ssa_value_get_name

    function ssa_value_get_type(this) result(value_type)
        class(ssa_value_info_t), intent(in) :: this
        type(mlir_type_t) :: value_type
        value_type = this%value_type
    end function ssa_value_get_type

    function ssa_value_equals(this, other) result(equal)
        class(ssa_value_info_t), intent(in) :: this
        type(ssa_value_info_t), intent(in) :: other
        logical :: equal
        equal = (this%id == other%id)
    end function ssa_value_equals

    ! Hash function for strings
    function hash_string(str) result(hash)
        character(len=*), intent(in) :: str
        integer :: hash
        integer :: i
        
        hash = 0
        do i = 1, len(str)
            hash = mod(hash * 31 + ichar(str(i:i)), huge(hash))
        end do
        hash = abs(hash)
    end function hash_string

    ! Add entry to hash table
    subroutine add_to_hash_table(manager, name, value_index)
        type(ssa_manager_t), intent(inout) :: manager
        character(len=*), intent(in) :: name
        integer, intent(in) :: value_index
        integer :: hash_index
        type(hash_entry_t), pointer :: new_entry, current
        
        hash_index = mod(hash_string(name), manager%hash_size) + 1
        
        ! Create new entry
        allocate(new_entry)
        new_entry%name = name
        new_entry%value_index = value_index
        new_entry%next => null()
        
        ! Insert at beginning of chain
        if (.not. associated(manager%name_hash(hash_index)%next)) then
            manager%name_hash(hash_index)%next => new_entry
        else
            new_entry%next => manager%name_hash(hash_index)%next
            manager%name_hash(hash_index)%next => new_entry
        end if
    end subroutine add_to_hash_table

    ! Find entry in hash table
    function find_in_hash_table(manager, name) result(value_index)
        type(ssa_manager_t), intent(in) :: manager
        character(len=*), intent(in) :: name
        integer :: value_index
        integer :: hash_index
        type(hash_entry_t), pointer :: current
        
        value_index = 0
        hash_index = mod(hash_string(name), manager%hash_size) + 1
        
        current => manager%name_hash(hash_index)%next
        do while (associated(current))
            if (allocated(current%name) .and. current%name == name) then
                value_index = current%value_index
                return
            end if
            current => current%next
        end do
    end function find_in_hash_table

    ! Debug helper: dump all values
    subroutine manager_dump_values(this)
        class(ssa_manager_t), intent(in) :: this
        integer :: i
        
        call log_debug("=== SSA Manager Value Dump ===")
        block
            character(len=32) :: num_str, id_str
            write(num_str, '(I0)') this%num_values
            write(id_str, '(I0)') this%next_id
            call log_debug("Total values: " // trim(num_str))
            call log_debug("Next ID: " // trim(id_str))
        end block
        
        do i = 1, this%num_values
            block
                character(len=32) :: val_str, id_str, use_str
                character(len=:), allocatable :: msg
                write(val_str, '(I0)') i
                write(id_str, '(I0)') this%values(i)%id
                write(use_str, '(I0)') this%values(i)%use_count
                
                msg = "Value " // trim(val_str) // ": ID=" // trim(id_str)
                if (allocated(this%values(i)%name)) then
                    msg = msg // ", Name=" // this%values(i)%name
                end if
                msg = msg // ", Uses=" // trim(use_str)
                msg = msg // ", DefOp=" // merge("valid  ", "invalid", this%values(i)%defining_op%is_valid())
                call log_debug(msg)
            end block
        end do
        call log_debug("==============================")
    end subroutine manager_dump_values

    ! Debug helper: get memory usage
    function manager_get_memory_usage(this) result(bytes)
        class(ssa_manager_t), intent(in) :: this
        integer :: bytes
        integer :: i
        
        bytes = 0
        
        ! Values array
        if (allocated(this%values)) then
            bytes = bytes + size(this%values) * 200  ! Rough estimate per value
        end if
        
        ! Hash table
        bytes = bytes + this%hash_size * 16  ! Rough estimate per hash entry
        
        ! Names and use lists
        do i = 1, this%num_values
            if (allocated(this%values(i)%name)) then
                bytes = bytes + len(this%values(i)%name)
            end if
            if (allocated(this%values(i)%use_ops)) then
                bytes = bytes + size(this%values(i)%use_ops) * 4
            end if
        end do
    end function manager_get_memory_usage

    ! Debug helper: validate internal integrity
    function manager_validate_integrity(this) result(valid)
        class(ssa_manager_t), intent(in) :: this
        logical :: valid
        integer :: i, found_index
        
        valid = .true.
        
        ! Check that all named values can be found in hash table
        do i = 1, this%num_values
            if (allocated(this%values(i)%name)) then
                found_index = find_in_hash_table(this, this%values(i)%name)
                if (found_index /= i) then
                    block
                        character(len=32) :: val_str
                        write(val_str, '(I0)') i
                        call log_error("Value " // trim(val_str) // " name hash lookup failed")
                    end block
                    valid = .false.
                end if
            end if
        end do
        
        ! Check for reasonable ID sequence
        do i = 1, this%num_values
            if (this%values(i)%id <= 0 .or. this%values(i)%id >= this%next_id) then
                block
                    character(len=32) :: val_str, id_str
                    write(val_str, '(I0)') i
                    write(id_str, '(I0)') this%values(i)%id
                    call log_error("Value " // trim(val_str) // " has invalid ID " // trim(id_str))
                end block
                valid = .false.
            end if
        end do
    end function manager_validate_integrity

end module ssa_manager