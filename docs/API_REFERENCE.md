# FortFC API Reference

## MLIR C API Bindings

### mlir_c_core Module

Core MLIR context and operation management.

#### Types

```fortran
type :: mlir_context_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => context_is_valid
end type mlir_context_t

type :: mlir_module_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => module_is_valid
    procedure :: add_operation => module_add_operation
end type mlir_module_t

type :: mlir_operation_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => operation_is_valid
    procedure :: get_entry_block => operation_get_entry_block
end type mlir_operation_t

type :: mlir_location_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => location_is_valid
end type mlir_location_t
```

#### Functions

```fortran
! Context Management
function create_mlir_context() result(context)
    type(mlir_context_t) :: context

subroutine destroy_mlir_context(context)
    type(mlir_context_t), intent(inout) :: context

! Module Management  
function create_empty_module(location) result(module)
    type(mlir_location_t), intent(in) :: location
    type(mlir_module_t) :: module

! Location Management
function create_unknown_location(context) result(location)
    type(mlir_context_t), intent(in) :: context
    type(mlir_location_t) :: location

function create_file_line_col_location(context, filename, line, col) result(location)
    type(mlir_context_t), intent(in) :: context
    character(len=*), intent(in) :: filename
    integer, intent(in) :: line, col
    type(mlir_location_t) :: location
```

### mlir_c_types Module

MLIR type system bindings.

#### Functions

```fortran
! Basic Types
function create_i1_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_i8_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_i16_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_i32_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_i64_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_f32_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

function create_f64_type(context) result(type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t) :: type

! Array Types
function create_array_type(context, shape, element_type) result(array_type)
    type(mlir_context_t), intent(in) :: context
    integer, intent(in) :: shape(:)
    type(mlir_type_t), intent(in) :: element_type
    type(mlir_type_t) :: array_type

! Reference Types
function create_reference_type(context, pointee_type) result(ref_type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t), intent(in) :: pointee_type
    type(mlir_type_t) :: ref_type

! Function Types
function create_function_type(context, inputs, outputs) result(func_type)
    type(mlir_context_t), intent(in) :: context
    type(mlir_type_t), intent(in) :: inputs(:), outputs(:)
    type(mlir_type_t) :: func_type
```

### mlir_c_operations Module

Operation building and manipulation.

#### Types

```fortran
type :: mlir_operation_state_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: add_operand => state_add_operand
    procedure :: add_result_type => state_add_result_type
    procedure :: add_attribute => state_add_attribute
    procedure :: add_region => state_add_region
end type mlir_operation_state_t
```

#### Functions

```fortran
! Operation State Management
function create_operation_state(name, location) result(state)
    type(mlir_string_ref_t), intent(in) :: name
    type(mlir_location_t), intent(in) :: location
    type(mlir_operation_state_t) :: state

function create_operation(state) result(operation)
    type(mlir_operation_state_t), intent(in) :: state
    type(mlir_operation_t) :: operation

function verify_operation(operation) result(valid)
    type(mlir_operation_t), intent(in) :: operation
    logical :: valid
```

## High-Level Builder API

### mlir_builder Module

High-level IR building interface.

#### Types

```fortran
type :: mlir_builder_t
    private
    type(mlir_context_t) :: context
    type(c_ptr) :: builder_ptr = c_null_ptr
    logical :: initialized = .false.
contains
    procedure :: init => builder_init
    procedure :: cleanup => builder_cleanup
    procedure :: is_valid => builder_is_valid
    procedure :: get_context => builder_get_context
    procedure :: get_unknown_location => builder_get_unknown_location
    procedure :: set_insertion_point_to_start => builder_set_insertion_point_to_start
    procedure :: create_constant_op => builder_create_constant_op
end type mlir_builder_t
```

#### Functions

```fortran
function create_mlir_builder(context) result(builder)
    type(mlir_context_t), intent(in) :: context
    type(mlir_builder_t) :: builder

subroutine destroy_mlir_builder(builder)
    type(mlir_builder_t), intent(inout) :: builder
```

### ssa_manager Module

SSA value generation and management.

#### Types

```fortran
type :: ssa_manager_t
    private
    integer :: counter = 0
    character(len=32), allocatable :: value_names(:)
    type(mlir_type_t), allocatable :: value_types(:)
    integer :: value_count = 0
    integer :: max_values = 10000
    logical :: initialized = .false.
contains
    procedure :: init => ssa_init
    procedure :: cleanup => ssa_cleanup
    procedure :: next_value => ssa_next_value
    procedure :: set_value_type => ssa_set_value_type
    procedure :: get_value_type => ssa_get_value_type
    procedure :: dump_values => ssa_dump_values
end type ssa_manager_t
```

#### Functions

```fortran
! Generate next SSA value name
function next_value(this) result(value_name)
    class(ssa_manager_t), intent(inout) :: this
    character(len=:), allocatable :: value_name
    ! Returns: "%1", "%2", "%3", ...

! Set type for SSA value
subroutine set_value_type(this, value_name, value_type)
    class(ssa_manager_t), intent(inout) :: this
    character(len=*), intent(in) :: value_name
    type(mlir_type_t), intent(in) :: value_type

! Get type for SSA value
function get_value_type(this, value_name) result(value_type)
    class(ssa_manager_t), intent(in) :: this
    character(len=*), intent(in) :: value_name
    type(mlir_type_t) :: value_type
```

## Type Conversion API

### fortfc_type_converter Module

Fortran to MLIR type conversion.

#### Types

```fortran
type :: mlir_type_converter_t
    private
    type(mlir_context_t) :: context
    logical :: initialized = .false.
    ! Type cache for performance
    character(len=64), allocatable :: cached_keys(:)
    type(mlir_type_t), allocatable :: cached_types(:)
    integer :: cache_count = 0
    integer :: cache_hits = 0
    integer :: cache_misses = 0
contains
    procedure :: init => converter_init
    procedure :: cleanup => converter_cleanup
    procedure :: get_mlir_type_string => converter_get_mlir_type_string
    procedure :: create_integer_type => converter_create_integer_type
    procedure :: create_float_type => converter_create_float_type
    procedure :: create_character_type => converter_create_character_type
    procedure :: create_array_type => converter_create_array_type
    procedure :: print_statistics => converter_print_statistics
end type mlir_type_converter_t
```

#### Functions

```fortran
! Get MLIR type string for Fortran type
function get_mlir_type_string(this, fortran_type, kind) result(mlir_type)
    class(mlir_type_converter_t), intent(inout) :: this
    character(len=*), intent(in) :: fortran_type
    integer, intent(in) :: kind
    character(len=:), allocatable :: mlir_type
    ! Examples:
    ! ("integer", 4) → "i32"
    ! ("real", 8) → "f64"
    ! ("character", 10) → "!fir.char<1,10>"

! Array descriptor generation
function get_array_descriptor(shape, elem_type, elem_kind, assumed_shape, allocatable, pointer) result(descriptor)
    integer, intent(in) :: shape(:)
    character(len=*), intent(in) :: elem_type
    integer, intent(in) :: elem_kind
    logical, intent(in) :: assumed_shape, allocatable, pointer
    character(len=:), allocatable :: descriptor
    ! Examples:
    ! ([10,20], "real", 4, .false., .false., .false.) → "!fir.array<10x20xf32>"
    ! ([-1], "integer", 4, .true., .false., .false.) → "!fir.box<!fir.array<?xi32>>"

! Derived type name mangling
function mangle_derived_type_name(type_name, module_name, parent_type) result(mangled)
    character(len=*), intent(in) :: type_name
    character(len=*), intent(in), optional :: module_name, parent_type
    character(len=:), allocatable :: mangled
    ! Examples:
    ! ("point") → "_QTpoint"
    ! ("point", "geometry") → "_QMgeometryTpoint"
```

## Dialect APIs

### hlfir_dialect Module

HLFIR dialect operations.

#### Functions

```fortran
! Register HLFIR dialect
subroutine register_hlfir_dialect(context)
    type(mlir_context_t), intent(in) :: context

! Create hlfir.declare operation
function create_hlfir_declare(builder, memref, var_name) result(declare_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: memref
    character(len=*), intent(in) :: var_name
    type(mlir_operation_t) :: declare_op

! Create hlfir.assign operation  
function create_hlfir_assign(builder, rhs, lhs) result(assign_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: rhs, lhs
    type(mlir_operation_t) :: assign_op

! Create hlfir.elemental operation
function create_hlfir_elemental(builder, shape, element_type) result(elemental_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: shape
    type(mlir_type_t), intent(in) :: element_type
    type(mlir_operation_t) :: elemental_op
```

### fir_dialect Module

FIR dialect operations.

#### Functions

```fortran
! Register FIR dialect
subroutine register_fir_dialect(context)
    type(mlir_context_t), intent(in) :: context

! Create fir.alloca operation
function create_fir_alloca(builder, var_type, var_name) result(alloca_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_type_t), intent(in) :: var_type
    character(len=*), intent(in) :: var_name
    type(mlir_operation_t) :: alloca_op

! Create fir.load operation
function create_fir_load(builder, memref) result(load_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: memref
    type(mlir_operation_t) :: load_op

! Create fir.store operation
function create_fir_store(builder, value, memref) result(store_op)
    type(mlir_builder_t), intent(in) :: builder
    type(mlir_value_t), intent(in) :: value, memref
    type(mlir_operation_t) :: store_op
```

## Memory Management API

### memory_tracker Module

Memory usage tracking and leak detection.

#### Types

```fortran
type :: memory_tracker_t
    private
    logical :: initialized = .false.
    integer(8) :: current_usage = 0
    integer(8) :: peak_usage = 0
    logical :: peak_tracking_enabled = .false.
    integer(8) :: total_allocations = 0
    integer(8) :: total_deallocations = 0
    integer(8) :: allocation_failures = 0
contains
    procedure :: init => tracker_init
    procedure :: cleanup => tracker_cleanup
    procedure :: is_initialized => tracker_is_initialized
    procedure :: record_allocation => tracker_record_allocation
    procedure :: record_deallocation => tracker_record_deallocation
    procedure :: get_current_usage => tracker_get_current_usage
    procedure :: get_peak_usage => tracker_get_peak_usage
    procedure :: has_memory_leaks => tracker_has_memory_leaks
    procedure :: verify_all_freed => tracker_verify_all_freed
    procedure :: enable_peak_tracking => tracker_enable_peak_tracking
    procedure :: start_phase => tracker_start_phase
    procedure :: end_phase => tracker_end_phase
    procedure :: print_leak_report => tracker_print_leak_report
    procedure :: print_statistics => tracker_print_statistics
end type memory_tracker_t
```

### memory_guard Module

RAII-style automatic resource management.

#### Types

```fortran
type :: memory_guard_t
    private
    logical :: active = .false.
    integer :: total_registered = 0
    integer :: total_freed = 0
    logical :: auto_cleanup = .true.
contains
    procedure :: init => guard_init
    procedure :: cleanup => guard_cleanup
    procedure :: is_active => guard_is_active
    procedure :: register_resource => guard_register_resource
    procedure :: free_resource_by_name => guard_free_resource_by_name
    procedure :: all_resources_freed => guard_all_resources_freed
    procedure :: set_auto_cleanup => guard_set_auto_cleanup
    final :: guard_destructor  ! Automatic cleanup
end type memory_guard_t
```

### resource_manager Module

Centralized resource lifecycle management.

#### Types

```fortran
type :: resource_manager_t
    private
    logical :: initialized = .false.
    integer :: resource_count = 0
    integer :: peak_resource_count = 0
    integer :: total_allocated = 0
    integer :: total_freed = 0
    integer :: pass_manager_count = 0
    integer :: pipeline_count = 0
    integer :: module_count = 0
    integer :: context_count = 0
contains
    procedure :: init => manager_init
    procedure :: cleanup => manager_cleanup
    procedure :: is_initialized => manager_is_initialized
    procedure :: register_pass_manager => manager_register_pass_manager
    procedure :: register_pipeline => manager_register_pipeline
    procedure :: cleanup_resource => manager_cleanup_resource
    procedure :: cleanup_all => manager_cleanup_all
    procedure :: cleanup_by_type => manager_cleanup_by_type
    procedure :: verify_all_freed => manager_verify_all_freed
    procedure :: get_resource_count => manager_get_resource_count
    procedure :: get_peak_resource_count => manager_get_peak_resource_count
    procedure :: get_type_count => manager_get_type_count
    procedure :: print_statistics => manager_print_statistics
    procedure :: print_detailed_report => manager_print_detailed_report
end type resource_manager_t
```

## Pass Management API

### ffc_pass_manager Module

MLIR pass pipeline management.

#### Types

```fortran
type :: mlir_pass_manager_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => pm_is_valid
end type mlir_pass_manager_t
```

#### Functions

```fortran
! Create pass manager
function create_pass_manager(context) result(pass_manager)
    type(mlir_context_t), intent(in) :: context
    type(mlir_pass_manager_t) :: pass_manager

! Run passes on module
function run_passes(pass_manager, module) result(success)
    type(mlir_pass_manager_t), intent(in) :: pass_manager
    type(mlir_module_t), intent(inout) :: module
    logical :: success

! Destroy pass manager
subroutine destroy_pass_manager(pass_manager)
    type(mlir_pass_manager_t), intent(inout) :: pass_manager
```

### lowering_pipeline Module

HLFIR to FIR to LLVM lowering pipelines.

#### Types

```fortran
type :: mlir_lowering_pipeline_t
    type(c_ptr) :: ptr = c_null_ptr
contains
    procedure :: is_valid => pipeline_is_valid
end type mlir_lowering_pipeline_t
```

#### Functions

```fortran
! Create lowering pipeline
function create_lowering_pipeline(context, pipeline_type) result(pipeline)
    type(mlir_context_t), intent(in) :: context
    character(len=*), intent(in) :: pipeline_type  ! "hlfir-to-fir", "fir-to-llvm"
    type(mlir_lowering_pipeline_t) :: pipeline

! Create complete lowering pipeline (HLFIR → FIR → LLVM)
function create_complete_lowering_pipeline(context) result(pipeline)
    type(mlir_context_t), intent(in) :: context
    type(mlir_lowering_pipeline_t) :: pipeline

! Apply lowering pipeline
function apply_lowering_pipeline(pipeline, module) result(success)
    type(mlir_lowering_pipeline_t), intent(in) :: pipeline
    type(mlir_module_t), intent(inout) :: module
    logical :: success

! Set optimization level
subroutine set_optimization_level(pipeline, level)
    type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
    integer, intent(in) :: level  ! 0, 1, 2, 3

! Destroy pipeline
subroutine destroy_lowering_pipeline(pipeline)
    type(mlir_lowering_pipeline_t), intent(inout) :: pipeline
```

## Backend API

### mlir_c_backend Module

Complete MLIR C API backend implementation.

#### Types

```fortran
type, extends(backend_t) :: mlir_c_backend_t
    private
    logical :: initialized = .false.
    type(mlir_context_t) :: context
    type(mlir_builder_t) :: builder  
    type(mlir_module_t) :: current_module
    character(len=:), allocatable :: error_buffer
    integer :: error_count = 0
    logical :: debug_mode = .false.
contains
    ! Backend interface
    procedure :: generate_code => mlir_c_generate_code
    procedure :: get_name => mlir_c_get_name
    procedure :: get_version => mlir_c_get_version
    ! Lifecycle
    procedure :: init => mlir_c_init
    procedure :: cleanup => mlir_c_cleanup
    procedure :: is_initialized => mlir_c_is_initialized
    ! Capabilities
    procedure :: uses_c_api_exclusively => mlir_c_uses_c_api_exclusively
    procedure :: has_llvm_integration => mlir_c_has_llvm_integration
    procedure :: supports_linking => mlir_c_supports_linking
end type mlir_c_backend_t

type :: backend_options_t
    logical :: compile_mode = .false.
    logical :: optimize = .false.
    logical :: emit_hlfir = .false.
    logical :: emit_fir = .false.
    logical :: generate_llvm = .false.
    logical :: generate_executable = .false.
    logical :: link_runtime = .false.
    character(len=:), allocatable :: output_file
end type backend_options_t
```

## Test Framework API

### test_harness Module

Comprehensive test framework.

#### Types

```fortran
type :: test_case_t
    character(len=128) :: name = ""
    procedure(test_function_interface), pointer, nopass :: test_func => null()
    logical :: enabled = .true.
end type test_case_t

type :: test_suite_t
    character(len=128) :: name = ""
    type(test_case_t), allocatable :: tests(:)
    integer :: test_count = 0
    integer :: max_tests = 1000
    integer :: passed = 0
    integer :: failed = 0
    integer :: skipped = 0
    real :: start_time = 0.0
    real :: end_time = 0.0
contains
    procedure :: init => suite_init
    procedure :: cleanup => suite_cleanup
    procedure :: add_test => suite_add_test
    procedure :: run => suite_run
    procedure :: print_summary => suite_print_summary
end type test_suite_t

abstract interface
    function test_function_interface() result(passed)
        logical :: passed
    end function test_function_interface
end interface
```

#### Functions

```fortran
! Create test suite
function create_test_suite(name) result(suite)
    character(len=*), intent(in) :: name
    type(test_suite_t) :: suite

! Add test case
subroutine add_test_case(suite, name, test_func, enabled)
    type(test_suite_t), intent(inout) :: suite
    character(len=*), intent(in) :: name
    procedure(test_function_interface) :: test_func
    logical, intent(in), optional :: enabled

! Run test suite
subroutine run_test_suite(suite, verbose)
    type(test_suite_t), intent(inout) :: suite
    logical, intent(in), optional :: verbose
```

## Usage Examples

### Basic HLFIR Generation

```fortran
program basic_example
    use mlir_c_core
    use mlir_builder
    use hlfir_dialect
    use memory_guard
    
    type(memory_guard_t) :: guard
    type(mlir_context_t) :: context
    type(mlir_builder_t) :: builder
    type(mlir_module_t) :: module
    
    call guard%init()
    
    context = create_mlir_context()
    call guard%register_resource(context, "context")
    call register_hlfir_dialect(context)
    
    builder = create_mlir_builder(context)
    call guard%register_resource(builder, "builder")
    
    module = create_empty_module(builder%get_unknown_location())
    
    ! Generate HLFIR operations
    call generate_hello_world_hlfir(builder, module)
    
    ! Resources automatically cleaned up by guard
end program basic_example
```

### Type Conversion

```fortran
program type_example
    use mlir_c_core
    use fortfc_type_converter
    
    type(mlir_context_t) :: context
    type(mlir_type_converter_t) :: converter
    character(len=:), allocatable :: mlir_type
    
    context = create_mlir_context()
    call converter%init(context)
    
    ! Convert Fortran types to MLIR
    mlir_type = converter%get_mlir_type_string("integer", 4)  ! "i32"
    mlir_type = converter%get_mlir_type_string("real", 8)     ! "f64"
    
    ! Generate array descriptors
    mlir_type = get_array_descriptor([10, 20], "real", 4, .false., .false., .false.)
    ! "!fir.array<10x20xf32>"
    
    call converter%cleanup()
    call destroy_mlir_context(context)
end program type_example
```

### Memory Management

```fortran
program memory_example
    use memory_tracker
    use memory_guard
    
    type(memory_tracker_t) :: tracker
    type(memory_guard_t) :: guard
    
    call tracker%init()
    call tracker%enable_peak_tracking()
    call guard%init()
    
    ! Track allocations
    call tracker%record_allocation("test_data", 1024_8)
    
    ! Use guard for automatic cleanup
    ! ... create resources ...
    
    ! Check for leaks
    if (tracker%has_memory_leaks()) then
        call tracker%print_leak_report()
    end if
    
    call tracker%print_statistics()
    call tracker%cleanup()
end program memory_example
```

This API reference provides comprehensive coverage of all FortFC modules and their interfaces for MLIR C API usage.