#include <stdlib.h>

// Stub implementations for MLIR C API functions
// These are temporary until we link with actual MLIR libraries

void* mlirContextCreate() {
    // Return a non-null pointer to simulate a valid context
    return malloc(1);
}

void mlirContextDestroy(void* context) {
    if (context) {
        free(context);
    }
}

void* mlirModuleCreateEmpty(void* location) {
    // Return a non-null pointer to simulate a valid module
    // Use a unique pointer value to avoid conflicts
    static int module_dummy = 0;
    return &module_dummy;
}

void* mlirLocationUnknownGet(void* context) {
    // Return a non-null pointer to simulate a valid location
    // Use a unique pointer value to avoid conflicts
    static int location_dummy = 0;
    return &location_dummy;
}

// Type system stubs - use global statics for type identity
static int int_type_dummy = 0;
static int signed_int_type_dummy = 0;
static int unsigned_int_type_dummy = 0;
static int f32_type_dummy = 0;
static int f64_type_dummy = 0;
static int memref_type_dummy = 0;

void* mlirIntegerTypeGet(void* context, int width) {
    return &int_type_dummy;
}

void* mlirIntegerTypeSignedGet(void* context, int width) {
    return &signed_int_type_dummy;
}

void* mlirIntegerTypeUnsignedGet(void* context, int width) {
    return &unsigned_int_type_dummy;
}

void* mlirF32TypeGet(void* context) {
    return &f32_type_dummy;
}

void* mlirF64TypeGet(void* context) {
    return &f64_type_dummy;
}

void* mlirMemRefTypeGet(void* element_type, long rank, void* shape, void* layout, void* memspace) {
    return &memref_type_dummy;
}

int mlirTypeIsAInteger(void* type) {
    return (type == &int_type_dummy || 
            type == &signed_int_type_dummy || 
            type == &unsigned_int_type_dummy) ? 1 : 0;
}

int mlirTypeIsAFloat(void* type) {
    return (type == &f32_type_dummy || type == &f64_type_dummy) ? 1 : 0;
}

int mlirTypeIsAMemRef(void* type) {
    return (type == &memref_type_dummy) ? 1 : 0;
}

int mlirIntegerTypeGetWidth(void* type) {
    // Stub - return 32 as default
    return 32;
}

// Attribute system stubs
static int int_attr_dummy = 0;
static int float_attr_dummy = 0;
static int string_attr_dummy = 0;
static int array_attr_dummy = 0;

void* mlirIntegerAttrGet(void* type, long value) {
    // Store the value for retrieval
    static long stored_value = 0;
    stored_value = value;
    return &stored_value;
}

void* mlirFloatAttrDoubleGet(void* context, void* type, double value) {
    return &float_attr_dummy;
}

void* mlirStringAttrGet(void* context, void* str_ref) {
    return &string_attr_dummy;
}

void* mlirArrayAttrGet(void* context, long num_elements, void* elements) {
    return &array_attr_dummy;
}

int mlirAttributeIsAInteger(void* attr) {
    // Check if it's an integer attribute by checking if it's not one of the other types
    return (attr != &float_attr_dummy && attr != &string_attr_dummy && attr != &array_attr_dummy && attr != NULL) ? 1 : 0;
}

int mlirAttributeIsAFloat(void* attr) {
    return (attr == &float_attr_dummy) ? 1 : 0;
}

int mlirAttributeIsAString(void* attr) {
    return (attr == &string_attr_dummy) ? 1 : 0;
}

int mlirAttributeIsAArray(void* attr) {
    return (attr == &array_attr_dummy) ? 1 : 0;
}

long mlirIntegerAttrGetValueInt(void* attr) {
    // Return the stored value
    if (attr) {
        return *(long*)attr;
    }
    return 0;
}

double mlirFloatAttrGetValueDouble(void* attr) {
    return 3.14;  // Stub value
}

long mlirArrayAttrGetNumElements(void* attr) {
    return 3;  // Stub value
}

// Operation system stubs
typedef struct {
    char* name;
    void* location;
    int num_operands;
    int num_results;
    int num_attributes;
} MlirOperationState;

void* mlirOperationStateCreate(void* name, void* location) {
    MlirOperationState* state = malloc(sizeof(MlirOperationState));
    state->name = "test.op";  // Simplified
    state->location = location;
    state->num_operands = 0;
    state->num_results = 0;
    state->num_attributes = 0;
    return state;
}

void mlirOperationStateDestroy(void* state) {
    if (state) {
        free(state);
    }
}

void mlirOperationStateAddOperands(void* state, long n, void* operands) {
    if (state) {
        ((MlirOperationState*)state)->num_operands += n;
    }
}

void mlirOperationStateAddResults(void* state, long n, void* results) {
    if (state) {
        ((MlirOperationState*)state)->num_results += n;
    }
}

void mlirOperationStateAddNamedAttribute(void* state, void* name, void* attribute) {
    if (state) {
        ((MlirOperationState*)state)->num_attributes++;
    }
}

void* mlirOperationCreate(void* state) {
    // Return the state as a dummy operation
    return state;
}

void mlirOperationDestroy(void* op) {
    // Don't free here as it's already freed by state destroy
}

int mlirOperationVerify(void* op) {
    return 1;  // Always valid for stub
}

long mlirOperationGetNumResults(void* op) {
    if (op) {
        return ((MlirOperationState*)op)->num_results;
    }
    return 0;
}

// FIR Dialect stubs
void* mlirGetDialectHandle__fir__() {
    static int fir_handle_dummy = 0;
    return &fir_handle_dummy;
}

void mlirDialectHandleRegisterDialect(void* handle, void* context) {
    // No-op for stub
}