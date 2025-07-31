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