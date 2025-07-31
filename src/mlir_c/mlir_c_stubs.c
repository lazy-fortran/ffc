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