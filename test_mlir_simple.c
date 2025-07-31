#include <stdio.h>
#include <stdlib.h>

// Test our real MLIR C API wrapper functions
extern void* ffc_mlirContextCreate();
extern void ffc_mlirContextDestroy(void* context);
extern void* ffc_mlirLocationUnknownGet(void* context);
extern void* ffc_mlirModuleCreateEmpty(void* location);

int main() {
    printf("=== Testing Real MLIR C API Integration ===\n");
    
    // Test 1: Context creation
    printf("Testing context creation...\n");
    void* context = ffc_mlirContextCreate();
    if (!context) {
        printf("FAIL: Context creation failed\n");
        return 1;
    }
    printf("PASS: Context created successfully\n");
    
    // Test 2: Location creation
    printf("Testing location creation...\n");
    void* location = ffc_mlirLocationUnknownGet(context);
    if (!location) {
        printf("FAIL: Location creation failed\n");
        ffc_mlirContextDestroy(context);
        return 1;
    }
    printf("PASS: Location created successfully\n");
    
    // Test 3: Module creation
    printf("Testing module creation...\n");
    void* module = ffc_mlirModuleCreateEmpty(location);
    if (!module) {
        printf("FAIL: Module creation failed\n");
        ffc_mlirContextDestroy(context);
        return 1;
    }
    printf("PASS: Module created successfully\n");
    
    // Test 4: Verify this is NOT a stub (real pointers should be different)
    printf("Testing real vs stub behavior...\n");
    void* context2 = ffc_mlirContextCreate();
    if (!context2 || context == context2) {
        printf("FAIL: Contexts should be different (not using stubs)\n");
        ffc_mlirContextDestroy(context);
        if (context2) ffc_mlirContextDestroy(context2);
        return 1;
    }
    printf("PASS: Different context pointers indicate real MLIR usage\n");
    
    // Cleanup
    ffc_mlirContextDestroy(context);
    ffc_mlirContextDestroy(context2);
    
    printf("=== ALL TESTS PASSED: Real MLIR Integration Working ===\n");
    return 0;
}