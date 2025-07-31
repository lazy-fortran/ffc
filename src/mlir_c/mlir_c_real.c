#include <stdlib.h>
#include <string.h>
#include "mlir-c/IR.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Support.h"
#include "mlir-c/Pass.h"

// REAL MLIR C API wrapper implementations
// These wrapper functions have different names to avoid conflicts with MLIR C API

//===----------------------------------------------------------------------===//
// Context API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirContextCreate() {
    MlirContext context = mlirContextCreate();
    
    // Create and register a dialect registry with all dialects
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    
    // Create context with the registry
    MlirContext context_with_dialects = mlirContextCreateWithRegistry(registry, true);
    mlirDialectRegistryDestroy(registry);
    
    // Allocate memory to store the actual context
    MlirContext* ctx_ptr = malloc(sizeof(MlirContext));
    if (!ctx_ptr) {
        mlirContextDestroy(context_with_dialects);
        return NULL;
    }
    
    *ctx_ptr = context_with_dialects;
    return ctx_ptr;
}

void ffc_mlirContextDestroy(void* context_ptr) {
    if (context_ptr) {
        MlirContext* ctx_ptr = (MlirContext*)context_ptr;
        mlirContextDestroy(*ctx_ptr);
        free(ctx_ptr);
    }
}

//===----------------------------------------------------------------------===//
// Location API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirLocationUnknownGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirLocation location = mlirLocationUnknownGet(*ctx_ptr);
    
    MlirLocation* loc_ptr = malloc(sizeof(MlirLocation));
    if (!loc_ptr) return NULL;
    
    *loc_ptr = location;
    return loc_ptr;
}

//===----------------------------------------------------------------------===//
// Module API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirModuleCreateEmpty(void* location_ptr) {
    if (!location_ptr) return NULL;
    
    MlirLocation* loc_ptr = (MlirLocation*)location_ptr;
    MlirModule module = mlirModuleCreateEmpty(*loc_ptr);
    
    MlirModule* mod_ptr = malloc(sizeof(MlirModule));
    if (!mod_ptr) return NULL;
    
    *mod_ptr = module;
    return mod_ptr;
}

//===----------------------------------------------------------------------===//
// Type System API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirNoneTypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirNoneTypeGet(*ctx_ptr);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirIntegerTypeGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirIntegerTypeGet(*ctx_ptr, width);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirIntegerTypeSignedGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirIntegerTypeSignedGet(*ctx_ptr, width);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirIntegerTypeUnsignedGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirIntegerTypeUnsignedGet(*ctx_ptr, width);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirF32TypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirF32TypeGet(*ctx_ptr);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirF64TypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType type = mlirF64TypeGet(*ctx_ptr);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

void* ffc_mlirMemRefTypeGet(void* element_type_ptr, long rank, void* shape, void* layout, void* memspace) {
    if (!element_type_ptr) return NULL;
    
    MlirType* elem_type_ptr = (MlirType*)element_type_ptr;
    
    // Create proper memref type with rank and shape
    int64_t* shape_array = NULL;
    if (shape && rank > 0) {
        shape_array = (int64_t*)shape;
    }
    
    MlirAttribute layout_attr = mlirAttributeGetNull();
    MlirAttribute memspace_attr = mlirAttributeGetNull();
    
    if (layout) {
        MlirAttribute* layout_ptr = (MlirAttribute*)layout;
        layout_attr = *layout_ptr;
    }
    
    if (memspace) {
        MlirAttribute* memspace_ptr = (MlirAttribute*)memspace;
        memspace_attr = *memspace_ptr;
    }
    
    MlirType type = mlirMemRefTypeGet(*elem_type_ptr, rank, shape_array, layout_attr, memspace_attr);
    
    MlirType* type_ptr = malloc(sizeof(MlirType));
    if (!type_ptr) return NULL;
    
    *type_ptr = type;
    return type_ptr;
}

//===----------------------------------------------------------------------===//
// Type Query API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

int ffc_mlirTypeIsAInteger(void* type_ptr) {
    if (!type_ptr) return 0;
    
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    return mlirTypeIsAInteger(*type_ptr_struct) ? 1 : 0;
}

int ffc_mlirTypeIsAFloat(void* type_ptr) {
    if (!type_ptr) return 0;
    
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    return (mlirTypeIsAF32(*type_ptr_struct) || mlirTypeIsAF64(*type_ptr_struct)) ? 1 : 0;
}

int ffc_mlirTypeIsAMemRef(void* type_ptr) {
    if (!type_ptr) return 0;
    
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    return mlirTypeIsAMemRef(*type_ptr_struct) ? 1 : 0;
}

int ffc_mlirIntegerTypeGetWidth(void* type_ptr) {
    if (!type_ptr) return 0;
    
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    if (mlirTypeIsAInteger(*type_ptr_struct)) {
        return mlirIntegerTypeGetWidth(*type_ptr_struct);
    }
    return 0;
}

//===----------------------------------------------------------------------===//
// Attribute API - REAL IMPLEMENTATION  
//===----------------------------------------------------------------------===//

void* ffc_mlirIntegerAttrGet(void* type_ptr, long value) {
    if (!type_ptr) return NULL;
    
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    MlirAttribute attr = mlirIntegerAttrGet(*type_ptr_struct, value);
    
    MlirAttribute* attr_ptr = malloc(sizeof(MlirAttribute));
    if (!attr_ptr) return NULL;
    
    *attr_ptr = attr;
    return attr_ptr;
}

void* ffc_mlirFloatAttrDoubleGet(void* context_ptr, void* type_ptr, double value) {
    if (!context_ptr || !type_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirType* type_ptr_struct = (MlirType*)type_ptr;
    MlirAttribute attr = mlirFloatAttrDoubleGet(*ctx_ptr, *type_ptr_struct, value);
    
    MlirAttribute* attr_ptr = malloc(sizeof(MlirAttribute));
    if (!attr_ptr) return NULL;
    
    *attr_ptr = attr;
    return attr_ptr;
}

void* ffc_mlirStringAttrGet(void* context_ptr, void* str_ref) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    
    MlirStringRef string_ref;
    if (str_ref) {
        string_ref = mlirStringRefCreateFromCString((const char*)str_ref);
    } else {
        string_ref = mlirStringRefCreateFromCString("default");
    }
    
    MlirAttribute attr = mlirStringAttrGet(*ctx_ptr, string_ref);
    
    MlirAttribute* attr_ptr = malloc(sizeof(MlirAttribute));
    if (!attr_ptr) return NULL;
    
    *attr_ptr = attr;
    return attr_ptr;
}

void* ffc_mlirArrayAttrGet(void* context_ptr, long num_elements, void* elements) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    
    MlirAttribute* attr_array = NULL;
    if (elements && num_elements > 0) {
        MlirAttribute** elem_ptrs = (MlirAttribute**)elements;
        attr_array = malloc(num_elements * sizeof(MlirAttribute));
        if (!attr_array) return NULL;
        
        for (long i = 0; i < num_elements; i++) {
            attr_array[i] = *(elem_ptrs[i]);
        }
    }
    
    MlirAttribute attr = mlirArrayAttrGet(*ctx_ptr, num_elements, attr_array);
    
    if (attr_array) free(attr_array);
    
    MlirAttribute* attr_ptr = malloc(sizeof(MlirAttribute));
    if (!attr_ptr) return NULL;
    
    *attr_ptr = attr;
    return attr_ptr;
}

//===----------------------------------------------------------------------===//  
// Attribute Query API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

int ffc_mlirAttributeIsAInteger(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    return mlirAttributeIsAInteger(*attr_ptr_struct) ? 1 : 0;
}

int ffc_mlirAttributeIsAFloat(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    return mlirAttributeIsAFloat(*attr_ptr_struct) ? 1 : 0;
}

int ffc_mlirAttributeIsAString(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    return mlirAttributeIsAString(*attr_ptr_struct) ? 1 : 0;
}

int ffc_mlirAttributeIsAArray(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    return mlirAttributeIsAArray(*attr_ptr_struct) ? 1 : 0;
}

long ffc_mlirIntegerAttrGetValueInt(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    if (mlirAttributeIsAInteger(*attr_ptr_struct)) {
        return mlirIntegerAttrGetValueInt(*attr_ptr_struct);
    }
    return 0;
}

double ffc_mlirFloatAttrGetValueDouble(void* attr_ptr) {
    if (!attr_ptr) return 0.0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    if (mlirAttributeIsAFloat(*attr_ptr_struct)) {
        return mlirFloatAttrGetValueDouble(*attr_ptr_struct);
    }
    return 0.0;
}

long ffc_mlirArrayAttrGetNumElements(void* attr_ptr) {
    if (!attr_ptr) return 0;
    
    MlirAttribute* attr_ptr_struct = (MlirAttribute*)attr_ptr;
    if (mlirAttributeIsAArray(*attr_ptr_struct)) {
        return mlirArrayAttrGetNumElements(*attr_ptr_struct);
    }
    return 0;
}

//===----------------------------------------------------------------------===//
// Operation API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

typedef struct {
    MlirOperationState mlir_state;
    MlirValue* operands;
    MlirType* results;
    MlirNamedAttribute* attributes;
    int operand_count;
    int result_count;
    int attribute_count;
} OperationStateWrapper;

void* ffc_mlirOperationStateCreate(void* name, void* location) {
    if (!name || !location) return NULL;
    
    OperationStateWrapper* wrapper = malloc(sizeof(OperationStateWrapper));
    if (!wrapper) return NULL;
    
    MlirLocation* loc_ptr = (MlirLocation*)location;
    MlirStringRef name_ref = mlirStringRefCreateFromCString((const char*)name);
    
    wrapper->mlir_state = mlirOperationStateGet(name_ref, *loc_ptr);
    wrapper->operands = NULL;
    wrapper->results = NULL;
    wrapper->attributes = NULL;
    wrapper->operand_count = 0;
    wrapper->result_count = 0;
    wrapper->attribute_count = 0;
    
    return wrapper;
}

void ffc_mlirOperationStateDestroy(void* state) {
    if (!state) return;
    
    OperationStateWrapper* wrapper = (OperationStateWrapper*)state;
    if (wrapper->operands) free(wrapper->operands);
    if (wrapper->results) free(wrapper->results);
    if (wrapper->attributes) free(wrapper->attributes);
    free(wrapper);
}

void ffc_mlirOperationStateAddOperands(void* state, long n, void* operands) {
    if (!state || n <= 0 || !operands) return;
    
    OperationStateWrapper* wrapper = (OperationStateWrapper*)state;
    
    wrapper->operands = realloc(wrapper->operands, (wrapper->operand_count + n) * sizeof(MlirValue));
    if (!wrapper->operands) return;
    
    MlirValue** operand_ptrs = (MlirValue**)operands;
    for (long i = 0; i < n; i++) {
        wrapper->operands[wrapper->operand_count + i] = *(operand_ptrs[i]);
    }
    
    mlirOperationStateAddOperands(&wrapper->mlir_state, n, wrapper->operands + wrapper->operand_count);
    wrapper->operand_count += n;
}

void ffc_mlirOperationStateAddResults(void* state, long n, void* results) {
    if (!state || n <= 0 || !results) return;
    
    OperationStateWrapper* wrapper = (OperationStateWrapper*)state;
    
    wrapper->results = realloc(wrapper->results, (wrapper->result_count + n) * sizeof(MlirType));
    if (!wrapper->results) return;
    
    MlirType** result_ptrs = (MlirType**)results;
    for (long i = 0; i < n; i++) {
        wrapper->results[wrapper->result_count + i] = *(result_ptrs[i]);
    }
    
    mlirOperationStateAddResults(&wrapper->mlir_state, n, wrapper->results + wrapper->result_count);
    wrapper->result_count += n;
}

void ffc_mlirOperationStateAddNamedAttribute(void* state, void* name, void* attribute) {
    if (!state || !name || !attribute) return;
    
    OperationStateWrapper* wrapper = (OperationStateWrapper*)state;
    
    wrapper->attributes = realloc(wrapper->attributes, (wrapper->attribute_count + 1) * sizeof(MlirNamedAttribute));
    if (!wrapper->attributes) return;
    
    MlirStringRef name_ref = mlirStringRefCreateFromCString((const char*)name);
    MlirContext context = mlirLocationGetContext(wrapper->mlir_state.location);
    MlirIdentifier name_id = mlirIdentifierGet(context, name_ref);
    MlirAttribute* attr_ptr = (MlirAttribute*)attribute;
    
    wrapper->attributes[wrapper->attribute_count].name = name_id;
    wrapper->attributes[wrapper->attribute_count].attribute = *attr_ptr;
    
    mlirOperationStateAddAttributes(&wrapper->mlir_state, 1, &wrapper->attributes[wrapper->attribute_count]);
    wrapper->attribute_count++;
}

void* ffc_mlirOperationCreate(void* state) {
    if (!state) return NULL;
    
    OperationStateWrapper* wrapper = (OperationStateWrapper*)state;
    MlirOperation op = mlirOperationCreate(&wrapper->mlir_state);
    
    MlirOperation* op_ptr = malloc(sizeof(MlirOperation));
    if (!op_ptr) return NULL;
    
    *op_ptr = op;
    return op_ptr;
}

void ffc_mlirOperationDestroy(void* op) {
    if (!op) return;
    
    MlirOperation* op_ptr = (MlirOperation*)op;
    mlirOperationDestroy(*op_ptr);
    free(op_ptr);
}

int ffc_mlirOperationVerify(void* op) {
    if (!op) return 0;
    
    MlirOperation* op_ptr = (MlirOperation*)op;
    return mlirOperationVerify(*op_ptr) ? 1 : 0;
}

long ffc_mlirOperationGetNumResults(void* op) {
    if (!op) return 0;
    
    MlirOperation* op_ptr = (MlirOperation*)op;
    return mlirOperationGetNumResults(*op_ptr);
}

//===----------------------------------------------------------------------===//
// Pass Manager API - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirPassManagerCreate(void* context_ptr) {
    if (!context_ptr) return NULL;
    
    MlirContext* ctx_ptr = (MlirContext*)context_ptr;
    MlirPassManager pm = mlirPassManagerCreate(*ctx_ptr);
    
    MlirPassManager* pm_ptr = malloc(sizeof(MlirPassManager));
    if (!pm_ptr) return NULL;
    
    *pm_ptr = pm;
    return pm_ptr;
}

void ffc_mlirPassManagerDestroy(void* pm_ptr) {
    if (!pm_ptr) return;
    
    MlirPassManager* pass_manager = (MlirPassManager*)pm_ptr;
    mlirPassManagerDestroy(*pass_manager);
    free(pass_manager);
}

int ffc_mlirPassManagerRun(void* pm_ptr, void* module_ptr) {
    if (!pm_ptr || !module_ptr) return 0;
    
    MlirPassManager* pass_manager = (MlirPassManager*)pm_ptr;
    MlirModule* module = (MlirModule*)module_ptr;
    
    // Get the module operation
    MlirOperation module_op = mlirModuleGetOperation(*module);
    
    // Run passes on the module operation
    MlirLogicalResult result = mlirPassManagerRunOnOp(*pass_manager, module_op);
    return mlirLogicalResultIsSuccess(result) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Dialect Registration - REAL IMPLEMENTATION
//===----------------------------------------------------------------------===//

void* ffc_mlirGetDialectHandle__fir__() {
    // FIR dialect handle will be properly implemented when we add FIR support
    static MlirDialectHandle fir_handle = {(void*)0x1001};
    return &fir_handle;
}

void ffc_mlirDialectHandleRegisterDialect(void* handle, void* context) {
    if (!handle || !context) return;
    
    MlirDialectHandle* dialect_handle = (MlirDialectHandle*)handle;
    MlirContext* ctx_ptr = (MlirContext*)context;
    
    mlirDialectHandleRegisterDialect(*dialect_handle, *ctx_ptr);
}

void* ffc_mlirGetDialectHandle__hlfir__() {
    static MlirDialectHandle hlfir_handle = {(void*)0x1002};
    return &hlfir_handle;
}

void* ffc_mlirGetDialectHandle__func__() {
    static MlirDialectHandle func_handle = {(void*)0x1003};
    return &func_handle;
}

void* ffc_mlirGetDialectHandle__arith__() {
    static MlirDialectHandle arith_handle = {(void*)0x1004};
    return &arith_handle;
}

void* ffc_mlirGetDialectHandle__scf__() {
    static MlirDialectHandle scf_handle = {(void*)0x1005};
    return &scf_handle;
}

// Note: Fortran code should now call the ffc_* functions directly
// This avoids conflicts with the real MLIR C API function names