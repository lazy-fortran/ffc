// Standalone MLIR-compatible implementation for ffc
// This implementation does NOT depend on MLIR C API libraries
// Instead, it tracks IR structures and can generate valid MLIR text output

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

// Internal type identifiers
typedef enum {
    TYPE_NONE = 0,
    TYPE_INTEGER,
    TYPE_FLOAT,
    TYPE_INDEX,
    TYPE_MEMREF,
    TYPE_TENSOR,
    TYPE_FUNCTION,
    TYPE_COMPLEX,
    TYPE_VECTOR,
    TYPE_TUPLE
} TypeKind;

// Internal attribute identifiers
typedef enum {
    ATTR_NONE = 0,
    ATTR_INTEGER,
    ATTR_FLOAT,
    ATTR_STRING,
    ATTR_ARRAY,
    ATTR_TYPE,
    ATTR_UNIT,
    ATTR_BOOL,
    ATTR_SYMBOL_REF
} AttrKind;

// Forward declarations
typedef struct FfcContext FfcContext;
typedef struct FfcLocation FfcLocation;
typedef struct FfcModule FfcModule;
typedef struct FfcType FfcType;
typedef struct FfcAttribute FfcAttribute;
typedef struct FfcValue FfcValue;
typedef struct FfcOperation FfcOperation;
typedef struct FfcBlock FfcBlock;
typedef struct FfcRegion FfcRegion;
typedef struct FfcOperationState FfcOperationState;
typedef struct FfcPassManager FfcPassManager;
typedef struct FfcDialectHandle FfcDialectHandle;

// Context structure
struct FfcContext {
    int id;
    int valid;
    int next_type_id;
    int next_value_id;
    int next_block_id;
};

// Location structure
struct FfcLocation {
    FfcContext* context;
    char* filename;
    int line;
    int column;
    int valid;
};

// Module structure
struct FfcModule {
    FfcContext* context;
    FfcOperation* body_op;
    int valid;
};

// Type structure
struct FfcType {
    FfcContext* context;
    TypeKind kind;
    int bit_width;
    int is_signed;
    int rank;
    int64_t* shape;
    FfcType* element_type;
    int valid;
    int id;
};

// Attribute structure
struct FfcAttribute {
    FfcContext* context;
    AttrKind kind;
    int64_t int_value;
    double float_value;
    char* string_value;
    FfcAttribute** array_elements;
    int array_size;
    FfcType* type;
    int valid;
};

// Value structure
struct FfcValue {
    FfcContext* context;
    FfcType* type;
    int id;
    int valid;
};

// Block structure
struct FfcBlock {
    FfcContext* context;
    FfcValue** arguments;
    int num_arguments;
    FfcOperation** operations;
    int num_operations;
    int valid;
    int id;
};

// Region structure
struct FfcRegion {
    FfcContext* context;
    FfcBlock** blocks;
    int num_blocks;
    int valid;
};

// Operation structure
struct FfcOperation {
    FfcContext* context;
    char* name;
    FfcValue** operands;
    int num_operands;
    FfcValue** results;
    int num_results;
    FfcRegion** regions;
    int num_regions;
    FfcAttribute** attributes;
    char** attribute_names;
    int num_attributes;
    FfcLocation* location;
    int valid;
};

// Operation state for building
struct FfcOperationState {
    FfcContext* context;
    char* name;
    FfcLocation* location;
    FfcValue** operands;
    int operands_capacity;
    int num_operands;
    FfcType** result_types;
    int results_capacity;
    int num_results;
    FfcAttribute** attributes;
    char** attribute_names;
    int attributes_capacity;
    int num_attributes;
    FfcRegion** regions;
    int regions_capacity;
    int num_regions;
    int valid;
};

// Pass manager
struct FfcPassManager {
    FfcContext* context;
    int valid;
};

// Dialect handle (simple pointer-based)
struct FfcDialectHandle {
    void* ptr;
};

// Global context counter
static int g_context_counter = 1;

//===----------------------------------------------------------------------===//
// Context API
//===----------------------------------------------------------------------===//

void* ffc_mlirContextCreate() {
    FfcContext* ctx = (FfcContext*)malloc(sizeof(FfcContext));
    if (!ctx) return NULL;

    ctx->id = g_context_counter++;
    ctx->valid = 1;
    ctx->next_type_id = 1;
    ctx->next_value_id = 1;
    ctx->next_block_id = 1;

    return ctx;
}

void ffc_mlirContextDestroy(void* context_ptr) {
    if (context_ptr) {
        FfcContext* ctx = (FfcContext*)context_ptr;
        ctx->valid = 0;
        free(ctx);
    }
}

int ffc_mlirContextIsNull(void* context_ptr) {
    return context_ptr == NULL;
}

//===----------------------------------------------------------------------===//
// Location API
//===----------------------------------------------------------------------===//

void* ffc_mlirLocationUnknownGet(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcLocation* loc = (FfcLocation*)malloc(sizeof(FfcLocation));
    if (!loc) return NULL;

    loc->context = (FfcContext*)context_ptr;
    loc->filename = NULL;
    loc->line = 0;
    loc->column = 0;
    loc->valid = 1;

    return loc;
}

void* ffc_mlirLocationFileLineColGet(void* context_ptr, const char* filename,
                                      int line, int column) {
    if (!context_ptr) return NULL;

    FfcLocation* loc = (FfcLocation*)malloc(sizeof(FfcLocation));
    if (!loc) return NULL;

    loc->context = (FfcContext*)context_ptr;
    loc->filename = filename ? strdup(filename) : NULL;
    loc->line = line;
    loc->column = column;
    loc->valid = 1;

    return loc;
}

int ffc_mlirLocationIsNull(void* location_ptr) {
    return location_ptr == NULL;
}

//===----------------------------------------------------------------------===//
// Module API
//===----------------------------------------------------------------------===//

void* ffc_mlirModuleCreateEmpty(void* location_ptr) {
    if (!location_ptr) return NULL;

    FfcLocation* loc = (FfcLocation*)location_ptr;

    FfcModule* module = (FfcModule*)malloc(sizeof(FfcModule));
    if (!module) return NULL;

    module->context = loc->context;
    module->body_op = NULL;
    module->valid = 1;

    return module;
}

int ffc_mlirModuleIsNull(void* module_ptr) {
    return module_ptr == NULL;
}

void ffc_mlirModuleDestroy(void* module_ptr) {
    if (module_ptr) {
        FfcModule* module = (FfcModule*)module_ptr;
        module->valid = 0;
        free(module);
    }
}

//===----------------------------------------------------------------------===//
// Type System API
//===----------------------------------------------------------------------===//

static FfcType* create_type(FfcContext* ctx, TypeKind kind) {
    FfcType* type = (FfcType*)malloc(sizeof(FfcType));
    if (!type) return NULL;

    type->context = ctx;
    type->kind = kind;
    type->bit_width = 0;
    type->is_signed = 1;
    type->rank = 0;
    type->shape = NULL;
    type->element_type = NULL;
    type->valid = 1;
    type->id = ctx->next_type_id++;

    return type;
}

void* ffc_mlirNoneTypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    return create_type((FfcContext*)context_ptr, TYPE_NONE);
}

void* ffc_mlirIntegerTypeGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_INTEGER);
    if (!type) return NULL;

    type->bit_width = width;
    type->is_signed = 1;
    return type;
}

void* ffc_mlirIntegerTypeSignedGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_INTEGER);
    if (!type) return NULL;

    type->bit_width = width;
    type->is_signed = 1;
    return type;
}

void* ffc_mlirIntegerTypeUnsignedGet(void* context_ptr, int width) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_INTEGER);
    if (!type) return NULL;

    type->bit_width = width;
    type->is_signed = 0;
    return type;
}

void* ffc_mlirF16TypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_FLOAT);
    if (!type) return NULL;

    type->bit_width = 16;
    return type;
}

void* ffc_mlirF32TypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_FLOAT);
    if (!type) return NULL;

    type->bit_width = 32;
    return type;
}

void* ffc_mlirF64TypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_FLOAT);
    if (!type) return NULL;

    type->bit_width = 64;
    return type;
}

void* ffc_mlirIndexTypeGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    return create_type((FfcContext*)context_ptr, TYPE_INDEX);
}

void* ffc_mlirMemRefTypeGet(void* element_type_ptr, long rank,
                             void* shape, void* layout, void* memspace) {
    if (!element_type_ptr) return NULL;

    FfcType* elem_type = (FfcType*)element_type_ptr;
    FfcType* type = create_type(elem_type->context, TYPE_MEMREF);
    if (!type) return NULL;

    type->element_type = elem_type;
    type->rank = (int)rank;

    if (shape && rank > 0) {
        type->shape = (int64_t*)malloc(rank * sizeof(int64_t));
        if (type->shape) {
            memcpy(type->shape, shape, rank * sizeof(int64_t));
        }
    }

    return type;
}

void* ffc_mlirMemRefTypeContiguousGet(void* element_type_ptr,
                                       long rank, void* shape) {
    return ffc_mlirMemRefTypeGet(element_type_ptr, rank, shape, NULL, NULL);
}

void* ffc_mlirRankedTensorTypeGet(long rank, void* shape,
                                   void* element_type_ptr, void* encoding) {
    if (!element_type_ptr) return NULL;

    FfcType* elem_type = (FfcType*)element_type_ptr;
    FfcType* type = create_type(elem_type->context, TYPE_TENSOR);
    if (!type) return NULL;

    type->element_type = elem_type;
    type->rank = (int)rank;

    if (shape && rank > 0) {
        type->shape = (int64_t*)malloc(rank * sizeof(int64_t));
        if (type->shape) {
            memcpy(type->shape, shape, rank * sizeof(int64_t));
        }
    }

    return type;
}

void* ffc_mlirFunctionTypeGet(void* context_ptr, long num_inputs,
                               void* inputs, long num_results, void* results) {
    if (!context_ptr) return NULL;

    FfcType* type = create_type((FfcContext*)context_ptr, TYPE_FUNCTION);
    if (!type) return NULL;

    // Store function signature info
    type->rank = (int)num_inputs;
    type->bit_width = (int)num_results;

    return type;
}

void* ffc_mlirComplexTypeGet(void* element_type_ptr) {
    if (!element_type_ptr) return NULL;

    FfcType* elem_type = (FfcType*)element_type_ptr;
    FfcType* type = create_type(elem_type->context, TYPE_COMPLEX);
    if (!type) return NULL;

    type->element_type = elem_type;
    return type;
}

//===----------------------------------------------------------------------===//
// Type Query API
//===----------------------------------------------------------------------===//

int ffc_mlirTypeIsNull(void* type_ptr) {
    return type_ptr == NULL;
}

int ffc_mlirTypeIsAInteger(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_INTEGER ? 1 : 0;
}

int ffc_mlirTypeIsAFloat(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_FLOAT ? 1 : 0;
}

int ffc_mlirTypeIsAIndex(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_INDEX ? 1 : 0;
}

int ffc_mlirTypeIsAMemRef(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_MEMREF ? 1 : 0;
}

int ffc_mlirTypeIsATensor(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_TENSOR ? 1 : 0;
}

int ffc_mlirTypeIsAFunction(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_FUNCTION ? 1 : 0;
}

int ffc_mlirTypeIsAComplex(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->kind == TYPE_COMPLEX ? 1 : 0;
}

int ffc_mlirIntegerTypeGetWidth(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->bit_width;
}

int ffc_mlirIntegerTypeIsSigned(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->is_signed;
}

int ffc_mlirShapedTypeGetRank(void* type_ptr) {
    if (!type_ptr) return 0;
    FfcType* type = (FfcType*)type_ptr;
    return type->rank;
}

//===----------------------------------------------------------------------===//
// Attribute API
//===----------------------------------------------------------------------===//

static FfcAttribute* create_attribute(FfcContext* ctx, AttrKind kind) {
    FfcAttribute* attr = (FfcAttribute*)malloc(sizeof(FfcAttribute));
    if (!attr) return NULL;

    attr->context = ctx;
    attr->kind = kind;
    attr->int_value = 0;
    attr->float_value = 0.0;
    attr->string_value = NULL;
    attr->array_elements = NULL;
    attr->array_size = 0;
    attr->type = NULL;
    attr->valid = 1;

    return attr;
}

void* ffc_mlirIntegerAttrGet(void* type_ptr, long value) {
    if (!type_ptr) return NULL;

    FfcType* type = (FfcType*)type_ptr;
    FfcAttribute* attr = create_attribute(type->context, ATTR_INTEGER);
    if (!attr) return NULL;

    attr->int_value = value;
    attr->type = type;
    return attr;
}

void* ffc_mlirFloatAttrDoubleGet(void* context_ptr, void* type_ptr, double value) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcAttribute* attr = create_attribute(ctx, ATTR_FLOAT);
    if (!attr) return NULL;

    attr->float_value = value;
    attr->type = (FfcType*)type_ptr;
    return attr;
}

void* ffc_mlirStringAttrGet(void* context_ptr, const char* str, long length) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcAttribute* attr = create_attribute(ctx, ATTR_STRING);
    if (!attr) return NULL;

    if (str) {
        attr->string_value = (char*)malloc(length + 1);
        if (attr->string_value) {
            memcpy(attr->string_value, str, length);
            attr->string_value[length] = '\0';
        }
    }

    return attr;
}

void* ffc_mlirArrayAttrGet(void* context_ptr, long num_elements, void* elements) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcAttribute* attr = create_attribute(ctx, ATTR_ARRAY);
    if (!attr) return NULL;

    attr->array_size = (int)num_elements;
    if (num_elements > 0 && elements) {
        attr->array_elements = (FfcAttribute**)malloc(num_elements * sizeof(FfcAttribute*));
        if (attr->array_elements) {
            memcpy(attr->array_elements, elements, num_elements * sizeof(FfcAttribute*));
        }
    }

    return attr;
}

void* ffc_mlirTypeAttrGet(void* type_ptr) {
    if (!type_ptr) return NULL;

    FfcType* type = (FfcType*)type_ptr;
    FfcAttribute* attr = create_attribute(type->context, ATTR_TYPE);
    if (!attr) return NULL;

    attr->type = type;
    return attr;
}

void* ffc_mlirUnitAttrGet(void* context_ptr) {
    if (!context_ptr) return NULL;
    return create_attribute((FfcContext*)context_ptr, ATTR_UNIT);
}

void* ffc_mlirBoolAttrGet(void* context_ptr, int value) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcAttribute* attr = create_attribute(ctx, ATTR_BOOL);
    if (!attr) return NULL;

    attr->int_value = value ? 1 : 0;
    return attr;
}

void* ffc_mlirSymbolRefAttrGet(void* context_ptr, const char* symbol, long length) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcAttribute* attr = create_attribute(ctx, ATTR_SYMBOL_REF);
    if (!attr) return NULL;

    if (symbol) {
        attr->string_value = (char*)malloc(length + 1);
        if (attr->string_value) {
            memcpy(attr->string_value, symbol, length);
            attr->string_value[length] = '\0';
        }
    }

    return attr;
}

void* ffc_mlirAttributeGetNull() {
    return NULL;
}

//===----------------------------------------------------------------------===//
// Attribute Query API
//===----------------------------------------------------------------------===//

int ffc_mlirAttributeIsNull(void* attr_ptr) {
    return attr_ptr == NULL;
}

int ffc_mlirAttributeIsAInteger(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->kind == ATTR_INTEGER ? 1 : 0;
}

int ffc_mlirAttributeIsAFloat(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->kind == ATTR_FLOAT ? 1 : 0;
}

int ffc_mlirAttributeIsAString(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->kind == ATTR_STRING ? 1 : 0;
}

int ffc_mlirAttributeIsAArray(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->kind == ATTR_ARRAY ? 1 : 0;
}

long ffc_mlirIntegerAttrGetValueInt(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return (long)attr->int_value;
}

double ffc_mlirFloatAttrGetValueDouble(void* attr_ptr) {
    if (!attr_ptr) return 0.0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->float_value;
}

long ffc_mlirArrayAttrGetNumElements(void* attr_ptr) {
    if (!attr_ptr) return 0;
    FfcAttribute* attr = (FfcAttribute*)attr_ptr;
    return attr->array_size;
}

//===----------------------------------------------------------------------===//
// Value API
//===----------------------------------------------------------------------===//

void* ffc_createDummyValue(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcValue* value = (FfcValue*)malloc(sizeof(FfcValue));
    if (!value) return NULL;

    value->context = ctx;
    value->type = NULL;
    value->id = ctx->next_value_id++;
    value->valid = 1;

    return value;
}

int ffc_mlirValueIsNull(void* value_ptr) {
    return value_ptr == NULL;
}

void* ffc_mlirValueGetType(void* value_ptr) {
    if (!value_ptr) return NULL;
    FfcValue* value = (FfcValue*)value_ptr;
    return value->type;
}

//===----------------------------------------------------------------------===//
// Block and Region API
//===----------------------------------------------------------------------===//

void* ffc_mlirBlockCreate(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcBlock* block = (FfcBlock*)malloc(sizeof(FfcBlock));
    if (!block) return NULL;

    block->context = ctx;
    block->arguments = NULL;
    block->num_arguments = 0;
    block->operations = NULL;
    block->num_operations = 0;
    block->valid = 1;
    block->id = ctx->next_block_id++;

    return block;
}

int ffc_mlirBlockIsNull(void* block_ptr) {
    return block_ptr == NULL;
}

void* ffc_mlirRegionCreate(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcContext* ctx = (FfcContext*)context_ptr;
    FfcRegion* region = (FfcRegion*)malloc(sizeof(FfcRegion));
    if (!region) return NULL;

    region->context = ctx;
    region->blocks = NULL;
    region->num_blocks = 0;
    region->valid = 1;

    return region;
}

int ffc_mlirRegionIsNull(void* region_ptr) {
    return region_ptr == NULL;
}

//===----------------------------------------------------------------------===//
// Operation State API
//===----------------------------------------------------------------------===//

void* ffc_mlirOperationStateCreate(const char* name, long name_len, void* location_ptr) {
    if (!location_ptr) return NULL;

    FfcLocation* loc = (FfcLocation*)location_ptr;
    FfcOperationState* state = (FfcOperationState*)malloc(sizeof(FfcOperationState));
    if (!state) return NULL;

    state->context = loc->context;
    state->name = (char*)malloc(name_len + 1);
    if (state->name) {
        memcpy(state->name, name, name_len);
        state->name[name_len] = '\0';
    }
    state->location = loc;

    state->operands = NULL;
    state->operands_capacity = 0;
    state->num_operands = 0;

    state->result_types = NULL;
    state->results_capacity = 0;
    state->num_results = 0;

    state->attributes = NULL;
    state->attribute_names = NULL;
    state->attributes_capacity = 0;
    state->num_attributes = 0;

    state->regions = NULL;
    state->regions_capacity = 0;
    state->num_regions = 0;

    state->valid = 1;

    return state;
}

void ffc_mlirOperationStateAddOperands(void* state_ptr, long n, void* operands) {
    if (!state_ptr || n <= 0 || !operands) return;

    FfcOperationState* state = (FfcOperationState*)state_ptr;

    // Ensure capacity
    int new_capacity = state->num_operands + (int)n;
    if (new_capacity > state->operands_capacity) {
        new_capacity = new_capacity * 2 + 4;
        FfcValue** new_operands = (FfcValue**)realloc(state->operands,
                                                       new_capacity * sizeof(FfcValue*));
        if (!new_operands) return;
        state->operands = new_operands;
        state->operands_capacity = new_capacity;
    }

    FfcValue** values = (FfcValue**)operands;
    for (long i = 0; i < n; i++) {
        state->operands[state->num_operands++] = values[i];
    }
}

void ffc_mlirOperationStateAddResults(void* state_ptr, long n, void* types) {
    if (!state_ptr || n <= 0 || !types) return;

    FfcOperationState* state = (FfcOperationState*)state_ptr;

    // Ensure capacity
    int new_capacity = state->num_results + (int)n;
    if (new_capacity > state->results_capacity) {
        new_capacity = new_capacity * 2 + 4;
        FfcType** new_types = (FfcType**)realloc(state->result_types,
                                                  new_capacity * sizeof(FfcType*));
        if (!new_types) return;
        state->result_types = new_types;
        state->results_capacity = new_capacity;
    }

    FfcType** type_array = (FfcType**)types;
    for (long i = 0; i < n; i++) {
        state->result_types[state->num_results++] = type_array[i];
    }
}

void ffc_mlirOperationStateAddNamedAttribute(void* state_ptr,
                                              const char* name, long name_len,
                                              void* attr_ptr) {
    if (!state_ptr || !name || !attr_ptr) return;

    FfcOperationState* state = (FfcOperationState*)state_ptr;

    // Ensure capacity
    if (state->num_attributes >= state->attributes_capacity) {
        int new_capacity = state->attributes_capacity * 2 + 4;
        FfcAttribute** new_attrs = (FfcAttribute**)realloc(state->attributes,
                                                            new_capacity * sizeof(FfcAttribute*));
        char** new_names = (char**)realloc(state->attribute_names,
                                           new_capacity * sizeof(char*));
        if (!new_attrs || !new_names) return;
        state->attributes = new_attrs;
        state->attribute_names = new_names;
        state->attributes_capacity = new_capacity;
    }

    char* name_copy = (char*)malloc(name_len + 1);
    if (name_copy) {
        memcpy(name_copy, name, name_len);
        name_copy[name_len] = '\0';
    }

    state->attributes[state->num_attributes] = (FfcAttribute*)attr_ptr;
    state->attribute_names[state->num_attributes] = name_copy;
    state->num_attributes++;
}

void ffc_mlirOperationStateAddRegion(void* state_ptr, void* region_ptr) {
    if (!state_ptr || !region_ptr) return;

    FfcOperationState* state = (FfcOperationState*)state_ptr;

    // Ensure capacity
    if (state->num_regions >= state->regions_capacity) {
        int new_capacity = state->regions_capacity * 2 + 4;
        FfcRegion** new_regions = (FfcRegion**)realloc(state->regions,
                                                        new_capacity * sizeof(FfcRegion*));
        if (!new_regions) return;
        state->regions = new_regions;
        state->regions_capacity = new_capacity;
    }

    state->regions[state->num_regions++] = (FfcRegion*)region_ptr;
}

//===----------------------------------------------------------------------===//
// Operation API
//===----------------------------------------------------------------------===//

void* ffc_mlirOperationCreate(void* state_ptr) {
    if (!state_ptr) return NULL;

    FfcOperationState* state = (FfcOperationState*)state_ptr;
    FfcOperation* op = (FfcOperation*)malloc(sizeof(FfcOperation));
    if (!op) return NULL;

    op->context = state->context;
    op->name = state->name ? strdup(state->name) : NULL;
    op->location = state->location;

    // Copy operands
    op->num_operands = state->num_operands;
    if (state->num_operands > 0) {
        op->operands = (FfcValue**)malloc(state->num_operands * sizeof(FfcValue*));
        if (op->operands) {
            memcpy(op->operands, state->operands, state->num_operands * sizeof(FfcValue*));
        }
    } else {
        op->operands = NULL;
    }

    // Create result values
    op->num_results = state->num_results;
    if (state->num_results > 0) {
        op->results = (FfcValue**)malloc(state->num_results * sizeof(FfcValue*));
        if (op->results) {
            for (int i = 0; i < state->num_results; i++) {
                op->results[i] = (FfcValue*)ffc_createDummyValue(state->context);
                if (op->results[i]) {
                    op->results[i]->type = state->result_types[i];
                }
            }
        }
    } else {
        op->results = NULL;
    }

    // Copy attributes
    op->num_attributes = state->num_attributes;
    if (state->num_attributes > 0) {
        op->attributes = (FfcAttribute**)malloc(state->num_attributes * sizeof(FfcAttribute*));
        op->attribute_names = (char**)malloc(state->num_attributes * sizeof(char*));
        if (op->attributes && op->attribute_names) {
            memcpy(op->attributes, state->attributes,
                   state->num_attributes * sizeof(FfcAttribute*));
            for (int i = 0; i < state->num_attributes; i++) {
                op->attribute_names[i] = state->attribute_names[i] ?
                    strdup(state->attribute_names[i]) : NULL;
            }
        }
    } else {
        op->attributes = NULL;
        op->attribute_names = NULL;
    }

    // Copy regions
    op->num_regions = state->num_regions;
    if (state->num_regions > 0) {
        op->regions = (FfcRegion**)malloc(state->num_regions * sizeof(FfcRegion*));
        if (op->regions) {
            memcpy(op->regions, state->regions, state->num_regions * sizeof(FfcRegion*));
        }
    } else {
        op->regions = NULL;
    }

    op->valid = 1;

    return op;
}

void ffc_mlirOperationDestroy(void* op_ptr) {
    if (!op_ptr) return;

    FfcOperation* op = (FfcOperation*)op_ptr;

    if (op->name) free(op->name);
    if (op->operands) free(op->operands);
    if (op->results) {
        for (int i = 0; i < op->num_results; i++) {
            if (op->results[i]) free(op->results[i]);
        }
        free(op->results);
    }
    if (op->attributes) free(op->attributes);
    if (op->attribute_names) {
        for (int i = 0; i < op->num_attributes; i++) {
            if (op->attribute_names[i]) free(op->attribute_names[i]);
        }
        free(op->attribute_names);
    }
    if (op->regions) free(op->regions);

    op->valid = 0;
    free(op);
}

int ffc_mlirOperationIsNull(void* op_ptr) {
    return op_ptr == NULL;
}

int ffc_mlirOperationVerify(void* op_ptr) {
    if (!op_ptr) return 0;
    FfcOperation* op = (FfcOperation*)op_ptr;
    return op->valid;
}

long ffc_mlirOperationGetNumResults(void* op_ptr) {
    if (!op_ptr) return 0;
    FfcOperation* op = (FfcOperation*)op_ptr;
    return op->num_results;
}

void* ffc_mlirOperationGetResult(void* op_ptr, long index) {
    if (!op_ptr) return NULL;
    FfcOperation* op = (FfcOperation*)op_ptr;
    if (index < 0 || index >= op->num_results) return NULL;
    return op->results[index];
}

// Get operation name (returns pointer to name and length via output parameter)
const char* ffc_mlirOperationGetName(void* op_ptr, long* length) {
    if (!op_ptr) {
        if (length) *length = 0;
        return "";
    }
    FfcOperation* op = (FfcOperation*)op_ptr;
    if (op->name) {
        if (length) *length = (long)strlen(op->name);
        return op->name;
    }
    if (length) *length = 0;
    return "";
}

//===----------------------------------------------------------------------===//
// Pass Manager API
//===----------------------------------------------------------------------===//

void* ffc_mlirPassManagerCreate(void* context_ptr) {
    if (!context_ptr) return NULL;

    FfcPassManager* pm = (FfcPassManager*)malloc(sizeof(FfcPassManager));
    if (!pm) return NULL;

    pm->context = (FfcContext*)context_ptr;
    pm->valid = 1;

    return pm;
}

void ffc_mlirPassManagerDestroy(void* pm_ptr) {
    if (pm_ptr) {
        FfcPassManager* pm = (FfcPassManager*)pm_ptr;
        pm->valid = 0;
        free(pm);
    }
}

int ffc_mlirPassManagerRun(void* pm_ptr, void* module_ptr) {
    // Always succeed for now
    return 1;
}

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

static FfcDialectHandle g_fir_handle = {(void*)0x1001};
static FfcDialectHandle g_hlfir_handle = {(void*)0x1002};
static FfcDialectHandle g_func_handle = {(void*)0x1003};
static FfcDialectHandle g_arith_handle = {(void*)0x1004};
static FfcDialectHandle g_scf_handle = {(void*)0x1005};
static FfcDialectHandle g_memref_handle = {(void*)0x1006};
static FfcDialectHandle g_llvm_handle = {(void*)0x1007};

void* ffc_mlirGetDialectHandle__fir__() {
    return &g_fir_handle;
}

void* ffc_mlirGetDialectHandle__hlfir__() {
    return &g_hlfir_handle;
}

void* ffc_mlirGetDialectHandle__func__() {
    return &g_func_handle;
}

void* ffc_mlirGetDialectHandle__arith__() {
    return &g_arith_handle;
}

void* ffc_mlirGetDialectHandle__scf__() {
    return &g_scf_handle;
}

void* ffc_mlirGetDialectHandle__memref__() {
    return &g_memref_handle;
}

void* ffc_mlirGetDialectHandle__llvm__() {
    return &g_llvm_handle;
}

void ffc_mlirDialectHandleRegisterDialect(void* handle, void* context) {
    // No-op for now - dialect registration is simulated
}

int ffc_mlirDialectIsNull(void* dialect_ptr) {
    return dialect_ptr == NULL;
}

void ffc_mlirOperationStateDestroy(void* state_ptr) {
    if (!state_ptr) return;

    FfcOperationState* state = (FfcOperationState*)state_ptr;

    if (state->name) free(state->name);
    if (state->operands) free(state->operands);
    if (state->result_types) free(state->result_types);
    if (state->attributes) free(state->attributes);
    if (state->attribute_names) {
        for (int i = 0; i < state->num_attributes; i++) {
            if (state->attribute_names[i]) free(state->attribute_names[i]);
        }
        free(state->attribute_names);
    }
    if (state->regions) free(state->regions);

    state->valid = 0;
    free(state);
}

//===----------------------------------------------------------------------===//
// Unprefixed MLIR C API compatibility functions
// These provide the standard MLIR C API function names
//===----------------------------------------------------------------------===//

// Type query functions (unprefixed)
int mlirTypeIsAFloat(void* type_ptr) {
    return ffc_mlirTypeIsAFloat(type_ptr);
}

int mlirTypeIsAMemRef(void* type_ptr) {
    return ffc_mlirTypeIsAMemRef(type_ptr);
}

int mlirTypeIsAInteger(void* type_ptr) {
    return ffc_mlirTypeIsAInteger(type_ptr);
}

int mlirIntegerTypeGetWidth(void* type_ptr) {
    return ffc_mlirIntegerTypeGetWidth(type_ptr);
}

void* mlirNoneTypeGet(void* context_ptr) {
    return ffc_mlirNoneTypeGet(context_ptr);
}

// Attribute query functions (unprefixed)
int mlirAttributeIsAInteger(void* attr_ptr) {
    return ffc_mlirAttributeIsAInteger(attr_ptr);
}

int mlirAttributeIsAFloat(void* attr_ptr) {
    return ffc_mlirAttributeIsAFloat(attr_ptr);
}

int mlirAttributeIsAString(void* attr_ptr) {
    return ffc_mlirAttributeIsAString(attr_ptr);
}

int mlirAttributeIsAArray(void* attr_ptr) {
    return ffc_mlirAttributeIsAArray(attr_ptr);
}

long mlirIntegerAttrGetValueInt(void* attr_ptr) {
    return ffc_mlirIntegerAttrGetValueInt(attr_ptr);
}

double mlirFloatAttrGetValueDouble(void* attr_ptr) {
    return ffc_mlirFloatAttrGetValueDouble(attr_ptr);
}

long mlirArrayAttrGetNumElements(void* attr_ptr) {
    return ffc_mlirArrayAttrGetNumElements(attr_ptr);
}

// Attribute creation functions (unprefixed)
// The Fortran code passes string_ref which is a struct with data pointer and length
// We need to handle this differently

typedef struct {
    const char* data;
    size_t length;
} MlirStringRef;

void* mlirFloatAttrDoubleGet(void* context_ptr, void* type_ptr, double value) {
    return ffc_mlirFloatAttrDoubleGet(context_ptr, type_ptr, value);
}

void* mlirStringAttrGet(void* context_ptr, void* string_ref_ptr) {
    if (!context_ptr || !string_ref_ptr) return NULL;
    MlirStringRef* str_ref = (MlirStringRef*)string_ref_ptr;
    return ffc_mlirStringAttrGet(context_ptr, str_ref->data, (long)str_ref->length);
}

void* mlirArrayAttrGet(void* context_ptr, long num_elements, void* elements) {
    return ffc_mlirArrayAttrGet(context_ptr, num_elements, elements);
}

void* mlirTypeAttrGet(void* type_ptr) {
    return ffc_mlirTypeAttrGet(type_ptr);
}
