#import "ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <atomic>
#include <cstring>

// ============================================================================
// Private API class references (resolved at runtime via dlopen)
// ============================================================================

static Class g_ANECInMemoryModelDescriptor = nil;
static Class g_ANECInMemoryModel = nil;
static Class g_ANECRequest = nil;
static Class g_ANECIOSurfaceObject = nil;

static bool g_ane_initialized = false;
static bool g_ane_available = false;
static std::atomic<int> g_compile_count{0};

// ============================================================================
// Opaque handle wrapping the compiled+loaded ANE model
// ============================================================================

struct ANEModelHandle {
    void *model; // _ANEInMemoryModel retained via __bridge_retained
};

// ============================================================================
// Initialization
// ============================================================================

int ane_init(void) {
    static dispatch_once_t once;
    __block int result = 0;

    dispatch_once(&once, ^{
      void *fw = dlopen("/System/Library/PrivateFrameworks/"
                        "AppleNeuralEngine.framework/AppleNeuralEngine",
                        RTLD_NOW);
      if (!fw) {
          NSLog(@"[Axiom ANE] Failed to load AppleNeuralEngine.framework: %s",
                dlerror());
          result = -1;
          return;
      }

      g_ANECInMemoryModelDescriptor =
          objc_getClass("_ANEInMemoryModelDescriptor");
      g_ANECInMemoryModel = objc_getClass("_ANEInMemoryModel");
      g_ANECRequest = objc_getClass("_ANERequest");
      g_ANECIOSurfaceObject = objc_getClass("_ANEIOSurfaceObject");

      if (!g_ANECInMemoryModelDescriptor || !g_ANECInMemoryModel) {
          NSLog(@"[Axiom ANE] Failed to resolve private classes. "
                 "Requires macOS 15+ on Apple Silicon.");
          result = -1;
          return;
      }

      g_ane_available = true;
      g_ane_initialized = true;
      NSLog(@"[Axiom ANE] Bridge initialized successfully");
    });

    return result;
}

bool ane_is_available(void) {
    if (!g_ane_initialized) {
        ane_init();
    }
    return g_ane_available;
}

// ============================================================================
// Weight blob construction
// ============================================================================

void *ane_build_weight_blob(const void *data, size_t data_size,
                            size_t *out_total_size) {
    // ANE weight blob format:
    // Bytes 0-127: header
    //   [0] = 0x01 (marker)
    //   [4] = 0x02 (format)
    //   [64-67] = 0xDEADBEEF (magic, little-endian)
    //   [68] = 0x01 (version)
    //   [72-75] = data_size (uint32, data size in bytes)
    //   [80-83] = 128 (uint32, data offset)
    // Bytes 128+: raw fp16 data
    size_t total = 128 + data_size;
    auto *buf = static_cast<uint8_t *>(calloc(1, total));
    if (!buf) {
        return nullptr;
    }

    buf[0] = 0x01;
    buf[4] = 0x02;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 0x01;

    auto data_size_32 = static_cast<uint32_t>(data_size);
    std::memcpy(buf + 72, &data_size_32, sizeof(uint32_t));

    uint32_t offset = 128;
    std::memcpy(buf + 80, &offset, sizeof(uint32_t));

    if (data && data_size > 0) {
        std::memcpy(buf + 128, data, data_size);
    }

    if (out_total_size) {
        *out_total_size = total;
    }
    return buf;
}

// ============================================================================
// Compilation
// ============================================================================

ANEModelHandle *ane_compile_with_weights(const char *mil_text,
                                          const ANEWeightEntry *weights,
                                          int num_weights) {
    if (!g_ane_available) {
        NSLog(@"[Axiom ANE] ANE not available");
        return nullptr;
    }

    int count = g_compile_count.fetch_add(1) + 1;
    if (count >= ANE_COMPILE_BUDGET_LIMIT) {
        NSLog(@"[Axiom ANE] Compile budget exhausted (%d/%d)", count,
              ANE_COMPILE_BUDGET_LIMIT);
    } else if (count >= ANE_COMPILE_BUDGET_WARNING) {
        NSLog(@"[Axiom ANE] Compile budget warning: %d/%d", count,
              ANE_COMPILE_BUDGET_LIMIT);
    }

    @try {
        // Convert MIL text to NSData (UTF-8)
        NSData *milData = [NSData dataWithBytes:mil_text
                                         length:strlen(mil_text)];

        // Build weight dictionary: path → {offset: 0, data: NSData}
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < num_weights; i++) {
            NSString *path = [NSString
                stringWithFormat:@"@model_path/weights/%s.bin",
                                 weights[i].name];
            NSData *blobData = [NSData dataWithBytes:weights[i].blob_data
                                              length:weights[i].blob_size];
            wdict[path] = @{@"offset" : @0, @"data" : blobData};
        }

        // Create model descriptor from MIL data + weights
        SEL createSel =
            sel_registerName("modelWithMILText:weights:optionsPlist:");

        id descriptor = ((id(*)(id, SEL, id, id, id))objc_msgSend)(
            (id)g_ANECInMemoryModelDescriptor, createSel, milData, wdict,
            (id)nil);
        if (!descriptor) {
            NSLog(@"[Axiom ANE] Failed to create model descriptor from MIL");
            return nullptr;
        }

        // Create in-memory model from descriptor
        SEL modelSel =
            sel_registerName("inMemoryModelWithDescriptor:");
        id model = ((id(*)(id, SEL, id))objc_msgSend)(
            (id)g_ANECInMemoryModel, modelSel, descriptor);
        if (!model) {
            NSLog(@"[Axiom ANE] Failed to create in-memory model");
            return nullptr;
        }

        // Get the hex identifier for temp directory setup
        SEL hexIdSel = sel_registerName("hexStringIdentifier");
        NSString *hexId =
            ((NSString * (*)(id, SEL)) objc_msgSend)(model, hexIdSel);

        if (hexId) {
            // Create temp directory structure for weight files
            NSString *tmpDir = [NSTemporaryDirectory()
                stringByAppendingPathComponent:hexId];
            NSString *weightsDir =
                [tmpDir stringByAppendingPathComponent:@"weights"];
            [[NSFileManager defaultManager] createDirectoryAtPath:weightsDir
                                      withIntermediateDirectories:YES
                                                       attributes:nil
                                                            error:nil];

            // Write model.mil
            NSString *milPath =
                [tmpDir stringByAppendingPathComponent:@"model.mil"];
            [milData writeToFile:milPath atomically:YES];

            // Write weight files
            for (int i = 0; i < num_weights; i++) {
                NSString *wpath = [weightsDir stringByAppendingPathComponent:
                    [NSString stringWithFormat:@"%s.bin", weights[i].name]];
                NSData *blobData =
                    [NSData dataWithBytes:weights[i].blob_data
                                   length:weights[i].blob_size];
                [blobData writeToFile:wpath atomically:YES];
            }
        }

        // Compile
        SEL compileSel = sel_registerName("compileWithQoS:options:error:");
        NSError *error = nil;

        BOOL compileOk =
            ((BOOL(*)(id, SEL, int, id,
                      NSError *__autoreleasing *))objc_msgSend)(
                model, compileSel, 21, @{}, &error);
        if (!compileOk || error) {
            NSLog(@"[Axiom ANE] Compile failed: %@",
                  error ? [error localizedDescription] : @"unknown error");
            return nullptr;
        }

        auto *handle = static_cast<ANEModelHandle *>(
            calloc(1, sizeof(ANEModelHandle)));
        if (!handle) {
            return nullptr;
        }

        handle->model = (__bridge_retained void *)model;

        NSLog(@"[Axiom ANE] Model compiled successfully (compile #%d)", count);
        return handle;

    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Compile exception: %@", exception);
        return nullptr;
    }
}

ANEModelHandle *ane_compile(const char *mil_text) {
    return ane_compile_with_weights(mil_text, nullptr, 0);
}

// ============================================================================
// Loading
// ============================================================================

int ane_load(ANEModelHandle *handle) {
    if (!handle || !handle->model) {
        return -1;
    }

    @try {
        id model = (__bridge id)handle->model;

        SEL loadSel = sel_registerName("loadWithQoS:options:error:");
        NSError *error = nil;

        BOOL loadOk =
            ((BOOL(*)(id, SEL, int, id,
                      NSError *__autoreleasing *))objc_msgSend)(
                model, loadSel, 21, @{}, &error);
        if (!loadOk || error) {
            NSLog(@"[Axiom ANE] Load failed: %@",
                  error ? [error localizedDescription] : @"unknown error");
            return -1;
        }

        return 0;

    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Load exception: %@", exception);
        return -1;
    }
}

// ============================================================================
// Evaluation
// ============================================================================

int ane_eval(ANEModelHandle *handle, IOSurfaceRef input_surface,
             IOSurfaceRef output_surface) {
    if (!handle || !handle->model || !input_surface || !output_surface) {
        return -1;
    }

    @try {
        id model = (__bridge id)handle->model;

        // Wrap IOSurfaces using class factory method (not init)
        SEL wrapSel = sel_registerName("objectWithIOSurface:");

        id inputObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANECIOSurfaceObject, wrapSel, input_surface);
        id outputObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            g_ANECIOSurfaceObject, wrapSel, output_surface);

        if (!inputObj || !outputObj) {
            NSLog(@"[Axiom ANE] Failed to wrap IOSurface objects");
            return -1;
        }

        // Create request using full class factory method
        SEL requestSel = sel_registerName(
            "requestWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:");

        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))
                          objc_msgSend)(
            g_ANECRequest, requestSel,
            @[ inputObj ],  // inputs
            @[ @0 ],        // inputIndices
            @[ outputObj ], // outputs
            @[ @0 ],        // outputIndices
            (id)nil,        // weightsBuffer
            (id)nil,        // perfStats
            @0);            // procedureIndex
        if (!request) {
            NSLog(@"[Axiom ANE] Failed to create request");
            return -1;
        }

        // Evaluate: -[_ANEInMemoryModel evaluateWithQoS:options:request:error:]
        SEL evalSel =
            sel_registerName("evaluateWithQoS:options:request:error:");
        NSError *error = nil;

        BOOL evalOk =
            ((BOOL(*)(id, SEL, unsigned int, id, id,
                      NSError *__autoreleasing *))objc_msgSend)(
                model, evalSel, 21u, @{}, request, &error);
        if (!evalOk || error) {
            NSLog(@"[Axiom ANE] Eval failed: %@",
                  error ? [error localizedDescription] : @"unknown error");
            return -1;
        }

        return 0;

    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Eval exception: %@", exception);
        return -1;
    }
}

// ============================================================================
// Cleanup
// ============================================================================

void ane_release(ANEModelHandle *handle) {
    if (!handle) {
        return;
    }

    @try {
        if (handle->model) {
            id model = (__bridge_transfer id)(handle->model);

            // unloadWithQoS:error: (2 params, not 3)
            SEL unloadSel = sel_registerName("unloadWithQoS:error:");
            NSError *error = nil;
            ((BOOL(*)(id, SEL, unsigned int,
                      NSError *__autoreleasing *))objc_msgSend)(
                model, unloadSel, 21u, &error);
            // model released by ARC via __bridge_transfer
        }
    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Release exception: %@", exception);
    }

    free(handle);
}

int ane_compile_count(void) { return g_compile_count.load(); }

bool ane_can_execute(void) {
    static int result = -1; // -1 = untested, 0 = no, 1 = yes
    if (result >= 0)
        return result == 1;

    if (!ane_is_available()) {
        result = 0;
        return false;
    }

    // Try compiling and evaluating a trivial MIL program
    const char *mil =
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, 1, 1, 1]> x) {\n"
        "        tensor<fp16, [1, 1, 1, 1]> y = relu(x=x)"
        "[name=string(\"r\")];\n"
        "    } -> (y);\n"
        "}\n";

    ANEModelHandle *handle = ane_compile(mil);
    if (!handle) {
        result = 0;
        return false;
    }

    int rc = ane_load(handle);
    ane_release(handle);

    result = (rc == 0) ? 1 : 0;
    return result == 1;
}
