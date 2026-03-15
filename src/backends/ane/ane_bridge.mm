#import "ane_bridge.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <os/log.h>

#include <atomic>

// ============================================================================
// Private API class references (resolved at runtime via dlopen)
// ============================================================================

static Class g_ANECInMemoryModelDescriptor = nil;
static Class g_ANECInMemoryModel = nil;
static Class g_ANECRequest = nil;
static Class g_ANECIOSurfaceObject = nil;
static Class g_ANECClient = nil;

static bool g_ane_initialized = false;
static bool g_ane_available = false;
static std::atomic<int> g_compile_count{0};

static os_log_t g_ane_log = nullptr;

// ============================================================================
// Opaque handle wrapping the compiled+loaded ANE model
// ============================================================================

struct ANEModelHandle {
    void *model;      // id retained via __bridge_retained
    void *descriptor; // id retained via __bridge_retained
};

// ============================================================================
// Initialization
// ============================================================================

int ane_init(void) {
    static dispatch_once_t once;
    __block int result = 0;

    dispatch_once(&once, ^{
      g_ane_log = os_log_create("com.axiom.ane", "bridge");

      // Load the private ANE framework
      void *fw = dlopen("/System/Library/PrivateFrameworks/"
                        "AppleNeuralEngine.framework/AppleNeuralEngine",
                        RTLD_NOW);
      if (!fw) {
          NSLog(@"[Axiom ANE] Failed to load AppleNeuralEngine.framework: %s",
                dlerror());
          result = -1;
          return;
      }

      // Resolve private classes
      g_ANECInMemoryModelDescriptor =
          objc_getClass("_ANEInMemoryModelDescriptor");
      g_ANECInMemoryModel = objc_getClass("_ANEInMemoryModel");
      g_ANECRequest = objc_getClass("_ANERequest");
      g_ANECIOSurfaceObject = objc_getClass("_ANEIOSurfaceObject");
      g_ANECClient = objc_getClass("_ANEClient");

      if (!g_ANECInMemoryModelDescriptor || !g_ANECInMemoryModel) {
          NSLog(@"[Axiom ANE] Failed to resolve ANE private classes. "
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
// Compilation
// ============================================================================

ANEModelHandle *ane_compile(const char *mil_text) {
    if (!g_ane_available) {
        NSLog(@"[Axiom ANE] ANE not available");
        return nullptr;
    }

    int count = g_compile_count.fetch_add(1) + 1;
    if (count >= ANE_COMPILE_BUDGET_LIMIT) {
        NSLog(@"[Axiom ANE] Compile budget exhausted (%d/%d). "
               "Resource leaks likely.",
              count, ANE_COMPILE_BUDGET_LIMIT);
    } else if (count >= ANE_COMPILE_BUDGET_WARNING) {
        NSLog(@"[Axiom ANE] Compile budget warning: %d/%d compilations used",
              count, ANE_COMPILE_BUDGET_LIMIT);
    }

    @try {
        NSString *milString = [NSString stringWithUTF8String:mil_text];
        if (!milString) {
            NSLog(@"[Axiom ANE] Invalid MIL text (not valid UTF-8)");
            return nullptr;
        }

        // Create model descriptor from MIL text
        SEL createSel =
            sel_registerName("modelWithMILText:weights:optionsPlist:");

        NSData *weightsData = [NSData data];
        NSDictionary *options = @{};

        id descriptor =
            ((id(*)(id, SEL, id, id, id))objc_msgSend)(
                (id)g_ANECInMemoryModelDescriptor, createSel, milString,
                weightsData, options);
        if (!descriptor) {
            NSLog(@"[Axiom ANE] Failed to create model descriptor from MIL");
            return nullptr;
        }

        // Compile the model
        SEL compileSel = sel_registerName("compileWithQoS:options:error:");
        NSError *error = nil;
        NSDictionary *compileOptions = @{};

        BOOL compileOk =
            ((BOOL(*)(id, SEL, int, id,
                      NSError *__autoreleasing *))objc_msgSend)(
                descriptor, compileSel, 0x19, compileOptions, &error);
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

        // Retain the descriptor (it becomes the model after compilation)
        handle->descriptor = (__bridge_retained void *)descriptor;
        handle->model = handle->descriptor; // Same object

        NSLog(@"[Axiom ANE] Model compiled successfully (compile #%d)", count);
        return handle;

    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Compile exception: %@", exception);
        return nullptr;
    }
}

// ============================================================================
// Loading
// ============================================================================

int ane_load(ANEModelHandle *handle) {
    if (!handle || !handle->descriptor) {
        return -1;
    }

    @try {
        id descriptor = (__bridge id)handle->descriptor;

        SEL loadSel = sel_registerName("loadWithQoS:options:error:");
        NSError *error = nil;
        NSDictionary *loadOptions = @{};

        BOOL loadOk =
            ((BOOL(*)(id, SEL, int, id,
                      NSError *__autoreleasing *))objc_msgSend)(
                descriptor, loadSel, 0x19, loadOptions, &error);
        if (!loadOk || error) {
            NSLog(@"[Axiom ANE] Load failed: %@",
                  error ? [error localizedDescription] : @"unknown error");
            return -1;
        }

        NSLog(@"[Axiom ANE] Model loaded onto hardware");
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

        // Create _ANEIOSurfaceObject wrappers
        SEL ioSurfaceInitSel = sel_registerName("initWithIOSurface:");

        id inputObj = ((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(
            [g_ANECIOSurfaceObject alloc], ioSurfaceInitSel, input_surface);
        id outputObj = ((id(*)(id, SEL, IOSurfaceRef))objc_msgSend)(
            [g_ANECIOSurfaceObject alloc], ioSurfaceInitSel, output_surface);

        if (!inputObj || !outputObj) {
            NSLog(@"[Axiom ANE] Failed to create IOSurface objects");
            return -1;
        }

        // Create request
        SEL requestInitSel = sel_registerName("initWithInputs:outputs:");
        NSArray *inputs = @[ inputObj ];
        NSArray *outputs = @[ outputObj ];

        id request = ((id(*)(id, SEL, id, id))objc_msgSend)(
            [g_ANECRequest alloc], requestInitSel, inputs, outputs);
        if (!request) {
            NSLog(@"[Axiom ANE] Failed to create request");
            return -1;
        }

        // Evaluate
        SEL evalSel = sel_registerName("evaluateWithRequest:qos:error:");
        NSError *error = nil;

        BOOL evalOk =
            ((BOOL(*)(id, SEL, id, int,
                      NSError *__autoreleasing *))objc_msgSend)(
                model, evalSel, request, 0x19, &error);
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

            // Try to unload from hardware
            SEL unloadSel = sel_registerName("unloadWithQoS:options:error:");
            NSError *error = nil;
            ((BOOL(*)(id, SEL, int, id, NSError *__autoreleasing *))
                 objc_msgSend)(model, unloadSel, 0x19, @{}, &error);
            // model released by ARC via __bridge_transfer
        }

        // If descriptor is a different object (shouldn't be with current
        // implementation, but guard for future changes)
        if (handle->descriptor && handle->descriptor != handle->model) {
            id desc = (__bridge_transfer id)(handle->descriptor);
            (void)desc; // Released by ARC
        }
    } @catch (NSException *exception) {
        NSLog(@"[Axiom ANE] Release exception: %@", exception);
    }

    free(handle);
}

int ane_compile_count(void) { return g_compile_count.load(); }
