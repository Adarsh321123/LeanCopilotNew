#include <iostream>
#include <vector>
#include <string>
#include "ct2.cpp"  // Include your original file

#include <ctranslate2/devices.h>
#include <ctranslate2/encoder.h>
#include <ctranslate2/ops/matmul.h>
#include <ctranslate2/ops/topk.h>
#include <ctranslate2/translator.h>
#include <lean/lean.h>

#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <vector>

#include <exception>
#include <iostream>

#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

#include <dlfcn.h>

#include "json.hpp"
#include "npy.hpp"

bool is_lean_string(lean_object* obj) {
    return obj != nullptr && lean_is_string(obj);
}

int main() {
    std::cout << "Starting program" << std::endl;

    void* lean_lib = dlopen("libleanshared.so", RTLD_LAZY);
    if (!lean_lib) {
        std::cerr << "Failed to load Lean library: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "Lean library loaded successfully" << std::endl;

    void (*init_func)() = (void(*)())dlsym(lean_lib, "lean_initialize_runtime_module");
    if (!init_func) {
        std::cerr << "Failed to find lean_initialize_runtime_module: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "lean_initialize_runtime_module found" << std::endl;

    std::cout << "Calling lean_initialize_runtime_module" << std::endl;
    init_func();
    std::cout << "lean_initialize_runtime_module called successfully" << std::endl;

    std::cout << "Loading Lean functions" << std::endl;
    lean_object* (*lean_mk_string_func)(char const*) = (lean_object*(*)(char const*))dlsym(lean_lib, "lean_mk_string");
    if (!lean_mk_string_func) {
        std::cerr << "Failed to load lean_mk_string: " << dlerror() << std::endl;
    } else {
        std::cout << "lean_mk_string loaded successfully" << std::endl;
    }

    lean_object* (*lean_mk_empty_array_func)() = (lean_object*(*)())dlsym(lean_lib, "lean_mk_array");
    if (!lean_mk_empty_array_func) {
        std::cerr << "Failed to load lean_mk_array: " << dlerror() << std::endl;
    } else {
        std::cout << "lean_mk_array loaded successfully" << std::endl;
    }

    lean_object* (*lean_array_push_func)(lean_object*, lean_object*) = (lean_object*(*)(lean_object*, lean_object*))dlsym(lean_lib, "lean_array_push");
    if (!lean_array_push_func) {
        std::cerr << "Failed to load lean_array_push: " << dlerror() << std::endl;
    } else {
        std::cout << "lean_array_push loaded successfully" << std::endl;
    }

    std::cout << "Searching for array-related functions:" << std::endl;
    void* symbol;
    symbol = dlsym(lean_lib, "lean_mk_array");
    if (symbol) std::cout << "Found lean_mk_array" << std::endl;
    symbol = dlsym(lean_lib, "lean_array_size");
    if (symbol) std::cout << "Found lean_array_size" << std::endl;
    symbol = dlsym(lean_lib, "lean_array_get_size");
    if (symbol) std::cout << "Found lean_array_get_size" << std::endl;
    symbol = dlsym(lean_lib, "lean_array_cptr");
    if (symbol) std::cout << "Found lean_array_cptr" << std::endl;

    if (!lean_mk_string_func || !lean_mk_empty_array_func || !lean_array_push_func) {
        std::cerr << "Failed to load all required Lean functions" << std::endl;
        return 1;
    }
    std::cout << "Lean functions loaded successfully" << std::endl;

    const char* name = "ct2-leandojo-lean4-tacgen-byt5-small";
    std::vector<std::string> input_tokens = {"a", "b", "c", ":", "Nat", "⊢", "a", "+", "b", "+", "c", "=", "a", "+", "c", "+", "b"};
    std::vector<std::string> target_prefix_tokens = {};

    std::cout << "Creating _name" << std::endl;
    lean_object* _name = lean_mk_string_func(name);
    if (!is_lean_string(_name)) {
        std::cerr << "Failed to create valid Lean string for _name" << std::endl;
        return 1;
    }
    std::cout << "Created _name" << std::endl;

    std::cout << "Creating _input_tokens" << std::endl;
    lean_object* _input_tokens = lean_mk_array(lean_box(0), lean_box(0));  // Create an empty array
    for (const auto& token : input_tokens) {
        _input_tokens = lean_array_push_func(_input_tokens, lean_mk_string_func(token.c_str()));
    }
    std::cout << "Created _input_tokens" << std::endl;

    std::cout << "Creating _target_prefix_tokens" << std::endl;
    lean_object* _target_prefix_tokens = lean_mk_array(lean_box(0), lean_box(0));  // Create an empty array
    for (const auto& token : target_prefix_tokens) {
        _target_prefix_tokens = lean_array_push_func(_target_prefix_tokens, lean_mk_string_func(token.c_str()));
    }
    std::cout << "Created _target_prefix_tokens" << std::endl;

    std::cout << "Initializing generator model" << std::endl;
    // const char* name = "ct2-leandojo-lean4-tacgen-byt5-small";
    const char* model_path = "/home/adarsh/.cache/lean_copilot/models/huggingface.co/kaiyuy/ct2-leandojo-lean4-tacgen-byt5-small"; // Replace with actual path
    const char* compute_type = "default";
    const char* device = "cpu"; // or "cuda" if you're using GPU
    std::vector<uint64_t> device_index = {0}; // Adjust as needed

    std::cout << "Initializing generator" << std::endl;
    // lean_object* _name = lean_mk_string_func(name);
    lean_object* _model_path = lean_mk_string_func(model_path);
    std::cout << "Created _model_path" << std::endl;
    lean_object* _compute_type = lean_mk_string_func(compute_type);
    std::cout << "Created _compute_type" << std::endl;
    lean_object* _device = lean_mk_string_func(device);
    std::cout << "Created _device" << std::endl;
    lean_object* _device_index = lean_mk_array(lean_box(device_index.size()), lean_box_uint64(0));
    std::cout << "Device index size: " << device_index.size() << std::endl;
    for (size_t i = 0; i < device_index.size(); ++i) {
        _device_index = lean_array_push_func(_device_index, lean_box_uint64(device_index[i]));
    }
    std::cout << "Created _device_index" << std::endl;

    // TODO: make sure lean call uses same args as this
    uint8_t init_result = init_generator(_name, _model_path, _compute_type, _device, _device_index);
    if (init_result == 0) {
        std::cerr << "Failed to initialize generator" << std::endl;
        return 1;
    }
    std::cout << "Generator initialized successfully" << std::endl;

    std::cout << "Calling generate function" << std::endl;
    lean_object* result = generate(_name, _input_tokens, _target_prefix_tokens, 
                                   1, 5, 1, 100, 1.0, 1.0, 1.0);

    if (result != nullptr) {
        std::cout << "Generate function completed successfully" << std::endl;
    } else {
        std::cout << "Generate function failed" << std::endl;
    }

    std::cout << "Program completed" << std::endl;
    return 0;
}



// #include <iostream>
// #include <vector>
// #include <string>
// #include "ct2.cpp"
// #include <lean/lean.h>
// #include <dlfcn.h>

// bool is_lean_string(lean_object* obj) {
//     return obj != nullptr && lean_is_string(obj);
// }

// int main() {
//     std::cout << "Starting program" << std::endl;

//     void* lean_lib = dlopen("libleanshared.so", RTLD_LAZY);
//     if (!lean_lib) {
//         std::cerr << "Failed to load Lean library: " << dlerror() << std::endl;
//         return 1;
//     }
//     std::cout << "Lean library loaded successfully" << std::endl;

//     void (*init_func)() = (void(*)())dlsym(lean_lib, "lean_initialize_runtime_module");
//     if (!init_func) {
//         std::cerr << "Failed to find lean_initialize_runtime_module: " << dlerror() << std::endl;
//         return 1;
//     }
//     init_func();
//     std::cout << "lean_initialize_runtime_module called successfully" << std::endl;

//     lean_object* (*lean_mk_string_func)(char const*) = (lean_object*(*)(char const*))dlsym(lean_lib, "lean_mk_string");
//     lean_object* (*lean_mk_empty_array_func)() = (lean_object*(*)())dlsym(lean_lib, "lean_mk_array");
//     lean_object* (*lean_array_push_func)(lean_object*, lean_object*) = (lean_object*(*)(lean_object*, lean_object*))dlsym(lean_lib, "lean_array_push");

//     std::cout << "Searching for array-related functions:" << std::endl;

//     if (!lean_mk_string_func || !lean_mk_empty_array_func || !lean_array_push_func) {
//         std::cerr << "Failed to load all required Lean functions" << std::endl;
//         return 1;
//     }

//     std::cout << "Lean functions loaded successfully" << std::endl;

//     const char* name = "ct2-leandojo-lean4-tacgen-byt5-small";
//     const char* model_path = "/home/adarsh/.cache/lean_copilot/models/huggingface.co/kaiyuy/ct2-leandojo-lean4-tacgen-byt5-small";
//     const char* compute_type = "default";
//     const char* device = "cpu";
//     std::vector<uint64_t> device_index = {0};

//     std::cout << "Creating _name" << std::endl;

//     lean_object* _name = lean_mk_string_func(name);
//     std::cout << "Created _name" << std::endl;
//     std::cout << "Creating _model_path" << std::endl;
//     lean_object* _model_path = lean_mk_string_func(model_path);
//     std::cout << "Created _model_path" << std::endl;
//     std::cout << "Creating _compute_type" << std::endl;
//     lean_object* _compute_type = lean_mk_string_func(compute_type);
//     std::cout << "Created _compute_type" << std::endl;
//     std::cout << "Creating _device" << std::endl;
//     lean_object* _device = lean_mk_string_func(device);
//     std::cout << "Created _device" << std::endl;
//     std::cout << "Creating _device_index" << std::endl;
//     lean_object* _device_index = lean_mk_empty_array_func();
//     std::cout << "Device index size: " << device_index.size() << std::endl;
//     for (size_t i = 0; i < device_index.size(); ++i) {
//         _device_index = lean_array_push_func(_device_index, lean_box_uint64(device_index[i]));
//     }
//     std::cout << "Created _device_index" << std::endl;

//     std::cout << "Initializing generator" << std::endl;
//     uint8_t init_result = init_generator(_name, _model_path, _compute_type, _device, _device_index);
//     if (init_result == 0) {
//         std::cerr << "Failed to initialize generator" << std::endl;
//         return 1;
//     }
//     std::cout << "Generator initialized successfully" << std::endl;

//     std::vector<std::string> input_tokens = {"test", "input"};
//     std::vector<std::string> target_prefix_tokens = {};

//     lean_object* _input_tokens = lean_mk_empty_array_func();
//     for (const auto& token : input_tokens) {
//         _input_tokens = lean_array_push_func(_input_tokens, lean_mk_string_func(token.c_str()));
//     }

//     lean_object* _target_prefix_tokens = lean_mk_empty_array_func();
//     for (const auto& token : target_prefix_tokens) {
//         _target_prefix_tokens = lean_array_push_func(_target_prefix_tokens, lean_mk_string_func(token.c_str()));
//     }

//     std::cout << "Calling generate function" << std::endl;
//     try {
//         // lean_object* result = generate(_name, _input_tokens, _target_prefix_tokens, 
//         //                                1, 5, 1, 100, 1.0, 1.0, 1.0);
//         lean_object* result = generate(_name, _input_tokens, _target_prefix_tokens, 
//                                1, 1, 1, 10, 1.0, 1.0, 1.0);
//         if (result == nullptr) {
//             std::cout << "Generate function returned null" << std::endl;
//         } else {
//             std::cout << "Generate function completed successfully" << std::endl;
//             // Here you would need to parse the Lean object to extract the results
//             // This depends on the exact structure of the returned Lean object
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "Exception caught in generate: " << e.what() << std::endl;
//     } catch (...) {
//         std::cerr << "Unknown exception caught in generate" << std::endl;
//     }

//     std::cout << "Program completed" << std::endl;
//     return 0;
// }