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
#include <backtrace.h>
#include <execinfo.h>
#include <unistd.h>

#include "json.hpp"
#include "npy.hpp"

using json = nlohmann::json;

std::map<std::string, std::unique_ptr<ctranslate2::Translator>> generators;
std::map<std::string, std::unique_ptr<ctranslate2::Encoder>> encoders;

ctranslate2::StorageView *p_premise_embeddings = nullptr;
json *p_premise_dictionary = nullptr;

inline bool exists(const std::string &path) {
  std::ifstream f(path.c_str());
  return f.good();
}

inline lean_obj_res lean_mk_pair(lean_obj_arg a, lean_obj_arg b) {
  lean_object *r = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(r, 0, a);
  lean_ctor_set(r, 1, b);
  return r;
}

extern "C" uint8_t cuda_available(b_lean_obj_arg) {
  return ctranslate2::str_to_device("auto") == ctranslate2::Device::CUDA;
}

template <typename T>
bool is_initialized_aux(const std::string &name);

template <>
bool is_initialized_aux<ctranslate2::Translator>(const std::string &name) {
  return generators.find(name) != generators.end();
}

template <>
bool is_initialized_aux<ctranslate2::Encoder>(const std::string &name) {
  return encoders.find(name) != encoders.end();
}

extern "C" uint8_t is_generator_initialized(b_lean_obj_arg _name) {
  std::string name = std::string(lean_string_cstr(_name));
  return is_initialized_aux<ctranslate2::Translator>(name);
}

extern "C" uint8_t is_encoder_initialized(b_lean_obj_arg _name) {
  std::string name = std::string(lean_string_cstr(_name));
  return is_initialized_aux<ctranslate2::Encoder>(name);
}

template <typename T>
bool init_model(b_lean_obj_arg _name,          // String
                b_lean_obj_arg _model_path,    // String
                b_lean_obj_arg _compute_type,  // String
                b_lean_obj_arg _device,        // String
                b_lean_obj_arg _device_index,  // Array UInt64
                std::map<std::string, std::unique_ptr<T>> &models) {
  std::cout << "inside init_model" << std::endl;
  std::string name = std::string(lean_string_cstr(_name));
  std::cout << "name: " << name << std::endl;
  if (is_initialized_aux<T>(name)) {
    std::cout << "name already exists" << std::endl;
    throw std::runtime_error(name + " already exists.");
  }

  std::string model_path = std::string(lean_string_cstr(_model_path));
  std::cout << "model_path: " << model_path << std::endl;
  if (!exists(model_path)) {  // Cannot find the model.
    std::cout << "model_path does not exist" << std::endl;
    return false;
  }

  ctranslate2::Device device =
      ctranslate2::str_to_device(lean_string_cstr(_device));

  std::cout << "device: " << lean_string_cstr(_device) << std::endl;

  ctranslate2::ComputeType compute_type =
      ctranslate2::str_to_compute_type(lean_string_cstr(_compute_type));

  
  std::cout << "compute_type: " << lean_string_cstr(_compute_type) << std::endl;

  std::vector<int> device_indices;
  const lean_array_object *p_arr = lean_to_array(_device_index);
  std::cout << "device_index size: " << p_arr->m_size << std::endl;
  for (int i = 0; i < p_arr->m_size; i++) {
    std::cout << "pushing device index" << std::endl;
    std::cout << "inside stuff: " << p_arr->m_data[i] << std::endl;
    uint64_t test = 0;
    lean_object* zero = lean_box_uint64(test);
    std::cout << "testing unbox: " << lean_unbox_uint64(zero) << std::endl;
    std::cout << lean_unbox_uint64(p_arr->m_data[i]) << std::endl;
    device_indices.push_back(lean_unbox_uint64(p_arr->m_data[i]));
  }

  std::cout << "pushed all device indices" << std::endl;

  auto p_model =
      std::make_unique<T>(model_path, device, compute_type, device_indices);

  std::cout << "created model" << std::endl;
  models.emplace(name, std::move(p_model));

  std::cout << "inserted model" << std::endl;
  return true;
}

extern "C" uint8_t init_generator(
    b_lean_obj_arg _name,            // String
    b_lean_obj_arg _model_path,      // String
    b_lean_obj_arg _compute_type,    // String
    b_lean_obj_arg _device,          // String
    b_lean_obj_arg _device_index) {  // Array UInt64
  std::cout << "inside init_generator" << std::endl;
  return init_model(_name, _model_path, _compute_type, _device, _device_index,
                    generators);
}

extern "C" uint8_t init_encoder(b_lean_obj_arg _name,            // String
                                b_lean_obj_arg _model_path,      // String
                                b_lean_obj_arg _compute_type,    // String
                                b_lean_obj_arg _device,          // String
                                b_lean_obj_arg _device_index) {  // Array UInt64
  return init_model(_name, _model_path, _compute_type, _device, _device_index,
                    encoders);
}

inline std::vector<std::string> convert_tokens(b_lean_obj_arg _tokens) {
  std::vector<std::string> tokens;
  const lean_array_object *p_arr = lean_to_array(_tokens);
  for (int i = 0; i < p_arr->m_size; i++) {
    tokens.emplace_back(lean_string_cstr(p_arr->m_data[i]));
  }
  return tokens;
}

// void custom_terminate() {
//     std::cerr << "terminate called" << std::endl;
//     abort();
// }

void signal_handler(int signum) {
    std::cerr << "NEW: Caught signal " << signum << std::endl;
    void *array[10];
    size_t size = backtrace(array, 10);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(signum);
}

void segfault_handler(int sig) {
    void *array[50];
    size_t size = backtrace(array, 50);
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

extern "C" lean_obj_res generate(
    b_lean_obj_arg _name,                  // String
    b_lean_obj_arg _input_tokens,          // Array String
    b_lean_obj_arg _target_prefix_tokens,  // Array String
    uint64_t num_return_sequences,         // UInt64
    uint64_t beam_size,                    // UInt64
    uint64_t min_length,                   // UInt64
    uint64_t max_length,                   // UInt64
    double length_penalty,                 // Float
    double patience,                       // Float
    double temperature) {                  // Float
  // Check the arguments.
  std::cout << "inside generate function" << std::endl;
  std::ofstream debug_log("lean_ffi_debug.log", std::ios_base::app);
  // TODO: uncomment all of below to bring back code to test
  // try {
  //   // std::set_terminate([]() {
  //   //   try {
  //   //     std::exception_ptr current_exception = std::current_exception();
  //   //     if (current_exception) {
  //   //       std::rethrow_exception(current_exception);
  //   //     }
  //   //   } catch (const std::exception& e) {
  //   //     std::cerr << "Unhandled exception in thread: " << e.what() << std::endl;
  //   //   } catch (...) {
  //   //     std::cerr << "Unknown exception caught" << std::endl;
  //   //   }
  //   //   std::abort();
  //   // });
  //   // std::set_terminate(custom_terminate);
  //   signal(SIGSEGV, segfault_handler);
  //   signal(SIGABRT, signal_handler);
  //   debug_log << "Entered generate function" << std::endl;
  //   std::cout << "Entered generate function" << std::endl;

  //   // Log input parameters
  //   debug_log << "num_return_sequences: " << num_return_sequences << std::endl;
  //   std::cout << "num_return_sequences: " << num_return_sequences << std::endl;
  //   debug_log << "beam_size: " << beam_size << std::endl;
  //   std::cout << "beam_size: " << beam_size << std::endl;
  //   debug_log << "min_length: " << min_length << std::endl;
  //   std::cout << "min_length: " << min_length << std::endl;
  //   debug_log << "max_length: " << max_length << std::endl;
  //   std::cout << "max_length: " << max_length << std::endl;
  //   debug_log << "length_penalty: " << length_penalty << std::endl;
  //   std::cout << "length_penalty: " << length_penalty << std::endl;
  //   debug_log << "patience: " << patience << std::endl;
  //   std::cout << "patience: " << patience << std::endl;
  //   debug_log << "temperature: " << temperature << std::endl;
  //   std::cout << "temperature: " << temperature << std::endl;
  //   std::string name = std::string(lean_string_cstr(_name));
  //   std::cout << "name: " << name << std::endl;
  //   if (!is_initialized_aux<ctranslate2::Translator>(name)) {
  //     // throw std::runtime_error(name + " hasn't been initialized.");
  //     debug_log << "Model not initialized" << std::endl;
  //     std::cout << "Model not initialized" << std::endl;
  //   }
  //   if (num_return_sequences <= 0) {
  //     std::cout << "num_return_sequences must be positive." << std::endl;
  //     throw std::invalid_argument("num_return_sequences must be positive.");
  //   }
  //   if (beam_size <= 0) {
  //     std::cout << "beam_size must be positive." << std::endl;
  //     throw std::invalid_argument("beam_size must be positive.");
  //   }
  //   if (min_length < 0 || max_length < 0 || min_length > max_length) {
  //     std::cout << "Invalid min_length or max_length." << std::endl;
  //     throw std::invalid_argument("Invalid min_length or max_length.");
  //   }
  //   if (patience < 1.0) {
  //     std::cout << "patience must be at least 1.0." << std::endl;
  //     throw std::invalid_argument("patience must be at least 1.0.");
  //   }
  //   if (temperature <= 0) {
  //     std::cout << "temperature must be positive." << std::endl;
  //     throw std::invalid_argument("temperature must be positive.");
  //   }

  //   // Set beam search's hyperparameters.
  //   ctranslate2::TranslationOptions opts;
  //   opts.num_hypotheses = num_return_sequences;
  //   opts.beam_size = beam_size;
  //   opts.patience = patience;
  //   opts.length_penalty = length_penalty;
  //   opts.min_decoding_length = min_length;
  //   opts.max_decoding_length = max_length;
  //   opts.sampling_temperature = temperature;
  //   opts.sampling_topk = 0;
  //   opts.sampling_topp = 1.0;
  //   opts.max_input_length = 0;
  //   opts.use_vmap = true;
  //   opts.disable_unk = true;
  //   opts.return_scores = true;

  //   // Get the input tokens ready.
  //   std::vector<std::string> input_tokens = convert_tokens(_input_tokens);
  //   std::cout << "input tokens size: " << input_tokens.size() << std::endl;
  //   std::vector<std::string> target_prefix_tokens =
  //       convert_tokens(_target_prefix_tokens);
  //   std::cout << "target prefix tokens size: " << target_prefix_tokens.size() << std::endl;

  //   // Generate tactics with beam search.
  //   debug_log << "About to generate tactics with beam search" << std::endl;
  //   debug_log << "model name: " << name << std::endl;
  //   debug_log << "input tokens: " << input_tokens.size() << std::endl;
  //   std::cout << "input tokens: " << std::endl;
  //   for (const auto& token : input_tokens) {
  //     std::cout << token << " ";
  //     debug_log << token << " ";
  //   }
  //   std::cout << std::endl;

  //   std::cout << "as hex" << std::endl;
  //   for (const auto& token : input_tokens) {
  //     debug_log << "Token: ";
  //     std::cout << "Token: ";
  //     for (unsigned char c : token) {
  //       std::cout << std::hex << (int)c << " ";
  //       debug_log << std::hex << (int)c << " ";
  //     }
  //     std::cout << std::endl;
  //     debug_log << std::endl;
  //   }

  //   if (target_prefix_tokens.empty()) {
  //     debug_log << "target prefix tokens empty" << std::endl;
  //     std::cout << "target prefix tokens empty" << std::endl;
  //     target_prefix_tokens.push_back("<pad>");
  //   }
  //   debug_log << "target prefix tokens: " << target_prefix_tokens.size() << std::endl;
  //   std::cout << "target prefix tokens size: " << target_prefix_tokens.size() << std::endl;
  //   debug_log << "opts: " << opts.num_hypotheses << " " << opts.beam_size << " " << opts.patience << " " << opts.length_penalty << " " << opts.min_decoding_length << " " << opts.max_decoding_length << " " << opts.sampling_temperature << " " << opts.sampling_topk << " " << opts.sampling_topp << " " << opts.max_input_length << " " << opts.use_vmap << " " << opts.disable_unk << " " << opts.return_scores << std::endl;
  //   std::cout << "opts: " << opts.num_hypotheses << " " << opts.beam_size << " " << opts.patience << " " << opts.length_penalty << " " << opts.min_decoding_length << " " << opts.max_decoding_length << " " << opts.sampling_temperature << " " << opts.sampling_topk << " " << opts.sampling_topp << " " << opts.max_input_length << " " << opts.use_vmap << " " << opts.disable_unk << " " << opts.return_scores << std::endl;
  //   std::unique_ptr<ctranslate2::Translator> generator = std::move(generators.at(name));
  //   std::cout << "moved generator" << std::endl;
  //   // std::vector<ctranslate2::TranslationResult> results = generator->translate_batch(
  //   //     {input_tokens}, {target_prefix_tokens}, opts);

  //   // TODO: rmeoce after testing
  //   std::vector<std::vector<std::string>> batch = {input_tokens};
  //   std::cout << "batch size: " << batch.size() << std::endl;
  //   if (!target_prefix_tokens.empty()) {
  //     std::cout << "target prefix tokens not empty" << std::endl;
  //     batch.push_back(target_prefix_tokens);
  //   }
  //   std::cout << "new batch size: " << batch.size() << std::endl;
  //   std::cout << "first sequence length: " << batch[0].size() << std::endl;
  //   debug_log << "About to call translate_batch" << std::endl;
  //   debug_log << "Batch size: " << batch.size() << std::endl;
  //   debug_log << "First sequence length: " << batch[0].size() << std::endl;

  //   try {
  //     if (!generator) {
  //       debug_log << "Error: generator is null" << std::endl;
  //       std::cout << "Error: generator is null" << std::endl;
  //       // Handle error...
  //     }
  //     // debug_log << "About to call translate_batch" << std::endl;
  //     std::cout << "about to call translate_batch" << std::endl;
  //     ctranslate2::Translator& deref_generator = *generator;
  //     std::cout << "dereferenced generator" << std::endl;
  //     deref_generator.translate_batch(batch, opts);
  //     std::cout << "translation completed successfully" << std::endl;
  //     std::vector<ctranslate2::TranslationResult> results = generator->translate_batch(batch, opts);
  //     // std::vector<ctranslate2::TranslationResult> results = generator->translate_batch(
  //     //     {input_tokens}, {target_prefix_tokens}, opts);
  //     std::cout << "Translation completed successfully" << std::endl;
  //     debug_log << "Translation completed successfully" << std::endl;
  //     debug_log << "Number of results: " << results.size() << std::endl;
  //     if (results.empty()) {
  //         debug_log << "No results returned" << std::endl;
  //     }


  //     const auto& firstResult = results[0];
  //     debug_log << "First result tokens: ";
  //     for (const auto& token : firstResult.output()) {
  //         debug_log << token << " ";
  //     }
  //     debug_log << std::endl;

  //     // Process results...
  //   } catch (const std::exception& e) {
  //     debug_log << "Exception caught: " << e.what() << std::endl;
  //     // Handle the error...
  //   } catch (...) {
  //     debug_log << "Unknown exception caught" << std::endl;
  //   }

  //   // ctranslate2::TranslationResult results = generator->translate_batch(
  //   //     {input_tokens}, {target_prefix_tokens}, opts)[0];

  //   // ctranslate2::TranslationResult results = generators.at(name)->translate_batch(
  //   //     {input_tokens}, {target_prefix_tokens}, opts)[0];
  //   // assert(results.hypotheses.size() == num_return_sequences &&
  //   //        results.scores.size() == num_return_sequences);

  //   // // Return the output.
  //   // lean_object *output = lean_mk_empty_array();

  //   // try {
  //   //   for (size_t i = 0; i < num_return_sequences; i++) {
  //   //     lean_object *tokens = lean_mk_empty_array();
  //   //     for (const auto& token : results.hypotheses[i]) {
  //   //       tokens = lean_array_push(tokens, lean_mk_string(token.c_str()));
  //   //     }
  //   //     output = lean_array_push(output, lean_mk_pair(tokens, lean_box_float(std::exp(results.scores[i]))));
  //   //   }
  //   // } catch (const std::exception& e) {
  //   //   std::cerr << "Exception in generating tactics: " << e.what() << std::endl;
  //   //   return lean_box(0); // Returning an empty box on error.
  //   // }

  //   // works but panic
  //   // lean_object* output = lean_mk_empty_array();
  //   // lean_object* tokens = lean_mk_empty_array();
  //   // tokens = lean_array_push(tokens, lean_mk_string("test"));
  //   // double score = 0.5;
  //   // output = lean_array_push(output, lean_mk_pair(tokens, lean_box_float(score)));

  //   lean_object* output = lean_mk_empty_array();

  //   const char* tactics[] = {
  //     "rw [add_comm]",
  //     "simp",
  //     "apply eq_comm",
  //     "ring"
  //   };
  //   int num_tactics = sizeof(tactics) / sizeof(tactics[0]);

  //   for (int i = 0; i < num_tactics; ++i) {
  //     lean_object* tokens = lean_mk_empty_array();
  //     for (const char* c = tactics[i]; *c; ++c) {
  //       char byte_str[2] = {*c, '\0'};
  //       tokens = lean_array_push(tokens, lean_mk_string(byte_str));
  //     }
  //     double score = 1.0 - (0.1 * i);  // Decreasing scores
  //     output = lean_array_push(output, lean_mk_pair(tokens, lean_box_float(score)));
  //   }

  //   debug_log << "Created output array with " << lean_array_size(output) << " elements" << std::endl;
  //   debug_log << "Exiting generate function" << std::endl;
  //   debug_log.close();

  //   return output;
  // } catch (const std::exception& e) {
  //   debug_log << "Top-level exception caught: " << e.what() << std::endl;
  //   lean_object* output = lean_mk_empty_array();
  //   return output;
  //   // Handle the error, perhaps return an error object to Lean
  // } catch (...) {
  //   debug_log << "Unknown top-level exception caught" << std::endl;
  //   lean_object* output = lean_mk_empty_array();
  //   return output;
  //   // Handle unknown exceptions
  // }

  lean_object* output = lean_mk_empty_array();

  const char* tactics[] = {
    "rw [add_comm]",
    "simp",
    "apply eq_comm",
    "ring"
  };
  int num_tactics = sizeof(tactics) / sizeof(tactics[0]);

  for (int i = 0; i < num_tactics; ++i) {
    lean_object* tokens = lean_mk_empty_array();
    for (const char* c = tactics[i]; *c; ++c) {
      char byte_str[2] = {*c, '\0'};
      tokens = lean_array_push(tokens, lean_mk_string(byte_str));
    }
    double score = 1.0 - (0.1 * i);  // Decreasing scores
    output = lean_array_push(output, lean_mk_pair(tokens, lean_box_float(score)));
  }

  debug_log << "Created output array with " << lean_array_size(output) << " elements" << std::endl;
  debug_log << "Exiting generate function" << std::endl;
  debug_log.close();

  return output;
}

extern "C" lean_obj_res encode(b_lean_obj_arg _name,            // String
                               b_lean_obj_arg _input_tokens) {  // Array String
  std::string name = std::string(lean_string_cstr(_name));
  if (!is_initialized_aux<ctranslate2::Encoder>(name)) {
    throw std::runtime_error(name + " hasn't been initialized.");
  }

  std::vector<std::string> input_tokens = convert_tokens(_input_tokens);
  ctranslate2::EncoderForwardOutput results =
      encoders.at(name)->forward_batch_async({input_tokens}).get();
  ctranslate2::StorageView hidden_state = results.last_hidden_state;

  assert(hidden_state.dim(0) == 1);
  int l = hidden_state.dim(1);
  int d = hidden_state.dim(2);
  lean_object *arr = lean_mk_empty_float_array(lean_box(d));

  for (ctranslate2::dim_t i = 0; i < d; i++) {
    double sum = 0.0;
    for (ctranslate2::dim_t j = 0; j < l; j++) {
      sum += hidden_state.scalar_at<float>({0, j, i});
    }
    lean_float_array_push(arr, sum / l);
  }

  return arr;
}

extern "C" uint8_t init_premise_embeddings(b_lean_obj_arg _path,      // String
                                           b_lean_obj_arg _device) {  // String
  std::string path = std::string(lean_string_cstr(_path));
  if (!exists(path)) {
    return false;
  }
  if (p_premise_embeddings != nullptr) {
    delete p_premise_embeddings;
  }

  // ctranslate2::Device device =
  // ctranslate2::str_to_device(lean_string_cstr(_device));
  // TODO: We should remove this line when everything can work well on CUDA.
  ctranslate2::Device device = ctranslate2::Device::CPU;

  const auto &d = npy::read_npy<double>(path);
  std::vector<double> data = d.data;
  std::vector<unsigned long> shape = d.shape;
  bool fortran_order = d.fortran_order;

  std::vector<float> data_f;
  data_f.resize(data.size());
  std::transform(data.begin(), data.end(), data_f.begin(),
                 [](double d) { return static_cast<float>(d); });

  std::vector<int64_t> shape_i64;
  shape_i64.resize(shape.size());
  std::transform(shape.begin(), shape.end(), shape_i64.begin(),
                 [](unsigned long ul) { return static_cast<int64_t>(ul); });

  p_premise_embeddings =
      new ctranslate2::StorageView(shape_i64, data_f, device);
  return true;
}

inline bool premise_embeddings_initialized_aux() {
  return p_premise_embeddings != nullptr;
}

extern "C" uint8_t premise_embeddings_initialized(lean_object *) {
  return premise_embeddings_initialized_aux();
}

extern "C" uint8_t init_premise_dictionary(b_lean_obj_arg _path) {
  std::string path = std::string(lean_string_cstr(_path));
  if (!exists(path)) {
    return false;
  }
  if (p_premise_dictionary != nullptr) {
    delete p_premise_dictionary;
  }

  std::ifstream f(path);
  p_premise_dictionary = new json(json::parse(f));

  return true;
}

inline bool premise_dictionary_initialized_aux() {
  return p_premise_dictionary != nullptr;
}

extern "C" uint8_t premise_dictionary_initialized(lean_object *) {
  return premise_dictionary_initialized_aux();
}

extern "C" lean_obj_res retrieve(b_lean_obj_arg _query_emb,
                                 uint64_t _k) {  // FloatArray
  // lean_object *arr
  // assert(p_premise_embeddings && static_cast<int64_t>(p_arr->m_size) ==
  // p_premise_embeddings->dim(1));

  int64_t d = lean_unbox(lean_float_array_size(_query_emb));
  std::vector<float> query_emb_data;
  for (int i = 0; i < d; i++) {
    query_emb_data.push_back(lean_float_array_uget(_query_emb, i));
  }

  ctranslate2::Device device = p_premise_embeddings->device();
  ctranslate2::StorageView query_emb =
      ctranslate2::StorageView({d, 1}, query_emb_data, device);

  ctranslate2::ops::MatMul matmul(false, false, 1.0);
  long int k = static_cast<long int>(_k);
  ctranslate2::ops::TopK topk(k, -1);

  int num_premises = p_premise_embeddings->dim(0);
  std::vector<int64_t> probs_shape{num_premises, 1};

  ctranslate2::StorageView probs = ctranslate2::StorageView(
      probs_shape, ctranslate2::DataType::FLOAT32, device);
  matmul(*p_premise_embeddings, query_emb, probs);
  probs.resize({num_premises});

  ctranslate2::StorageView topk_values =
      ctranslate2::StorageView({k}, ctranslate2::DataType::FLOAT32, device);
  ctranslate2::StorageView topk_indices =
      ctranslate2::StorageView({k}, ctranslate2::DataType::INT32, device);
  topk(probs, topk_values, topk_indices);

  lean_object *output = lean_mk_empty_array();
  const int *p_topk_indices = topk_indices.data<int>();
  const float *p_topk_values = topk_values.data<float>();

  for (int i = 0; i < k; i++) {
    int idx = p_topk_indices[i];
    assert(0 < idx && idx < num_premises);
    // [NOTE]: This is where the server crash occurs on CUDA.
    const std::string this_premise =
        (*p_premise_dictionary)[std::to_string(idx)]["full_name"];
    const std::string this_path =
        (*p_premise_dictionary)[std::to_string(idx)]["path"];
    const std::string this_code =
        (*p_premise_dictionary)[std::to_string(idx)]["code"];

    output = lean_array_push(
        output,
        lean_mk_pair(
            lean_mk_string(this_premise.c_str()),
            lean_mk_pair(lean_mk_string(this_path.c_str()),
                         lean_mk_pair(lean_mk_string(this_code.c_str()),
                                      lean_box_float(p_topk_values[i])))));
  }

  return output;
}
