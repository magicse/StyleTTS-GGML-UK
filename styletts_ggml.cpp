#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <iomanip>

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm> // std::max_element
#include <set>

#include "styletts_ggml.h"

#ifndef GGML_COMMIT_HASH
#define GGML_COMMIT_HASH "unknown"
#endif

struct ggml_context* create_ggml_context(size_t mem_size) {
    struct ggml_init_params params = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    return ggml_init(params);
}

bool load_model_weights_custom(StyleTTSModel& model, const std::string& model_path,
                        const std::set<std::string>& components_to_load) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file " << model_path << std::endl;
        return false;
    }

    // Читаем метаинформацию
    file.read(reinterpret_cast<char*>(&model.vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.hidden_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.mel_channels), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.max_seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.num_speakers), sizeof(int));

    // Читаем конфигурацию
    file.read(reinterpret_cast<char*>(&model.config), sizeof(model.config));

    std::cout << "Model configuration:" << std::endl;
    std::cout << "  vocab_size: " << model.vocab_size << std::endl;
    std::cout << "  hidden_dim: " << model.hidden_dim << std::endl;
    std::cout << "  mel_channels: " << model.mel_channels << std::endl;
    std::cout << "  max_seq_len: " << model.max_seq_len << std::endl;
    std::cout << "  num_speakers: " << model.num_speakers << std::endl;

    // Функция для загрузки или пропуска компонента
    auto load_component = [&](const std::string& component_name,
                              std::map<std::string, struct ggml_tensor*>& weights) -> bool {
        int num_tensors;
        file.read(reinterpret_cast<char*>(&num_tensors), sizeof(int));

        if (components_to_load.count(component_name) == 0) {
            // Пропускаем тензоры компонента
            for (int i = 0; i < num_tensors; ++i) {
                int name_len;
                file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
                file.seekg(name_len, std::ios::cur);

                int n_dims;
                file.read(reinterpret_cast<char*>(&n_dims), sizeof(int));
                std::vector<int64_t> dims(n_dims);
                for (int j = 0; j < n_dims; ++j)
                    file.read(reinterpret_cast<char*>(&dims[j]), sizeof(int64_t));

                size_t tensor_size = sizeof(float);
                for (auto d : dims) tensor_size *= d;
                file.seekg(tensor_size, std::ios::cur);
            }
            std::cout << "[Skipping component: " << component_name << "]\n";
            return true;
        }

        std::cout << "\n========= Loading " << component_name << " with " << num_tensors << " tensors =========" << std::endl;

        auto print_progress = [](int current, int total, int bar_width = 40) {
            float progress = float(current) / float(total);
            int pos = static_cast<int>(bar_width * progress);

            std::cout << "[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "%\r";
            std::cout.flush();
        };

        for (int p = 0; p < num_tensors; ++p) {
            int name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(int));
            std::string tensor_name(name_len, '\0');
            file.read(&tensor_name[0], name_len);

            int n_dims;
            file.read(reinterpret_cast<char*>(&n_dims), sizeof(int));
            std::vector<int64_t> dims(n_dims);
            for (int j = 0; j < n_dims; ++j)
                file.read(reinterpret_cast<char*>(&dims[j]), sizeof(int64_t));

			// std::reverse(dims.begin(), dims.end()); // инверсия осей если тензоры из Python были сохранены в (float32, C-order)
			// так как GGML интерпретирует размерности в обратном порядке,

            struct ggml_tensor* tensor;
            if (n_dims == 0) {
                tensor = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 1);
                std::cout << "  [info] Interpreting scalar tensor as 1D tensor with size 1\n";
            } else if (n_dims == 1) {
                tensor = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, dims[0]);
            } else if (n_dims == 2) {
                tensor = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, dims[0], dims[1]);
            } else if (n_dims == 3) {
                tensor = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, dims[0], dims[1], dims[2]);
            } else if (n_dims == 4) {
                tensor = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
            } else {
                std::cerr << "Unsupported tensor rank: " << n_dims << " for tensor " << tensor_name << std::endl;
                return false;
            }

            if (!tensor || !tensor->data) {
                std::cerr << "tensor->data is nullptr! Cannot read weights." << std::endl;
                return false;
            }

            file.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));

            print_progress(p + 1, num_tensors);
            weights[tensor_name] = tensor;
        }
        std::cout << std::endl;
        return true;
    };

    // Загружаем компоненты по выбору
    if (!load_component("predictor", model.predictor_weights)) return false;
    if (!load_component("decoder", model.predictor_weights)) return false;
    if (!load_component("pitch_extractor", model.predictor_weights)) return false;
    if (!load_component("text_encoder", model.text_encoder_weights)) return false;
    if (!load_component("style_encoder", model.style_encoder_weights)) return false;
    if (!load_component("text_aligner", model.text_aligner_weights)) return false;

    file.close();
    return true;
};

struct ggml_tensor* weight_norm_conv1d(struct ggml_context* ctx,
                                             struct ggml_tensor* weight_v,
                                             struct ggml_tensor* weight_g) {
    const int64_t out_channels = weight_v->ne[0];
    const int64_t in_channels  = weight_v->ne[1];
    const int64_t kernel_size  = weight_v->ne[2];

    struct ggml_tensor* v_squared = ggml_mul(ctx, weight_v, weight_v);

    struct ggml_tensor* v_sq_reshaped = ggml_reshape_2d(ctx, v_squared, out_channels, in_channels * kernel_size);
    struct ggml_tensor* v_sq_cont = ggml_cont(ctx, v_sq_reshaped);


    struct ggml_tensor* v_sq_t = ggml_transpose(ctx, v_sq_cont);
  	v_sq_t = ggml_cont(ctx, v_sq_t);
  	print_tensor_info(v_sq_t, "v_sq_t");


    struct ggml_tensor* sumsq = ggml_sum_rows(ctx, v_sq_t);  // [oc]
  	sumsq = ggml_transpose(ctx, sumsq);
  	sumsq = ggml_cont(ctx, sumsq);
	
	//DEBUG_TENSOR(sumsq, "sumsq");
	//DEBUG_TENSOR_TO_FILE(sumsq,"norm_sq_sum_cpp.txt");

    struct ggml_tensor* norm = ggml_sqrt(ctx, sumsq);
	
	//DEBUG_TENSOR(norm, "norm_sqrt");
	//DEBUG_TENSOR_TO_FILE(norm,"norm_sqrt_sum_cpp.txt");	
	
  struct ggml_tensor* eps = ggml_new_f32(ctx, 1e-8f);
  struct ggml_tensor* safe_norm = ggml_add(ctx, norm, eps);
  safe_norm = ggml_cont(ctx, safe_norm);
	
  print_tensor_info(safe_norm, "safe_norm");

	//DEBUG_TENSOR(weight_v, "weight_v");
	auto safe_norm_broadcast = ggml_reshape_3d(ctx, safe_norm, 1, 1, 512); // [1, 1, 512]
	safe_norm_broadcast = ggml_cont(ctx, safe_norm_broadcast);
	struct ggml_tensor* weight_v_perm = ggml_permute(ctx, weight_v, 2, 0, 1, 3); 
	weight_v_perm = ggml_cont(ctx, weight_v_perm);
	
    struct ggml_tensor* v_normalized = ggml_div(ctx, weight_v_perm, safe_norm_broadcast);
	v_normalized = ggml_reshape_3d(ctx, v_normalized, in_channels, kernel_size, out_channels);
	v_normalized = ggml_cont(ctx, v_normalized); 
	
	//DEBUG_TENSOR_TO_FILE(v_normalized,"norm_div_sqrt_sum_cpp.txt");
	DEBUG_TENSOR(v_normalized, "v / norm_safe");	
	
	// Broadcast weight_g: [out_c] -> [1, 1, out_c]
	auto weight_g_broadcast = ggml_reshape_3d(ctx, weight_g, 1, 1, out_channels);
    weight_g_broadcast = ggml_cont(ctx, weight_g_broadcast);
	
	struct ggml_tensor* result = ggml_mul(ctx, v_normalized, weight_g_broadcast);
	DEBUG_TENSOR(result, "result");
	result = ggml_permute(ctx, result, 2, 0, 1, 3);  // → [out_c, in_c, k]

	result = ggml_cont(ctx, result);
	DEBUG_TENSOR(result, "result");
    return result;
}
