// styletts_ggml.h
#pragma once


#include "ggml.h"
#include <vector>
#include <map>
#include <string>

#include <stdio.h>

const char* ggml_type_names[GGML_TYPE_COUNT] = {
    "F32",      // 0
    "F16",      // 1
    "Q4_0",     // 2
    "Q4_1",     // 3
    "UNKNOWN",  // 4 - removed
    "UNKNOWN",  // 5 - removed
    "Q5_0",     // 6
    "Q5_1",     // 7
    "Q8_0",     // 8
    "Q8_1",     // 9
    "Q2_K",     // 10
    "Q3_K",     // 11
    "Q4_K",     // 12
    "Q5_K",     // 13
    "Q6_K",     // 14
    "Q8_K",     // 15
    "IQ2_XXS",  // 16
    "IQ2_XS",   // 17
    "IQ3_XXS",  // 18
    "IQ1_S",    // 19
    "IQ4_NL",   // 20
    "IQ3_S",    // 21
    "IQ2_S",    // 22
    "IQ4_XS",   // 23
    "I8",       // 24
    "I16",      // 25
    "I32",      // 26
    "I64",      // 27
    "F64",      // 28
    "IQ1_M",    // 29
    "BF16",     // 30
    "UNKNOWN",  // 31 - removed
    "UNKNOWN",  // 32 - removed
    "UNKNOWN",  // 33 - removed
    "TQ1_0",    // 34
    "TQ2_0",    // 35
    "UNKNOWN",  // 36 - removed
    "UNKNOWN",  // 37 - removed
    "UNKNOWN",  // 38 - removed
};

const char* ggml_type_name_by_index(int type) {
    if (type >= 0 && type < GGML_TYPE_COUNT) {
        return ggml_type_names[type];
    } else {
        return "INVALID_TYPE";
    }
}

// Model Structure
struct StyleTTSModel {
    struct ggml_context* ctx;
    std::map<std::string, struct ggml_tensor*> predictor_weights;
    std::map<std::string, struct ggml_tensor*> decoder_weights;
    std::map<std::string, struct ggml_tensor*> pitch_extractor_weights;
    std::map<std::string, struct ggml_tensor*> text_encoder_weights;
    std::map<std::string, struct ggml_tensor*> style_encoder_weights;
    std::map<std::string, struct ggml_tensor*> text_aligner_weights;
    std::map<std::string, struct ggml_tensor*> discriminator_weights;

    int vocab_size;
    int hidden_dim;
    int mel_channels;
    int max_seq_len;
    int num_speakers;

    struct {
        int text_encoder_dim;
        int style_encoder_dim;
        int decoder_dim;
        int predictor_dim;
        int pitch_extractor_dim;
    } config;
};

// LSTM State structure
struct LSTMState {
    struct ggml_tensor* hidden;
    struct ggml_tensor* cell;
    int hidden_size;
    int batch_size;
    int seq_len;
};

// Create LSTM state
LSTMState* create_lstm_state(struct ggml_context* ctx, int batch_size, int hidden_size, int seq_len) {
    LSTMState* state = new LSTMState();
    state->hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, batch_size);
    state->cell = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, batch_size);
    state->hidden_size = hidden_size;
    state->batch_size = batch_size;
    state->seq_len = seq_len;
    
    // Initialize with zeros
    ggml_set_f32(state->hidden, 0.0f);
    ggml_set_f32(state->cell, 0.0f);
    
    return state;
}
