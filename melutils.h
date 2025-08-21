#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>
#include "ggml.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
Example of usage
#include "meldata.h"

// In main program:
struct ggml_context* ctx = ggml_init(...);
std::vector<float> audio_data = load_your_audio(); // your audio loading
int sample_rate = 22400; // or any other sample rate

struct ggml_tensor* mel_tensor = compute_mel_spectrogram_tensor(
    ctx,
    audio_data,
    sample_rate,
    80,     // n_mels
    2048,   // n_fft  
    1200,   // win_length
    300,    // hop_length
	16000,   // mel_sr = 16000	// mel_sr - must be 24000 as target_sr but due mismatch in StyleTts it is 16000
    -4.0f,  // mean
    4.0f,   // std
    true,   // trim
    30.0f,  // top_db
    24000   // target_sr
);
*/

namespace MelSpectrogramUtils {
    
    using complex_d = std::complex<double>;

    // Structure for trimming result
    struct TrimResult {
        float* trimmed_signal;
        int trimmed_length;
        int start_index;
        int end_index;
    };

    // Hann window function
    inline double hann_window(int n, int samples) {
        return 0.5 * (1.0 - cos((2.0 * M_PI * n) / (samples - 1.0)));
    }
    
    // Bit reversal for FFT
    inline uint32_t reverse_bits(uint32_t val, int power) {
        int reversed = 0;
        for (int i = 0; i < power; i++) {
            bool cur_bit = (1 << i) & val;
            reversed |= (cur_bit << (power - i - 1));
        }
        return reversed;
    }
    
    // Omega function for FFT
    inline std::complex<double> omega(float p, float q) {
        const float trig_arg = 2 * M_PI * q / p;
        return {cos(trig_arg), sin(trig_arg)};
    }
    
    // Padding to the nearest power of two
    inline int pad_to_power2(std::vector<std::complex<double>>& signal, int min_size) {
        int power;
        int new_size = 2;
        
        for (power = 1; new_size < min_size; power++, new_size <<= 1) {
            // Do nothing
        }
        
        if (new_size > signal.size()) {
            signal.resize(new_size, std::complex<double>(0.0, 0.0));
        }
        return power;
    }
    
    // FFT implementation
    inline void fft_transform(std::vector<std::complex<double>>& signal) {
        int min_size = signal.size();
        int power = pad_to_power2(signal, min_size);
        
        if (!power) return;
        
        std::vector<std::complex<double>> transformed(signal.size(), 0);
        
        // Apply window function and reorder by reversed bits
        for (int i = 0; i < signal.size(); i++) {
            transformed[reverse_bits(i, power)] = signal[i];
        }
        
        int n = 2;
        while (n <= transformed.size()) {
            // Iterate over segments of length n
            for (int i = 0; i <= transformed.size() - n; i += n) {
                // Combine each half of the segment
                for (int m = i; m < i + n/2; m++) {
                    complex_d term1 = transformed[m];
                    complex_d term2 = omega(n, -m) * transformed[m + n/2];
                    
                    transformed[m]       = term1 + term2;
                    transformed[m + n/2] = term1 - term2;
                }
            }
            n *= 2;
        }
        signal = std::move(transformed);
    }

    // Function to compute RMS (Root Mean Square) for a frame
    inline float compute_rms(const float* signal, int signal_length, int start, int frame_length, bool center = true) {
        float sum = 0.0f;
        int pad = center ? frame_length / 2 : 0;
        int frame_start = start - pad;

        for (int i = 0; i < frame_length; i++) {
            int idx = frame_start + i;
            float sample = 0.0f;
            if (idx >= 0 && idx < signal_length) {
                sample = signal[idx];
            }
            sum += sample * sample;
        }
        return sqrtf(sum / frame_length);
    }

    // Convert amplitude to decibels
    inline float amplitude_to_db(float amplitude, float ref) {
        if (amplitude <= 0.0f) {
            return -INFINITY;
        }
        return 20.0f * log10f(amplitude / ref);
    }

    // Main function for trimming audio
    inline TrimResult* audio_trim(const float* signal, int signal_length, 
                                  float top_db, int frame_length, int hop_length) {
        
        if (signal == nullptr || signal_length <= 0 || frame_length <= 0 || hop_length <= 0) {
            return nullptr;
        }
        
        // Calculate number of frames
        int num_frames;
        if (signal_length >= frame_length) {
            num_frames = 1 + (signal_length - frame_length) / hop_length;
        } else {
            num_frames = 1;
        }
        
        if (num_frames <= 0) {
            return nullptr;
        }
        
        // Array to store RMS values of each frame
        float* rms_values = (float*)malloc(num_frames * sizeof(float));
        
        // Compute RMS for each frame
        for (int frame = 0; frame < num_frames; frame++) {
            int start_pos = frame * hop_length;
            int current_frame_length = frame_length;
            
            if (start_pos + frame_length > signal_length) {
                current_frame_length = signal_length - start_pos;
            }

            rms_values[frame] = compute_rms(signal, signal_length, start_pos, current_frame_length);
        }
        
        // Find maximum RMS value as reference
        float ref_rms = 0.0f;
        for (int i = 0; i < num_frames; i++) {
            if (rms_values[i] > ref_rms) {
                ref_rms = rms_values[i];
            }
        }
        
        TrimResult* result = (TrimResult*)malloc(sizeof(TrimResult));
        
        if (ref_rms == 0.0f) {
            result->trimmed_signal = nullptr;
            result->trimmed_length = 0;
            result->start_index = 0;
            result->end_index = 0;
            free(rms_values);
            return result;
        }
        
        // Array to store information about non-silent frames
        int* non_silent = (int*)malloc(num_frames * sizeof(int));
        
        // Determine non-silent frames
        for (int frame = 0; frame < num_frames; frame++) {
            float db = amplitude_to_db(rms_values[frame], ref_rms);
            non_silent[frame] = (db > -top_db) ? 1 : 0;
        }
        
        // Find first and last non-silent frames
        int first_non_silent = -1;
        int last_non_silent = -1;
        
        for (int i = 0; i < num_frames; i++) {
            if (non_silent[i]) {
                if (first_non_silent == -1) {
                    first_non_silent = i;
                }
                last_non_silent = i;
            }
        }
        
        if (first_non_silent == -1) {
            result->trimmed_signal = nullptr;
            result->trimmed_length = 0;
            result->start_index = 0;
            result->end_index = 0;
        } else {
            // Calculate boundaries in samples
            int start_sample = first_non_silent * hop_length;
            int end_sample = (last_non_silent + 1) * hop_length;
            
            if (end_sample > signal_length) {
                end_sample = signal_length;
            }
            
            result->start_index = start_sample;
            result->end_index = end_sample;
            result->trimmed_length = end_sample - start_sample;
            
            // Copy trimmed part of signal
            result->trimmed_signal = (float*)malloc(result->trimmed_length * sizeof(float));
            memcpy(result->trimmed_signal, &signal[start_sample], 
                   result->trimmed_length * sizeof(float));
        }
        
        free(rms_values);
        free(non_silent);
        return result;
    }

    // Free memory function
    inline void free_trim_result(TrimResult* result) {
        if (result) {
            if (result->trimmed_signal) {
                free(result->trimmed_signal);
            }
            free(result);
        }
    }

    // Trim wrapper
    inline std::vector<float> trim_audio(const std::vector<float>& audio, float top_db = 30.0f) {
        if (audio.empty()) return audio;

        const int frame_length = 2048;
        const int hop_length = 512;

        TrimResult* result = audio_trim(audio.data(), static_cast<int>(audio.size()), top_db, frame_length, hop_length);
        
        if (!result || result->trimmed_signal == nullptr || result->trimmed_length <= 0) {
            if (result) free_trim_result(result);
            return audio;
        }

        std::vector<float> trimmed(audio.data() + result->start_index, 
                                   audio.data() + result->end_index);

        free_trim_result(result);
        return trimmed;
    }

    // Improved resampling with Kaiser window
    inline std::vector<float> resample_audio(
        const std::vector<float>& input,
        int input_sr,
        int output_sr,
        int kaiser_beta = 8
    ) {
        if (input_sr == output_sr) return input;

        double ratio = static_cast<double>(output_sr) / input_sr;
        int output_len = static_cast<int>(input.size() * ratio);
        std::vector<float> output(output_len, 0.0f);

        // Kaiser window for anti-aliasing
        auto kaiser = [kaiser_beta](double x, int width) {
            if (x < -width/2 || x > width/2) return 0.0;
            double arg = kaiser_beta * sqrt(1.0 - pow(2.0 * x / width, 2));
            // Simplified approximation of I0 (modified Bessel function)
            double i0_arg = 1.0 + pow(arg / 2, 2) / 4 + pow(arg / 2, 4) / 64;
            double i0_beta = 1.0 + pow(kaiser_beta / 2, 2) / 4 + pow(kaiser_beta / 2, 4) / 64;
            return i0_arg / i0_beta;
        };

        auto sinc = [](double x) {
            if (abs(x) < 1e-10) return 1.0;
            x *= M_PI;
            return sin(x) / x;
        };

        int filter_width = 64;
        double cutoff = std::min(1.0, 1.0 / ratio) * 0.95;  // Anti-aliasing

        for (int i = 0; i < output_len; i++) {
            double center = i / ratio;
            int left = static_cast<int>(center) - filter_width / 2;

            float sum = 0.0f;
            float norm = 0.0f;

            for (int j = 0; j < filter_width; j++) {
                int idx = left + j;
                if (idx < 0 || idx >= input.size()) continue;

                double x = (center - idx) * cutoff;
                double sinc_val = sinc(x) * cutoff;
                double window = kaiser(center - idx, filter_width);
                double coeff = sinc_val * window;

                sum += input[idx] * coeff;
                norm += coeff;
            }

            output[i] = (norm != 0.0f) ? sum / norm : 0.0f;
        }

        return output;
    }
    
    // Create Mel filterbank
    inline std::vector<std::vector<double>> create_mel_filterbank(int sr, int n_fft, int n_mels) {
        double min_freq = 0;
        double max_freq = (double)sr / 2.0;
        
        // Convert Hz to Mel
        float mel_fmin = 2595.0 * log10(1.0 + min_freq / 700.0);
        float mel_fmax = 2595.0 * log10(1.0 + max_freq / 700.0);
        
        std::vector<float> mel_points(n_mels + 2);
        
        for (int i = 0; i < mel_points.size(); ++i) {
            // Mel bins to Hz
            float mel = mel_fmin + i * (mel_fmax - mel_fmin) / (n_mels + 1);
            float freq = 700.0 * (pow(10, mel / 2595.0) - 1.0);
            mel_points[i] = freq / static_cast<float>(sr) * n_fft;
        }
        
        std::vector<std::vector<double>> filterbank(n_mels, std::vector<double>(n_fft / 2 + 1));
        for (int i = 0; i < n_mels; ++i) {
            for (int j = 0; j < n_fft / 2 + 1; ++j) {
                double h = 0.0;
                if (mel_points[i] <= j && j <= mel_points[i + 1]) {
                    h = (double)(j - mel_points[i]) / (mel_points[i + 1] - mel_points[i]);
                } else if (mel_points[i + 1] <= j && j <= mel_points[i + 2]) {
                    h = (double)(mel_points[i + 2] - j) / (mel_points[i + 2] - mel_points[i + 1]);
                }
                filterbank[i][j] = h;
            }
        }
        
        return filterbank;
    }
    
    // Matrix transpose
    template<typename T>
    inline std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return {};
        std::vector<std::vector<T>> result(matrix[0].size(), std::vector<T>(matrix.size()));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

} // namespace MelSpectrogramUtils

// Main function to compute mel spectrogram and return GGML tensor
inline struct ggml_tensor* compute_mel_spectrogram_tensor(
    struct ggml_context* ctx,
    const std::vector<float>& audio_data,
    int current_sr,
    int n_mels = 80,
    int n_fft = 2048,
    int win_length = 1200,
    int hop_length = 300,
	int	mel_sr = 16000,	// mel_sr - must be 24000 as target_sr but due mismatch in StyleTts it is 16000	
    float mean = -4.0f,
    float std_dev = 4.0f,
    bool trim_audio = true,
    float top_db = 30.0f,
    int target_sr = 24000
) {
    using namespace MelSpectrogramUtils;
    
    if (audio_data.empty()) {
        return nullptr;
    }
    
    // Copy data for processing
    std::vector<float> processed_audio = audio_data;
    //int current_sr = sample_rate;
    
    // 1. Trim audio
    if (trim_audio) {
        processed_audio = MelSpectrogramUtils::trim_audio(processed_audio, top_db);
        if (processed_audio.empty()) {
            return nullptr;
        }
    }
    
    // 2. Resampling
    if (current_sr != target_sr) {
        processed_audio = MelSpectrogramUtils::resample_audio(processed_audio, current_sr, target_sr);
        current_sr = target_sr;
    }

    // 2.1 Padding
    int pad = win_length / 2;
    processed_audio.insert(processed_audio.begin(), pad, 0.0f);
    processed_audio.insert(processed_audio.end(), pad, 0.0f);
    
    // 3. Compute STFT
    int num_frames = (processed_audio.size() - win_length) / hop_length + 1;
    if (num_frames <= 0) {
        return nullptr;
    }
    
    std::vector<std::vector<std::complex<double>>> stft_result;
    stft_result.reserve(num_frames);
    
    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        int start_idx = i * hop_length;
        
        std::vector<std::complex<double>> frame;
        frame.reserve(win_length);
        
        for (int j = 0; j < win_length; j++) {
            float sample = (start_idx + j < processed_audio.size()) ? processed_audio[start_idx + j] : 0.0f;
            double windowed = sample * MelSpectrogramUtils::hann_window(j, win_length);
            frame.push_back(std::complex<double>(windowed, 0.0));
        }       
        
        // Pad to n_fft if needed
        if (frame.size() < n_fft) {
            frame.resize(n_fft, std::complex<double>(0.0, 0.0));
        }
        
        // Apply FFT
        fft_transform(frame);
        stft_result.push_back(std::move(frame));
    }
    
    // 4. Compute power spectrogram
    std::vector<std::vector<float>> power_spec(stft_result.size(), 
        std::vector<float>(n_fft / 2 + 1));
    
    for (int i = 0; i < stft_result.size(); i++) {
        for (int j = 0; j < n_fft / 2 + 1; j++) {
            float magnitude = std::abs(stft_result[i][j]);
            power_spec[i][j] = magnitude * magnitude;  // Power = |X|^2
        }
    }
    
    // 5. Create mel filterbank
    auto mel_filterbank = create_mel_filterbank(mel_sr, n_fft, n_mels);
    
    // 6. Transpose power spectrogram
    power_spec = transpose(power_spec);
    
    // 7. Apply mel filterbank
    std::vector<std::vector<float>> mel_spec(n_mels, 
        std::vector<float>(power_spec[0].size()));
    
    for (int i = 0; i < n_mels; i++) {
        for (int j = 0; j < power_spec[0].size(); j++) {
            float sum = 0.0f;
            for (int k = 0; k < power_spec.size(); k++) {
                sum += power_spec[k][j] * static_cast<float>(mel_filterbank[i][k]);
            }
            mel_spec[i][j] = sum;
        }
    }
    
    // 8. Apply log and normalization
    const float epsilon = 1e-5f;
    
    for (int i = 0; i < n_mels; i++) {
        for (int j = 0; j < mel_spec[i].size(); j++) {
            float log_mel = std::log(epsilon + mel_spec[i][j]);
            mel_spec[i][j] = (log_mel - mean) / std_dev;
        }
    }
    
    // 9. Create GGML tensor
    int64_t ne[2] = {static_cast<int64_t>(mel_spec[0].size()), static_cast<int64_t>(n_mels)};
    struct ggml_tensor* result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
    
    // Copy data to tensor
    float* data = (float*)result->data;
    for (int i = 0; i < n_mels; i++) {
        for (int j = 0; j < mel_spec[i].size(); j++) {
            data[i * ne[0] + j] = mel_spec[i][j];
        }
    }
    
    return result;
}
