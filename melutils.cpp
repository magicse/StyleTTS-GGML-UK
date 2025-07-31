#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>
#include "AudioFile.h"
#include <fstream>
#include <windows.h>

// Data types
using complex_d = std::complex<double>;

// Helper functions for FFT
namespace MelSpectrogramUtils {
    
    // Hann window function
    double hann_window(int n, int samples) {
        return 0.5 * (1.0 - cos((2.0 * M_PI * n) / (samples - 1.0)));
    }
    
    // Bit reversal for FFT
    uint32_t reverse_bits(uint32_t val, int power) {
        int reversed = 0;
        for (int i = 0; i < power; i++) {
            bool cur_bit = (1 << i) & val;
            reversed |= (cur_bit << (power - i - 1));
        }
        return reversed;
    }
    
    // Omega function for FFT
    std::complex<double> omega(float p, float q) {
        const float trig_arg = 2 * M_PI * q / p;
        return {cos(trig_arg), sin(trig_arg)};
    }
    
    // Padding to next power of two
    int pad_to_power2(std::vector<std::complex<double>>& signal, int min_size) {
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
    void fft_transform(std::vector<std::complex<double>>& signal) {
        int min_size = signal.size();
        int power = pad_to_power2(signal, min_size);
        
        if (!power) return;
        
        std::vector<std::complex<double>> transformed(signal.size(), 0);
        
        // Apply window function and reorder by bit-reversed indices
        for (int i = 0; i < signal.size(); i++) {
            //transformed[reverse_bits(i, power)] = signal[i] * hann_window(i, signal.size());
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
    
    // Create mel filterbank
    std::vector<std::vector<double>> create_mel_filterbank(int sr, int n_fft, int n_mels) {
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
    
    // Transpose matrix
    template<typename T>
    std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return {};
        std::vector<std::vector<T>> result(matrix[0].size(), std::vector<T>(matrix.size()));
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix[0].size(); j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    // Audio trimming (trim silence)
    std::vector<float> trim_audio(const std::vector<float>& audio, float top_db = 30.0f) {
        if (audio.empty()) return audio;
        
        const int frame_length = 2048;
        const int hop_length = 512;
        
        std::vector<float> rms_values;
        
        for (int i = 0; i + frame_length <= audio.size(); i += hop_length) {
            float rms = 0.0f;
            for (int j = 0; j < frame_length; j++) {
                rms += audio[i + j] * audio[i + j];
            }
            rms = std::sqrt(rms / frame_length);
            rms_values.push_back(rms);
        }
        
        if (rms_values.empty()) return audio;
        
        float max_rms = *std::max_element(rms_values.begin(), rms_values.end());
        float threshold = max_rms * std::pow(10.0f, -top_db / 20.0f);
        
        int start_idx = 0;
        int end_idx = audio.size() - 1;
        
        // Find start index
        for (int i = 0; i < rms_values.size(); i++) {
            if (rms_values[i] > threshold) {
                start_idx = i * hop_length;
                break;
            }
        }
        
        // Find end index
        for (int i = rms_values.size() - 1; i >= 0; i--) {
            if (rms_values[i] > threshold) {
                end_idx = std::min(static_cast<int>(audio.size() - 1), 
                                  (i + 1) * hop_length + frame_length);
                break;
            }
        }
        
        if (end_idx > start_idx) {
            return std::vector<float>(audio.begin() + start_idx, audio.begin() + end_idx + 1);
        }
        
        return audio;
    }

    std::vector<float> trim_audio_v2(const std::vector<float>& audio, float top_db = 30.0f) {
        if (audio.empty()) return audio;

        const int frame_length = 2048;
        const int hop_length = 512;
        const int pad = frame_length / 2;

        // Zero-padding at the edges
        std::vector<float> padded_audio = audio;
        padded_audio.insert(padded_audio.begin(), pad, 0.0f);
        padded_audio.insert(padded_audio.end(), pad, 0.0f);

        std::vector<float> rms_values;

        for (int i = 0; i + frame_length <= padded_audio.size(); i += hop_length) {
            float rms = 0.0f;
            for (int j = 0; j < frame_length; j++) {
                rms += padded_audio[i + j] * padded_audio[i + j];
            }
            rms = std::sqrt(rms / frame_length);
            rms_values.push_back(rms);
        }

        if (rms_values.empty()) return audio;

        float max_rms = *std::max_element(rms_values.begin(), rms_values.end());
        float threshold = max_rms * std::pow(10.0f, -top_db / 20.0f);

        int start_idx = 0;
        int end_idx = static_cast<int>(audio.size()) - 1;

        // Find start index
        for (int i = 0; i < rms_values.size(); i++) {
            if (rms_values[i] > threshold) {
                start_idx = std::max(0, i * hop_length - pad);
                break;
            }
        }

        // Find end index
        for (int i = rms_values.size() - 1; i >= 0; i--) {
            if (rms_values[i] > threshold) {
                end_idx = std::min(static_cast<int>(audio.size()) - 1, (i * hop_length + frame_length - pad));
                break;
            }
        }

        if (end_idx > start_idx) {
            return std::vector<float>(audio.begin() + start_idx, audio.begin() + end_idx + 1);
        }

        return audio;
    }

    // Simple resampling (linear interpolation)
    std::vector<float> resample_audio(const std::vector<float>& audio, int current_sr, int target_sr) {
        if (current_sr == target_sr) return audio;
        
        float ratio = static_cast<float>(target_sr) / current_sr;
        int new_size = static_cast<int>(audio.size() * ratio);
        
        std::vector<float> resampled_data(new_size);
        
        for (int i = 0; i < new_size; i++) {
            float src_index = i / ratio;
            int src_idx = static_cast<int>(src_index);
            float frac = src_index - src_idx;
            
            if (src_idx + 1 < audio.size()) {
                resampled_data[i] = audio[src_idx] * (1.0f - frac) + audio[src_idx + 1] * frac;
            } else if (src_idx < audio.size()) {
                resampled_data[i] = audio[src_idx];
            }
        }
        
        return resampled_data;
    }
    
    std::vector<float> resample_audio_high_quality(
        const std::vector<float>& input,
        int input_sr,
        int output_sr,
        int filter_width = 32)
    {
        if (input_sr == output_sr) return input;

        double ratio = static_cast<double>(output_sr) / input_sr;
        int output_len = static_cast<int>(input.size() * ratio);
        std::vector<float> output(output_len, 0.0f);

        auto sinc = [](double x) {
            if (x == 0.0) return 1.0;
            x *= M_PI;
            return sin(x) / x;
        };

        auto hann = [](double x, int width) {
            return 0.5 * (1.0 + cos(2 * M_PI * x / width));
        };

        for (int i = 0; i < output_len; i++) {
            double center = i / ratio;
            int left = static_cast<int>(center) - filter_width / 2;

            float sum = 0.0f;
            float norm = 0.0f;

            for (int j = 0; j < filter_width; j++) {
                int idx = left + j;
                if (idx < 0 || idx >= input.size()) continue;

                double x = center - idx;
                double window = hann(x, filter_width);
                double sinc_val = sinc(x) * window;

                sum += input[idx] * sinc_val;
                norm += sinc_val;
            }

            output[i] = (norm != 0.0f) ? sum / norm : 0.0f;
        }

        return output;
    }   
    
}

// Main function
std::vector<std::vector<float>> compute_mel_spectrogram(
    const std::vector<float>& audio_data,
    int sample_rate,
    int n_mels = 80,
    int n_fft = 2048,
    int win_length = 1200,
    int hop_length = 300,
    float mean = -4.0f,
    float std_dev = 4.0f,
    bool trim_audio = true,
    float top_db = 30.0f,
    int target_sr = 24000
) {
    using namespace MelSpectrogramUtils;
    
    std::cout << "Computing mel spectrogram..." << std::endl;
    std::cout << "Input audio size: " << audio_data.size() << " samples" << std::endl;
    std::cout << "Sample rate: " << sample_rate << " Hz" << std::endl;
    
    // Copy data for processing
    std::vector<float> processed_audio = audio_data;
    int current_sr = sample_rate;
    
    // 1. Trim audio (librosa.effects.trim)
    if (trim_audio) {
        std::cout << "Trimming audio..." << std::endl;
        processed_audio = MelSpectrogramUtils::trim_audio(processed_audio, top_db);
        std::cout << "Audio after trim: " << processed_audio.size() << " samples" << std::endl;
    }
    
    // 2. Resampling (librosa.resample)
    if (current_sr != target_sr) {
        std::cout << "Resampling from " << current_sr << " to " << target_sr << " Hz..." << std::endl;
        //processed_audio = MelSpectrogramUtils::resample_audio(processed_audio, current_sr, target_sr);
        processed_audio = MelSpectrogramUtils::resample_audio_high_quality(processed_audio, current_sr, target_sr);		
        current_sr = target_sr;
        std::cout << "Audio after resample: " << processed_audio.size() << " samples" << std::endl;
    }

    // Padding with zeros at the beginning and end
	int pad = win_length / 2;
	processed_audio.insert(processed_audio.begin(), pad, 0.0f);
	processed_audio.insert(processed_audio.end(), pad, 0.0f);
    
    // 3. Compute STFT
    std::cout << "Computing STFT..." << std::endl;
    std::cout << "Parameters: win_length=" << win_length << ", hop_length=" << hop_length << std::endl;
    
    // Number of frames
    int num_frames = (processed_audio.size() - win_length) / hop_length + 1;
    if (num_frames <= 0) {
        std::cerr << "Error: Audio too short for given parameters" << std::endl;
        return {};
    }
    
    std::vector<std::vector<std::complex<double>>> stft_result;
    stft_result.reserve(num_frames);
    
    // Process each frame
    for (int i = 0; i < num_frames; i++) {
        int start_idx = i * hop_length;
        
        // Extract frame
        std::vector<std::complex<double>> frame;
        frame.reserve(win_length);
        
		//=======
		/*
        for (int j = 0; j < win_length; j++) {
            if (start_idx + j < processed_audio.size()) {
                frame.push_back(std::complex<double>(processed_audio[start_idx + j], 0.0));
            } else {
                frame.push_back(std::complex<double>(0.0, 0.0));
            }
        }
        */
		for (int j = 0; j < win_length; j++) {
			float sample = (start_idx + j < processed_audio.size()) ? processed_audio[start_idx + j] : 0.0f;
			double windowed = sample * MelSpectrogramUtils::hann_window(j, win_length);
			frame.push_back(std::complex<double>(windowed, 0.0));
		}		
		//=======
		
        // Zero-pad to n_fft if needed
        if (frame.size() < n_fft) {
            frame.resize(n_fft, std::complex<double>(0.0, 0.0));
        }
        
        // Apply FFT
        fft_transform(frame);
        stft_result.push_back(std::move(frame));
    }
    
    std::cout << "STFT computed. Shape: " << stft_result.size() << " x " << stft_result[0].size() << std::endl;
    
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
    std::cout << "Creating mel filterbank..." << std::endl;
    auto mel_filterbank = create_mel_filterbank(current_sr, n_fft, n_mels);
    
    // 6. Transpose power spectrogram for matrix multiplication
    power_spec = transpose(power_spec);
    
    // 7. Apply mel filterbank
    std::cout << "Applying mel filterbank..." << std::endl;
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
    
    // 8. Apply logarithm and normalization
    std::cout << "Applying log transform and normalization..." << std::endl;
    const float epsilon = 1e-5f;
    
    for (int i = 0; i < n_mels; i++) {
        for (int j = 0; j < mel_spec[i].size(); j++) {
            // torch.log(1e-5 + mel_tensor)
            float log_mel = std::log(epsilon + mel_spec[i][j]);
            
            // (log_mel - mean) / std_dev
            mel_spec[i][j] = (log_mel - mean) / std_dev;
        }
    }
    
    std::cout << "Mel spectrogram computed successfully!" << std::endl;
    std::cout << "Output shape: " << n_mels << " x " << mel_spec[0].size() << std::endl;
    
    return mel_spec;
}

//Example
void print_utf16(const std::wstring& text) {
    DWORD written;
    WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE), text.c_str(), text.size(), &written, nullptr);
}

int main() {
    std::locale::global(std::locale(""));
    setlocale(LC_ALL, "");
    // Load WAV file
    AudioFile<float> audioFile;
    //std::wstring filename = L"212_ukr_speedup.wav";
    std::wstring filename = L"212_ukr_speedup_24.wav";

    //if (!audioFile.load(filename)) {
    if (!audioFile.load(std::string(filename.begin(), filename.end()))) {    
        //std::cerr << "Failed to load " << filename << std::endl;
        std::wcout << L"Failed to load " << filename << std::endl;
        return 1;
    }

    // Get audio data
    int sampleRate = audioFile.getSampleRate();
    int numChannels = audioFile.getNumChannels();
    
    if (numChannels > 1) {
        //std::cerr << "Only mono WAV is supported. Using only 1 channel." << std::endl;
        std::wcout << L"Only mono WAV is supported. Using only 1 channel." << std::endl;
    }

    //std::vector<float> audio_data = audioFile.samples[0];  // First channel
    
    std::vector<float> audio_data;

    if (numChannels == 1) {
        audio_data = audioFile.samples[0];
    } else {
        std::wcout << L"File contains " << numChannels << L" channels. Averaging to mono..." << std::endl;
        size_t numSamples = audioFile.samples[0].size();
        audio_data.resize(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < numChannels; ++ch) {
                sum += audioFile.samples[ch][i];
            }
            audio_data[i] = sum / numChannels;
        }
    }    

    // Compute mel spectrogram
    auto mel_tensor = compute_mel_spectrogram(
        audio_data,
        sampleRate,
        80,     // n_mels
        2048,   // n_fft
        1200,   // win_length
        300,    // hop_length
        -4.0f,  // mean
        4.0f,   // std_dev
        false,  // trim
        30.0f,  // top_db
        24000   // target_sr
    );

    // Save result to CSV file
    std::ofstream out("mel_output_cpp.csv");
    if (!out.is_open()) {
        //std::cerr << "Failed to open file for writing." << std::endl;
        std::wcout << L"Failed to open file for writing." << std::endl;
        return 1;
    }

    for (int i = 0; i < mel_tensor.size(); i++) {
        for (int j = 0; j < mel_tensor[i].size(); j++) {
            out << mel_tensor[i][j];
            if (j != mel_tensor[i].size() - 1)
                out << ",";
        }
        out << "\n";
    }

    out.close();
    //std::cout << "Mel spectrogram saved to mel_output.csv" << std::endl;
    std::wcout << L"Mel spectrogram saved to mel_output_cpp.csv" << std::endl; 

    return 0;
}

