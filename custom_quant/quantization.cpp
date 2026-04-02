#include <torch/extension.h>
#include <vector>
#include <cmath>

// A very simplified block-wise 4-bit quantization simulation.
// In true NF4, the quantiles are optimized for Normal distributions.
// Here, we'll map floats linearly to 16 buckets for educational simplicity.
// For memory-tightness, we pack two 4-bit values into one 8-bit unsigned char.

std::pair<torch::Tensor, torch::Tensor> quantize_4bit(torch::Tensor input, int block_size) {
    // Ensuring input is on CPU and float32 and contiguous for this basic implementation.
    auto in = input.contiguous().to(torch::kFloat32).cpu();
    int64_t numel = in.numel();
    int num_blocks = (numel + block_size - 1) / block_size;
    
    // Outputs
    // packed_weights: 1 uint8_t stores 2 4-bit values
    auto packed_weights = torch::empty({(numel + 1) / 2}, torch::kUInt8);
    auto block_scales = torch::empty({num_blocks}, torch::kFloat32);

    float* in_data = in.data_ptr<float>();
    uint8_t* out_data = packed_weights.data_ptr<uint8_t>();
    float* scale_data = block_scales.data_ptr<float>();

    for(int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = std::min((int)numel, start + block_size);
        
        // Find max absolute value in the block for scaling (AbsMax quantization)
        float max_abs = 0.0f;
        for(int i = start; i < end; i++) {
            if(std::abs(in_data[i]) > max_abs) max_abs = std::abs(in_data[i]);
        }
        
        // Quantization step: the maximum value mapping to 4-bit signed [-8, 7]
        float scale = max_abs / 7.0f;
        if (scale == 0) scale = 1e-9f; // prevent division by zero
        scale_data[b] = scale;

        // Pack the values into 4-bit
        for(int i = start; i < end; i += 2) {
            // value 1
            float v1 = in_data[i] / scale;
            int8_t q1 = std::clamp((int)std::round(v1), -8, 7);
            
            // value 2 (if exists)
            int8_t q2 = 0;
            if(i + 1 < end) {
                float v2 = in_data[i+1] / scale;
                q2 = std::clamp((int)std::round(v2), -8, 7);
            }
            
            // Pack q1 into lower 4 bits, q2 into upper 4 bits
            // Offset by 8 to make them unsigned [0, 15]
            uint8_t u1 = (uint8_t)(q1 + 8) & 0x0F;
            uint8_t u2 = (uint8_t)(q2 + 8) & 0x0F;
            
            out_data[i / 2] = u1 | (u2 << 4);
        }
    }

    return {packed_weights, block_scales};
}

// Dequantize the 4-bit values back to Float32
torch::Tensor dequantize_4bit(torch::Tensor packed_weights, torch::Tensor block_scales, std::vector<int64_t> original_shape, int block_size) {
    auto packed = packed_weights.contiguous().cpu();
    auto scales = block_scales.contiguous().cpu();
    
    // Calculate total elements based on original shape
    int64_t numel = 1;
    for(auto dim : original_shape) numel *= dim;
    int num_blocks = scales.numel();
    
    auto output = torch::empty({numel}, torch::kFloat32);

    uint8_t* packed_data = packed.data_ptr<uint8_t>();
    float* scale_data = scales.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    for(int b = 0; b < num_blocks; b++) {
        int start = b * block_size;
        int end = std::min((int)numel, start + block_size);
        float scale = scale_data[b];

        for(int i = start; i < end; i += 2) {
            uint8_t p = packed_data[i / 2];
            uint8_t u1 = p & 0x0F;
            uint8_t u2 = (p >> 4) & 0x0F;
            
            int8_t q1 = (int8_t)u1 - 8;
            int8_t q2 = (int8_t)u2 - 8;
            
            out_data[i] = (float)q1 * scale;
            if(i + 1 < end) {
                out_data[i+1] = (float)q2 * scale;
            }
        }
    }

    return output.view(original_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_4bit", &quantize_4bit, "Quantize a float tensor to 4-bit packed layout");
    m.def("dequantize_4bit", &dequantize_4bit, "Dequantize 4-bit packed layout to float tensor");
}
