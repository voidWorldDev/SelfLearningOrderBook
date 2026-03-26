#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// gpu/rl_agent_gpu.cuh
//
// GPU-accelerated RL agent using CUDA.
// Implements a batched neural network policy (Actor-Critic / PPO) that
// evaluates many parallel market states simultaneously on the GPU.
//
// ► Requires: CUDA ≥ 11.0, cuBLAS, cuDNN (optional for conv layers)
// ► Enable in CMakeLists.txt:
//       enable_language(CUDA)
//       find_package(CUDAToolkit REQUIRED)
//       target_link_libraries(drut PRIVATE CUDA::cublas CUDA::cudart)
//
// Architecture overview:
//
//   Host (CPU)                          Device (GPU)
//   ──────────────────────────────────────────────────────
//   StateVec[BATCH]  ──cudaMemcpyHtD──► d_states[BATCH × STATE_DIM]
//                                         │
//                                    ┌────▼────────────┐
//                                    │  MLP Forward    │  (fc1→ReLU→fc2→ReLU→fc3)
//                                    │  Batched matmul │  (cuBLAS sgemm)
//                                    └────┬────────────┘
//                                         │
//   ActionVec[BATCH] ◄─cudaMemcpyDtH─── d_actions[BATCH × ACTION_DIM]
//   Value[BATCH]     ◄─cudaMemcpyDtH─── d_values[BATCH]
//
// ─────────────────────────────────────────────────────────────────────────────

// NOTE: This file intentionally left as a stub.
// Uncomment and implement when CUDA is available in your build environment.

/*
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/rl_agent.hpp"

namespace lob {
namespace gpu {

// ── Network Hyperparameters ───────────────────────────────────────────────────
static constexpr int HIDDEN1     = 128;
static constexpr int HIDDEN2     = 64;
static constexpr int BATCH_SIZE  = 256;   // parallel environments

// ── Device Weight Buffers ─────────────────────────────────────────────────────
struct NetworkWeights {
    float* fc1_w;  // [HIDDEN1 × STATE_DIM]
    float* fc1_b;  // [HIDDEN1]
    float* fc2_w;  // [HIDDEN2 × HIDDEN1]
    float* fc2_b;  // [HIDDEN2]
    float* actor_w;// [ACTION_DIM × HIDDEN2]
    float* actor_b;// [ACTION_DIM]
    float* critic_w;// [1 × HIDDEN2]
    float* critic_b;// [1]
};

// ── CUDA Kernels ──────────────────────────────────────────────────────────────

// ReLU activation in-place
__global__ void relu_inplace(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = fmaxf(0.0f, x[idx]);
}

// Softmax over action logits for stochastic policy
__global__ void softmax(float* logits, float* probs, int batch, int actions) {
    int b = blockIdx.x;
    if (b >= batch) return;
    float* row = logits + b * actions;
    float* out = probs  + b * actions;
    float max_v = row[0];
    for (int i = 1; i < actions; ++i) max_v = fmaxf(max_v, row[i]);
    float sum = 0.0f;
    for (int i = 0; i < actions; ++i) { out[i] = expf(row[i] - max_v); sum += out[i]; }
    for (int i = 0; i < actions; ++i) out[i] /= sum;
}

// ── GPURLAgent class ──────────────────────────────────────────────────────────
class GPURLAgent : public lob::RLAgent {
public:
    GPURLAgent(uint32_t id, int action_dim = 2);
    ~GPURLAgent();

    ActionVec act(const StateVec& state)          override;
    void      observe_reward(float reward, bool)  override;

    // Batch inference: evaluate BATCH_SIZE states in one GPU pass
    void batch_act(const float* h_states, float* h_actions, float* h_values,
                   int batch);

    // PPO update step (on GPU)
    void ppo_update(const float* h_states, const float* h_actions,
                    const float* h_returns, const float* h_advantages,
                    int n_samples, int n_epochs = 4);

    void save(const std::string& path) const override;
    void load(const std::string& path)       override;

private:
    cublasHandle_t cublas_;
    NetworkWeights weights_;

    float* d_states_;   // device buffer
    float* d_hidden1_;
    float* d_hidden2_;
    float* d_actions_;
    float* d_values_;

    int action_dim_;

    void alloc_device_buffers();
    void free_device_buffers();
    void forward(int batch);
};

} // namespace gpu
} // namespace lob
*/

// To enable GPU support:
//  1. Rename this file to rl_agent_gpu.cu
//  2. Uncomment the code above
//  3. Add to CMakeLists.txt:
//       enable_language(CUDA)
//       set(CMAKE_CUDA_STANDARD 17)
//       find_package(CUDAToolkit REQUIRED)
//       file(GLOB_RECURSE CUDA_SRC "${CMAKE_SOURCE_DIR}/src/gpu/*.cu")
//       target_sources(drut PRIVATE ${CUDA_SRC})
//       target_link_libraries(drut PRIVATE CUDA::cublas CUDA::cudart)
