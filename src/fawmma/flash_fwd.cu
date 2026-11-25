// FlashAttention-2 forward pass on tensor cores via WMMA (sm_70+; written for the T4, sm_75).
//
// One thread block computes a 64-row tile of queries for one (batch, head). Its 4 warps own
// 16 query rows each. For every 64-row tile of keys and values:
//   1. K and V tiles are copied to shared memory (all 128 threads, 16-byte vector loads).
//   2. S = Q Kᵀ: each warp multiplies its 16×64 Q fragments (kept in registers for the
//      whole loop) by Kᵀ, i.e. 4 output fragments × 4 k-steps of 16×16×16 HMMA.
//   3. Online softmax, in shared memory. Two lanes per row hold 32 scores each (even and
//      odd columns) and combine their max and sum with one shuffle. The running max m and
//      denominator l are updated, P = exp(S − m_new) is written as fp16 over the K tile,
//      which is no longer needed, and the warp's rows of the fp32 output accumulator O are
//      rescaled by exp(m_old − m_new).
//   4. O += P V: each of the 4 O fragments is loaded as the accumulator, gets 4 k-steps of
//      HMMA, and is stored back.
// At the end O / l is written as fp16, along with the log-sum-exp per row.
//
// Two layouts are compiled from one template:
//   * unpadded (v1): row strides of 64 elements. A warp's 16 rows then start on the same
//     shared-memory bank, so the softmax and rescale loops, where each lane walks its own
//     row, hit 16-way bank conflicts.
//   * padded (v2): fp32 rows are 68 floats and fp16 rows 72 halves (the smallest strides
//     WMMA accepts that shift each row by 4 banks), which leaves 2-way conflicts.
// Shared memory: 48.5 KiB unpadded, 52.5 KiB padded; both need the opt-in above 48 KiB.
//
// Constraints: head dim 64, sequence length a multiple of 64, contiguous (B, H, N, 64)
// fp16 tensors. Causal masking skips key tiles past the query tile entirely and masks only
// on the diagonal tile.

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

constexpr int D = 64;    // head dim
constexpr int BR = 64;   // query rows per block
constexpr int BC = 64;   // key rows per tile
constexpr int WARPS = 4; // 16 query rows each
constexpr int THREADS = WARPS * 32;

template <int HS, int FS>  // row stride of fp16 tiles (halves) and fp32 tiles (floats)
struct Smem {
  half kp[BC * HS];  // K tile, then P (fp16) for the same tile
  half v[BC * HS];
  float s[BR * FS];  // scores
  float o[BR * FS];  // running output accumulator
  float m[BR];
  float l[BR];
};

template <int HS>
__device__ __forceinline__ void load_tile(half* dst, const half* src, int tid) {
  // 64×64 halves = 512 × 16-byte vectors (8 per row); 128 threads × 4
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int idx = tid + i * THREADS, r = idx / 8, c = (idx % 8) * 8;
    *reinterpret_cast<int4*>(dst + r * HS + c) = *reinterpret_cast<const int4*>(src + r * D + c);
  }
}

template <int HS, int FS>
__global__ void __launch_bounds__(THREADS)
flash_fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                 const half* __restrict__ V, half* __restrict__ O, float* __restrict__ LSE,
                 int N, float scale_log2, bool causal) {
  extern __shared__ __align__(32) unsigned char raw[];
  Smem<HS, FS>& sm = *reinterpret_cast<Smem<HS, FS>*>(raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int q_tile = blockIdx.x, bh = blockIdx.y;
  const size_t base = (size_t)bh * N * D;
  const half* q = Q + base + (size_t)q_tile * BR * D;

  // Q fragments for this warp's 16 rows: 4 k-steps of 16 columns, kept in registers.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) wmma::load_matrix_sync(qf[kk], q + warp * 16 * D + kk * 16, D);

  for (int i = tid; i < BR * FS; i += THREADS) sm.o[i] = 0.f;
  for (int i = tid; i < BR; i += THREADS) { sm.m[i] = -INFINITY; sm.l[i] = 0.f; }

  const int row_in_warp = lane / 2, h = lane % 2;  // 2 lanes per row: even / odd columns
  const int row = warp * 16 + row_in_warp;         // row within the 64-row tile
  const int q_pos = q_tile * BR + row;
  const int n_tiles = causal ? q_tile + 1 : N / BC;
  float* srow = sm.s + row * FS;
  float* orow = sm.o + row * FS;
  half* prow = sm.kp + row * HS;

  for (int t = 0; t < n_tiles; ++t) {
    __syncthreads();  // previous tile's P and V are no longer in use
    load_tile<HS>(sm.kp, K + base + (size_t)t * BC * D, tid);
    load_tile<HS>(sm.v, V + base + (size_t)t * BC * D, tid);
    __syncthreads();

    // ---- S = Q Kᵀ for this warp's 16 rows × 64 key columns
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        // element (d, n) of Kᵀ is K[n][d]: column-major with leading dim HS
        wmma::load_matrix_sync(kf, sm.kp + j * 16 * HS + kk * 16, HS);
        wmma::mma_sync(acc, qf[kk], kf, acc);
      }
      wmma::store_matrix_sync(sm.s + warp * 16 * FS + j * 16, acc, FS, wmma::mem_row_major);
    }
    __syncwarp();

    // ---- online softmax on this lane's 32 scores (columns 2c + h) of row `row`
    const bool diag = causal && t == q_tile;
    float mx = -INFINITY;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int col = 2 * c + h;
      float x = srow[col] * scale_log2;
      if (diag && t * BC + col > q_pos) x = -INFINITY;
      srow[col] = x;
      mx = fmaxf(mx, x);
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    const float m_old = sm.m[row];
    const float m_new = fmaxf(m_old, mx);  // finite: every row sees at least its own key
    float sum = 0.f;
    __syncthreads();  // P overwrites K: every warp must finish reading K first
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int col = 2 * c + h;
      const float p = exp2f(srow[col] - m_new);
      sum += p;
      prow[col] = __float2half(p);
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    const float alpha = exp2f(m_old - m_new);
#pragma unroll
    for (int c = 0; c < 32; ++c) orow[2 * c + h] *= alpha;
    __syncwarp();
    if (h == 0) { sm.m[row] = m_new; sm.l[row] = sm.l[row] * alpha + sum; }
    __syncwarp();

    // ---- O += P V  (P: 16×64 rows of this warp; V: 64×64), accumulating in place
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      float* o_tile = sm.o + warp * 16 * FS + j * 16;
      wmma::load_matrix_sync(acc, o_tile, FS, wmma::mem_row_major);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        wmma::load_matrix_sync(pf, sm.kp + warp * 16 * HS + kk * 16, HS);
        wmma::load_matrix_sync(vf, sm.v + kk * 16 * HS + j * 16, HS);
        wmma::mma_sync(acc, pf, vf, acc);
      }
      wmma::store_matrix_sync(o_tile, acc, FS, wmma::mem_row_major);
    }
    __syncwarp();
  }

  // ---- epilogue: O / l, and LSE in natural-log units
  const float inv_l = 1.f / sm.l[row];
  half* out = O + base + (size_t)q_pos * D;
#pragma unroll
  for (int c = 0; c < 32; ++c) out[2 * c + h] = __float2half(orow[2 * c + h] * inv_l);
  if (h == 0) LSE[(size_t)bh * N + q_pos] = (sm.m[row] + log2f(sm.l[row])) * 0.69314718f;
}

template <int HS, int FS>
std::vector<torch::Tensor> launch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  bool causal) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf, "q must be CUDA fp16");
  TORCH_CHECK(k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf, "fp16 only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous only");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == D, "expected (B, H, N, 64)");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k, v shapes must match");
  const int B = q.size(0), H = q.size(1), N = q.size(2);
  TORCH_CHECK(N % BR == 0, "sequence length must be a multiple of 64");
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, N}, q.options().dtype(torch::kFloat));
  const size_t smem = sizeof(Smem<HS, FS>);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(flash_fwd_kernel<HS, FS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  const float scale_log2 = 1.4426950408889634f / sqrtf((float)D);
  dim3 grid(N / BR, B * H);
  flash_fwd_kernel<HS, FS><<<grid, THREADS, smem>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(o.data_ptr<at::Half>()), lse.data_ptr<float>(), N, scale_log2,
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, lse};
}

std::vector<torch::Tensor> flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                     bool causal, bool padded) {
  return padded ? launch<72, 68>(q, k, v, causal) : launch<64, 64>(q, k, v, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_fwd", &flash_fwd, "FlashAttention-2 forward (WMMA)");
}
