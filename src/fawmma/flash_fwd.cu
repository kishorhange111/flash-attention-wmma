// FlashAttention-2 forward pass on tensor cores via WMMA (sm_70+; written for the T4, sm_75).
//
// One thread block computes a 64-row tile of queries for one (batch, head). Its 4 warps own
// 16 query rows each. For every 64-row tile of keys and values:
//   1. K and V tiles are copied to shared memory (all 128 threads, 16-byte vector loads).
//   2. S = Q Kᵀ: each warp multiplies its 16×64 Q fragments (kept in registers for the
//      whole loop) by Kᵀ, i.e. 4 output fragments × 4 k-steps of 16×16×16 HMMA.
//   3. Online softmax, in shared memory. Two lanes per row hold 32 scores each and
//      combine their max and sum with one shuffle. The running max m and denominator l
//      are updated, P = exp(S − m_new) is written as fp16 over the K tile, which is no
//      longer needed, and the warp's rows of the fp32 output accumulator O are rescaled by
//      exp(m_old − m_new).
//   4. O += P V: 4 output fragments × 4 k-steps of HMMA. The result lands in the S buffer
//      and is added to O.
// At the end O / l is written as fp16, along with the log-sum-exp per row.
//
// Shared memory: K/P 8 KiB + V 8 KiB + S 16 KiB + O 16 KiB + m, l = 48.5 KiB, which needs
// the opt-in above the 48 KiB static limit.
//
// Constraints: head dim 64, sequence length a multiple of 64, contiguous (B, H, N, 64)
// fp16 tensors. Causal masking skips key tiles past the query tile entirely and masks only
// on the diagonal tile.

#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

constexpr int D = 64;    // head dim
constexpr int BR = 64;   // query rows per block
constexpr int BC = 64;   // key rows per tile
constexpr int WARPS = 4; // 16 query rows each
constexpr int THREADS = WARPS * 32;

struct Smem {
  half kp[BC * D];   // K tile, then P (fp16) for the same tile
  half v[BC * D];
  float s[BR * BC];  // scores, then P·V partial result
  float o[BR * D];   // running output accumulator
  float m[BR];
  float l[BR];
};

__device__ __forceinline__ void load_tile(half* dst, const half* src, int tid) {
  // 64×64 halves = 512 × 16-byte vectors; 128 threads × 4
  const int4* s4 = reinterpret_cast<const int4*>(src);
  int4* d4 = reinterpret_cast<int4*>(dst);
#pragma unroll
  for (int i = 0; i < 4; ++i) d4[tid + i * THREADS] = s4[tid + i * THREADS];
}

__global__ void __launch_bounds__(THREADS)
flash_fwd_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                 const half* __restrict__ V, half* __restrict__ O, float* __restrict__ LSE,
                 int N, float scale_log2, bool causal) {
  extern __shared__ __align__(32) unsigned char raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int q_tile = blockIdx.x, bh = blockIdx.y;
  const size_t base = (size_t)bh * N * D;
  const half* q = Q + base + (size_t)q_tile * BR * D;

  // Q fragments for this warp's 16 rows: 4 k-steps of 16 columns, kept in registers.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) wmma::load_matrix_sync(qf[kk], q + warp * 16 * D + kk * 16, D);

  for (int i = tid; i < BR * D; i += THREADS) sm.o[i] = 0.f;
  for (int i = tid; i < BR; i += THREADS) { sm.m[i] = -INFINITY; sm.l[i] = 0.f; }

  const int row_in_warp = lane / 2, half_id = lane % 2;  // 2 lanes per row, 32 cols each
  const int row = warp * 16 + row_in_warp;                // row within the 64-row tile
  const int q_pos = q_tile * BR + row;
  const int n_tiles = causal ? q_tile + 1 : N / BC;

  for (int t = 0; t < n_tiles; ++t) {
    __syncthreads();  // previous tile's P and V are no longer in use
    load_tile(sm.kp, K + base + (size_t)t * BC * D, tid);
    load_tile(sm.v, V + base + (size_t)t * BC * D, tid);
    __syncthreads();

    // ---- S = Q Kᵀ for this warp's 16 rows × 64 key columns
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        // element (d, n) of Kᵀ is K[n][d]: column-major with leading dim D
        wmma::load_matrix_sync(kf, sm.kp + j * 16 * D + kk * 16, D);
        wmma::mma_sync(acc, qf[kk], kf, acc);
      }
      wmma::store_matrix_sync(sm.s + warp * 16 * BC + j * 16, acc, BC, wmma::mem_row_major);
    }
    __syncwarp();

    // ---- online softmax on this lane's 32 scores of row `row`
    float* srow = sm.s + row * BC + half_id * 32;
    const bool diag = causal && t == q_tile;
    float mx = -INFINITY;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const int k_pos = t * BC + half_id * 32 + c;
      float x = srow[c] * scale_log2;
      if (diag && k_pos > q_pos) x = -INFINITY;
      srow[c] = x;
      mx = fmaxf(mx, x);
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    const float m_old = sm.m[row];
    const float m_new = fmaxf(m_old, mx);  // finite: every row sees at least its own key
    float sum = 0.f;
    half* prow = sm.kp + row * BC + half_id * 32;  // overwrite K: only this warp's rows
    __syncthreads();  // ...but every warp must finish reading K before anyone writes P
#pragma unroll
    for (int c = 0; c < 32; ++c) {
      const float p = exp2f(srow[c] - m_new);
      sum += p;
      prow[c] = __float2half(p);
    }
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    const float alpha = exp2f(m_old - m_new);
    float* orow = sm.o + row * D + half_id * 32;
#pragma unroll
    for (int c = 0; c < 32; ++c) orow[c] *= alpha;
    __syncwarp();
    if (half_id == 0) { sm.m[row] = m_new; sm.l[row] = sm.l[row] * alpha + sum; }
    __syncwarp();

    // ---- O += P V  (P: 16×64 rows of this warp; V: 64×64)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        wmma::load_matrix_sync(pf, sm.kp + warp * 16 * BC + kk * 16, BC);
        wmma::load_matrix_sync(vf, sm.v + kk * 16 * D + j * 16, D);
        wmma::mma_sync(acc, pf, vf, acc);
      }
      wmma::store_matrix_sync(sm.s + warp * 16 * BC + j * 16, acc, BC, wmma::mem_row_major);
    }
    __syncwarp();
    const float* pv = sm.s + row * BC + half_id * 32;
#pragma unroll
    for (int c = 0; c < 32; ++c) orow[c] += pv[c];
    __syncwarp();
  }

  // ---- epilogue: O / l, and LSE in natural-log units
  const float inv_l = 1.f / sm.l[row];
  half* out = O + base + (size_t)q_pos * D + half_id * 32;
  const float* orow = sm.o + row * D + half_id * 32;
#pragma unroll
  for (int c = 0; c < 32; ++c) out[c] = __float2half(orow[c] * inv_l);
  if (half_id == 0) LSE[(size_t)bh * N + q_pos] = (sm.m[row] + log2f(sm.l[row])) * 0.69314718f;
}

std::vector<torch::Tensor> flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                     bool causal) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf, "q must be CUDA fp16");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "contiguous only");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == D, "expected (B, H, N, 64)");
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k, v shapes must match");
  const int B = q.size(0), H = q.size(1), N = q.size(2);
  TORCH_CHECK(N % BR == 0, "sequence length must be a multiple of 64");
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, N}, q.options().dtype(torch::kFloat));
  const size_t smem = sizeof(Smem);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(flash_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  const float scale_log2 = 1.4426950408889634f / sqrtf((float)D);
  dim3 grid(N / BR, B * H);
  flash_fwd_kernel<<<grid, THREADS, smem>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(o.data_ptr<at::Half>()), lse.data_ptr<float>(), N, scale_log2,
      causal);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("flash_fwd", &flash_fwd, "FlashAttention-2 fwd (WMMA)"); }
