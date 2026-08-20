#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def replace_or_accept(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"unexpected upstream source while patching {label}: {path}")
    path.write_text(text.replace(old, new, 1))


def patch_max_action_heads(kernels: Path) -> None:
    text = kernels.read_text()
    target = "#define MAX_ATN_HEADS 64"
    if target in text:
        return
    for old in ("#define MAX_ATN_HEADS 16", "#define MAX_ATN_HEADS 32"):
        if old in text:
            kernels.write_text(text.replace(old, target, 1))
            return
    raise SystemExit(f"unexpected MAX_ATN_HEADS definition: {kernels}")


def patch_pufferlib(pufferlib: Path) -> None:
    close_old = '''    cudaGraphExecDestroy(pufferl.train_cudagraph);\n    for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {\n        cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);\n    }'''
    close_new = '''    if (pufferl.train_cudagraph) cudaGraphExecDestroy(pufferl.train_cudagraph);\n    if (pufferl.fused_rollout_cudagraphs) {\n        for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {\n            if (pufferl.fused_rollout_cudagraphs[i])\n                cudaGraphExecDestroy(pufferl.fused_rollout_cudagraphs[i]);\n        }\n    }'''
    replace_or_accept(pufferlib, close_old, close_new, "cudagraph cleanup guard")

    train_graph_old = "            bool capturing = pufferl.train_warmup == hypers.cudagraphs;"
    train_graph_new = (
        "            // PufferLib 4.0 train graphs capture a fixed selected minibatch/gradient path.\n"
        "            // Replaying that graph changes weights but reuses stale training work, so keep\n"
        "            // CUDA graphs for rollout inference only and execute each train update normally.\n"
        "            bool capturing = false;"
    )
    replace_or_accept(pufferlib, train_graph_old, train_graph_new, "rollout-only cudagraph mode")

    text = pufferlib.read_text()
    kernel_name = "sample_logits_parallel_heads"
    if kernel_name not in text:
        marker = "// Single step rollout forward pass. Called by each environment worker in their\n"
        if marker not in text:
            raise SystemExit("rollout callback marker not found")
        kernel = r'''
// Parallel MultiDiscrete sampler for large action vectors. One block per agent,
// one thread per action head. Philox draws and per-head sequential softmax math
// preserve the original sampler ordering while heads execute concurrently.
__global__ void sample_logits_parallel_heads(
        PrecisionTensor dec_out,
        IntTensor act_sizes_puf,
        precision_t* __restrict__ actions,
        precision_t* __restrict__ logprobs,
        precision_t* __restrict__ value_out,
        curandStatePhilox4_32_10_t* __restrict__ rng_states) {
    const int idx = blockIdx.x;
    const int h = threadIdx.x;
    const int B = dec_out.shape[0];
    if (idx >= B) return;

    const int num_atns = numel(act_sizes_puf.shape);
    const int* act_sizes = act_sizes_puf.data;
    const int fused_cols = dec_out.shape[1];
    const precision_t* logits = dec_out.data;
    const int logits_base = idx * fused_cols;

    __shared__ int offsets[MAX_ATN_HEADS + 1];
    __shared__ float rand_vals[MAX_ATN_HEADS];
    __shared__ float head_logp[MAX_ATN_HEADS];

    if (h == 0) {
        curandStatePhilox4_32_10_t state = rng_states[idx];
        int offset = 0;
        for (int j = 0; j < num_atns; ++j) {
            offsets[j] = offset;
            rand_vals[j] = curand_uniform(&state);
            offset += act_sizes[j];
        }
        offsets[num_atns] = offset;
        rng_states[idx] = state;
    }
    __syncthreads();

    if (h < num_atns) {
        const int A = act_sizes[h];
        const int logits_offset = offsets[h];
        float max_val = -INFINITY;
        float sum_exp = 0.0f;
        for (int a = 0; a < A; ++a) {
            float l = safe_logit(logits, logits_base, logits_offset, a);
            if (l > max_val) {
                sum_exp *= expf(max_val - l);
                max_val = l;
            }
            sum_exp += expf(l - max_val);
        }
        const float logsumexp = max_val + logf(sum_exp);
        const float rand_val = rand_vals[h];
        float cumsum = 0.0f;
        int sampled_action = A - 1;
        for (int a = 0; a < A; ++a) {
            const float l = safe_logit(logits, logits_base, logits_offset, a);
            const float prob = expf(l - logsumexp);
            cumsum += prob;
            if (rand_val < cumsum) {
                sampled_action = a;
                break;
            }
        }
        const float sampled_logit = safe_logit(logits, logits_base, logits_offset, sampled_action);
        actions[idx * num_atns + h] = from_float(sampled_action);
        head_logp[h] = sampled_logit - logsumexp;
    }
    __syncthreads();

    if (h == 0) {
        float total_log_prob = 0.0f;
        for (int j = 0; j < num_atns; ++j) total_log_prob += head_logp[j];
        logprobs[idx] = from_float(total_log_prob);
        value_out[idx] = dec_out.data[idx * fused_cols + (fused_cols - 1)];
    }
}

'''
        text = text.replace(marker, kernel + marker, 1)

    serial_launch = '''    sample_logits<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        dec_puf, p_logstd, pufferl->act_sizes_puf,
        act_slice.data, lp_slice.data, val_slice.data,
        pufferl->rng_states[buf]);'''
    parallel_launch = '''    if (!dw->continuous) {
        sample_logits_parallel_heads<<<block_size, MAX_ATN_HEADS, 0, stream>>>(
            dec_puf, pufferl->act_sizes_puf, act_slice.data, lp_slice.data, val_slice.data,
            pufferl->rng_states[buf]);
    } else {
        sample_logits<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
            dec_puf, p_logstd, pufferl->act_sizes_puf,
            act_slice.data, lp_slice.data, val_slice.data,
            pufferl->rng_states[buf]);
    }'''
    if parallel_launch not in text:
        if serial_launch not in text:
            raise SystemExit("serial sampler launch block not found")
        text = text.replace(serial_launch, parallel_launch, 1)

    pufferlib.write_text(text)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_puffer_backend.py /path/to/pufferlib")
    root = Path(sys.argv[1]).resolve()
    patch_max_action_heads(root / "src/kernels.cu")
    patch_pufferlib(root / "src/pufferlib.cu")


if __name__ == "__main__":
    main()
