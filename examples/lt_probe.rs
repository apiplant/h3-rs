use candle_core::cuda_backend::cudarc::cublaslt::{result::*, sys::*};
fn main() {
    let _dev = candle_core::Device::new_cuda(0).unwrap();
    let h = create_handle().unwrap();
    for (name, ct, st, dt) in [
        ("32I/32F -> 32F", cublasComputeType_t::CUBLAS_COMPUTE_32I, cudaDataType::CUDA_R_32F, cudaDataType::CUDA_R_32F),
        ("32I/32F -> 16BF", cublasComputeType_t::CUBLAS_COMPUTE_32I, cudaDataType::CUDA_R_32F, cudaDataType::CUDA_R_16BF),
        ("32I/32I -> 32I", cublasComputeType_t::CUBLAS_COMPUTE_32I, cudaDataType::CUDA_R_32I, cudaDataType::CUDA_R_32I),
    ] {
        unsafe {
            let desc = create_matmul_desc(ct, st).unwrap();
            let opt = candle_core::cuda_backend::cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
            set_matmul_desc_attribute(desc, cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &opt as *const _ as *const _, 4).unwrap();
            let (k, n, m) = (5376u64, 28672u64, 4096u64);
            let la = create_matrix_layout(cudaDataType::CUDA_R_8I, k, n, k as i64).unwrap();
            let lb = create_matrix_layout(cudaDataType::CUDA_R_8I, k, m, k as i64).unwrap();
            let lc = create_matrix_layout(dt, n, m, n as i64).unwrap();
            let pref = create_matmul_pref().unwrap();
            let r = get_matmul_algo_heuristic(h, desc, la, lb, lc, lc, pref);
            println!("{name}: {}", if r.is_ok() { "supported" } else { "NOT supported" });
        }
    }
}
