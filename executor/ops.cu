#include <cudnn.h>
#include <assert.h>
#include "ops.h"
#include "utils.h"


__global__ void assign_with_stride_dst(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size) {
    CUDA_KERNEL_LOOP(i, n) {
        int blk_idx = i / dst_blk_size;
        int blk_offset = i % dst_blk_size;
        int src_offset = blk_idx * src_blk_size + blk_offset;
        int dst_offset = blk_idx * dst_blk_size + blk_offset;
        dst[dst_offset] = src[src_offset];
    }
}


__global__ void assign_with_stride_src(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size) {
    CUDA_KERNEL_LOOP(i, n) {
        int blk_idx = i / src_blk_size;
        int blk_offset = i % src_blk_size;
        int src_offset = blk_idx * src_blk_size + blk_offset;
        int dst_offset = blk_idx * dst_blk_size + blk_offset;
        dst[dst_offset] = src[src_offset];
    }
}




__global__ void accumulate_sum_2(data_type *dst, const data_type *src1, const data_type *src2, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i];
    }
}




__global__ void accumulate_sum_3(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i];
    }
}



__global__ void accumulate_sum_4(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i] + src4[i];
    }
}



__global__ void accumulate_sum_5(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, const data_type *src5, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        dst[i] = src1[i] + src2[i] + src3[i] + src4[i] + src5[i];
    }
}





extern void assign_with_stride_dst_call(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size, cudaStream_t stream ){
        assign_with_stride_dst<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src, n, dst_blk_size, src_blk_size);
}
extern void assign_with_stride_src_call(data_type *dst, const data_type *src, int n, int dst_blk_size, int src_blk_size, cudaStream_t stream ) {
        assign_with_stride_src<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src, n, dst_blk_size, src_blk_size);
}
extern void accumulate_sum_2_call(data_type *dst, const data_type *src1, const data_type *src2, int n, cudaStream_t stream ){
        accumulate_sum_2 <<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src1, src2, n);
}
extern void accumulate_sum_3_call(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, int n, cudaStream_t stream ){
        accumulate_sum_3<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src1, src2, src3, n);
}
extern void accumulate_sum_4_call(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, int n, cudaStream_t stream ){
        accumulate_sum_4<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src1, src2, src3, src4, n);
}
extern void accumulate_sum_5_call(data_type *dst, const data_type *src1, const data_type *src2, const data_type *src3, const data_type *src4, const data_type *src5, int n, cudaStream_t stream ){
        accumulate_sum_5<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>> (dst, src1, src2, src3, src4, src5, n);
}

