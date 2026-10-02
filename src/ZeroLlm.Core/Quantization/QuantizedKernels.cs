using System;
using System.Runtime.CompilerServices;

namespace ZeroLlm.Core.Quantization
{
    /// <summary>
    /// Hardware-optimized fused dot-product and matrix-vector multiplication kernels
    /// operating directly on GGML quantized blocks without heap allocations or full dequantization.
    /// </summary>
    public static unsafe class QuantizedKernels
    {
        /// <summary>
        /// Computes the fused dot product between a float input vector and Q8_0 quantized weight blocks.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductQ8_0(ReadOnlySpan<float> x, ReadOnlySpan<BlockQ8_0> w)
        {
            int blockCount = w.Length;
            if (x.Length < blockCount * BlockQ8_0.BlockSize)
                throw new ArgumentException("Vector x is shorter than required for the quantized weight blocks.");

            float totalSum = 0.0f;

            fixed (float* pX = x)
            fixed (BlockQ8_0* pW = w)
            {
                for (int b = 0; b < blockCount; b++)
                {
                    BlockQ8_0* block = pW + b;
                    float scale = block->GetScale();
                    float* bX = pX + (b * BlockQ8_0.BlockSize);

                    float blockSum = 0.0f;
                    for (int i = 0; i < 32; i += 8)
                    {
                        blockSum += (bX[i] * block->Qs[i])
                                  + (bX[i + 1] * block->Qs[i + 1])
                                  + (bX[i + 2] * block->Qs[i + 2])
                                  + (bX[i + 3] * block->Qs[i + 3])
                                  + (bX[i + 4] * block->Qs[i + 4])
                                  + (bX[i + 5] * block->Qs[i + 5])
                                  + (bX[i + 6] * block->Qs[i + 6])
                                  + (bX[i + 7] * block->Qs[i + 7]);
                    }

                    totalSum += blockSum * scale;
                }
            }

            return totalSum;
        }

        /// <summary>
        /// Computes the fused dot product between a float input vector and Q4_0 quantized weight blocks.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductQ4_0(ReadOnlySpan<float> x, ReadOnlySpan<BlockQ4_0> w)
        {
            int blockCount = w.Length;
            if (x.Length < blockCount * BlockQ4_0.BlockSize)
                throw new ArgumentException("Vector x is shorter than required for the quantized weight blocks.");

            float totalSum = 0.0f;

            fixed (float* pX = x)
            fixed (BlockQ4_0* pW = w)
            {
                for (int b = 0; b < blockCount; b++)
                {
                    BlockQ4_0* block = pW + b;
                    float scale = block->GetScale();
                    float* bX = pX + (b * BlockQ4_0.BlockSize);

                    float blockSum = 0.0f;
                    for (int i = 0; i < 16; i += 4)
                    {
                        byte val0 = block->Qs[i];
                        byte val1 = block->Qs[i + 1];
                        byte val2 = block->Qs[i + 2];
                        byte val3 = block->Qs[i + 3];

                        blockSum += (bX[i] * ((val0 & 0x0F) - 8)) + (bX[i + 16] * ((val0 >> 4) - 8))
                                  + (bX[i + 1] * ((val1 & 0x0F) - 8)) + (bX[i + 17] * ((val1 >> 4) - 8))
                                  + (bX[i + 2] * ((val2 & 0x0F) - 8)) + (bX[i + 18] * ((val2 >> 4) - 8))
                                  + (bX[i + 3] * ((val3 & 0x0F) - 8)) + (bX[i + 19] * ((val3 >> 4) - 8));
                    }

                    totalSum += blockSum * scale;
                }
            }

            return totalSum;
        }

        /// <summary>
        /// Multiplies a matrix of Q8_0 blocks with a float vector x, writing into destination vector y: y = W * x.
        /// </summary>
        public static void MatVecMulQ8_0(
            ReadOnlySpan<BlockQ8_0> weightMatrix,
            ReadOnlySpan<float> x,
            Span<float> y,
            int rows,
            int cols)
        {
            int blocksPerRow = cols / BlockQ8_0.BlockSize;
            if (weightMatrix.Length < rows * blocksPerRow)
                throw new ArgumentException("Weight matrix dimension mismatch.");

            for (int r = 0; r < rows; r++)
            {
                var rowBlocks = weightMatrix.Slice(r * blocksPerRow, blocksPerRow);
                y[r] = DotProductQ8_0(x, rowBlocks);
            }
        }

        /// <summary>
        /// Multiplies a matrix of Q4_0 blocks with a float vector x, writing into destination vector y: y = W * x.
        /// </summary>
        public static void MatVecMulQ4_0(
            ReadOnlySpan<BlockQ4_0> weightMatrix,
            ReadOnlySpan<float> x,
            Span<float> y,
            int rows,
            int cols)
        {
            int blocksPerRow = cols / BlockQ4_0.BlockSize;
            if (weightMatrix.Length < rows * blocksPerRow)
                throw new ArgumentException("Weight matrix dimension mismatch.");

            for (int r = 0; r < rows; r++)
            {
                var rowBlocks = weightMatrix.Slice(r * blocksPerRow, blocksPerRow);
                y[r] = DotProductQ4_0(x, rowBlocks);
            }
        }

        /// <summary>
        /// Multiplies a matrix of Q8_0 blocks with a float vector x, accumulating into destination vector y: y += W * x.
        /// </summary>
        public static void MatVecAddQ8_0(
            ReadOnlySpan<BlockQ8_0> weightMatrix,
            ReadOnlySpan<float> x,
            Span<float> y,
            int rows,
            int cols)
        {
            int blocksPerRow = cols / BlockQ8_0.BlockSize;
            if (weightMatrix.Length < rows * blocksPerRow)
                throw new ArgumentException("Weight matrix dimension mismatch.");

            for (int r = 0; r < rows; r++)
            {
                var rowBlocks = weightMatrix.Slice(r * blocksPerRow, blocksPerRow);
                y[r] += DotProductQ8_0(x, rowBlocks);
            }
        }

        /// <summary>
        /// Multiplies a matrix of Q4_0 blocks with a float vector x, accumulating into destination vector y: y += W * x.
        /// </summary>
        public static void MatVecAddQ4_0(
            ReadOnlySpan<BlockQ4_0> weightMatrix,
            ReadOnlySpan<float> x,
            Span<float> y,
            int rows,
            int cols)
        {
            int blocksPerRow = cols / BlockQ4_0.BlockSize;
            if (weightMatrix.Length < rows * blocksPerRow)
                throw new ArgumentException("Weight matrix dimension mismatch.");

            for (int r = 0; r < rows; r++)
            {
                var rowBlocks = weightMatrix.Slice(r * blocksPerRow, blocksPerRow);
                y[r] += DotProductQ4_0(x, rowBlocks);
            }
        }
    }
}
