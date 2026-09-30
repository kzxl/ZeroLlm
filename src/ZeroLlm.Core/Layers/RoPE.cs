using System;

namespace ZeroLlm.Core.Layers
{
    /// <summary>
    /// Pure C# Rotary Position Embedding (RoPE) kernel for Query and Key projections.
    /// </summary>
    public static class RoPE
    {
        /// <summary>
        /// Applies in-place rotary position embedding to a head vector of length headDim.
        /// </summary>
        public static void ApplyInplace(Span<float> vector, int pos, int headDim, float freqBase = 10000.0f)
        {
            if (headDim % 2 != 0)
            {
                throw new ArgumentException("Head dimension must be even for RoPE.", nameof(headDim));
            }

            int halfDim = headDim / 2;
            for (int i = 0; i < halfDim; i++)
            {
                double exponent = 2.0 * i / headDim;
                double freq = 1.0 / Math.Pow(freqBase, exponent);
                double theta = pos * freq;

                float cosTheta = (float)Math.Cos(theta);
                float sinTheta = (float)Math.Sin(theta);

                int idx0 = 2 * i;
                int idx1 = 2 * i + 1;

                float x0 = vector[idx0];
                float x1 = vector[idx1];

                vector[idx0] = x0 * cosTheta - x1 * sinTheta;
                vector[idx1] = x0 * sinTheta + x1 * cosTheta;
            }
        }
    }
}
