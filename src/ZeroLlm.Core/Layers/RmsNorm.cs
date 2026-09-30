using System;

namespace ZeroLlm.Core.Layers
{
    /// <summary>
    /// Pure C# Root Mean Square Layer Normalization (RMSNorm) kernel.
    /// </summary>
    public static class RmsNorm
    {
        public static void Forward(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output, float epsilon = 1e-5f)
        {
            if (input.Length != weight.Length || input.Length != output.Length)
            {
                throw new ArgumentException("Input, weight, and output spans must have identical lengths.");
            }

            int len = input.Length;
            float sumSquares = 0.0f;
            for (int i = 0; i < len; i++)
            {
                float v = input[i];
                sumSquares += v * v;
            }

            float meanSquare = sumSquares / len;
            float invRms = 1.0f / (float)Math.Sqrt(meanSquare + epsilon);

            for (int i = 0; i < len; i++)
            {
                output[i] = input[i] * invRms * weight[i];
            }
        }
    }
}
