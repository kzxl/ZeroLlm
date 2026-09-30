using System;

namespace ZeroLlm.Core.Layers
{
    /// <summary>
    /// Pure C# Swish-Gated Linear Unit (SwiGLU) activation kernel: Swish(gate) * up.
    /// </summary>
    public static class SwiGLU
    {
        public static void Forward(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output)
        {
            if (gate.Length != up.Length || gate.Length != output.Length)
            {
                throw new ArgumentException("Gate, up, and output spans must have identical lengths.");
            }

            int len = gate.Length;
            for (int i = 0; i < len; i++)
            {
                float g = gate[i];
                // Swish(g) = g / (1 + exp(-g))
                float sigmoid = 1.0f / (1.0f + (float)Math.Exp(-g));
                float swish = g * sigmoid;
                output[i] = swish * up[i];
            }
        }
    }
}
