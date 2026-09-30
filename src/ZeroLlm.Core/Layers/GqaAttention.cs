using System;
using ZeroLlm.Core.Memory;

namespace ZeroLlm.Core.Layers
{
    /// <summary>
    /// Pure C# Grouped-Query Attention (GQA) execution kernel with Paged KV-Cache retrieval.
    /// </summary>
    public static class GqaAttention
    {
        public static void Forward(
            ReadOnlySpan<float> q,
            int seqId,
            int layer,
            PagedKvCache kvCache,
            int qHeadCount,
            int kvHeadCount,
            int headDim,
            Span<float> output)
        {
            int groupSize = qHeadCount / kvHeadCount;
            float scale = 1.0f / (float)Math.Sqrt(headDim);
            int seqLen = kvCache.GetSequenceLength(seqId);

            if (seqLen == 0)
            {
                output.Clear();
                return;
            }

            // Scratch buffer for attention weights across context length
            Span<float> scores = stackalloc float[Math.Min(seqLen, 1024)];
            float[]? rentedScores = null;
            if (seqLen > 1024)
            {
                rentedScores = new float[seqLen];
                scores = rentedScores;
            }

            try
            {
                for (int qh = 0; qh < qHeadCount; qh++)
                {
                    int kvHead = qh / groupSize;
                    var qSpan = q.Slice(qh * headDim, headDim);
                    var outSpan = output.Slice(qh * headDim, headDim);
                    outSpan.Clear();

                    float maxScore = float.NegativeInfinity;

                    // 1. Compute dot-product attention scores
                    for (int t = 0; t < seqLen; t++)
                    {
                        var keyAllHeads = kvCache.GetKey(seqId, layer, t);
                        var kSpan = keyAllHeads.Slice(kvHead * headDim, headDim);

                        float dot = 0.0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += qSpan[d] * kSpan[d];
                        }

                        float s = dot * scale;
                        scores[t] = s;
                        if (s > maxScore) maxScore = s;
                    }

                    // 2. Softmax normalization
                    float sumExp = 0.0f;
                    for (int t = 0; t < seqLen; t++)
                    {
                        float exp = (float)Math.Exp(scores[t] - maxScore);
                        scores[t] = exp;
                        sumExp += exp;
                    }

                    float invSum = sumExp > 0 ? 1.0f / sumExp : 0.0f;

                    // 3. Weighted accumulation of Value vectors
                    for (int t = 0; t < seqLen; t++)
                    {
                        float weight = scores[t] * invSum;
                        if (weight <= 0.0f) continue;

                        var valAllHeads = kvCache.GetValue(seqId, layer, t);
                        var vSpan = valAllHeads.Slice(kvHead * headDim, headDim);

                        for (int d = 0; d < headDim; d++)
                        {
                            outSpan[d] += weight * vSpan[d];
                        }
                    }
                }
            }
            finally
            {
                rentedScores = null;
            }
        }
    }
}
