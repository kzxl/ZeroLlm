using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ZeroLlm.Core.Layers;
using ZeroLlm.Core.Memory;
using ZeroLlm.Core.Quantization;
using ZeroLlm.Core.Sampling;
using ZeroTokenizer.Core.Abstractions;

namespace ZeroLlm.Core.Engine
{
    /// <summary>
    /// Pure C# Small Language Model execution engine.
    /// </summary>
    public sealed class LlmEngine : ILlmEngine
    {
        private readonly LlmModel _model;
        private readonly ITokenizer _tokenizer;
        private readonly PagedKvCache _kvCache;
        private readonly SamplingConfig _defaultSampling;
        private int _nextSeqId = 1;
        private bool _disposed;

        public LlmModel Model => _model;
        public ITokenizer Tokenizer => _tokenizer;
        public PagedKvCache KvCache => _kvCache;

        public LlmEngine(LlmModel model, ITokenizer tokenizer, SamplingConfig? defaultSampling = null, PagedKvCache? kvCache = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _defaultSampling = defaultSampling ?? new SamplingConfig();

            var cfg = _model.Config;
            _kvCache = kvCache ?? new PagedKvCache(
                totalBlocks: 256,
                blockSize: 16,
                layerCount: cfg.LayerCount,
                kvHeadCount: cfg.HeadCountKv,
                headDim: cfg.HeadDim);

            // Precompute RoPE trigonometric cache for context length
            int maxPositions = Math.Max(1024, cfg.ContextLength);
            RoPE.EnsureCache(maxPositions, cfg.HeadDim, cfg.RopeFreqBase);
        }

        public Task<string> CompleteAsync(string prompt, CancellationToken cancellationToken = default)
            => CompleteAsync(prompt, _defaultSampling, null, cancellationToken);

        public async Task<string> CompleteAsync(
            string prompt,
            SamplingConfig? samplingConfig,
            LogitProcessor? logitProcessor = null,
            CancellationToken cancellationToken = default)
        {
            var sb = new StringBuilder();
            await foreach (var piece in GenerateStreamAsync(prompt, samplingConfig, logitProcessor, cancellationToken).ConfigureAwait(false))
            {
                sb.Append(piece);
            }
            return sb.ToString();
        }

        public IAsyncEnumerable<string> GenerateStreamAsync(string prompt, CancellationToken cancellationToken = default)
            => GenerateStreamAsync(prompt, _defaultSampling, null, cancellationToken);

        public async IAsyncEnumerable<string> GenerateStreamAsync(
            string prompt,
            SamplingConfig? samplingConfig,
            LogitProcessor? logitProcessor = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(prompt)) yield break;

            var sampling = samplingConfig ?? _defaultSampling;

            int seqId = Interlocked.Increment(ref _nextSeqId);
            int[] promptTokens = _tokenizer.Encode(prompt);

            if (promptTokens.Length == 0) yield break;

            var cfg = _model.Config;
            var pastTokens = new List<int>(promptTokens.Length + sampling.MaxTokens);
            pastTokens.AddRange(promptTokens);

            // Scratch memory buffers
            float[] x = new float[cfg.EmbeddingDim];
            float[] xNorm = new float[cfg.EmbeddingDim];
            float[] q = new float[cfg.HeadCount * cfg.HeadDim];
            float[] k = new float[cfg.HeadCountKv * cfg.HeadDim];
            float[] v = new float[cfg.HeadCountKv * cfg.HeadDim];
            float[] attnOut = new float[cfg.HeadCount * cfg.HeadDim];
            float[] gate = new float[cfg.FeedForwardDim];
            float[] up = new float[cfg.FeedForwardDim];
            float[] swiglu = new float[cfg.FeedForwardDim];
            float[] logits = new float[cfg.VocabSize];

            try
            {
                // 1. Prefill Phase
                for (int pos = 0; pos < promptTokens.Length; pos++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int tok = promptTokens[pos];
                    ForwardToken(tok, pos, seqId, x, xNorm, q, k, v, attnOut, gate, up, swiglu, logits);
                }

                // 2. Autoregressive Generation Loop
                int currentPos = promptTokens.Length;
                int generatedCount = 0;

                while (generatedCount < sampling.MaxTokens)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Sample next token with grammar/custom logit processing
                    int nextToken = LlmSampler.Sample(logits, pastTokens.ToArray(), sampling, logitProcessor: logitProcessor);
                    if (nextToken == cfg.EosTokenId || sampling.StopTokens.Contains(nextToken))
                    {
                        break;
                    }

                    pastTokens.Add(nextToken);
                    generatedCount++;

                    // Decode and yield piece
                    string piece = _tokenizer.Decode(new[] { nextToken });
                    yield return piece;

                    // Advance single step
                    ForwardToken(nextToken, currentPos, seqId, x, xNorm, q, k, v, attnOut, gate, up, swiglu, logits);
                    currentPos++;

                    // Yield execution slice
                    await Task.Yield();
                }
            }
            finally
            {
                _kvCache.FreeSequence(seqId);
            }
        }

        private void ForwardToken(
            int tokenId,
            int pos,
            int seqId,
            Span<float> x,
            Span<float> xNorm,
            Span<float> q,
            Span<float> k,
            Span<float> v,
            Span<float> attnOut,
            Span<float> gate,
            Span<float> up,
            Span<float> swiglu,
            Span<float> logits)
        {
            var cfg = _model.Config;
            int embDim = cfg.EmbeddingDim;
            int qDim = cfg.HeadCount * cfg.HeadDim;
            int kvDim = cfg.HeadCountKv * cfg.HeadDim;
            int headDim = cfg.HeadDim;

            // 1. Token Embedding lookup
            int safeTokenId = (tokenId >= 0 && tokenId < cfg.VocabSize) ? tokenId : 0;
            if (_model.TokenEmbeddings_Q8 != null)
            {
                int blocksPerToken = embDim / BlockQ8_0.BlockSize;
                int blockOffset = safeTokenId * blocksPerToken;
                for (int b = 0; b < blocksPerToken; b++)
                {
                    _model.TokenEmbeddings_Q8[blockOffset + b].Dequantize(x.Slice(b * BlockQ8_0.BlockSize, BlockQ8_0.BlockSize));
                }
            }
            else if (_model.TokenEmbeddings_Q4 != null)
            {
                int blocksPerToken = embDim / BlockQ4_0.BlockSize;
                int blockOffset = safeTokenId * blocksPerToken;
                for (int b = 0; b < blocksPerToken; b++)
                {
                    _model.TokenEmbeddings_Q4[blockOffset + b].Dequantize(x.Slice(b * BlockQ4_0.BlockSize, BlockQ4_0.BlockSize));
                }
            }
            else
            {
                new ReadOnlySpan<float>(_model.TokenEmbeddings, safeTokenId * embDim, embDim).CopyTo(x);
            }

            // 2. Transformer Layers
            for (int l = 0; l < cfg.LayerCount; l++)
            {
                var layer = _model.Layers[l];

                // Attention RMSNorm
                RmsNorm.Forward(x, layer.AttnNorm, xNorm, cfg.RmsNormEps);

                // Q, K, V Projections (Matrix x Vector)
                if (layer.Wq_Q8 != null) QuantizedKernels.MatVecMulQ8_0(layer.Wq_Q8, xNorm, q, qDim, embDim);
                else if (layer.Wq_Q4 != null) QuantizedKernels.MatVecMulQ4_0(layer.Wq_Q4, xNorm, q, qDim, embDim);
                else MatVec(xNorm, layer.Wq, embDim, qDim, q);

                if (layer.Wk_Q8 != null) QuantizedKernels.MatVecMulQ8_0(layer.Wk_Q8, xNorm, k, kvDim, embDim);
                else if (layer.Wk_Q4 != null) QuantizedKernels.MatVecMulQ4_0(layer.Wk_Q4, xNorm, k, kvDim, embDim);
                else MatVec(xNorm, layer.Wk, embDim, kvDim, k);

                if (layer.Wv_Q8 != null) QuantizedKernels.MatVecMulQ8_0(layer.Wv_Q8, xNorm, v, kvDim, embDim);
                else if (layer.Wv_Q4 != null) QuantizedKernels.MatVecMulQ4_0(layer.Wv_Q4, xNorm, v, kvDim, embDim);
                else MatVec(xNorm, layer.Wv, embDim, kvDim, v);

                // RoPE on Q and K
                for (int h = 0; h < cfg.HeadCount; h++)
                {
                    RoPE.ApplyInplace(q.Slice(h * headDim, headDim), pos, headDim, cfg.RopeFreqBase);
                }
                for (int h = 0; h < cfg.HeadCountKv; h++)
                {
                    RoPE.ApplyInplace(k.Slice(h * headDim, headDim), pos, headDim, cfg.RopeFreqBase);
                }

                // Append Key and Value to Paged KV-Cache
                _kvCache.AppendToken(seqId, l, k, v);

                // GQA Attention
                GqaAttention.Forward(q, seqId, l, _kvCache, cfg.HeadCount, cfg.HeadCountKv, headDim, attnOut);

                // Output projection + Residual Add
                if (layer.Wo_Q8 != null) QuantizedKernels.MatVecAddQ8_0(layer.Wo_Q8, attnOut, x, embDim, qDim);
                else if (layer.Wo_Q4 != null) QuantizedKernels.MatVecAddQ4_0(layer.Wo_Q4, attnOut, x, embDim, qDim);
                else MatVecAdd(attnOut, layer.Wo, qDim, embDim, x);

                // FFN RMSNorm
                RmsNorm.Forward(x, layer.FfnNorm, xNorm, cfg.RmsNormEps);

                if (layer.IsMoE)
                {
                    // Sparse Mixture-of-Experts (MoE) FFN with Top-K Gating and Shared Experts (DeepSeek style)
                    SparseMoeLayer.Forward(
                        xNorm,
                        layer.Wrouter,
                        layer.Wrouter_Q8,
                        layer.Wrouter_Q4,
                        layer.Experts,
                        cfg.ExpertUsedCount > 0 ? cfg.ExpertUsedCount : 2,
                        embDim,
                        cfg.FeedForwardDim,
                        gate,
                        up,
                        swiglu,
                        x,
                        layer.SharedExperts);
                }
                else
                {
                    // Standard Dense SwiGLU FFN
                    if (layer.Wgate_Q8 != null) QuantizedKernels.MatVecMulQ8_0(layer.Wgate_Q8, xNorm, gate, cfg.FeedForwardDim, embDim);
                    else if (layer.Wgate_Q4 != null) QuantizedKernels.MatVecMulQ4_0(layer.Wgate_Q4, xNorm, gate, cfg.FeedForwardDim, embDim);
                    else MatVec(xNorm, layer.Wgate, embDim, cfg.FeedForwardDim, gate);

                    if (layer.Wup_Q8 != null) QuantizedKernels.MatVecMulQ8_0(layer.Wup_Q8, xNorm, up, cfg.FeedForwardDim, embDim);
                    else if (layer.Wup_Q4 != null) QuantizedKernels.MatVecMulQ4_0(layer.Wup_Q4, xNorm, up, cfg.FeedForwardDim, embDim);
                    else MatVec(xNorm, layer.Wup, embDim, cfg.FeedForwardDim, up);

                    SwiGLU.Forward(gate, up, swiglu);

                    // Down projection + Residual Add
                    if (layer.Wdown_Q8 != null) QuantizedKernels.MatVecAddQ8_0(layer.Wdown_Q8, swiglu, x, embDim, cfg.FeedForwardDim);
                    else if (layer.Wdown_Q4 != null) QuantizedKernels.MatVecAddQ4_0(layer.Wdown_Q4, swiglu, x, embDim, cfg.FeedForwardDim);
                    else MatVecAdd(swiglu, layer.Wdown, cfg.FeedForwardDim, embDim, x);
                }
            }

            // 3. Final RMSNorm
            RmsNorm.Forward(x, _model.FinalNorm, xNorm, cfg.RmsNormEps);

            // 4. LM Head (Logits projection)
            if (_model.LmHead_Q8 != null) QuantizedKernels.MatVecMulQ8_0(_model.LmHead_Q8, xNorm, logits, cfg.VocabSize, embDim);
            else if (_model.LmHead_Q4 != null) QuantizedKernels.MatVecMulQ4_0(_model.LmHead_Q4, xNorm, logits, cfg.VocabSize, embDim);
            else MatVec(xNorm, _model.LmHead, embDim, cfg.VocabSize, logits);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe float DotProductSimd(float* pA, float* pB, int len)
        {
            int vecSize = Vector<float>.Count;
            int step4 = vecSize * 4;
            int limit4 = len - (len % step4);
            int limit1 = len - (len % vecSize);

            Vector<float> acc0 = Vector<float>.Zero;
            Vector<float> acc1 = Vector<float>.Zero;
            Vector<float> acc2 = Vector<float>.Zero;
            Vector<float> acc3 = Vector<float>.Zero;

            int i = 0;
            for (; i < limit4; i += step4)
            {
                acc0 += *(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i);
                acc1 += *(Vector<float>*)(pA + i + vecSize) * *(Vector<float>*)(pB + i + vecSize);
                acc2 += *(Vector<float>*)(pA + i + (vecSize * 2)) * *(Vector<float>*)(pB + i + (vecSize * 2));
                acc3 += *(Vector<float>*)(pA + i + (vecSize * 3)) * *(Vector<float>*)(pB + i + (vecSize * 3));
            }

            acc0 = (acc0 + acc1) + (acc2 + acc3);

            for (; i < limit1; i += vecSize)
            {
                acc0 += *(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i);
            }

            float sum = Vector.Dot(acc0, Vector<float>.One);

            for (; i < len; i++)
            {
                sum += pA[i] * pB[i];
            }

            return sum;
        }

        internal static unsafe void MatVec(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int inDim, int outDim, Span<float> output)
        {
            fixed (float* pIn = input)
            fixed (float* pW = weight)
            fixed (float* pOut = output)
            {
                if (outDim >= 1024)
                {
                    IntPtr ptrIn = (IntPtr)pIn;
                    IntPtr ptrW = (IntPtr)pW;
                    IntPtr ptrOut = (IntPtr)pOut;
                    Parallel.For(0, outDim, o =>
                    {
                        float* inPtr = (float*)ptrIn;
                        float* wPtr = (float*)ptrW;
                        float* outPtr = (float*)ptrOut;
                        outPtr[o] = DotProductSimd(inPtr, wPtr + (o * inDim), inDim);
                    });
                }
                else
                {
                    for (int o = 0; o < outDim; o++)
                    {
                        pOut[o] = DotProductSimd(pIn, pW + (o * inDim), inDim);
                    }
                }
            }
        }

        internal static unsafe void MatVecAdd(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int inDim, int outDim, Span<float> accOutput)
        {
            fixed (float* pIn = input)
            fixed (float* pW = weight)
            fixed (float* pOut = accOutput)
            {
                if (outDim >= 1024)
                {
                    IntPtr ptrIn = (IntPtr)pIn;
                    IntPtr ptrW = (IntPtr)pW;
                    IntPtr ptrOut = (IntPtr)pOut;
                    Parallel.For(0, outDim, o =>
                    {
                        float* inPtr = (float*)ptrIn;
                        float* wPtr = (float*)ptrW;
                        float* outPtr = (float*)ptrOut;
                        outPtr[o] += DotProductSimd(inPtr, wPtr + (o * inDim), inDim);
                    });
                }
                else
                {
                    for (int o = 0; o < outDim; o++)
                    {
                        pOut[o] += DotProductSimd(pIn, pW + (o * inDim), inDim);
                    }
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                _kvCache.Dispose();
            }
        }
    }
}
