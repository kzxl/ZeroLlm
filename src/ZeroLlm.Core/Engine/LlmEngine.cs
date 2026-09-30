using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ZeroLlm.Core.Layers;
using ZeroLlm.Core.Memory;
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
        }

        public async Task<string> CompleteAsync(string prompt, CancellationToken cancellationToken = default)
        {
            var sb = new StringBuilder();
            await foreach (var piece in GenerateStreamAsync(prompt, cancellationToken).ConfigureAwait(false))
            {
                sb.Append(piece);
            }
            return sb.ToString();
        }

        public async IAsyncEnumerable<string> GenerateStreamAsync(string prompt, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(prompt)) yield break;

            int seqId = Interlocked.Increment(ref _nextSeqId);
            int[] promptTokens = _tokenizer.Encode(prompt);

            if (promptTokens.Length == 0) yield break;

            var cfg = _model.Config;
            var pastTokens = new List<int>(promptTokens.Length + _defaultSampling.MaxTokens);
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

                while (generatedCount < _defaultSampling.MaxTokens)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Sample next token
                    int nextToken = LlmSampler.Sample(logits, pastTokens.ToArray(), _defaultSampling);
                    if (nextToken == cfg.EosTokenId || _defaultSampling.StopTokens.Contains(nextToken))
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
            new ReadOnlySpan<float>(_model.TokenEmbeddings, safeTokenId * embDim, embDim).CopyTo(x);

            // 2. Transformer Layers
            for (int l = 0; l < cfg.LayerCount; l++)
            {
                var layer = _model.Layers[l];

                // Attention RMSNorm
                RmsNorm.Forward(x, layer.AttnNorm, xNorm, cfg.RmsNormEps);

                // Q, K, V Projections (Matrix x Vector)
                MatVec(xNorm, layer.Wq, embDim, qDim, q);
                MatVec(xNorm, layer.Wk, embDim, kvDim, k);
                MatVec(xNorm, layer.Wv, embDim, kvDim, v);

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
                MatVecAdd(attnOut, layer.Wo, qDim, embDim, x);

                // FFN RMSNorm
                RmsNorm.Forward(x, layer.FfnNorm, xNorm, cfg.RmsNormEps);

                // SwiGLU FFN
                MatVec(xNorm, layer.Wgate, embDim, cfg.FeedForwardDim, gate);
                MatVec(xNorm, layer.Wup, embDim, cfg.FeedForwardDim, up);
                SwiGLU.Forward(gate, up, swiglu);

                // Down projection + Residual Add
                MatVecAdd(swiglu, layer.Wdown, cfg.FeedForwardDim, embDim, x);
            }

            // 3. Final RMSNorm
            RmsNorm.Forward(x, _model.FinalNorm, xNorm, cfg.RmsNormEps);

            // 4. LM Head (Logits projection)
            MatVec(xNorm, _model.LmHead, embDim, cfg.VocabSize, logits);
        }

        private static void MatVec(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int inDim, int outDim, Span<float> output)
        {
            for (int o = 0; o < outDim; o++)
            {
                float dot = 0.0f;
                int rowOffset = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    dot += input[i] * weight[rowOffset + i];
                }
                output[o] = dot;
            }
        }

        private static void MatVecAdd(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int inDim, int outDim, Span<float> accOutput)
        {
            for (int o = 0; o < outDim; o++)
            {
                float dot = 0.0f;
                int rowOffset = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    dot += input[i] * weight[rowOffset + i];
                }
                accOutput[o] += dot;
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
