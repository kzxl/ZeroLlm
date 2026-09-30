using System;
using System.Collections.Generic;
using ZeroLlm.Core.Format;

namespace ZeroLlm.Core.Engine
{
    public sealed class LlmLayerWeights
    {
        public float[] AttnNorm { get; set; } = Array.Empty<float>();
        public float[] Wq { get; set; } = Array.Empty<float>();
        public float[] Wk { get; set; } = Array.Empty<float>();
        public float[] Wv { get; set; } = Array.Empty<float>();
        public float[] Wo { get; set; } = Array.Empty<float>();
        public float[] FfnNorm { get; set; } = Array.Empty<float>();
        public float[] Wgate { get; set; } = Array.Empty<float>();
        public float[] Wup { get; set; } = Array.Empty<float>();
        public float[] Wdown { get; set; } = Array.Empty<float>();
    }

    /// <summary>
    /// In-memory representation of an SLM/LLM's weights and architecture.
    /// </summary>
    public sealed class LlmModel
    {
        public LlmModelConfig Config { get; }
        public float[] TokenEmbeddings { get; set; } = Array.Empty<float>();
        public LlmLayerWeights[] Layers { get; set; } = Array.Empty<LlmLayerWeights>();
        public float[] FinalNorm { get; set; } = Array.Empty<float>();
        public float[] LmHead { get; set; } = Array.Empty<float>();

        public LlmModel(LlmModelConfig config)
        {
            Config = config ?? throw new ArgumentNullException(nameof(config));
        }

        /// <summary>
        /// Creates a initialized model with deterministic synthetic weights for tests and simulation.
        /// </summary>
        public static LlmModel CreateSynthetic(LlmModelConfig config, int seed = 42)
        {
            var rand = new Random(seed);
            var model = new LlmModel(config)
            {
                TokenEmbeddings = CreateRandomArray(config.VocabSize * config.EmbeddingDim, rand, 0.02f),
                Layers = new LlmLayerWeights[config.LayerCount],
                FinalNorm = CreateConstantArray(config.EmbeddingDim, 1.0f),
                LmHead = CreateRandomArray(config.EmbeddingDim * config.VocabSize, rand, 0.02f)
            };

            int qDim = config.HeadCount * config.HeadDim;
            int kvDim = config.HeadCountKv * config.HeadDim;

            for (int l = 0; l < config.LayerCount; l++)
            {
                model.Layers[l] = new LlmLayerWeights
                {
                    AttnNorm = CreateConstantArray(config.EmbeddingDim, 1.0f),
                    Wq = CreateRandomArray(config.EmbeddingDim * qDim, rand, 0.02f),
                    Wk = CreateRandomArray(config.EmbeddingDim * kvDim, rand, 0.02f),
                    Wv = CreateRandomArray(config.EmbeddingDim * kvDim, rand, 0.02f),
                    Wo = CreateRandomArray(qDim * config.EmbeddingDim, rand, 0.02f),
                    FfnNorm = CreateConstantArray(config.EmbeddingDim, 1.0f),
                    Wgate = CreateRandomArray(config.EmbeddingDim * config.FeedForwardDim, rand, 0.02f),
                    Wup = CreateRandomArray(config.EmbeddingDim * config.FeedForwardDim, rand, 0.02f),
                    Wdown = CreateRandomArray(config.FeedForwardDim * config.EmbeddingDim, rand, 0.02f)
                };
            }

            return model;
        }

        private static float[] CreateRandomArray(int size, Random rand, float scale)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++)
            {
                arr[i] = (float)(rand.NextDouble() * 2.0 - 1.0) * scale;
            }
            return arr;
        }

        private static float[] CreateConstantArray(int size, float value)
        {
            var arr = new float[size];
            for (int i = 0; i < size; i++) arr[i] = value;
            return arr;
        }
    }
}
