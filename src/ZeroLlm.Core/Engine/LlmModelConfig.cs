namespace ZeroLlm.Core.Engine
{
    /// <summary>
    /// Architecture hyperparameters for a Transformer model.
    /// </summary>
    public sealed class LlmModelConfig
    {
        public string Architecture { get; set; } = "llama";
        public int VocabSize { get; set; } = 32000;
        public int ContextLength { get; set; } = 2048;
        public int EmbeddingDim { get; set; } = 256;
        public int LayerCount { get; set; } = 4;
        public int HeadCount { get; set; } = 4;
        public int HeadCountKv { get; set; } = 2;
        public int HeadDim => EmbeddingDim / HeadCount;
        public int FeedForwardDim { get; set; } = 512;
        public float RmsNormEps { get; set; } = 1e-5f;
        public float RopeFreqBase { get; set; } = 10000.0f;
        public int BosTokenId { get; set; } = 1;
        public int EosTokenId { get; set; } = 2;

        /// <summary>
        /// Creates a canonical configuration for a generic Micro-SLM (~11.5M parameters).
        /// Tailored for high-speed sub-10ms CPU inference and compact memory footprint (~5.8MB INT4, ~11.5MB INT8).
        /// </summary>
        public static LlmModelConfig CreateMicro(
            int vocabSize = 16000,
            int contextLength = 1024,
            int embeddingDim = 256,
            int layerCount = 6,
            int headCount = 8,
            int headCountKv = 2,
            int feedForwardDim = 512)
        {
            return new LlmModelConfig
            {
                Architecture = "llama",
                VocabSize = vocabSize,
                ContextLength = contextLength,
                EmbeddingDim = embeddingDim,
                LayerCount = layerCount,
                HeadCount = headCount,
                HeadCountKv = headCountKv,
                FeedForwardDim = feedForwardDim,
                RmsNormEps = 1e-5f,
                RopeFreqBase = 10000.0f,
                BosTokenId = 1,
                EosTokenId = 2
            };
        }

        /// <summary>
        /// Convenience preset for the internal Vietnamese ERP Micro-SLM (~11.5M parameters).
        /// </summary>
        public static LlmModelConfig CreateVietnameseErpMicro(int vocabSize = 16000, int contextLength = 1024)
            => CreateMicro(vocabSize, contextLength);
    }
}
