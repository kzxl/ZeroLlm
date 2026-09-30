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
    }
}
