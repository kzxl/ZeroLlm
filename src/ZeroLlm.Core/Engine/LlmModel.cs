using System;
using System.Collections.Generic;
using System.IO;
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

        /// <summary>
        /// Serializes the model weights and architecture hyperparameters into standard GGUF v3 binary format.
        /// </summary>
        public void SaveGguf(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var writer = new GgufWriter();
            var arch = Config.Architecture ?? "llama";

            // Metadata
            writer.AddMetadata("general.architecture", arch);
            writer.AddMetadata($"{arch}.vocab_size", (uint)Config.VocabSize);
            writer.AddMetadata($"{arch}.context_length", (uint)Config.ContextLength);
            writer.AddMetadata($"{arch}.embedding_length", (uint)Config.EmbeddingDim);
            writer.AddMetadata($"{arch}.block_count", (uint)Config.LayerCount);
            writer.AddMetadata($"{arch}.feed_forward_length", (uint)Config.FeedForwardDim);
            writer.AddMetadata($"{arch}.attention.head_count", (uint)Config.HeadCount);
            writer.AddMetadata($"{arch}.attention.head_count_kv", (uint)Config.HeadCountKv);
            writer.AddMetadata($"{arch}.attention.layer_norm_rms_epsilon", Config.RmsNormEps);
            writer.AddMetadata($"{arch}.rope.freq_base", Config.RopeFreqBase);
            writer.AddMetadata("tokenizer.ggml.bos_token_id", (uint)Config.BosTokenId);
            writer.AddMetadata("tokenizer.ggml.eos_token_id", (uint)Config.EosTokenId);

            // Tensors
            // 1. Embeddings
            writer.AddTensor("token_embd.weight",
                new ulong[] { (ulong)Config.EmbeddingDim, (ulong)Config.VocabSize },
                GgufTensorType.F32,
                FloatArrayToBytes(TokenEmbeddings));

            // 2. Layers
            int qDim = Config.HeadCount * Config.HeadDim;
            int kvDim = Config.HeadCountKv * Config.HeadDim;

            for (int l = 0; l < Config.LayerCount; l++)
            {
                var layer = Layers[l];
                writer.AddTensor($"blk.{l}.attn_norm.weight", new ulong[] { (ulong)Config.EmbeddingDim }, GgufTensorType.F32, FloatArrayToBytes(layer.AttnNorm));
                writer.AddTensor($"blk.{l}.attn_q.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)qDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wq));
                writer.AddTensor($"blk.{l}.attn_k.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)kvDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wk));
                writer.AddTensor($"blk.{l}.attn_v.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)kvDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wv));
                writer.AddTensor($"blk.{l}.attn_output.weight", new ulong[] { (ulong)qDim, (ulong)Config.EmbeddingDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wo));
                writer.AddTensor($"blk.{l}.ffn_norm.weight", new ulong[] { (ulong)Config.EmbeddingDim }, GgufTensorType.F32, FloatArrayToBytes(layer.FfnNorm));
                writer.AddTensor($"blk.{l}.ffn_gate.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)Config.FeedForwardDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wgate));
                writer.AddTensor($"blk.{l}.ffn_up.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)Config.FeedForwardDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wup));
                writer.AddTensor($"blk.{l}.ffn_down.weight", new ulong[] { (ulong)Config.FeedForwardDim, (ulong)Config.EmbeddingDim }, GgufTensorType.F32, FloatArrayToBytes(layer.Wdown));
            }

            // 3. Final norm & LM head
            writer.AddTensor("output_norm.weight", new ulong[] { (ulong)Config.EmbeddingDim }, GgufTensorType.F32, FloatArrayToBytes(FinalNorm));
            writer.AddTensor("output.weight", new ulong[] { (ulong)Config.EmbeddingDim, (ulong)Config.VocabSize }, GgufTensorType.F32, FloatArrayToBytes(LmHead));

            writer.WriteTo(stream);
        }

        public void SaveGguf(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            using (var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                SaveGguf(fs);
            }
        }

        /// <summary>
        /// Deserializes a model from standard GGUF binary format stream.
        /// </summary>
        public static LlmModel LoadGguf(Stream stream)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            var reader = GgufReader.FromStream(stream);
            var arch = reader.Metadata.Architecture ?? "llama";

            int vocabSize = (int)reader.Metadata.GetValueOrDefault($"{arch}.vocab_size", 32000u);
            int contextLength = (int)reader.Metadata.GetValueOrDefault($"{arch}.context_length", 2048u);
            int embeddingDim = (int)reader.Metadata.GetValueOrDefault($"{arch}.embedding_length", 256u);
            int layerCount = (int)reader.Metadata.GetValueOrDefault($"{arch}.block_count", 4u);
            int feedForwardDim = (int)reader.Metadata.GetValueOrDefault($"{arch}.feed_forward_length", 512u);
            int headCount = (int)reader.Metadata.GetValueOrDefault($"{arch}.attention.head_count", 4u);
            int headCountKv = (int)reader.Metadata.GetValueOrDefault($"{arch}.attention.head_count_kv", (uint)headCount);
            float rmsNormEps = reader.Metadata.GetValueOrDefault($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
            float ropeFreqBase = reader.Metadata.GetValueOrDefault($"{arch}.rope.freq_base", 10000.0f);
            int bosTokenId = (int)reader.Metadata.GetValueOrDefault("tokenizer.ggml.bos_token_id", 1u);
            int eosTokenId = (int)reader.Metadata.GetValueOrDefault("tokenizer.ggml.eos_token_id", 2u);

            var tensorMap = new Dictionary<string, GgufTensorInfo>(StringComparer.OrdinalIgnoreCase);
            foreach (var t in reader.Tensors)
            {
                tensorMap[t.Name] = t;
            }

            if (tensorMap.TryGetValue("token_embd.weight", out var embTensor) && embTensor.Dimensions.Length >= 2)
            {
                embeddingDim = (int)embTensor.Dimensions[0];
                vocabSize = (int)embTensor.Dimensions[1];
            }

            var config = new LlmModelConfig
            {
                Architecture = arch,
                VocabSize = vocabSize,
                ContextLength = contextLength,
                EmbeddingDim = embeddingDim,
                LayerCount = layerCount,
                FeedForwardDim = feedForwardDim,
                HeadCount = headCount,
                HeadCountKv = headCountKv,
                RmsNormEps = rmsNormEps,
                RopeFreqBase = ropeFreqBase,
                BosTokenId = bosTokenId,
                EosTokenId = eosTokenId
            };

            var model = new LlmModel(config)
            {
                Layers = new LlmLayerWeights[layerCount]
            };

            if (tensorMap.TryGetValue("token_embd.weight", out var tEmb))
                model.TokenEmbeddings = ReadTensorFloats(stream, reader.TensorDataOffset, tEmb);

            for (int l = 0; l < layerCount; l++)
            {
                var layer = new LlmLayerWeights();
                if (tensorMap.TryGetValue($"blk.{l}.attn_norm.weight", out var tAttnNorm))
                    layer.AttnNorm = ReadTensorFloats(stream, reader.TensorDataOffset, tAttnNorm);
                if (tensorMap.TryGetValue($"blk.{l}.attn_q.weight", out var tWq))
                    layer.Wq = ReadTensorFloats(stream, reader.TensorDataOffset, tWq);
                if (tensorMap.TryGetValue($"blk.{l}.attn_k.weight", out var tWk))
                    layer.Wk = ReadTensorFloats(stream, reader.TensorDataOffset, tWk);
                if (tensorMap.TryGetValue($"blk.{l}.attn_v.weight", out var tWv))
                    layer.Wv = ReadTensorFloats(stream, reader.TensorDataOffset, tWv);
                if (tensorMap.TryGetValue($"blk.{l}.attn_output.weight", out var tWo))
                    layer.Wo = ReadTensorFloats(stream, reader.TensorDataOffset, tWo);
                if (tensorMap.TryGetValue($"blk.{l}.ffn_norm.weight", out var tFfnNorm))
                    layer.FfnNorm = ReadTensorFloats(stream, reader.TensorDataOffset, tFfnNorm);
                if (tensorMap.TryGetValue($"blk.{l}.ffn_gate.weight", out var tWgate))
                    layer.Wgate = ReadTensorFloats(stream, reader.TensorDataOffset, tWgate);
                if (tensorMap.TryGetValue($"blk.{l}.ffn_up.weight", out var tWup))
                    layer.Wup = ReadTensorFloats(stream, reader.TensorDataOffset, tWup);
                if (tensorMap.TryGetValue($"blk.{l}.ffn_down.weight", out var tWdown))
                    layer.Wdown = ReadTensorFloats(stream, reader.TensorDataOffset, tWdown);

                model.Layers[l] = layer;
            }

            if (tensorMap.TryGetValue("output_norm.weight", out var tFinalNorm))
                model.FinalNorm = ReadTensorFloats(stream, reader.TensorDataOffset, tFinalNorm);

            if (tensorMap.TryGetValue("output.weight", out var tLmHead))
                model.LmHead = ReadTensorFloats(stream, reader.TensorDataOffset, tLmHead);
            else
                model.LmHead = model.TokenEmbeddings; // Tied weights fallback

            return model;
        }

        public static LlmModel LoadGguf(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) throw new ArgumentNullException(nameof(filePath));
            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return LoadGguf(fs);
            }
        }

        private static byte[] FloatArrayToBytes(float[] array)
        {
            var bytes = new byte[array.Length * sizeof(float)];
            Buffer.BlockCopy(array, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        private static float[] BytesToFloatArray(byte[] bytes)
        {
            var floats = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
            return floats;
        }

        private static float[] ReadTensorFloats(Stream stream, long tensorDataOffset, GgufTensorInfo tensor)
        {
            stream.Seek(tensorDataOffset + (long)tensor.Offset, SeekOrigin.Begin);
            int byteLen = (int)tensor.GetSizeInBytes();
            byte[] bytes = new byte[byteLen];
            int read = 0;
            while (read < byteLen)
            {
                int r = stream.Read(bytes, read, byteLen - read);
                if (r <= 0) break;
                read += r;
            }
            return BytesToFloatArray(bytes);
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
