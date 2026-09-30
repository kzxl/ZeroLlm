using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using ZeroLlm.Core.Engine;
using ZeroLlm.Core.Sampling;
using ZeroTokenizer.Core.Tiktoken;

namespace ZeroLlm.Tests
{
    public class LlmEngineTests
    {
        [Fact]
        public async Task LlmEngine_Should_Generate_Tokens_End_To_End()
        {
            var config = new LlmModelConfig
            {
                VocabSize = 256,
                EmbeddingDim = 32,
                LayerCount = 2,
                HeadCount = 2,
                HeadCountKv = 2,
                FeedForwardDim = 64,
                EosTokenId = 255
            };

            var model = LlmModel.CreateSynthetic(config, seed: 123);
            var tokenizer = TiktokenTokenizer.CreateCl100kBase();

            var sampling = new SamplingConfig
            {
                Temperature = 0.0f,
                MaxTokens = 5
            };

            using (var engine = new LlmEngine(model, tokenizer, sampling))
            {
                var completion = await engine.CompleteAsync("Hello");
                Assert.NotNull(completion);

                var streamTokens = new List<string>();
                await foreach (var piece in engine.GenerateStreamAsync("Test"))
                {
                    streamTokens.Add(piece);
                }

                Assert.NotEmpty(streamTokens);
            }
        }
    }
}
