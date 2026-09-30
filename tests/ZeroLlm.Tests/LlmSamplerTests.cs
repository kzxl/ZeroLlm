using System;
using Xunit;
using ZeroLlm.Core.Sampling;

namespace ZeroLlm.Tests
{
    public class LlmSamplerTests
    {
        [Fact]
        public void GreedySampling_Should_Always_Pick_Maximum_Logit()
        {
            float[] logits = new float[] { 1.2f, 5.8f, 0.4f, 3.1f };
            var config = SamplingConfig.Greedy;

            int selected = LlmSampler.Sample(logits, ReadOnlySpan<int>.Empty, config);
            Assert.Equal(1, selected);
        }

        [Fact]
        public void RepetitionPenalty_Should_Penalize_Past_Tokens()
        {
            float[] logits = new float[] { 2.0f, 2.1f };
            var config = new SamplingConfig
            {
                Temperature = 0.0f,
                RepetitionPenalty = 2.0f // Halves token 1 from 2.1 to 1.05
            };

            int selected = LlmSampler.Sample(logits, new int[] { 1 }, config);
            Assert.Equal(0, selected); // Token 0 (2.0) should now beat Token 1 (1.05)
        }
    }
}
