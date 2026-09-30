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

        [Fact]
        public void LogitProcessor_Should_Mask_Tokens_Before_Sampling()
        {
            // Token 1 has highest logit (10.0), but custom logit processor masks it to -infinity
            float[] logits = new float[] { 2.0f, 10.0f, 4.0f };
            var config = SamplingConfig.Greedy;

            int selected = LlmSampler.Sample(logits, ReadOnlySpan<int>.Empty, config, logitProcessor: l =>
            {
                l[1] = float.NegativeInfinity; // Mask token 1
            });

            Assert.Equal(2, selected); // Token 2 (4.0) should now win
        }

        [Fact]
        public void MinP_Sampling_Should_Filter_Out_Low_Probability_Tokens()
        {
            float[] logits = new float[] { 10.0f, 5.0f, 0.1f };
            var config = new SamplingConfig
            {
                Temperature = 1.0f,
                TopP = 1.0f,
                MinP = 0.1f // Filters tokens with p < 0.1 * max_p
            };

            int selected = LlmSampler.Sample(logits, ReadOnlySpan<int>.Empty, config, new Random(42));
            Assert.Equal(0, selected);
        }
    }
}
