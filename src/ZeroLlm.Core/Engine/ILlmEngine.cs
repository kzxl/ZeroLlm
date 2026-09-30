using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace ZeroLlm.Core.Engine
{
    /// <summary>
    /// High-level Small Language Model (SLM) generation and completion engine.
    /// </summary>
    public interface ILlmEngine : IDisposable
    {
        /// <summary>
        /// Generates a complete text completion for the provided prompt.
        /// </summary>
        Task<string> CompleteAsync(string prompt, CancellationToken cancellationToken = default);

        /// <summary>
        /// Streams generated text tokens asynchronously token-by-token.
        /// </summary>
        IAsyncEnumerable<string> GenerateStreamAsync(string prompt, CancellationToken cancellationToken = default);
    }
}
