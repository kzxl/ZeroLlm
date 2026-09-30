# 🧠 ZeroLlm: Sovereign Pure C# Small Language Model (SLM) Runtime

[![ZeroPlatform Tier](https://img.shields.io/badge/ZeroPlatform-Tier%203%20(Perception%20%26%20AI)-7c3aed.svg)](https://github.com/kzxl/ZeroPlatform)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0%20(Pure%20C%23)-brightgreen.svg)]()
[![Multi-Targeting](https://img.shields.io/badge/.NET-8.0%20%7C%204.6.2%20%7C%20Standard%202.0-orange.svg)]()

> **Architectural Standard**: 100% Pure C#, Zero Unmanaged Wrappers (no `llama.cpp` or Python runtime required), Zero External Dependencies, Multi-Targeting across `.NET 8.0`, `.NET Framework 4.6.2`, and `.NET Standard 2.0`.

`ZeroLlm` is an in-process, high-throughput Small Language Model (SLM) runtime and Transformer execution engine engineered for mission-critical edge gateways, industrial controllers, and enterprise agent systems. Operating within **Tier 3 (Perception & AI)** of the **ZeroPlatform** ecosystem, it bridges high-speed tokenization (`ZeroTokenizer`), high-dimensional similarity (`ZeroVector`), and cognitive swarms (`ZeroAgent`).

---

## 🏛️ Key Subsystem Capabilities

| Component | Namespace | Description |
| :--- | :--- | :--- |
| **`GgufReader` & `GgufWriter`** | `ZeroLlm.Core.Format` | Pure C# binary reader and serializer for GGUF v2/v3 model formats, tensor metadata tables, and aligned tensor offsets. |
| **`PagedKvCache`** | `ZeroLlm.Core.Memory` | Virtual memory page-block allocator (16 tokens/block) preventing context memory fragmentation and enabling instant prompt prefix caching. |
| **`RmsNorm`** | `ZeroLlm.Core.Layers` | Allocation-free Root Mean Square Layer Normalization kernel with epsilon stability. |
| **`RoPE`** | `ZeroLlm.Core.Layers` | Rotary Position Embedding kernel rotating Query and Key projections with customizable frequency scaling. |
| **`SwiGLU`** | `ZeroLlm.Core.Layers` | Swish-Gated Linear Unit activation kernel for feed-forward neural layers ($Swish(gate) \odot up$). |
| **`GqaAttention`** | `ZeroLlm.Core.Layers` | Grouped-Query Attention (GQA) kernel with multi-query head mapping, causal attention masking, and Paged KV-Cache retrieval. |
| **`LlmSampler`** | `ZeroLlm.Core.Sampling` | Logit sampler supporting Greedy (ArgMax), Temperature scaling, Top-K, Top-P (Nucleus), and Repetition penalty. |
| **`LlmEngine`** | `ZeroLlm.Core.Engine` | End-to-end execution pipeline supporting synchronous completions and asynchronous token streaming (`IAsyncEnumerable<string>`). |

---

## 🚀 Quickstart

### 1. Execute Text Completion

```csharp
using ZeroLlm.Core.Engine;
using ZeroLlm.Core.Sampling;
using ZeroTokenizer.Core.Bpe;

// Configure model architecture
var config = new LlmModelConfig
{
    VocabSize = 32000,
    EmbeddingDim = 256,
    LayerCount = 4,
    HeadCount = 4,
    HeadCountKv = 2
};

var model = LlmModel.CreateSynthetic(config);
var tokenizer = BpeTokenizer.CreateDefault();

using var engine = new LlmEngine(model, tokenizer, SamplingConfig.Greedy);
string response = await engine.CompleteAsync("System Status: OK. Commencing pipeline diagnosis.");
Console.WriteLine(response);
```

### 2. Stream Tokens Asynchronously

```csharp
await foreach (string token in engine.GenerateStreamAsync("Inspect PLC telemetry"))
{
    Console.Write(token);
}
```

---

## 📄 License

MIT License. Copyright © 2026 Phong Võ.
