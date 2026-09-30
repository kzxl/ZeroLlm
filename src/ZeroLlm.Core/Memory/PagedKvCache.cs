using System;
using System.Collections.Generic;

namespace ZeroLlm.Core.Memory
{
    /// <summary>
    /// High-throughput Paged KV-Cache block allocator preventing memory fragmentation in LLM inference.
    /// </summary>
    public sealed class PagedKvCache : IDisposable
    {
        public int BlockSize { get; }
        public int LayerCount { get; }
        public int KvHeadCount { get; }
        public int HeadDim { get; }
        public int TotalBlocks { get; }

        private readonly KvBlock[] _allBlocks;
        private readonly Stack<int> _freeBlocks;
        private readonly Dictionary<int, List<int>> _blockTables = new Dictionary<int, List<int>>();
        private readonly Dictionary<int, int> _seqLengths = new Dictionary<int, int>();
        private readonly object _lock = new object();

        public PagedKvCache(int totalBlocks, int blockSize, int layerCount, int kvHeadCount, int headDim)
        {
            if (totalBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(totalBlocks));
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));

            TotalBlocks = totalBlocks;
            BlockSize = blockSize;
            LayerCount = layerCount;
            KvHeadCount = kvHeadCount;
            HeadDim = headDim;

            _allBlocks = new KvBlock[totalBlocks];
            _freeBlocks = new Stack<int>(totalBlocks);

            for (int i = 0; i < totalBlocks; i++)
            {
                _allBlocks[i] = new KvBlock(i, blockSize, layerCount, kvHeadCount, headDim);
                _freeBlocks.Push(i);
            }
        }

        public int FreeBlockCount
        {
            get
            {
                lock (_lock)
                {
                    return _freeBlocks.Count;
                }
            }
        }

        public int GetSequenceLength(int seqId)
        {
            lock (_lock)
            {
                return _seqLengths.TryGetValue(seqId, out int len) ? len : 0;
            }
        }

        /// <summary>
        /// Appends key and value states for a specific layer at the current sequence token position.
        /// When layer == LayerCount - 1, the sequence length advances.
        /// </summary>
        public void AppendToken(int seqId, int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> val)
        {
            lock (_lock)
            {
                if (!_blockTables.TryGetValue(seqId, out var table))
                {
                    table = new List<int>();
                    _blockTables[seqId] = table;
                    _seqLengths[seqId] = 0;
                }

                int currentLen = _seqLengths[seqId];
                int tokenIdx = currentLen;
                int blockIndexInSeq = tokenIdx / BlockSize;
                int slotIndex = tokenIdx % BlockSize;

                // Ensure physical block is allocated
                while (table.Count <= blockIndexInSeq)
                {
                    if (_freeBlocks.Count == 0)
                    {
                        throw new InvalidOperationException($"Paged KV-cache exhausted! (All {TotalBlocks} blocks in use)");
                    }

                    int newBlockId = _freeBlocks.Pop();
                    _allBlocks[newBlockId].RefCount = 1;
                    table.Add(newBlockId);
                }

                var block = _allBlocks[table[blockIndexInSeq]];
                key.CopyTo(block.GetKeySpan(layer, slotIndex));
                val.CopyTo(block.GetValueSpan(layer, slotIndex));

                // If this is the last layer in the Transformer block, advance token counter
                if (layer == LayerCount - 1)
                {
                    _seqLengths[seqId] = currentLen + 1;
                }
            }
        }

        public ReadOnlySpan<float> GetKey(int seqId, int layer, int tokenPos)
        {
            lock (_lock)
            {
                if (!_blockTables.TryGetValue(seqId, out var table))
                {
                    throw new ArgumentException($"Sequence {seqId} does not exist.");
                }

                int blockIdx = tokenPos / BlockSize;
                int slotIdx = tokenPos % BlockSize;
                return _allBlocks[table[blockIdx]].GetKeySpan(layer, slotIdx);
            }
        }

        public ReadOnlySpan<float> GetValue(int seqId, int layer, int tokenPos)
        {
            lock (_lock)
            {
                if (!_blockTables.TryGetValue(seqId, out var table))
                {
                    throw new ArgumentException($"Sequence {seqId} does not exist.");
                }

                int blockIdx = tokenPos / BlockSize;
                int slotIdx = tokenPos % BlockSize;
                return _allBlocks[table[blockIdx]].GetValueSpan(layer, slotIdx);
            }
        }

        public void FreeSequence(int seqId)
        {
            lock (_lock)
            {
                if (_blockTables.TryGetValue(seqId, out var table))
                {
                    foreach (int blockId in table)
                    {
                        var block = _allBlocks[blockId];
                        block.RefCount--;
                        if (block.RefCount <= 0)
                        {
                            block.Reset();
                            _freeBlocks.Push(blockId);
                        }
                    }

                    _blockTables.Remove(seqId);
                    _seqLengths.Remove(seqId);
                }
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _blockTables.Clear();
                _seqLengths.Clear();
                _freeBlocks.Clear();
                for (int i = 0; i < _allBlocks.Length; i++)
                {
                    _allBlocks[i].Reset();
                    _freeBlocks.Push(i);
                }
            }
        }
    }
}
