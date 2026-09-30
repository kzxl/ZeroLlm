using System;

namespace ZeroLlm.Core.Format
{
    /// <summary>
    /// Metadata descriptor for a tensor stored inside a GGUF model file.
    /// </summary>
    public sealed class GgufTensorInfo
    {
        public string Name { get; }
        public ulong[] Dimensions { get; }
        public GgufTensorType Type { get; }
        public ulong Offset { get; }

        public GgufTensorInfo(string name, ulong[] dimensions, GgufTensorType type, ulong offset)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
            Type = type;
            Offset = offset;
        }

        /// <summary>
        /// Total number of elements across all dimensions.
        /// </summary>
        public ulong ElementCount
        {
            get
            {
                if (Dimensions.Length == 0) return 0;
                ulong count = 1;
                for (int i = 0; i < Dimensions.Length; i++)
                {
                    count *= Dimensions[i];
                }
                return count;
            }
        }

        public override string ToString() =>
            $"{Name} [{string.Join("x", Dimensions)}] ({Type}, offset: {Offset})";
    }
}
