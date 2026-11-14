/// The namespace for this package's public types: `Moonshine.Tokenizer`, `Moonshine.Error` and so on.
public enum Moonshine {}

// Swift's Float16, which the decoder's cross-attention buffers are made of, doesn't exist on Intel Macs. Watches build
// for arm64_32, not arm64, so this checks for Intel rather than for arm64.
#if os(macOS) && arch(x86_64)
#error("MoonshineCoreML needs Apple silicon on the Mac: Float16 isn't available on Intel. Build for arm64 only (for a universal macOS app, remove x86_64 from the target's Architectures).")
#endif
