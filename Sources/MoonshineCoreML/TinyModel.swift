/// The fixed sizes of Moonshine Streaming Tiny as converted to Core ML. 
/// 
/// They are contracts with the five compiled models:
/// model loading checks each model against them, so a mismatched model fails at load time, not as wrong text.
/// See the README's glossary for what each term means.

enum TinyModel {
    enum Audio {
        static let sampleRateHz = 16_000
        static let samplesPerChunk = 1280  // 80 ms
    }

    enum Frontend {
        static let featureFramesPerChunk = 4  // one feature frame per 20 ms
        
        // The frontend's two convolutions each need their last 4 inputs from the previous chunk. The model returns
        // these buffers and takes them back on the next call.
        static let conv1BufferShape = [1, 320, 4]
        static let conv2BufferShape = [1, 640, 4]
    }

    enum Encoder {
        /// Numbers per feature frame and per encoded frame.
        static let frameWidth = 320
        /// Frames the encoder looks back: 16 per layer across 6 layers.
        static let leftContextFrameCount = 96
        /// Frames the encoder looks ahead: 4 per layer across 4 layers. The newest 16 encoded frames aren't final
        /// until more audio arrives, so they're held back until the next chunk or the end of the utterance.
        static let heldBackFrameCount = 16
        /// Every encoder call sees this many frames: left context, one chunk's new frames, then the held-back frames.
        static let windowFrameCount = leftContextFrameCount + Frontend.featureFramesPerChunk + heldBackFrameCount  // 116
    }

    enum Adapter {
        /// The adapter and cross-attention key/value models each take 4 encoded frames per call.
        static let memoryFramesPerCall = Frontend.featureFramesPerChunk
        /// Numbers per memory frame.
        static let memoryFrameWidth = 320
    }

    enum Decoder {
        static let layerCount = 6
        static let headCount = 8
        static let headWidth = 40
        
        // The self-attention cache and the cross-attention keys and values pad each head's 40 numbers to 64. With 40,
        // the Neural Engine fails every prediction on the cache, and Core ML re-lays out the cross-attention inputs on
        // every call.
        static let paddedHeadWidth = 64
        /// Rows of the cross-attention keys and values: the longest utterance the decoder can attend to.
        static let maximumMemoryFrameCount = 768
        /// Rows of the self-attention cache: the most tokens one decode can produce.
        static let maximumTokenCount = 128

        /// [layer, batch, head, row, number]: cached keys or values for rowCount rows (memory frames for
        /// cross-attention, tokens for self-attention).
        static func attentionCacheShape(rowCount: Int, numbersPerRow: Int = headWidth) -> [Int] {
            [layerCount, 1, headCount, rowCount, numbersPerRow]
        }
    }

    enum Tokens {
        static let startTokenID = 1
        static let endTokenID = 2
        static let vocabularySize = 32_768
        /// Moonshine's cap on decoding: at most this many tokens per second of audio heard, so a decode that never
        /// picks the end token still stops.
        static let maximumTokensPerSecondOfAudio = 6.5
    }

    enum Utterance {
        /// 192 chunks, 245,760 samples, 15.36 s: the audio that fills every memory frame the decoder can attend to.
        static let maximumSampleCount =
            Decoder.maximumMemoryFrameCount / Adapter.memoryFramesPerCall * Audio.samplesPerChunk
    }
}
