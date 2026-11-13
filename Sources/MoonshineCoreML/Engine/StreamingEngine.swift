import CoreML

/// Everything kept between Core ML calls for one utterance at a time, and the four things done with it: begin an
/// utterance, add a chunk, decode an update, end the utterance. Listening is incremental: each chunk adds memory frames
/// for good. Decoding isn't: every update decodes from scratch over all memory frames so far, as Moonshine does.
///
/// The work is split by stage: this file holds the state and the four operations, StreamingEngine+Listening turns
/// chunks into memory frames, and StreamingEngine+Decoding turns memory frames into tokens.
///
/// Not thread-safe: use it from one thread or actor at a time. Predictions that share the decoder's MLState must run one
/// at a time anyway.
final class StreamingEngine {
    let models: LoadedModels
    let tokenizer: Moonshine.Tokenizer

    // Allocated and zeroed once, never cleared. In a later utterance, the rows past its memory frames still hold an
    // earlier utterance's keys and values, but memoryFrameIsReal hides them from the decoder, so they change nothing.
    // The keys and values are float16, as the decoder takes them. They're IOSurface-backed: with ordinary arrays,
    // Core ML copies both into its own buffers on every decoder step (on an Apple Watch Series 9, surfaces brought the
    // median step from 15.8 to 12.4 ms).
    /// [6, 1, 8, 768, 64]: each row's 40 numbers, then zeros.
    let crossAttentionKeys: MLMultiArray
    let crossAttentionValues: MLMultiArray
    /// [1, 768] float32: 1 for each memory frame of this utterance, 0 after.
    let memoryFrameIsReal: MLMultiArray
    /// Every feature frame of the utterance, 768 × 320 row-major. Each becomes one encoded frame and one memory frame.
    var featureFrames: [Float]

    // Inputs rewritten before each prediction, allocated once.
    let audioChunkInput: MLMultiArray  // [1, 1280]
    let encoderWindowInput: MLMultiArray  // [1, 116, 320]: the window's real frames first, then zeros
    let encoderWindowFrameIsRealInput: MLMultiArray  // [1, 116]: 1 for each real frame, 0 for each padding frame
    let adapterEncodedFramesInput: MLMultiArray  // [1, 4, 320]
    let adapterPositionOffsetInput: MLMultiArray  // [1] int32
    let decoderTokenInput: MLMultiArray  // [1, 1] int32
    let decoderPositionInput: MLMultiArray  // [1] int32

    // Kept per utterance; beginUtterance() starts them over.
    /// The frontend's convolution buffers: each prediction's output is the next prediction's input.
    var conv1Buffer: MLMultiArray
    var conv2Buffer: MLMultiArray
    var featureFrameCount = 0
    var encodedFrameCount = 0
    /// The adapter's position offset, and how many rows of cross-attention keys and values are real.
    var memoryFrameCount = 0
    /// Real samples so far, not the last chunk's padding. Sets each decode's maximum token count.
    private(set) var heardSampleCount = 0
    /// The decoder's self-attention cache, one per utterance. nil when no utterance is in progress.
    var decoderState: MLState?

    init(models: LoadedModels, tokenizer: Moonshine.Tokenizer) throws {
        typealias Shape = TinyModel
        self.models = models
        self.tokenizer = tokenizer
        let crossAttentionShape = Shape.Decoder.attentionCacheShape(
            rowCount: Shape.Decoder.maximumMemoryFrameCount, numbersPerRow: Shape.Decoder.paddedHeadWidth)
        crossAttentionKeys = try .makeZeroedSurfaceBackedFloat16(shape: crossAttentionShape)
        crossAttentionValues = try .makeZeroedSurfaceBackedFloat16(shape: crossAttentionShape)
        memoryFrameIsReal = try .makeZeroed(shape: [1, Shape.Decoder.maximumMemoryFrameCount], dataType: .float32)
        featureFrames = [Float](repeating: 0, count: Shape.Decoder.maximumMemoryFrameCount * Shape.Encoder.frameWidth)
        audioChunkInput = try .makeZeroed(shape: [1, Shape.Audio.samplesPerChunk], dataType: .float32)
        encoderWindowInput = try .makeZeroed(
            shape: [1, Shape.Encoder.windowFrameCount, Shape.Encoder.frameWidth], dataType: .float32)
        encoderWindowFrameIsRealInput = try .makeZeroed(shape: [1, Shape.Encoder.windowFrameCount], dataType: .float32)
        adapterEncodedFramesInput = try .makeZeroed(
            shape: [1, Shape.Adapter.memoryFramesPerCall, Shape.Encoder.frameWidth], dataType: .float32)
        adapterPositionOffsetInput = try .makeZeroed(shape: [1], dataType: .int32)
        decoderTokenInput = try .makeZeroed(shape: [1, 1], dataType: .int32)
        decoderPositionInput = try .makeZeroed(shape: [1], dataType: .int32)
        conv1Buffer = try .makeZeroed(shape: Shape.Frontend.conv1BufferShape, dataType: .float32)
        conv2Buffer = try .makeZeroed(shape: Shape.Frontend.conv2BufferShape, dataType: .float32)
    }

    var isUtteranceInProgress: Bool { decoderState != nil }

    /// Every memory frame is used: another chunk won't fit, so the utterance has to end.
    var hasReachedMaximumUtteranceLength: Bool {
        featureFrameCount + TinyModel.Frontend.featureFramesPerChunk > TinyModel.Decoder.maximumMemoryFrameCount
    }

    /// Starts over everything kept per utterance. The cross-attention buffers keep their old rows, hidden by
    /// memoryFrameIsReal.
    func beginUtterance() throws {
        conv1Buffer = try .makeZeroed(shape: TinyModel.Frontend.conv1BufferShape, dataType: .float32)
        conv2Buffer = try .makeZeroed(shape: TinyModel.Frontend.conv2BufferShape, dataType: .float32)
        featureFrameCount = 0
        encodedFrameCount = 0
        memoryFrameCount = 0
        heardSampleCount = 0
        memoryFrameIsReal.withUnsafeMutableBufferPointer(ofType: Float.self) { isReal, _ in isReal.update(repeating: 0) }
        decoderState = models.decoderStep.makeState()
    }

    /// One chunk, exactly 1280 samples, through the frontend, encoder, adapter and cross-attention keys and values.
    func addChunk(_ audioChunk: [Float]) throws {
        try requireUtteranceInProgress()
        guard audioChunk.count == TinyModel.Audio.samplesPerChunk else {
            throw Moonshine.Error("a chunk is \(TinyModel.Audio.samplesPerChunk) samples, got \(audioChunk.count)")
        }
        try listen(to: audioChunk, isFinal: false)
        heardSampleCount += TinyModel.Audio.samplesPerChunk
    }

    /// A decode from scratch over the memory frames so far.
    func update() throws -> Moonshine.TranscriptUpdate {
        let decoderState = try requireUtteranceInProgress()
        let maximumTokenCount = Self.maximumTokenCount(heardSampleCount: heardSampleCount)
        guard memoryFrameCount > 0 else {  // nothing to decode yet; Moonshine returns "" here too
            return Moonshine.TranscriptUpdate(
                tokenIDs: [], text: "", maximumTokenCount: maximumTokenCount, heardSampleCount: heardSampleCount,
                decodeDuration: .zero)
        }
        let clock = ContinuousClock()
        var tokenIDs: [Int] = []
        let decodeDuration = try clock.measure {
            tokenIDs = try decode(maximumTokenCount: maximumTokenCount, decoderState: decoderState)
        }
        return Moonshine.TranscriptUpdate(
            tokenIDs: tokenIDs,
            text: try tokenizer.text(forTokenIDs: tokenIDs),
            maximumTokenCount: maximumTokenCount,
            heardSampleCount: heardSampleCount,
            decodeDuration: decodeDuration)
    }

    /// remainingSamples: what arrived after the last whole chunk, 0 to 1279 samples. They're padded with zeros to a
    /// whole chunk, and the encoder releases its 16 held-back frames, so every frame reaches the decoder. Then the final
    /// decode. No utterance is in progress afterwards.
    func endUtterance(remainingSamples: [Float]) throws -> Moonshine.TranscriptUpdate {
        try requireUtteranceInProgress()
        guard remainingSamples.count < TinyModel.Audio.samplesPerChunk else {
            throw Moonshine.Error("\(remainingSamples.count) remaining samples: whole chunks go through addChunk")
        }
        if remainingSamples.isEmpty {
            try encode(isFinal: true)
        } else {
            let padding = [Float](repeating: 0, count: TinyModel.Audio.samplesPerChunk - remainingSamples.count)
            try listen(to: remainingSamples + padding, isFinal: true)
        }
        heardSampleCount += remainingSamples.count
        defer { decoderState = nil }
        return try update()
    }

    /// Moonshine's limit, min(ceil(seconds × 6.5), 256), with 128 in place of 256 since the self-attention cache has
    /// 128 rows. It counts seconds of audio heard, not memory frames, which lag the audio by 0.32 s (the held-back
    /// frames) until the utterance ends.
    static func maximumTokenCount(heardSampleCount: Int) -> Int {
        let heardSeconds = Double(heardSampleCount) / Double(TinyModel.Audio.sampleRateHz)
        let tokenCountForAudio = Int((heardSeconds * TinyModel.Tokens.maximumTokensPerSecondOfAudio).rounded(.up))
        return min(tokenCountForAudio, TinyModel.Decoder.maximumTokenCount)
    }

    @discardableResult
    private func requireUtteranceInProgress() throws -> MLState {
        guard let decoderState else {
            throw Moonshine.Error("no utterance in progress: call beginUtterance() first")
        }
        return decoderState
    }
}
