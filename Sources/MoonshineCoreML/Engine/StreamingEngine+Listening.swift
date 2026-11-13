import CoreML

/// Audio in, memory frames out: frontend → encoder → adapter → cross-attention keys and values. Each chunk adds 4
/// feature frames, and each 4 encoded frames that become final add 4 memory frames.
extension StreamingEngine {
    /// The frontend turns one chunk into 4 feature frames; then the encoder runs.
    func listen(to audioChunk: [Float], isFinal: Bool) throws {
        guard !hasReachedMaximumUtteranceLength else {
            let maximumSeconds = Double(TinyModel.Utterance.maximumSampleCount) / Double(TinyModel.Audio.sampleRateHz)
            throw Moonshine.Error("an utterance holds at most \(maximumSeconds) s of audio: end it before adding more")
        }
        audioChunkInput.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, _ in
            audioChunk.withUnsafeBufferPointer { source in
                destination.baseAddress!.update(from: source.baseAddress!, count: source.count)
            }
        }
        let outputs = try models.frontend.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            FeatureName.Frontend.audioChunk: audioChunkInput,
            FeatureName.Frontend.conv1Buffer: conv1Buffer,
            FeatureName.Frontend.conv2Buffer: conv2Buffer,
        ]))
        let newFeatureFrames = try outputs.float32Output(
            named: FeatureName.Frontend.featureFrames, of: .frontend,
            shape: [1, TinyModel.Frontend.featureFramesPerChunk, TinyModel.Encoder.frameWidth])
        conv1Buffer = try outputs.float32Output(
            named: FeatureName.Frontend.conv1BufferOut, of: .frontend, shape: TinyModel.Frontend.conv1BufferShape)
        conv2Buffer = try outputs.float32Output(
            named: FeatureName.Frontend.conv2BufferOut, of: .frontend, shape: TinyModel.Frontend.conv2BufferShape)

        let frameWidth = TinyModel.Encoder.frameWidth
        let firstNewFrame = featureFrameCount
        let sourceStrides = newFeatureFrames.integerStrides
        newFeatureFrames.withUnsafeBufferPointer(ofType: Float.self) { source in
            for frame in 0 ..< TinyModel.Frontend.featureFramesPerChunk {
                for number in 0 ..< frameWidth {
                    featureFrames[(firstNewFrame + frame) * frameWidth + number] =
                        source[frame * sourceStrides[1] + number * sourceStrides[2]]
                }
            }
        }
        featureFrameCount += TinyModel.Frontend.featureFramesPerChunk
        try encode(isFinal: isFinal)
    }

    /// Runs the encoder over a window of the feature frames so far and passes each group of 4 newly final encoded frames
    /// on to the adapter. The newest 16 encoded frames aren't final until more audio arrives (or the utterance ends,
    /// isFinal), so nothing comes out for the first 4 chunks.
    func encode(isFinal: Bool) throws {
        typealias Encoder = TinyModel.Encoder
        let finalFrameCount = isFinal ? featureFrameCount : max(0, featureFrameCount - Encoder.heldBackFrameCount)
        let newFrameCount = finalFrameCount - encodedFrameCount
        guard newFrameCount > 0 else { return }
        guard newFrameCount.isMultiple(of: TinyModel.Adapter.memoryFramesPerCall) else {
            throw Moonshine.Error(
                "\(newFrameCount) new encoded frames is not a whole number of "
                    + "\(TinyModel.Adapter.memoryFramesPerCall)-frame adapter calls")
        }
        let windowStart = max(0, encodedFrameCount - Encoder.leftContextFrameCount)
        let windowFrameCount = featureFrameCount - windowStart
        guard windowFrameCount <= Encoder.windowFrameCount else {
            throw Moonshine.Error("a window of \(windowFrameCount) frames is larger than \(Encoder.windowFrameCount)")
        }

        let frameWidth = Encoder.frameWidth
        encoderWindowInput.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, _ in
            featureFrames.withUnsafeBufferPointer { source in
                destination.baseAddress!.update(
                    from: source.baseAddress! + windowStart * frameWidth, count: windowFrameCount * frameWidth)
            }
            (destination.baseAddress! + windowFrameCount * frameWidth).update(
                repeating: 0, count: (Encoder.windowFrameCount - windowFrameCount) * frameWidth)
        }
        encoderWindowFrameIsRealInput.withUnsafeMutableBufferPointer(ofType: Float.self) { isReal, _ in
            for frame in 0 ..< Encoder.windowFrameCount {
                isReal[frame] = frame < windowFrameCount ? 1 : 0
            }
        }
        let outputs = try models.encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            FeatureName.Encoder.featureFrames: encoderWindowInput,
            FeatureName.Encoder.featureFrameIsReal: encoderWindowFrameIsRealInput,
        ]))
        let encodedFrames = try outputs.float32Output(
            named: FeatureName.Encoder.encodedFrames, of: .encoder, shape: [1, Encoder.windowFrameCount, frameWidth])

        let firstNewFrameInWindow = encodedFrameCount - windowStart
        encodedFrameCount += newFrameCount
        for firstFrameOfCall in stride(
            from: firstNewFrameInWindow, to: firstNewFrameInWindow + newFrameCount,
            by: TinyModel.Adapter.memoryFramesPerCall
        ) {
            copyEncodedFramesIntoAdapterInput(encodedFrames, startingAt: firstFrameOfCall)
            try appendMemoryFrames()
        }
    }

    private func copyEncodedFramesIntoAdapterInput(_ encodedFrames: MLMultiArray, startingAt firstFrame: Int) {
        let frameWidth = TinyModel.Encoder.frameWidth
        let sourceStrides = encodedFrames.integerStrides
        encodedFrames.withUnsafeBufferPointer(ofType: Float.self) { source in
            adapterEncodedFramesInput.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, _ in
                for frame in 0 ..< TinyModel.Adapter.memoryFramesPerCall {
                    for number in 0 ..< frameWidth {
                        destination[frame * frameWidth + number] =
                            source[(firstFrame + frame) * sourceStrides[1] + number * sourceStrides[2]]
                    }
                }
            }
        }
    }

    /// The adapter, at a position offset of the memory frames so far, makes 4 memory frames. The cross-attention model
    /// turns them into keys and values, written into the buffers at the same 4 rows, which memoryFrameIsReal then
    /// shows to the decoder.
    private func appendMemoryFrames() throws {
        let firstNewFrame = memoryFrameCount
        adapterPositionOffsetInput.withUnsafeMutableBufferPointer(ofType: Int32.self) { offset, _ in
            offset[0] = Int32(firstNewFrame)
        }
        let adapterOutputs = try models.adapter.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            FeatureName.Adapter.encodedFrames: adapterEncodedFramesInput,
            FeatureName.Adapter.positionOffset: adapterPositionOffsetInput,
        ]))
        let memoryFrames = try adapterOutputs.float32Output(
            named: FeatureName.Adapter.memoryFrames, of: .adapter,
            shape: [1, TinyModel.Adapter.memoryFramesPerCall, TinyModel.Adapter.memoryFrameWidth])
        let crossKeyValueOutputs = try models.crossKeyValue.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [FeatureName.CrossKeyValue.memoryFrames: memoryFrames]))
        let newRowsShape = TinyModel.Decoder.attentionCacheShape(rowCount: TinyModel.Adapter.memoryFramesPerCall)
        copyNewRows(
            try crossKeyValueOutputs.float32Output(
                named: FeatureName.CrossKeyValue.newKeys, of: .crossKeyValue, shape: newRowsShape),
            intoFloat16Buffer: crossAttentionKeys, atRow: firstNewFrame)
        copyNewRows(
            try crossKeyValueOutputs.float32Output(
                named: FeatureName.CrossKeyValue.newValues, of: .crossKeyValue, shape: newRowsShape),
            intoFloat16Buffer: crossAttentionValues, atRow: firstNewFrame)
        memoryFrameIsReal.withUnsafeMutableBufferPointer(ofType: Float.self) { isReal, _ in
            for frame in firstNewFrame ..< firstNewFrame + TinyModel.Adapter.memoryFramesPerCall {
                isReal[frame] = 1
            }
        }
        memoryFrameCount += TinyModel.Adapter.memoryFramesPerCall
    }

    /// The cross-attention model's 4 new rows, [6, 1, 8, 4, 40] float32, written on the row axis (axis 3) of a
    /// [6, 1, 8, 768, 64] float16 buffer at firstRow ..< firstRow + 4. Each number is rounded to the nearest float16
    /// (ties to even); each row's other 24 numbers stay zero from when the buffer was made.
    private func copyNewRows(_ newRows: MLMultiArray, intoFloat16Buffer buffer: MLMultiArray, atRow firstRow: Int) {
        let sourceStrides = newRows.integerStrides
        newRows.withUnsafeBufferPointer(ofType: Float.self) { source in
            buffer.withUnsafeMutableBufferPointer(ofType: Float16.self) { destination, destinationStrides in
                for layer in 0 ..< TinyModel.Decoder.layerCount {
                    for head in 0 ..< TinyModel.Decoder.headCount {
                        for row in 0 ..< TinyModel.Adapter.memoryFramesPerCall {
                            let sourceRowStart = layer * sourceStrides[0] + head * sourceStrides[2] + row * sourceStrides[3]
                            let destinationRowStart = layer * destinationStrides[0] + head * destinationStrides[2]
                                + (firstRow + row) * destinationStrides[3]
                            for number in 0 ..< TinyModel.Decoder.headWidth {
                                destination[destinationRowStart + number * destinationStrides[4]] =
                                    Float16(source[sourceRowStart + number * sourceStrides[4]])
                            }
                        }
                    }
                }
            }
        }
    }
}
