import CoreML

/// Memory frames in, tokens out: Moonshine's greedy decoding loop.
extension StreamingEngine {
    /// The start token at position 0; each step's highest-scoring token goes back in at the next position; stop after
    /// the end token or maximumTokenCount steps. The decoder state isn't reset between updates: each step writes its
    /// own row before reading it, and later rows are masked. If the calling task is cancelled, the decode stops before
    /// its next step with CancellationError, and for the same reason the state is still safe to decode from again.
    func decode(maximumTokenCount: Int, decoderState: MLState) throws -> [Int] {
        var inputTokenID = TinyModel.Tokens.startTokenID
        var pickedTokenIDs: [Int] = []
        for position in 0 ..< maximumTokenCount {
            try Task.checkCancellation()
            decoderTokenInput.withUnsafeMutableBufferPointer(ofType: Int32.self) { input, _ in
                input[0] = Int32(inputTokenID)
            }
            decoderPositionInput.withUnsafeMutableBufferPointer(ofType: Int32.self) { input, _ in
                input[0] = Int32(position)
            }
            let outputs = try models.decoderStep.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    FeatureName.DecoderStep.token: decoderTokenInput,
                    FeatureName.DecoderStep.position: decoderPositionInput,
                    FeatureName.DecoderStep.crossAttentionKeys: crossAttentionKeys,
                    FeatureName.DecoderStep.crossAttentionValues: crossAttentionValues,
                    FeatureName.DecoderStep.memoryFrameIsReal: memoryFrameIsReal,
                ]),
                using: decoderState)
            let logits = try outputs.float32Output(
                named: FeatureName.DecoderStep.logits, of: .decoderStep, shape: [1, 1, TinyModel.Tokens.vocabularySize])
            let pickedTokenID = try highestScoringTokenID(in: logits, position: position)
            pickedTokenIDs.append(pickedTokenID)
            if pickedTokenID == TinyModel.Tokens.endTokenID {
                break
            }
            inputTokenID = pickedTokenID
        }
        return pickedTokenIDs
    }

    /// The first of the highest scores, as Moonshine picks it (a strict > starting from token 0). Any NaN fails: it
    /// would mean the float16 math broke, and since NaN never wins a comparison it would otherwise pass silently.
    private func highestScoringTokenID(in logits: MLMultiArray, position: Int) throws -> Int {
        let tokenStride = logits.integerStrides[2]
        return try logits.withUnsafeBufferPointer(ofType: Float.self) { scores in
            var bestTokenID = 0
            var bestScore = scores[0]
            var sawNaN = bestScore.isNaN
            for tokenID in 1 ..< TinyModel.Tokens.vocabularySize {
                let score = scores[tokenID * tokenStride]
                if score > bestScore {
                    bestScore = score
                    bestTokenID = tokenID
                } else if score.isNaN {
                    sawNaN = true
                }
            }
            guard !sawNaN else {
                throw Moonshine.Error("the decoder returned NaN scores at position \(position)")
            }
            return bestTokenID
        }
    }
}
