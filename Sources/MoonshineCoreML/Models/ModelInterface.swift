import CoreML

/// One input, output or state of a model: what the model loader checks and what reading an output expects.
struct FeatureDescription: Hashable, CustomStringConvertible {
    enum Kind: String { case input, output, state }

    let kind: Kind
    let name: String
    let shape: [Int]
    let dataType: MLMultiArrayDataType

    var description: String { "\(kind.rawValue) \(name) \(shape) \(dataType.name)" }
}

extension ModelName {
    /// Every input, output and state this model must have. The cross-attention keys and values go into the decoder as
    /// float16, and the decoder's state and cross-attention rows are padded from 40 numbers to 64 (see TinyModel.Decoder).
    var expectedFeatures: Set<FeatureDescription> {
        typealias Shape = TinyModel
        let fourFrameShape = [1, Shape.Adapter.memoryFramesPerCall, Shape.Encoder.frameWidth]
        let memoryFrameShape = [1, Shape.Adapter.memoryFramesPerCall, Shape.Adapter.memoryFrameWidth]
        let encoderWindowLength = Shape.Encoder.windowFrameCount
        let crossAttentionShape = Shape.Decoder.attentionCacheShape(
            rowCount: Shape.Decoder.maximumMemoryFrameCount, numbersPerRow: Shape.Decoder.paddedHeadWidth)
        let selfAttentionShape = Shape.Decoder.attentionCacheShape(
            rowCount: Shape.Decoder.maximumTokenCount, numbersPerRow: Shape.Decoder.paddedHeadWidth)
        let newCrossAttentionRowsShape = Shape.Decoder.attentionCacheShape(rowCount: Shape.Adapter.memoryFramesPerCall)

        let features: [FeatureDescription] = switch self {
        case .frontend: [
            .init(kind: .input, name: FeatureName.Frontend.audioChunk, shape: [1, Shape.Audio.samplesPerChunk], dataType: .float32),
            .init(kind: .input, name: FeatureName.Frontend.conv1Buffer, shape: Shape.Frontend.conv1BufferShape, dataType: .float32),
            .init(kind: .input, name: FeatureName.Frontend.conv2Buffer, shape: Shape.Frontend.conv2BufferShape, dataType: .float32),
            .init(kind: .output, name: FeatureName.Frontend.featureFrames, shape: fourFrameShape, dataType: .float32),
            .init(kind: .output, name: FeatureName.Frontend.conv1BufferOut, shape: Shape.Frontend.conv1BufferShape, dataType: .float32),
            .init(kind: .output, name: FeatureName.Frontend.conv2BufferOut, shape: Shape.Frontend.conv2BufferShape, dataType: .float32),
        ]
        case .encoder: [
            .init(kind: .input, name: FeatureName.Encoder.featureFrames, shape: [1, encoderWindowLength, Shape.Encoder.frameWidth], dataType: .float32),
            .init(kind: .input, name: FeatureName.Encoder.featureFrameIsReal, shape: [1, encoderWindowLength], dataType: .float32),
            .init(kind: .output, name: FeatureName.Encoder.encodedFrames, shape: [1, encoderWindowLength, Shape.Encoder.frameWidth], dataType: .float32),
        ]
        case .adapter: [
            .init(kind: .input, name: FeatureName.Adapter.encodedFrames, shape: fourFrameShape, dataType: .float32),
            .init(kind: .input, name: FeatureName.Adapter.positionOffset, shape: [1], dataType: .int32),
            .init(kind: .output, name: FeatureName.Adapter.memoryFrames, shape: memoryFrameShape, dataType: .float32),
        ]
        case .crossKeyValue: [
            .init(kind: .input, name: FeatureName.CrossKeyValue.memoryFrames, shape: memoryFrameShape, dataType: .float32),
            .init(kind: .output, name: FeatureName.CrossKeyValue.newKeys, shape: newCrossAttentionRowsShape, dataType: .float32),
            .init(kind: .output, name: FeatureName.CrossKeyValue.newValues, shape: newCrossAttentionRowsShape, dataType: .float32),
        ]
        case .decoderStep: [
            .init(kind: .input, name: FeatureName.DecoderStep.token, shape: [1, 1], dataType: .int32),
            .init(kind: .input, name: FeatureName.DecoderStep.position, shape: [1], dataType: .int32),
            .init(kind: .input, name: FeatureName.DecoderStep.crossAttentionKeys, shape: crossAttentionShape, dataType: .float16),
            .init(kind: .input, name: FeatureName.DecoderStep.crossAttentionValues, shape: crossAttentionShape, dataType: .float16),
            .init(kind: .input, name: FeatureName.DecoderStep.memoryFrameIsReal, shape: [1, Shape.Decoder.maximumMemoryFrameCount], dataType: .float32),
            .init(kind: .output, name: FeatureName.DecoderStep.logits, shape: [1, 1, Shape.Tokens.vocabularySize], dataType: .float32),
            .init(kind: .state, name: FeatureName.DecoderStep.selfAttentionKeys, shape: selfAttentionShape, dataType: .float16),
            .init(kind: .state, name: FeatureName.DecoderStep.selfAttentionValues, shape: selfAttentionShape, dataType: .float16),
        ]
        }
        return Set(features)
    }

    /// Fails unless the loaded model has exactly the expected features, listing what's missing and what's extra.
    func verifyInterface(of model: MLModel) throws {
        let modelDescription = model.modelDescription
        var actualFeatures = Set<String>()
        for (kind, descriptionsByName) in [
            (FeatureDescription.Kind.input, modelDescription.inputDescriptionsByName),
            (.output, modelDescription.outputDescriptionsByName),
            (.state, modelDescription.stateDescriptionsByName),
        ] {
            for (featureName, featureDescription) in descriptionsByName {
                if let array = featureDescription.multiArrayConstraint {
                    actualFeatures.insert(FeatureDescription(
                        kind: kind, name: featureName, shape: array.shape.map(\.intValue), dataType: array.dataType
                    ).description)
                } else if let state = featureDescription.stateConstraint {
                    actualFeatures.insert(FeatureDescription(
                        kind: kind, name: featureName, shape: state.bufferShape, dataType: state.dataType
                    ).description)
                } else {
                    actualFeatures.insert(
                        "\(kind.rawValue) \(featureName): not an array (feature type \(featureDescription.type.rawValue))")
                }
            }
        }
        let expectedFeatures = Set(expectedFeatures.map(\.description))
        guard actualFeatures == expectedFeatures else {
            throw Moonshine.Error(
                "\(rawValue) doesn't have the expected interface, so it's likely a different version of the model. "
                    + "Expected but missing: \(expectedFeatures.subtracting(actualFeatures).sorted()); "
                    + "present but not expected: \(actualFeatures.subtracting(expectedFeatures).sorted())")
        }
    }
}

extension MLFeatureProvider {
    /// One float32 output of a prediction, checked against the shape the caller expects.
    func float32Output(named outputName: String, of modelName: ModelName, shape: [Int]) throws -> MLMultiArray {
        guard let array = featureValue(for: outputName)?.multiArrayValue else {
            throw Moonshine.Error("\(modelName.rawValue) returned no \(outputName)")
        }
        guard array.dataType == .float32, array.integerShape == shape else {
            throw Moonshine.Error(
                "\(modelName.rawValue)'s \(outputName) is \(array.integerShape) \(array.dataType.name), "
                    + "expected \(shape) float32")
        }
        return array
    }
}
