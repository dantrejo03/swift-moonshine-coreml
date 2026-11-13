/// The names of every model input, output and state, as the compiled models spell them. They're a contract with the
/// model files and can't be renamed; Swift code refers to them only through these constants.
enum FeatureName {
    enum Frontend {
        static let audioChunk = "audio_chunk"
        static let conv1Buffer = "conv1_buffer"
        static let conv2Buffer = "conv2_buffer"
        static let featureFrames = "features"
        static let conv1BufferOut = "conv1_buffer_out"
        static let conv2BufferOut = "conv2_buffer_out"
    }

    enum Encoder {
        static let featureFrames = "features"
        static let featureFrameIsReal = "valid"
        static let encodedFrames = "encoded"
    }

    enum Adapter {
        static let encodedFrames = "encoded"
        static let positionOffset = "pos_offset"
        static let memoryFrames = "memory"
    }

    enum CrossKeyValue {
        static let memoryFrames = "memory"
        static let newKeys = "k_new"
        static let newValues = "v_new"
    }

    enum DecoderStep {
        static let token = "token"
        static let position = "position"
        static let crossAttentionKeys = "k_cross"
        static let crossAttentionValues = "v_cross"
        static let memoryFrameIsReal = "cross_valid"
        static let logits = "logits"
        static let selfAttentionKeys = "k_self"
        static let selfAttentionValues = "v_self"
    }
}
