/// The five Core ML models, in the order audio flows through them.
enum ModelName: String, CaseIterable, Sendable {
    case frontend
    case encoder
    case adapter
    case crossKeyValue = "cross_kv"
    case decoderStep = "decoder_step"

    /// The model as released on Hugging Face: what you add to an app's target.
    var packageFileName: String { "\(rawValue).mlpackage" }
    /// What Xcode compiles packageFileName into, and what the transcriber loads.
    var compiledFileName: String { "\(rawValue).mlmodelc" }
}
