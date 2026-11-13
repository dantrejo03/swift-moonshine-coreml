/// The five Core ML models, named as their compiled files are (frontend.mlmodelc … decoder_step.mlmodelc), in the
/// order audio flows through them.
enum ModelName: String, CaseIterable, Sendable {
    case frontend
    case encoder
    case adapter
    case crossKeyValue = "cross_kv"
    case decoderStep = "decoder_step"
}
