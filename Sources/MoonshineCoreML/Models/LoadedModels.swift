import CoreML
import Foundation

/// The five compiled models, each loaded once and checked against its expected interface, so a stale or mismatched
/// model fails here instead of producing wrong text later.
struct LoadedModels {
    /// Every model runs on the CPU and Neural Engine except these, which run on the CPU only. The frontend: an Apple
    /// Watch Series 9's Neural Engine computes its features wrongly (likely float16 underflow in its per-frame
    /// normalization), and it's the smallest model, so the CPU costs little.
    static let cpuOnlyModelNames: Set<ModelName> = [.frontend]

    let frontend: MLModel
    let encoder: MLModel
    let adapter: MLModel
    let crossKeyValue: MLModel
    let decoderStep: MLModel
    /// Loading is when Core ML prepares a model for this device's chips: seconds the first time, then it's cached.
    let loadDurationByName: [ModelName: Duration]

    /// compiledModelsDirectory holds <name>.mlmodelc for all five models, as Xcode puts them in an app bundle.
    init(compiledModelsDirectory: URL) throws {
        var loadDurationByName: [ModelName: Duration] = [:]
        func load(_ name: ModelName) throws -> MLModel {
            let compiledModelURL = compiledModelsDirectory.appending(path: "\(name.rawValue).mlmodelc")
            guard FileManager.default.fileExists(atPath: compiledModelURL.path) else {
                throw Moonshine.Error("\(compiledModelURL.path) is missing")
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = Self.cpuOnlyModelNames.contains(name) ? .cpuOnly : .cpuAndNeuralEngine
            let clock = ContinuousClock()
            let loadStart = clock.now
            let model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
            loadDurationByName[name] = loadStart.duration(to: clock.now)
            try name.verifyInterface(of: model)
            return model
        }
        frontend = try load(.frontend)
        encoder = try load(.encoder)
        adapter = try load(.adapter)
        crossKeyValue = try load(.crossKeyValue)
        decoderStep = try load(.decoderStep)
        self.loadDurationByName = loadDurationByName
    }
}
