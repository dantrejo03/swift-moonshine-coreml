import CoreML
import Foundation
import os

/// The five compiled models, each loaded once and checked, so a stale or mismatched model fails here instead of
/// producing wrong text later: first that they're one release this package works with (ModelRelease), then that each
/// has its expected interface.
struct LoadedModels {
    /// Every model runs on the CPU and Neural Engine except these, which run on the CPU only. The frontend: an Apple
    /// Watch Series 9's Neural Engine computes its features wrongly (likely float16 underflow in its per-frame
    /// normalization), and it's the smallest model, so the CPU costs little.
    static let cpuOnlyModelNames: Set<ModelName> = [.frontend]

    private static let logger = Logger(subsystem: "MoonshineCoreML", category: "Models")

    let frontend: MLModel
    let encoder: MLModel
    let adapter: MLModel
    let crossKeyValue: MLModel
    let decoderStep: MLModel
    /// Loading is when Core ML prepares a model for this device's chips: seconds the first time, then it's cached.
    let loadDurationByName: [ModelName: Duration]
    /// The release all five carry, or nil if they carry none or several (possible only with checksModelVersions false).
    let releaseVersion: String?

    /// compiledModelsDirectory holds <name>.mlmodelc for all five models, as Xcode puts them in an app bundle. The
    /// transcriber checks they're all there first (ModelsDirectory), so a missing one gets an error that says why.
    ///
    /// checksModelVersions false skips the release check, never the interface check, and logs a warning on every load.
    init(compiledModelsDirectory: URL, checksModelVersions: Bool) throws {
        var loadDurationByName: [ModelName: Duration] = [:]
        func load(_ name: ModelName) throws -> MLModel {
            let compiledModelURL = compiledModelsDirectory.appending(path: name.compiledFileName)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = Self.cpuOnlyModelNames.contains(name) ? .cpuOnly : .cpuAndNeuralEngine
            let clock = ContinuousClock()
            let loadStart = clock.now
            let model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
            loadDurationByName[name] = loadStart.duration(to: clock.now)
            return model
        }
        frontend = try load(.frontend)
        encoder = try load(.encoder)
        adapter = try load(.adapter)
        crossKeyValue = try load(.crossKeyValue)
        decoderStep = try load(.decoderStep)
        self.loadDurationByName = loadDurationByName

        let modelByName: [(name: ModelName, model: MLModel)] = [
            (.frontend, frontend), (.encoder, encoder), (.adapter, adapter), (.crossKeyValue, crossKeyValue),
            (.decoderStep, decoderStep),
        ]
        // The release first: a model from another release says so more plainly through its version than through a
        // list of mismatched features.
        let versions = modelByName.map { ModelRelease.version(of: $0.model) }
        releaseVersion = Set(versions).count == 1 ? versions[0] : nil
        let release = ModelRelease.assess(zip(modelByName, versions).map { (name: $0.0.name, version: $0.1) })
        if !checksModelVersions {
            let warning = "The model version check is off (checksModelVersions: false). \(release.summary) Models "
                + "from a release this package doesn't list can transcribe wrongly without any error."
            Self.logger.warning("\(warning, privacy: .public)")
        } else if !release.isCompatible {
            throw Moonshine.Error(
                "\(compiledModelsDirectory.path): \(release.summary) \(ModelsDirectory.downloadInstructionsURL) says "
                    + "how to download a compatible release, or pass checksModelVersions: false to load these anyway.")
        }
        for (name, model) in modelByName {
            try name.verifyInterface(of: model)
        }
    }
}
