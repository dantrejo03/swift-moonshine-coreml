import CoreML
import Foundation
import Testing
@testable import MoonshineCoreML

/// Checks against the real models, which aren't in the repo. They run once scripts/download_models.sh has put the
/// models in Models/, and are skipped until then.
@Suite(.enabled(
    if: FileManager.default.fileExists(atPath: RepositoryPaths.downloadedModels.path),
    "needs the models: run scripts/download_models.sh"))
struct DownloadedModelsTests {
    @Test func downloadedModelsPassEveryLoadCheck() async throws {
        let compiledModelsDirectory = try await Self.compileDownloadedModels()
        defer { try? FileManager.default.removeItem(at: compiledModelsDirectory) }

        _ = try await Moonshine.Transcriber(modelsDirectory: compiledModelsDirectory)
    }

    /// Compiles each downloaded .mlpackage into <name>.mlmodelc in a new temporary folder, with tokenizer.bin beside
    /// them, the way Xcode lays them out in an app bundle. Returns the folder.
    static func compileDownloadedModels() async throws -> URL {
        let fileManager = FileManager.default
        let compiledModelsDirectory = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
        try fileManager.createDirectory(at: compiledModelsDirectory, withIntermediateDirectories: true)
        for name in ModelName.allCases {
            let compiledModelURL = try await MLModel.compileModel(
                at: RepositoryPaths.downloadedModels.appending(path: name.packageFileName))
            try fileManager.moveItem(
                at: compiledModelURL, to: compiledModelsDirectory.appending(path: name.compiledFileName))
        }
        let tokenizerFileName = Moonshine.Transcriber.tokenizerFileName
        try fileManager.copyItem(
            at: RepositoryPaths.downloadedModels.appending(path: tokenizerFileName),
            to: compiledModelsDirectory.appending(path: tokenizerFileName))
        return compiledModelsDirectory
    }
}
