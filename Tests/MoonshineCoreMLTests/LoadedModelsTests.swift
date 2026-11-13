import Foundation
import Testing
@testable import MoonshineCoreML

struct LoadedModelsTests {
    @Test func missingModelFileNamesTheFile() throws {
        let emptyDirectory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDirectory) }

        let error = #expect(throws: Moonshine.Error.self) {
            try LoadedModels(compiledModelsDirectory: emptyDirectory)
        }
        #expect(error?.description.contains("frontend.mlmodelc") == true)
    }

    @Test func everyModelExpectsDistinctFeatureNames() {
        for modelName in ModelName.allCases {
            let features = modelName.expectedFeatures
            let names = features.map { "\($0.kind) \($0.name)" }
            #expect(Set(names).count == features.count, "\(modelName) lists a feature twice")
        }
    }
}
