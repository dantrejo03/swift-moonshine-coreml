import Testing
@testable import MoonshineCoreML

struct LoadedModelsTests {
    @Test func everyModelExpectsDistinctFeatureNames() {
        for modelName in ModelName.allCases {
            let features = modelName.expectedFeatures
            let names = features.map { "\($0.kind) \($0.name)" }
            #expect(Set(names).count == features.count, "\(modelName) lists a feature twice")
        }
    }
}
