import Foundation
import Testing
@testable import MoonshineCoreML

struct ModelsDirectoryTests {
    @Test func missingFilesAreAllNamedWithWhereToGetThem() {
        let error = #expect(throws: Moonshine.Error.self) {
            try ModelsDirectory.verifyEveryFileIsPresent(in: URL(filePath: "/nonexistent"))
        }
        for fileName in ModelName.allCases.map(\.compiledFileName) + ["tokenizer.bin"] {
            #expect(error?.description.contains(fileName) == true, "doesn't name \(fileName)")
        }
        #expect(error?.description.contains(ModelsDirectory.downloadInstructionsURL) == true)
    }

    @Test func downloadInstructionsLinkToTheReadmesModelsSection() throws {
        #expect(ModelsDirectory.downloadInstructionsURL.hasSuffix("#models"))
        let readme = try String(contentsOf: RepositoryPaths.root.appending(path: "README.md"), encoding: .utf8)
        #expect(readme.contains("\n## Models\n"))
    }

    @Test func uncompiledModelsSayToCompileThem() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory.appending(path: ModelName.frontend.packageFileName), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let error = #expect(throws: Moonshine.Error.self) {
            try ModelsDirectory.verifyEveryFileIsPresent(in: directory)
        }
        #expect(error?.description.contains("frontend.mlpackage") == true)
        #expect(error?.description.contains("MLModel.compileModel") == true)
    }
}
