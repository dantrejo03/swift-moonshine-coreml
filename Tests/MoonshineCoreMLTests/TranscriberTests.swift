import Foundation
import Testing
@testable import MoonshineCoreML

struct TranscriberTests {
    @Test func maximumUtteranceDurationIs15Point36Seconds() {
        #expect(Moonshine.Transcriber.maximumUtteranceDuration == .milliseconds(15_360))
    }

    @Test func zeroUpdateIntervalIsAnError() async throws {
        let error = await #expect(throws: Moonshine.Error.self) {
            try await Moonshine.Transcriber(modelsDirectory: URL(filePath: "/nonexistent"), updateInterval: .zero)
        }
        #expect(error?.description.contains("update interval") == true)
    }

    @Test func missingTokenizerIsAnError() async throws {
        let emptyDirectory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDirectory) }

        await #expect(throws: (any Error).self) {
            try await Moonshine.Transcriber(modelsDirectory: emptyDirectory)
        }
    }
}
