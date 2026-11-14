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

    @Test func missingModelsSayWhereToGetThem() async throws {
        let error = await #expect(throws: Moonshine.Error.self) {
            try await Moonshine.Transcriber(modelsDirectory: URL(filePath: "/nonexistent"))
        }
        #expect(error?.description.contains(ModelsDirectory.downloadInstructionsURL) == true)
    }
}
