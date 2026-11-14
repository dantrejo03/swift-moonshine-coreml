import Foundation
import Testing
@testable import MoonshineCoreML

/// scripts/download_models.sh pins one model release. The package must list it as compatible, and the reference
/// transcripts must be for it, or the end-to-end checks compare the downloaded models against another release's output.
struct DownloadScriptTests {
    let script: String

    init() throws {
        script = try String(contentsOf: RepositoryPaths.downloadScript, encoding: .utf8)
    }

    func scriptVariable(_ name: String) -> String? {
        script.split(separator: "\n").first { $0.hasPrefix("\(name)=") }.map { String($0.dropFirst(name.count + 1)) }
    }

    @Test func downloadsACompatibleRelease() throws {
        let releaseTag = try #require(scriptVariable("release_tag"))
        #expect(ModelRelease.compatibleVersions.contains(releaseTag))
    }

    @Test func downloadsTheReleaseTheReferenceTranscriptsAreFor() throws {
        let referenceModels = try ReferenceTranscripts.load().models

        #expect(scriptVariable("huggingface_repository") == referenceModels.huggingfaceRepository)
        #expect(scriptVariable("release_tag") == referenceModels.revision)
    }
}
