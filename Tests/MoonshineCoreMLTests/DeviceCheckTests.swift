import Foundation
import Testing
@testable import MoonshineCoreML

/// The on-device check's data, without the models. DownloadedModelsTests runs the check itself.
struct DeviceCheckTests {
    @Test func everyCompatibleReleaseHasReferenceTokens() {
        let releasesWithoutTokens = ModelRelease.compatibleVersions
            .subtracting(DeviceCheck.referenceTokenIDsByModelVersion.keys)
        #expect(releasesWithoutTokens.isEmpty, "add reference tokens for \(releasesWithoutTokens.sorted())")
    }

    /// The check plays the whole recording, so the reference has to be one utterance covering all of it.
    @Test func referenceTokensMatchTheFixtures() throws {
        let reference = try ReferenceTranscripts.load()
        let recording = try #require(reference.recordings.first { $0.audioFile == DeviceCheck.recordingFileName })
        let utterance = try #require(recording.utterances.first)
        #expect(recording.utterances.count == 1)
        #expect(utterance.startSample == 0 && utterance.endSample == recording.sampleCount)
        #expect(DeviceCheck.referenceTokenIDsByModelVersion[reference.models.revision] == utterance.finalTokenIDs)
    }

    /// What the check loads from the package's resources is exactly the recording the reference tokens are for.
    @Test func bundledRecordingIsTheReferenceRecording() throws {
        let reference = try ReferenceTranscripts.load()
        let recording = try #require(reference.recordings.first { $0.audioFile == DeviceCheck.recordingFileName })
        let samples = try AudioFile.loadSamples(at: DeviceCheck.recordingURL())
        #expect(samples.count == recording.sampleCount)
        #expect(float32SamplesSha256(samples) == recording.float32SamplesSha256)
    }

    @Test func modelsWithoutOneReleaseCantBeChecked() {
        let error = #expect(throws: Moonshine.Error.self) { try DeviceCheck.checkedModelVersion(nil) }
        #expect(error?.description.contains("no reference tokens") == true)
    }

    @Test func aReleaseWithoutReferenceTokensCantBeChecked() {
        let error = #expect(throws: Moonshine.Error.self) { try DeviceCheck.referenceTokenIDs(forModelVersion: "v99.0") }
        #expect(error?.description.contains("v99.0") == true)
        #expect(error?.description.contains("v1.1") == true)
    }
}
