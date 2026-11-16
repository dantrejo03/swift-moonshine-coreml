import Foundation

extension Moonshine {
    /// A passed on-device check (`Transcriber.checkOnThisDevice()`): what was checked and how fast it ran.
    public struct DeviceCheckResult: Sendable {
        /// The model release the reference tokens are for, such as "v1.1".
        public let modelVersion: String
        /// The test recording's length.
        public let recordingDuration: Duration
        /// How long the whole check took, tentative updates included.
        public let checkDuration: Duration
        /// Tentative updates that ran while the recording went in, one per update interval of audio.
        public let tentativeUpdateCount: Int

        /// Seconds of work per second of audio: below 1 is faster than real time.
        public var realTimeFactor: Double { checkDuration / recordingDuration }
    }
}

extension Moonshine.Transcriber {
    /// Transcribes a 10 s test recording that ships with the package and checks that the final tokens are exactly the
    /// ones the reference pipeline gives for the loaded model release. Throws, naming the first difference, if they
    /// aren't.
    ///
    /// Core ML can compute wrong numbers on a particular device without any error: a model that's correct on a Mac can
    /// turn speech into nothing on a watch. The model checks at load can't see that; only running the models can. Run
    /// this once per device model, OS version and model release before trusting the transcripts, and remember a pass:
    /// it takes a few seconds on a watch. The audio goes in as a microphone would deliver it, with tentative updates,
    /// so it also exercises the decoder state the way live use does.
    ///
    /// It ends with no utterance in progress; the next `beginUtterance()` starts fresh. Models loaded with
    /// `checksModelVersions: false` can't be checked unless they carry a release the package has reference tokens for.
    public func checkOnThisDevice() throws -> Moonshine.DeviceCheckResult {
        let modelVersion = try DeviceCheck.checkedModelVersion(modelVersion)
        let referenceTokenIDs = try DeviceCheck.referenceTokenIDs(forModelVersion: modelVersion)
        let samples = try AudioFile.loadSamples(at: DeviceCheck.recordingURL())

        let clock = ContinuousClock()
        let checkStart = clock.now
        try beginUtterance()
        var tentativeUpdateCount = 0
        for blockStart in stride(from: 0, to: samples.count, by: DeviceCheck.blockSampleCount) {
            let blockEnd = min(blockStart + DeviceCheck.blockSampleCount, samples.count)
            tentativeUpdateCount += try receive(Array(samples[blockStart ..< blockEnd])).count
        }
        let final = try endUtterance()
        let checkDuration = checkStart.duration(to: clock.now)

        if let firstDifference = describeFirstDifference(
            between: final.tokenIDs, and: referenceTokenIDs, tokenizer: engine.tokenizer
        ) {
            throw Moonshine.Error(
                "This device transcribes the check recording differently from the reference for models "
                    + "\(modelVersion): \(firstDifference). Core ML computes these models differently here, so "
                    + "transcripts on this device can't be trusted.")
        }
        return Moonshine.DeviceCheckResult(
            modelVersion: modelVersion,
            recordingDuration: .milliseconds(samples.count * 1000 / TinyModel.Audio.sampleRateHz),
            checkDuration: checkDuration,
            tentativeUpdateCount: tentativeUpdateCount)
    }
}

/// The on-device check's recording and the tokens each model release must produce for it.
enum DeviceCheck {
    /// 100 ms: about what one microphone callback delivers. Updates are scheduled by samples received, so the block size
    /// doesn't change the result.
    static let blockSampleCount = 1600

    static let recordingFileName = "beckett.wav"

    /// The final tokens for the whole of beckett.wav, from the Python reference pipeline on the same float16 models.
    /// Every release in ModelRelease.compatibleVersions needs an entry; the tests check that, and that these match
    /// Tests/MoonshineCoreMLTests/Fixtures/reference_transcripts.json.
    static let referenceTokenIDsByModelVersion: [String: [Int]] = [
        "v1.1": [
            18274, 1898, 29892, 3926, 5229, 29892, 694, 4383, 29892, 1018, 1449, 29892, 4418, 1449, 29892, 4418, 2253,
            29889, 2,
        ],
    ]

    static func recordingURL() throws -> URL {
        guard let url = Bundle.module.url(forResource: recordingFileName, withExtension: nil) else {
            throw Moonshine.Error("\(recordingFileName) is missing from the package's resources")
        }
        return url
    }

    /// The loaded models' release, or an error saying why there's nothing to check against.
    static func checkedModelVersion(_ modelVersion: String?) throws -> String {
        guard let modelVersion else {
            throw Moonshine.Error(
                "The on-device check needs models from one release, and these carry none (or several), so there are no "
                    + "reference tokens to check them against. It can check "
                    + referenceTokenIDsByModelVersion.keys.sorted().joined(separator: ", ") + ".")
        }
        return modelVersion
    }

    static func referenceTokenIDs(forModelVersion modelVersion: String) throws -> [Int] {
        guard let referenceTokenIDs = referenceTokenIDsByModelVersion[modelVersion] else {
            throw Moonshine.Error(
                "The on-device check has no reference tokens for models \(modelVersion). It can check "
                    + referenceTokenIDsByModelVersion.keys.sorted().joined(separator: ", ") + ".")
        }
        return referenceTokenIDs
    }
}

/// nil when the token ids are identical; otherwise where they first differ, with each side's token and its text.
func describeFirstDifference(
    between tokenIDs: [Int], and referenceTokenIDs: [Int], tokenizer: Moonshine.Tokenizer
) -> String? {
    guard tokenIDs != referenceTokenIDs else { return nil }
    let position = zip(tokenIDs, referenceTokenIDs).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        ?? min(tokenIDs.count, referenceTokenIDs.count)
    func describeToken(at position: Int, of tokenIDs: [Int]) -> String {
        guard position < tokenIDs.count else { return "(none: the decode had stopped)" }
        return "\(tokenIDs[position]) \((try? tokenizer.text(forTokenIDs: [tokenIDs[position]])).debugDescription)"
    }
    return "first difference at position \(position) of \(tokenIDs.count) tokens (reference \(referenceTokenIDs.count)): "
        + "this run \(describeToken(at: position, of: tokenIDs)), reference \(describeToken(at: position, of: referenceTokenIDs))"
}
