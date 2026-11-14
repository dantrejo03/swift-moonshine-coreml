import CoreML
import Foundation
import Testing
@testable import MoonshineCoreML

/// Checks against the real models, which aren't in the repo, end to end: they load and pass every check, the tokenizer
/// and the audio match the reference, and every reference utterance transcribes to exactly the reference tokens. They
/// run once scripts/download_models.sh has put the models in Models/, and are skipped until then.
@Suite(.enabled(
    if: FileManager.default.fileExists(atPath: RepositoryPaths.downloadedModels.path),
    "needs the models: run scripts/download_models.sh"))
struct DownloadedModelsTests {
    /// 100 ms: about what one microphone callback delivers. Updates are scheduled by samples received, so the block size
    /// doesn't change the results.
    static let microphoneBlockSampleCount = 1600

    /// Compiled once per test run and shared, since compiling all five takes seconds.
    static let compiledModelsDirectory = Task { try await compileDownloadedModels() }

    @Test func downloadedModelsPassEveryLoadCheck() async throws {
        _ = try await Moonshine.Transcriber(modelsDirectory: Self.compiledModelsDirectory.value)
    }

    @Test func tokenizerGivesTheReferenceTextForEveryToken() throws {
        let reference = try ReferenceTranscripts.load()
        let tokenizer = try Self.loadDownloadedTokenizer()
        #expect(reference.singleTokenTexts.count == TinyModel.Tokens.vocabularySize)

        var differences: [String] = []
        for (tokenID, expectedText) in reference.singleTokenTexts.enumerated() {
            let text = try? tokenizer.text(forTokenIDs: [tokenID])  // nil where it throws, as expected for empty entries
            if text != expectedText {
                differences.append("token \(tokenID): \(text.debugDescription), expected \(expectedText.debugDescription)")
            }
        }
        #expect(differences.isEmpty, "\(differences.count) tokens differ, including \(differences.prefix(5))")
    }

    @Test func recordingsReadAsTheReferenceSamples() throws {
        for recording in try ReferenceTranscripts.load().recordings {
            let samples = try recording.loadSamples()
            #expect(samples.count == recording.sampleCount, "\(recording.name)")
            #expect(float32SamplesSha256(samples) == recording.float32SamplesSha256, "\(recording.name)")
        }
    }

    /// Every utterance through one transcriber, in order, as an app would run them: audio in microphone-sized blocks,
    /// the tentative updates that fall due, then the final transcript. The tentative updates aren't compared, but they
    /// run, and each one decodes with the same decoder state the final transcript then reuses.
    @Test func everyUtteranceTranscribesToTheReferenceTokens() async throws {
        let reference = try ReferenceTranscripts.load()
        let tokenizer = try Self.loadDownloadedTokenizer()
        let transcriber = try await Moonshine.Transcriber(modelsDirectory: Self.compiledModelsDirectory.value)

        for recording in reference.recordings {
            let samples = try recording.loadSamples()
            for (utteranceIndex, utterance) in recording.utterances.enumerated() {
                let label = "\(recording.name) utterance \(utteranceIndex + 1)"
                try await transcriber.beginUtterance()
                for blockStart in stride(
                    from: utterance.startSample, to: utterance.endSample, by: Self.microphoneBlockSampleCount
                ) {
                    let blockEnd = min(blockStart + Self.microphoneBlockSampleCount, utterance.endSample)
                    _ = try await transcriber.receive(Array(samples[blockStart ..< blockEnd]))
                }
                let final = try await transcriber.endUtterance()

                let firstDifference = describeFirstDifference(
                    between: final.tokenIDs, and: utterance.finalTokenIDs, tokenizer: tokenizer)
                #expect(firstDifference == nil, "\(label): \(firstDifference ?? "")")
                #expect(final.text == utterance.finalText, "\(label)")
                #expect(final.heardSampleCount == utterance.endSample - utterance.startSample, "\(label)")
            }
        }
    }

    static func loadDownloadedTokenizer() throws -> Moonshine.Tokenizer {
        try Moonshine.Tokenizer(
            contentsOf: RepositoryPaths.downloadedModels.appending(path: Moonshine.Transcriber.tokenizerFileName))
    }

    /// Compiles each downloaded .mlpackage into <name>.mlmodelc, with tokenizer.bin beside them, the way Xcode lays them
    /// out in an app bundle. Always the same temporary folder, replaced each run, so runs don't pile up copies.
    static func compileDownloadedModels() async throws -> URL {
        let fileManager = FileManager.default
        let compiledModelsDirectory = fileManager.temporaryDirectory.appending(path: "MoonshineCoreMLTests-CompiledModels")
        try? fileManager.removeItem(at: compiledModelsDirectory)
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
