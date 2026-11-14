import AVFoundation
import CryptoKit
import Foundation
@testable import MoonshineCoreML

/// Fixtures/reference_transcripts.json: the recordings, and what the models must produce for them. The Fixtures README
/// describes the format.
struct ReferenceTranscripts: Decodable {
    struct Models: Decodable {
        let huggingfaceRepository: String
        let revision: String
    }

    struct Recording: Decodable {
        let name: String
        let audioFile: String
        let sampleCount: Int
        let float32SamplesSha256: String
        let utterances: [Utterance]
    }

    struct Utterance: Decodable {
        let startSample: Int
        let endSample: Int
        let finalTokenIDs: [Int]
        let finalText: String

        // The decoder converts final_token_ids to finalTokenIds before matching these.
        enum CodingKeys: String, CodingKey {
            case startSample, endSample, finalText
            case finalTokenIDs = "finalTokenIds"
        }
    }

    let models: Models
    /// Each token id's text on its own; nil where the tokenizer must fail (an empty entry).
    let singleTokenTexts: [String?]
    let recordings: [Recording]

    static func load() throws -> ReferenceTranscripts {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ReferenceTranscripts.self, from: Data(contentsOf: RepositoryPaths.referenceTranscripts))
    }
}

extension ReferenceTranscripts.Recording {
    /// The recording as float32 samples in [-1, 1], read with AVAudioFile. 16-bit samples become sample / 32768, as
    /// the reference pipeline read them, so the SHA-256 in the reference confirms both started from the same numbers.
    func loadSamples() throws -> [Float] {
        let audioURL = RepositoryPaths.fixtures.appending(path: audioFile)
        let file = try AVAudioFile(forReading: audioURL)
        let format = file.processingFormat
        guard format.sampleRate == Double(Moonshine.Transcriber.sampleRateHz), format.channelCount == 1,
              format.commonFormat == .pcmFormatFloat32
        else {
            throw Moonshine.Error("\(audioFile): expected 16 kHz mono, read as \(format)")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw Moonshine.Error("\(audioFile): couldn't allocate \(file.length) frames")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0], Int64(buffer.frameLength) == file.length else {
            throw Moonshine.Error("\(audioFile): read \(buffer.frameLength) of \(file.length) frames")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }
}

/// The SHA-256 of the samples' bytes as little-endian float32 (every Apple platform is little-endian), in hex.
func float32SamplesSha256(_ samples: [Float]) -> String {
    SHA256.hash(data: samples.withUnsafeBytes { Data($0) }).map { String(format: "%02x", $0) }.joined()
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
