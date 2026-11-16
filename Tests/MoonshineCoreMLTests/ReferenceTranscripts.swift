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
    /// The recording as float32 samples in [-1, 1], read as the on-device check reads its recording.
    func loadSamples() throws -> [Float] {
        try AudioFile.loadSamples(at: RepositoryPaths.recording(named: audioFile))
    }
}

/// The SHA-256 of the samples' bytes as little-endian float32 (every Apple platform is little-endian), in hex.
func float32SamplesSha256(_ samples: [Float]) -> String {
    SHA256.hash(data: samples.withUnsafeBytes { Data($0) }).map { String(format: "%02x", $0) }.joined()
}
