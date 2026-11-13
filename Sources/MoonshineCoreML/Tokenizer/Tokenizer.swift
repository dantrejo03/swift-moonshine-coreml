import Foundation

extension Moonshine {
    /// Turns token ids into text exactly as Moonshine's own runtime does (`BinTokenizer::tokens_to_text`, then
    /// `Transcriber::sanitize_text`), so transcripts match Moonshine's byte for byte. In order: join the tokens' bytes,
    /// turn each ▁ into a space, trim spaces and tabs from both ends, then replace each byte that doesn't start a
    /// whole UTF-8 sequence with "?".
    public struct Tokenizer: Sendable {
        /// SentencePiece's word-start marker: "a space goes here".
        private static let wordStartMarkerBytes: [UInt8] = Array("▁".utf8)
        /// What Moonshine's trim() removes.
        private static let trimmedBytes: Set<UInt8> = [UInt8(ascii: " "), UInt8(ascii: "\t")]
        private static let invalidByteReplacement = UInt8(ascii: "?")

        /// Each token's bytes, indexed by token id. Ids 3-258 are the 256 single bytes: a character the vocabulary
        /// lacks is spelled as its UTF-8 bytes, one token each. That's why text is decoded from all tokens' bytes at
        /// once rather than token by token.
        private let bytesByTokenID: [[UInt8]]

        /// Reads Moonshine's `tokenizer.bin`.
        public init(contentsOf tokenizerFileURL: URL) throws {
            try self.init(tokenizerFileBytes: [UInt8](Data(contentsOf: tokenizerFileURL)))
        }

        /// `tokenizer.bin`'s format: one entry per token id, in id order. Each entry is a length, then that many
        /// bytes. A length under 128 is one byte; otherwise it's two, and length = second × 128 + first − 128.
        /// A length of 0 is an empty entry.
        public init(tokenizerFileBytes bytes: [UInt8]) throws {
            var entries: [[UInt8]] = []
            var offset = 0
            while offset < bytes.count {
                let firstLengthByte = Int(bytes[offset])
                offset += 1
                var entryByteCount = firstLengthByte
                if firstLengthByte >= 128 {
                    guard offset < bytes.count else {
                        throw Error("tokenizer entry \(entries.count)'s length runs past the end of the file")
                    }
                    entryByteCount = Int(bytes[offset]) * 128 + firstLengthByte - 128
                    offset += 1
                }
                guard offset + entryByteCount <= bytes.count else {
                    throw Error("tokenizer entry \(entries.count) runs past the end of the file")
                }
                entries.append(Array(bytes[offset ..< offset + entryByteCount]))
                offset += entryByteCount
            }
            guard entries.count == TinyModel.Tokens.vocabularySize else {
                throw Error("the tokenizer file has \(entries.count) entries, expected \(TinyModel.Tokens.vocabularySize)")
            }
            bytesByTokenID = entries
        }

        /// The text for these tokens. Special tokens (`<unk>`, `<s>`, `</s>`, `<<ST_0>>` …) contribute nothing.
        /// A token whose entry is empty is an error, as it is in Moonshine ("Invalid token").
        public func text(forTokenIDs tokenIDs: [Int]) throws -> String {
            var textBytes: [UInt8] = []
            for tokenID in tokenIDs {
                guard bytesByTokenID.indices.contains(tokenID) else {
                    throw Error("token \(tokenID) is outside the vocabulary")
                }
                let tokenBytes = bytesByTokenID[tokenID]
                guard !tokenBytes.isEmpty else {
                    throw Error("invalid token \(tokenID): its tokenizer entry is empty")
                }
                if !Self.isSpecialToken(tokenBytes) {
                    textBytes += tokenBytes
                }
            }
            return Self.sanitizedText(Self.trimmed(Self.replacingWordStartMarkersWithSpaces(textBytes)))
        }

        /// Moonshine's rule: more than 2 bytes, starting with < and ending with >.
        private static func isSpecialToken(_ tokenBytes: [UInt8]) -> Bool {
            tokenBytes.count > 2 && tokenBytes.first == UInt8(ascii: "<") && tokenBytes.last == UInt8(ascii: ">")
        }

        /// Left to right; matches don't overlap.
        private static func replacingWordStartMarkersWithSpaces(_ bytes: [UInt8]) -> [UInt8] {
            var result: [UInt8] = []
            result.reserveCapacity(bytes.count)
            var index = 0
            while index < bytes.count {
                if bytes[index...].starts(with: wordStartMarkerBytes) {
                    result.append(UInt8(ascii: " "))
                    index += wordStartMarkerBytes.count
                } else {
                    result.append(bytes[index])
                    index += 1
                }
            }
            return result
        }

        private static func trimmed(_ bytes: [UInt8]) -> ArraySlice<UInt8> {
            guard let first = bytes.firstIndex(where: { !trimmedBytes.contains($0) }),
                  let last = bytes.lastIndex(where: { !trimmedBytes.contains($0) })
            else { return [] }
            return bytes[first ... last]
        }

        /// Moonshine's sanitize_text: a lead byte followed by the right number of continuation bytes (10xxxxxx) is
        /// kept whole; any other byte becomes "?" and reading resumes at the next byte. Like Moonshine, it checks only
        /// a sequence's shape. A well-shaped sequence that still isn't valid UTF-8 (overlong, a surrogate, past
        /// U+10FFFF) can't be held in a Swift String, so it becomes U+FFFD here where Moonshine would pass it through.
        private static func sanitizedText(_ bytes: ArraySlice<UInt8>) -> String {
            var result: [UInt8] = []
            result.reserveCapacity(bytes.count)
            var index = bytes.startIndex
            while index < bytes.endIndex {
                let leadByte = bytes[index]
                let sequenceLength: Int
                if leadByte < 0x80 {
                    sequenceLength = 1
                } else if (leadByte & 0xE0) == 0xC0 {
                    sequenceLength = 2
                } else if (leadByte & 0xF0) == 0xE0 {
                    sequenceLength = 3
                } else if (leadByte & 0xF8) == 0xF0 {
                    sequenceLength = 4
                } else {
                    sequenceLength = 0  // a continuation byte, or 0xF8-0xFF: never a lead byte
                }
                let isWholeSequence = sequenceLength > 0
                    && bytes.endIndex - index >= sequenceLength
                    && bytes[(index + 1) ..< (index + sequenceLength)].allSatisfy { ($0 & 0xC0) == 0x80 }
                if isWholeSequence {
                    result += bytes[index ..< index + sequenceLength]
                    index += sequenceLength
                } else {
                    result.append(invalidByteReplacement)
                    index += 1
                }
            }
            return String(decoding: result, as: UTF8.self)
        }
    }
}
