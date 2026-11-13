import Testing
@testable import MoonshineCoreML

/// The tests build a synthetic tokenizer.bin: a full-size vocabulary of placeholder entries, with the ids under test
/// set to the bytes each test needs. No real tokenizer file is required.
struct TokenizerTests {
    static let wordStartMarker = "▁"

    /// Token ids used by the tests, chosen away from the real special and byte-fallback ranges.
    enum TestTokenID {
        static let hello = 1000
        static let world = 1001
        static let comma = 1002
        static let special = 1003
        static let empty = 1004
        static let lone0xE2 = 1005
        static let lone0x82 = 1006
        static let lone0xAC = 1007
        static let long = 1008
    }

    static let longEntry = String(repeating: "a", count: 300)

    static let entriesByTokenID: [Int: [UInt8]] = [
        TestTokenID.hello: Array("\(wordStartMarker)Hello".utf8),
        TestTokenID.world: Array("\(wordStartMarker)world".utf8),
        TestTokenID.comma: Array(",".utf8),
        TestTokenID.special: Array("<<ST_0>>".utf8),
        TestTokenID.empty: [],
        TestTokenID.lone0xE2: [0xE2],
        TestTokenID.lone0x82: [0x82],
        TestTokenID.lone0xAC: [0xAC],
        TestTokenID.long: Array(longEntry.utf8),
    ]

    static func tokenizerFileBytes(entryCount: Int = TinyModel.Tokens.vocabularySize) -> [UInt8] {
        var fileBytes: [UInt8] = []
        for tokenID in 0 ..< entryCount {
            let entry = entriesByTokenID[tokenID] ?? Array("x".utf8)
            if entry.count < 128 {
                fileBytes.append(UInt8(entry.count))
            } else {
                fileBytes += [UInt8(entry.count % 128 + 128), UInt8(entry.count / 128)]
            }
            fileBytes += entry
        }
        return fileBytes
    }

    let tokenizer: Moonshine.Tokenizer

    init() throws {
        tokenizer = try Moonshine.Tokenizer(tokenizerFileBytes: Self.tokenizerFileBytes())
    }

    @Test func wordStartMarkersBecomeSpacesAndEndsAreTrimmed() throws {
        let text = try tokenizer.text(forTokenIDs: [TestTokenID.hello, TestTokenID.comma, TestTokenID.world])
        #expect(text == "Hello, world")
    }

    @Test func specialTokensContributeNothing() throws {
        let text = try tokenizer.text(forTokenIDs: [TestTokenID.special, TestTokenID.hello, TestTokenID.special])
        #expect(text == "Hello")
    }

    @Test func noTokensMakeEmptyText() throws {
        #expect(try tokenizer.text(forTokenIDs: []) == "")
    }

    @Test func byteFallbackTokensJoinIntoOneCharacter() throws {
        // "€" is E2 82 AC: three single-byte tokens that only make a character together.
        let text = try tokenizer.text(forTokenIDs: [TestTokenID.lone0xE2, TestTokenID.lone0x82, TestTokenID.lone0xAC])
        #expect(text == "€")
    }

    @Test func incompleteUTF8SequenceBecomesQuestionMarks() throws {
        // E2 82 without its third byte: neither byte starts a whole sequence.
        let text = try tokenizer.text(forTokenIDs: [TestTokenID.hello, TestTokenID.lone0xE2, TestTokenID.lone0x82])
        #expect(text == "Hello??")
    }

    @Test func entryLongerThan127BytesUsesTwoLengthBytes() throws {
        #expect(try tokenizer.text(forTokenIDs: [TestTokenID.long]) == Self.longEntry)
    }

    @Test func emptyEntryIsAnError() {
        #expect(throws: Moonshine.Error.self) { try tokenizer.text(forTokenIDs: [TestTokenID.empty]) }
    }

    @Test func tokenOutsideTheVocabularyIsAnError() {
        #expect(throws: Moonshine.Error.self) {
            try tokenizer.text(forTokenIDs: [TinyModel.Tokens.vocabularySize])
        }
        #expect(throws: Moonshine.Error.self) { try tokenizer.text(forTokenIDs: [-1]) }
    }

    @Test func wrongEntryCountIsAnError() {
        #expect(throws: Moonshine.Error.self) {
            try Moonshine.Tokenizer(tokenizerFileBytes: Self.tokenizerFileBytes(entryCount: 100))
        }
    }

    @Test func truncatedFileIsAnError() {
        let fileBytes = Self.tokenizerFileBytes()
        #expect(throws: Moonshine.Error.self) {
            try Moonshine.Tokenizer(tokenizerFileBytes: Array(fileBytes.dropLast()))
        }
    }
}
