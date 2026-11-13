extension Moonshine {
    /// One decode from scratch over the utterance so far: what the screen shows. Tentative while audio is still
    /// arriving; the final transcript when the utterance ends.
    public struct TranscriptUpdate: Sendable {
        /// The decoder's picks, in order. The last is the end token unless the token limit stopped the decode first.
        public let tokenIDs: [Int]
        public let text: String
        /// The most tokens this decode was allowed: 6.5 per second heard, rounded up, at most 128.
        public let maximumTokenCount: Int
        /// Every sample of the utterance so far, not counting the zeros that pad the last chunk.
        public let heardSampleCount: Int
        public let decodeDuration: Duration

        public var endedWithEndToken: Bool { tokenIDs.last == TinyModel.Tokens.endTokenID }
        public var heardSeconds: Double { Double(heardSampleCount) / Double(TinyModel.Audio.sampleRateHz) }
    }
}
