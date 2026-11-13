import Dispatch
import Foundation

extension Moonshine {
    /// Streaming speech-to-text for one utterance at a time: audio in, as the microphone delivers it, tentative
    /// transcripts while it arrives, and a final transcript when the utterance ends.
    ///
    /// An actor, so its work never runs on the caller's thread: a final decode can take over a second. Calls run one at
    /// a time on the transcriber's own serial queue, which keeps long Core ML predictions off Swift's shared
    /// concurrency threads.
    public actor Transcriber {
        /// What `receive(_:runningDueUpdates:)` expects: 16 kHz mono samples in [-1, 1].
        public static let sampleRateHz = TinyModel.Audio.sampleRateHz
        /// The longest utterance, 15.36 s. Audio past it is dropped: check `hasReachedMaximumUtteranceLength` and end
        /// the utterance.
        public static let maximumUtteranceDuration = Duration.milliseconds(
            TinyModel.Utterance.maximumSampleCount * 1000 / TinyModel.Audio.sampleRateHz)
        /// The file in the models directory that holds the tokenizer.
        public static let tokenizerFileName = "tokenizer.bin"

        private let executorQueue = DispatchSerialQueue(label: "Moonshine.Transcriber")
        public nonisolated var unownedExecutor: UnownedSerialExecutor { executorQueue.asUnownedSerialExecutor() }

        private let engine: StreamingEngine
        /// How much audio arrives between tentative updates. Counted in samples, not on a clock, so a replay of a
        /// recording and a live microphone produce the same updates.
        private let updateIntervalSampleCount: Int
        /// Arrived but not yet a whole chunk.
        private var pendingSamples: [Float] = []
        private var nextUpdateSampleCount = 0

        /// Loads and checks the five compiled models (frontend.mlmodelc … decoder_step.mlmodelc) and tokenizer.bin, all
        /// from modelsDirectory. In an app that bundles them, that's `Bundle.main.resourceURL`. The first load on a
        /// device takes seconds while Core ML prepares the models for its chips; later loads are cached.
        ///
        /// updateInterval: how much audio arrives between tentative updates.
        public init(modelsDirectory: URL, updateInterval: Duration = .milliseconds(500)) async throws {
            let updateIntervalSeconds = Double(updateInterval.components.seconds)
                + Double(updateInterval.components.attoseconds) / 1e18
            let updateIntervalSampleCount = Int((updateIntervalSeconds * Double(Self.sampleRateHz)).rounded())
            guard updateIntervalSampleCount > 0 else {
                throw Error("the update interval must be at least one sample long, got \(updateInterval)")
            }
            let tokenizer = try Tokenizer(contentsOf: modelsDirectory.appending(path: Self.tokenizerFileName))
            let models = try LoadedModels(compiledModelsDirectory: modelsDirectory)
            self.engine = try StreamingEngine(models: models, tokenizer: tokenizer)
            self.updateIntervalSampleCount = updateIntervalSampleCount
        }

        /// Every memory frame is used and further audio is dropped: end the utterance.
        public var hasReachedMaximumUtteranceLength: Bool { engine.hasReachedMaximumUtteranceLength }

        /// Starts a new utterance. Anything left of a previous one that wasn't ended is discarded.
        public func beginUtterance() throws {
            try engine.beginUtterance()
            pendingSamples.removeAll(keepingCapacity: true)
            nextUpdateSampleCount = updateIntervalSampleCount
        }

        /// Samples as they arrive, any number at a time. Returns the tentative updates that fell due, in order: usually
        /// none or one, more only if several update intervals of audio arrive at once. An update sees only the chunks
        /// that had fully arrived when it fell due: with a 0.5 s interval, the first update sees 6 chunks (0.48 s).
        ///
        /// With runningDueUpdates false, updates that fall due are skipped, for example when the caller has fallen
        /// behind the microphone. If the calling task is cancelled, an update in progress is abandoned at its next
        /// decoder step and the rest are skipped. Either way the audio still goes in, and the final transcript, which
        /// decodes over all of it, is unaffected.
        public func receive(_ samples: [Float], runningDueUpdates: Bool = true) throws -> [TranscriptUpdate] {
            guard engine.isUtteranceInProgress else {
                throw Error("no utterance in progress: call beginUtterance() first")
            }
            guard !hasReachedMaximumUtteranceLength else { return [] }
            pendingSamples += samples
            var tentativeUpdates: [TranscriptUpdate] = []
            let samplesPerChunk = TinyModel.Audio.samplesPerChunk
            while pendingSamples.count >= samplesPerChunk, !hasReachedMaximumUtteranceLength {
                let sampleCountAfterChunk = engine.heardSampleCount + samplesPerChunk
                while nextUpdateSampleCount < sampleCountAfterChunk {
                    if runningDueUpdates {
                        do {
                            tentativeUpdates.append(try engine.update())
                        } catch is CancellationError {
                            // Abandoned: the caller no longer wants it.
                        }
                    }
                    nextUpdateSampleCount += updateIntervalSampleCount
                }
                try engine.addChunk(Array(pendingSamples[..<samplesPerChunk]))
                pendingSamples.removeFirst(samplesPerChunk)
            }
            return tentativeUpdates
        }

        /// Ends the utterance and returns its final transcript. The samples left over (less than a chunk) are padded
        /// and included, unless the utterance is already at its maximum length. An update that fell due within those
        /// last samples isn't run first: the final transcript replaces it immediately.
        public func endUtterance() throws -> TranscriptUpdate {
            let remainingSamples = hasReachedMaximumUtteranceLength ? [] : pendingSamples
            pendingSamples.removeAll(keepingCapacity: true)
            return try engine.endUtterance(remainingSamples: remainingSamples)
        }
    }
}
