import Testing
@testable import MoonshineCoreML

struct StreamingEngineTests {
    @Test(arguments: [
        (heardSampleCount: 0, expectedMaximumTokenCount: 0),
        (heardSampleCount: 1, expectedMaximumTokenCount: 1),  // any audio allows at least one token
        (heardSampleCount: 16_000, expectedMaximumTokenCount: 7),  // 1 s × 6.5, rounded up
        (heardSampleCount: 32_000, expectedMaximumTokenCount: 13),  // 2 s × 6.5 exactly
        (heardSampleCount: TinyModel.Utterance.maximumSampleCount, expectedMaximumTokenCount: 100),  // 15.36 s
        (heardSampleCount: 16_000 * 60, expectedMaximumTokenCount: 128),  // capped by the self-attention cache
    ])
    func maximumTokenCountIsSixAndAHalfPerSecondCappedAt128(heardSampleCount: Int, expectedMaximumTokenCount: Int) {
        #expect(StreamingEngine.maximumTokenCount(heardSampleCount: heardSampleCount) == expectedMaximumTokenCount)
    }
}
