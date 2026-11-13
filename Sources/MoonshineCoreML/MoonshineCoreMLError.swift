/// Every failure in this package, explained in words. The package fails fast: a wrong model, shape or token stops
/// with a description of what was expected instead of producing a wrong transcript.
public struct MoonshineCoreMLError: Error, CustomStringConvertible {
    public let description: String

    init(_ description: String) {
        self.description = description
    }
}