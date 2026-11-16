import Foundation

/// Places in this repo, found from this file's location: the tests always run from a checkout of the repo.
enum RepositoryPaths {
    static let root = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let downloadScript = root.appending(path: "scripts/download_models.sh")
    /// Where scripts/download_models.sh puts the models.
    static let downloadedModels = root.appending(path: "Models")
    static let fixtures = root.appending(path: "Tests/MoonshineCoreMLTests/Fixtures")
    static let referenceTranscripts = fixtures.appending(path: "reference_transcripts.json")
    /// The library's resources: beckett.wav lives there, since the on-device check ships it.
    static let libraryResources = root.appending(path: "Sources/MoonshineCoreML/Resources")

    /// A recording named in reference_transcripts.json: a fixture, or the library's own recording.
    static func recording(named fileName: String) -> URL {
        let fixture = fixtures.appending(path: fileName)
        return FileManager.default.fileExists(atPath: fixture.path) ? fixture : libraryResources.appending(path: fileName)
    }
}
