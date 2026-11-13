import CoreML

/// Which Hugging Face releases of the models this version of the package works with. Each model carries its release's
/// tag as its Core ML version (metadata.versionString), which stays with it when Xcode compiles it into an app, so the
/// check is on the files actually loaded.
enum ModelRelease {
    /// Releases this code was checked against, spelled exactly as their tags. A release that changes the models'
    /// inputs, outputs or state needs a new package version. One that only changes weights or what the inputs mean
    /// still loads, which is why releases are listed here explicitly.
    static let compatibleVersions: Set<String> = ["v1.1"]

    /// The model's release tag, or nil if it has none (v1.0 doesn't: releases have carried one since v1.1).
    static func version(of model: MLModel) -> String? {
        guard let version = model.modelDescription.metadata[.versionString] as? String, !version.isEmpty else {
            return nil
        }
        return version
    }

    /// Whether the five models are one release on compatibleVersions, and a summary of what they are, for the error
    /// when they aren't, or for the warning when the check is off.
    static func assess(
        _ versionByName: [(name: ModelName, version: String?)]
    ) -> (isCompatible: Bool, summary: String) {
        let worksWith = "This version of MoonshineCoreML works with "
            + compatibleVersions.sorted().joined(separator: ", ") + "."
        let versions = versionByName.map(\.version)
        guard Set(versions).count == 1 else {
            let versionList = versionByName
                .map { "\($0.name.rawValue) \($0.version ?? "without a version")" }
                .joined(separator: ", ")
            return (false, "The models come from different releases: \(versionList). \(worksWith)")
        }
        guard let version = versions.first ?? nil else {
            return (false, "The models have no version, so they're v1.0 (releases have carried one since v1.1) or "
                + "not from a release. \(worksWith)")
        }
        guard compatibleVersions.contains(version) else {
            return (false, "The models are \(version). \(worksWith)")
        }
        return (true, "The models are \(version), which this version of MoonshineCoreML works with.")
    }
}
