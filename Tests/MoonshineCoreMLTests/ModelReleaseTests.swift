import Testing
@testable import MoonshineCoreML

struct ModelReleaseTests {
    static func versions(_ version: String?) -> [(name: ModelName, version: String?)] {
        ModelName.allCases.map { (name: $0, version: version) }
    }

    @Test func oneCompatibleReleaseIsCompatible() throws {
        let compatibleVersion = try #require(ModelRelease.compatibleVersions.first)
        let release = ModelRelease.assess(Self.versions(compatibleVersion))
        #expect(release.isCompatible)
        #expect(release.summary.contains(compatibleVersion))
    }

    @Test func anUnlistedReleaseIsNamedWithTheCompatibleOnes() throws {
        let compatibleVersion = try #require(ModelRelease.compatibleVersions.first)
        let release = ModelRelease.assess(Self.versions("v99.0"))
        #expect(!release.isCompatible)
        #expect(release.summary.contains("v99.0"))
        #expect(release.summary.contains(compatibleVersion))
    }

    @Test func modelsWithoutAVersionAreIncompatible() {
        let release = ModelRelease.assess(Self.versions(nil))
        #expect(!release.isCompatible)
        #expect(release.summary.contains("no version"))
    }

    @Test func mixedReleasesAreIncompatibleEvenIfEachIsListed() throws {
        let compatibleVersion = try #require(ModelRelease.compatibleVersions.first)
        var versionByName = Self.versions(compatibleVersion)
        versionByName[versionByName.count - 1].version = nil
        let release = ModelRelease.assess(versionByName)
        #expect(!release.isCompatible)
        #expect(release.summary.contains("different releases"))
        #expect(release.summary.contains("decoder_step without a version"))
    }
}
