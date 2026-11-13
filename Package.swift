// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MoonshineCoreML",
    // MLState (the decoder's self-attention cache) needs iOS 18, watchOS 11 and macOS 15.
    platforms: [.iOS(.v18), .macOS(.v15), .watchOS(.v11)],
    products: [
        .library(name: "MoonshineCoreML", targets: ["MoonshineCoreML"]),
    ],
    targets: [
        .target(name: "MoonshineCoreML"),
        .testTarget(
            name: "MoonshineCoreMLTests",
            dependencies: ["MoonshineCoreML"],
            // Recordings and expected transcripts for end-to-end checks against real models.
            resources: [.copy("Fixtures")]
        ),
    ]
)
