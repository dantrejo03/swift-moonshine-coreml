import Foundation

/// The folder the transcriber loads from: the five compiled models and the tokenizer, side by side, as Xcode puts them
/// in an app bundle. They aren't part of this package; scripts/download_models.sh downloads them.
enum ModelsDirectory {
    /// Fails unless every file is there, naming all the missing ones and saying where they come from, so a wrong folder
    /// or a skipped download doesn't show up as one model's load error.
    static func verifyEveryFileIsPresent(in directory: URL) throws {
        let requiredFileNames = ModelName.allCases.map(\.compiledFileName) + [Moonshine.Transcriber.tokenizerFileName]
        let missingFileNames = requiredFileNames.filter { !fileExists(named: $0, in: directory) }
        guard !missingFileNames.isEmpty else { return }

        let uncompiledModelFileNames = ModelName.allCases.map(\.packageFileName)
            .filter { fileExists(named: $0, in: directory) }
        if !uncompiledModelFileNames.isEmpty {
            throw Moonshine.Error(
                "\(directory.path) is missing \(missingFileNames.joined(separator: ", ")). It has "
                    + "\(uncompiledModelFileNames.joined(separator: ", ")), but the transcriber loads compiled models "
                    + "(.mlmodelc). In an app, add the .mlpackage files to the target and Xcode compiles them into the "
                    + "bundle; on a Mac, compile each one with MLModel.compileModel(at:).")
        }
        throw Moonshine.Error(
            "\(directory.path) is missing \(missingFileNames.joined(separator: ", ")). Is it the right folder? The models "
                + "aren't part of this package: download them with scripts/download_models.sh (see the README's Models "
                + "section). In an app, add the five .mlpackage files and tokenizer.bin to the target and pass "
                + "Bundle.main.resourceURL.")
    }

    private static func fileExists(named fileName: String, in directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appending(path: fileName).path)
    }
}
