import AVFoundation

/// Reads the package's test recordings: the on-device check's, and the tests' fixtures.
enum AudioFile {
    /// A 16 kHz mono file as float32 samples in [-1, 1], read with AVAudioFile. 16-bit samples become sample / 32768,
    /// as the reference pipeline read them, so the reference's SHA-256 can confirm both started from the same numbers.
    static func loadSamples(at audioURL: URL) throws -> [Float] {
        let fileName = audioURL.lastPathComponent
        let file = try AVAudioFile(forReading: audioURL)
        let format = file.processingFormat
        guard format.sampleRate == Double(TinyModel.Audio.sampleRateHz), format.channelCount == 1,
              format.commonFormat == .pcmFormatFloat32
        else {
            throw Moonshine.Error("\(fileName): expected 16 kHz mono, read as \(format)")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw Moonshine.Error("\(fileName): couldn't allocate \(file.length) frames")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0], Int64(buffer.frameLength) == file.length else {
            throw Moonshine.Error("\(fileName): read \(buffer.frameLength) of \(file.length) frames")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }
}
