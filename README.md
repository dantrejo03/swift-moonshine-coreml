# swift-moonshine-coreml

Streaming speech-to-text on Apple devices, with no network: [Moonshine Streaming
Tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) (English) converted to Core ML, and the Swift code
that runs it. It was built to show the words on an Apple Watch while the person is still talking, and builds for iOS and
macOS too. It's tested on an Apple Watch Series 9 and an Apple silicon Mac.

Audio goes in as it arrives. Every 0.5 s of audio (you can change this), you get a tentative transcript of the utterance
so far, and when the utterance ends, its final transcript.

## Requirements

- iOS 18, macOS 15 or watchOS 11, for `MLState`, which holds the decoder's self-attention cache.
- On a watch, a Series 9 or later. A Series 8's 2-core Neural Engine rejects the models, and Core ML's CPU fallback is
  about 4 times slower than real time there.
- On the Mac, Apple silicon. The package uses Swift's `Float16`, which doesn't exist on Intel Macs, so building for
  Intel macOS stops with an error that says so. A universal macOS app has to remove x86_64 from its Architectures.
  iPhones, every Apple Watch (arm64 and arm64_32) and the iOS Simulator, even on an Intel Mac, aren't affected.

## Installation

In Xcode, choose File > Add Package Dependencies and enter `https://github.com/dantrejo03/swift-moonshine-coreml.git`.
In a `Package.swift`:

```swift
.package(url: "https://github.com/dantrejo03/swift-moonshine-coreml.git", from: "0.1.0"),
```

and add the `MoonshineCoreML` product to your target. The models are a separate download, below.

## Models

The Core ML models and the tokenizer aren't in this repo or in the Swift package. They're released on Hugging Face as
[dantrejo03/moonshine-streaming-tiny-coreml](https://huggingface.co/dantrejo03/moonshine-streaming-tiny-coreml), and
`scripts/download_models.sh` downloads the release this version of the package works with:

```sh
python3 -m venv .venv && .venv/bin/pip install huggingface_hub   # once, for the Hugging Face CLI
scripts/download_models.sh
```

That puts the five `.mlpackage` folders and `tokenizer.bin` in `Models/` at the top of the repo (gitignored).
**Keep them there:** code in this repo that needs the models looks in `Models/` and nowhere else. The script pins
one release, by tag and commit. Run it again whenever a change to it comes in: it replaces `Models/` with that release.
Hugging Face caches what it downloads, so a run that changes nothing is instant. (Instead of the virtual environment,
`brew install huggingface-cli` works too.)

Each model carries its release's tag as its Core ML version, and the transcriber checks at load that all five are one
release this version of the package works with (the list is `ModelRelease.compatibleVersions`). If they aren't,
loading fails with an error that says which release it found. To try other models anyway, such as a new export, pass
`checksModelVersions: false` to `Moonshine.Transcriber`. The models then load as long as their inputs, outputs and
state match what the code expects, and a warning is logged every time. Models from a release the package doesn't list
can transcribe wrongly without any error.

To use the models in an app, download them into a folder of the app's own:

```sh
scripts/download_models.sh ~/Developer/MyApp/Models
```

Add the five `.mlpackage` folders and `tokenizer.bin` to the app's target, and pass `Bundle.main.resourceURL` to
`Moonshine.Transcriber(modelsDirectory:)`. Xcode compiles each `.mlpackage` into the `.mlmodelc` the transcriber loads;
a watch can't compile models itself. The script replaces the folder it's given, so don't point it at a folder that holds
anything else.

## Usage

```swift
import MoonshineCoreML

// Load once, for example at launch. The first load on a device takes a few seconds while Core ML prepares the models.
let transcriber = try await Moonshine.Transcriber(modelsDirectory: Bundle.main.resourceURL!)

try await transcriber.beginUtterance()
for await samples in microphoneSamples {  // 16 kHz mono Floats in [-1, 1], in blocks of any size
    for update in try await transcriber.receive(samples) {
        show(update.text)  // tentative
    }
    if await transcriber.hasReachedMaximumUtteranceLength { break }
}
let finalTranscript = try await transcriber.endUtterance()
show(finalTranscript.text)
```

The package doesn't record audio. Capture it with `AVAudioEngine` and convert it to 16 kHz mono `Float` with
`AVAudioConverter`.

`Moonshine.Transcriber` is an actor. Its Core ML work runs on its own serial queue, never on the caller's thread, so
calling it from the main actor doesn't block the UI. One transcriber handles one utterance at a time, and it can be
reused for as many as you like.

- **Utterance length:** at most 15.36 s (`Moonshine.Transcriber.maximumUtteranceDuration`). After that, audio is
  dropped and `hasReachedMaximumUtteranceLength` is true; end the utterance and begin another.
- **Falling behind:** each update decodes the whole utterance so far, so updates take longer as it grows. If audio
  arrives faster than updates finish, pass `runningDueUpdates: false` to `receive` to skip them. The audio still goes
  in, and the final transcript covers all of it. Cancelling the task that called `receive` also stops an update
  partway through.
- **Each update** (`Moonshine.TranscriptUpdate`) has its `text`, `tokenIDs`, `heardSeconds` and `decodeDuration`.

## Checking the device

Core ML can compute wrong numbers on a particular device without any error: on an Apple Watch Series 9, the
Neural Engine turned a test recording into nothing until the frontend moved to the CPU. The checks at load can't see
that; only running the models can. So before trusting transcripts on a device, run the package's check once:

```swift
let result = try await transcriber.checkOnThisDevice()  // throws, naming the first wrong token, if it fails
print("models \(result.modelVersion): \(result.realTimeFactor)x real time")
```

It transcribes a 10 s recording that ships with the package (312 KB) and compares the final tokens with the
reference ones for the loaded model release. It takes about 4 s on an Apple Watch Series 9. Remember a pass for the
device model, OS version and model release (`transcriber.modelVersion`), and run it again when any of them changes:
an OS update can change how Core ML runs a model.

## Checking your models

The tests in this repo check the downloaded models end to end. From a clone, with the models in `Models/`:

```sh
swift test
```

They check that the models are a release this version of the package works with, and that their inputs and outputs
are what the code expects. They also check that the tokenizer gives the reference text for all 32,768 tokens, and that
5 utterances from 2 recordings transcribe to exactly the reference tokens. That takes about 30 s on an Apple silicon Mac.
Without `Models/`, the tests that need the models are skipped and the rest still run.

## Versions

Each package version works with particular model releases, and its `scripts/download_models.sh` downloads one of them:

| Package | Model release |
|---|---|
| 0.1.0 | v1.1 |
| 0.2.0 | v1.1 |

## Performance

From an Apple Watch Series 9:

- Median decoder step: 12.4 ms.
- From the end of an utterance to the final transcript: 424 ms for 4.8 s of speech, 629 ms for 8.4 s, and about
  1.26 s for 14–15 s.

## Limitations

- English only, like the original model. It can make up or repeat words, especially on short or noisy audio.
- Utterances of at most 15.36 s. Split longer audio into several.
- Decoding is greedy: the top-scoring token at each step.
- Word error rate wasn't measured for this conversion. The original model averages 12.01% on the Open ASR Leaderboard
  ([its card](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)).

## Glossary

These terms mean one thing each, in code, comments and docs.

| Term | Means |
|---|---|
| **sample** | One 16 kHz mono audio value, a `Float` in [-1, 1]. |
| **chunk** | 1280 samples (80 ms): what the frontend model takes in one call. |
| **feature frame** | One row of frontend output, covering 20 ms of audio. A chunk makes 4. |
| **encoded frame** | One row of encoder output. Each feature frame becomes one encoded frame. |
| **memory frame** | One row of adapter output: what the decoder attends to. Each encoded frame becomes one memory frame. |
| **token** | One id the decoder picks; the tokenizer turns tokens into text. |
| **utterance** | One span of audio from begin to end, at most 15.36 s (768 memory frames). |
| **update** | A tentative decode over the utterance so far, shown while audio is still arriving. |
| **final transcript** | The decode at the end of an utterance, over all of its audio. |

## License

MIT; see [LICENSE](LICENSE). The model weights and `tokenizer.bin` are Moonshine AI's, also under the MIT License
([moonshine-ai/moonshine-v2](https://github.com/moonshine-ai/moonshine-v2)). This repo only converts and runs them.
