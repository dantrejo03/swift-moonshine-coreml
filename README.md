# swift-moonshine-coreml

Work in progress. This README is a placeholder and will be updated once the project contents are imported.

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
