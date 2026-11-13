# swift-moonshine-coreml

Work in progress. This README is a placeholder and will be updated once the project contents are imported.

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
