# Test fixtures

Reference data for checking the package against real models end to end. About 2 MB.

| File | What it is |
|---|---|
| `beckett.wav` | 10 s of speech, 16 kHz mono 16-bit. One utterance. |
| `two_cities_16k.wav` | 44 s of speech, 16 kHz mono 16-bit, the opening of *A Tale of Two Cities*. Four utterances. |
| `reference_transcripts.json` | What the models must produce for those recordings, and each token's text on its own. |

## `reference_transcripts.json`

- `models`: the Hugging Face repository and the release the expected tokens are for. A release with other weights can
  transcribe differently, so this data is only valid for that release. The tokens were produced with v1.0, whose
  weights v1.1 shares unchanged.
- `single_token_texts`: the tokenizer's text for each of the 32,768 token ids on its own; `null` where the tokenizer
  must fail (an empty entry).
- `recordings[]`: `audio_file`, `sample_count` and the SHA-256 of the samples as little-endian float32 (to confirm the
  audio was read identically), and `utterances[]`: each one's `start_sample` and `end_sample` within the recording, and
  the `final_token_ids` and `final_text` its final transcript must have.

The expected tokens come from the Python reference pipeline the Core ML conversion was validated with, run on the same
float16 models.

## Attribution

Both recordings are from the test assets of [Moonshine](https://github.com/moonshine-ai/moonshine-v2), Copyright (c)
2025 Moonshine AI, released under the MIT License, whose terms are the same as this package's [LICENSE](../../../LICENSE).
*A Tale of Two Cities* (Charles Dickens, 1859) is in the public domain.
