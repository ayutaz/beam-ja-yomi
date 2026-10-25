---
title: Beam Japanese Reading Fixer
emoji: 🗣️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
license: mit
short_description: Can Beam fix Japanese TTS misreadings from OpenJTalk?
---

# Beam Japanese Reading Fixer

Japanese text-to-speech converts text into a katakana reading with a dictionary-based converter
(OpenJTalk) before a voice model speaks it. The converter misreads numbers and units, words whose
reading depends on context, and proper nouns. This Space asks Reflection AI's **Beam** to correct
the converter's reading.

- **Results**: exact-match rates on 300 Japanese sentences with human-approved readings
  ([dataset](https://huggingface.co/datasets/ayousanz/beam-ja-yomi-bench)), by category and by
  reasoning effort, plus how many readings Beam fixed and how many correct ones it broke
- **Examples**: the correct reading, OpenJTalk's reading and Beam's corrected reading side by side,
  with differences highlighted
- **Try it**: type a sentence and compare OpenJTalk's reading with Beam's correction

Live correction uses the Reflection API (secret `REFLECTION_API_KEY`). Without it, the Space shows
OpenJTalk's reading only.

**How Beam is used.** Beam gets the sentence and OpenJTalk's reading and returns the corrected
reading (`reasoning_effort` low; the prompts are in the code repository). The draft helps for
numbers and homographs and hurts for place names, where Beam tends to keep OpenJTalk's マチ for 町.
On the 300 sentences, asking Beam a second time without the draft and keeping the reading only when
both answers agree left three quarters of the sentences 97% right (a post-hoc check; see the dataset card). The
follow-up, [Yomi-Distill](https://huggingface.co/ayousanz/yomi-distill-ja), confirmed it on 99 new sentences (97.0%), and tried to
distill Beam's fixes into a 1.8B model: it did not beat OpenJTalk.
