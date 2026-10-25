---
title: Yomi-Distill
emoji: 🗣️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.26.0
python_version: "3.12"
app_file: app.py
license: mit
short_description: A small model trained on Beam's Japanese reading fixes
---

# Yomi-Distill

Japanese text-to-speech converts text into a katakana reading with a dictionary-based converter
(OpenJTalk) before a voice model speaks it. Reflection AI's **Beam** corrects many of the
converter's misreadings ([Part 1](https://huggingface.co/datasets/ayousanz/beam-ja-yomi-bench)),
but it is far too large to run inside a speech pipeline. This Space runs a 1.8B model
([ayousanz/yomi-distill-ja](https://huggingface.co/ayousanz/yomi-distill-ja)) trained on Beam's
corrections of Japanese Wikipedia sentences
([teaching data](https://huggingface.co/datasets/ayousanz/beam-yomi-teacher)).

**In the pre-registered test it did not beat OpenJTalk** (74.7% on both, on Part 1's 300 sentences;
Beam correcting OpenJTalk: 86.0%). It mostly keeps OpenJTalk's reading, reads 町 in place names as チョウ,
and sometimes breaks a name.

- **Try it**: type a sentence and compare OpenJTalk's reading with the student's, and listen to both
- **Results**: the student, OpenJTalk and Beam on Part 1's 300 human-approved sentences and on 99 new
  sentences whose readings Claude wrote (not checked by a person)

Beam is the teacher only. This Space never calls Beam and needs no API key; the student runs on
ZeroGPU.
