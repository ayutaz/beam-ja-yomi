# Results

## Overall

| condition | n | exact match | mean CER | format issues | truncated | mean reasoning tokens |
|---|---|---|---|---|---|---|
| beam_direct-high-en-r0 | 300 | 84.0% | 0.018 | 0 | 0 | 592 |
| beam_direct-low-en-r0 | 300 | 82.3% | 0.021 | 0 | 0 | 289 |
| beam_direct-low-en-r1 | 300 | 82.3% | 0.023 | 0 | 0 | 297 |
| beam_direct-low-en-r2 | 300 | 84.3% | 0.020 | 0 | 0 | 300 |
| beam_direct-low-ja-r0 | 300 | 80.7% | 0.023 | 0 | 0 | 293 |
| beam_direct-medium-en-r0 | 300 | 85.3% | 0.016 | 0 | 0 | 504 |
| beam_fix-high-en-r0 | 300 | 87.3% | 0.044 | 2 | 0 | 852 |
| beam_fix-low-en-r0 | 300 | 86.0% | 0.034 | 1 | 0 | 339 |
| beam_fix-low-en-r1 | 300 | 83.0% | 0.054 | 1 | 1 | 310 |
| beam_fix-low-en-r2 | 300 | 85.7% | 0.023 | 0 | 0 | 345 |
| beam_fix-medium-en-r0 | 300 | 86.7% | 0.016 | 0 | 0 | 657 |
| ojt | 300 | 74.7% | 0.042 | 1 | 0 |  |
| ojt_vanilla | 300 | 73.7% | 0.043 | 1 | 0 |  |

## By category (exact match)

| category | beam_direct-high-en-r0 | beam_direct-low-en-r0 | beam_direct-low-en-r1 | beam_direct-low-en-r2 | beam_direct-low-ja-r0 | beam_direct-medium-en-r0 | beam_fix-high-en-r0 | beam_fix-low-en-r0 | beam_fix-low-en-r1 | beam_fix-low-en-r2 | beam_fix-medium-en-r0 | ojt | ojt_vanilla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| disaster | 85.0% | 80.0% | 75.0% | 82.5% | 80.0% | 85.0% | 90.0% | 92.5% | 87.5% | 95.0% | 90.0% | 70.0% | 67.5% |
| government | 93.3% | 96.7% | 100.0% | 100.0% | 93.3% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| homograph | 88.3% | 85.0% | 85.0% | 83.3% | 81.7% | 90.0% | 86.7% | 88.3% | 83.3% | 80.0% | 86.7% | 63.3% | 61.7% |
| medical | 96.7% | 90.0% | 96.7% | 100.0% | 96.7% | 90.0% | 100.0% | 93.3% | 93.3% | 96.7% | 100.0% | 96.7% | 96.7% |
| number | 85.0% | 80.0% | 80.0% | 83.3% | 85.0% | 88.3% | 91.7% | 91.7% | 88.3% | 93.3% | 93.3% | 85.0% | 83.3% |
| person | 100.0% | 96.7% | 100.0% | 93.3% | 90.0% | 100.0% | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 100.0% |
| place | 54.0% | 62.0% | 58.0% | 64.0% | 52.0% | 56.0% | 58.0% | 50.0% | 48.0% | 54.0% | 52.0% | 36.0% | 36.0% |

## Beam fix vs OpenJTalk

fixed = OpenJTalk wrong → Beam right; broken = OpenJTalk right → Beam wrong.

| condition | fixed | broken | kept correct | still wrong |
|---|---|---|---|---|
| beam_fix-high-en-r0 | 48 | 10 | 214 | 28 |
| beam_fix-low-en-r0 | 45 | 11 | 213 | 31 |
| beam_fix-low-en-r1 | 38 | 13 | 211 | 38 |
| beam_fix-low-en-r2 | 39 | 6 | 218 | 37 |
| beam_fix-medium-en-r0 | 46 | 10 | 214 | 30 |

## Run-to-run spread (exact match)

| condition | runs | min | max |
|---|---|---|---|
| beam_direct-low-en | 3 | 82.3% | 84.3% |
| beam_fix-low-en | 3 | 83.0% | 86.0% |
