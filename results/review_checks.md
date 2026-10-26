# Review-response experiments

## 1. Capacity match

| structure | parameters | runs | level-4 IoU |
|---|---:|---:|---:|
| concat, base 16 (reported) | 117,937 | 4 | 0.6339 |
| concat, base 24 (matched) | 264,073 | 2 | 0.6284 |
| gate, base 16 (reported) | 262,866 | 4 | 0.6360 |

The gate inside the base-24 runs reaches 0.6360 at 2 runs; its width is larger there, so the matched row is the concatenated one.

## 2. Boundary gap

- levels 0-2: reported split 0.5466 (n=3), gap of four frames 0.5382 (n=1), change -0.0083
- levels 0-4: reported split 0.6339 (n=4), gap of four frames 0.6394 (n=1), change +0.0056

## 3. Severe level only (no diversity, same sample budget)

| training set | level-4 IoU (concat) |
|---|---:|
| levels 0-2 (reported) | 0.5466 |
| level 4 only | 0.6460 |
| levels 0-4 (reported) | 0.6339 |

If s=4 alone lands near the 0-4 row, severity exposure is what moved the number
and diversity is not needed to explain it; if it lands at or below the 0-2 row,
the worst level alone is not enough and the whole ladder matters.

## 4. Detection, mAP at s=4

| trained on | structure | runs | mAP@0.5 | sd |
|---|---|---:|---:|---:|
| 0,1,2 | concat | 3 | 0.0620 | 0.0270 |
| 0,1,2 | gate | 3 | 0.0727 | 0.0062 |
| 0,1,2,3,4 | concat | 3 | 0.1409 | 0.0121 |
| 0,1,2,3,4 | gate | 3 | 0.1441 | 0.0110 |

- 0,1,2: gate minus concat +0.0108 (paired, n=3, range -0.0175..+0.0468)
- 0,1,2,3,4: gate minus concat +0.0032 (paired, n=3, range +0.0024..+0.0045)
