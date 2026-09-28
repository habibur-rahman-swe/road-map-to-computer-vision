# Paper 3 — Urgency from aligned text and face

Publish this paper third. Start its experiments only after Paper 1 (text LLM) and Paper 2 (face expression) are both public: a frozen PDF and a public code repository for each.

You may read this file earlier. Do not train the fusion model, pick fusion weights, or write results into this paper before those two releases. Changing Paper 1 or Paper 2 because fusion looked better is not allowed.

This file is only the guide for the combined paper.

## What this paper is

The question is: when a public example contains both a face and the words aligned with that face, does combining the text method from Paper 1 with the face method from Paper 2 reduce urgency error compared with either method alone?

Your reporting scale, same as the earlier papers:

| Score | Meaning |
| --- | --- |
| 1 | Highest urgency |
| 10 | Lowest urgency |

## What you may not combine

Paper 1’s posts and Paper 2’s faces are not pairs. A TREC-IS post does not come with the author’s facial expression. A FER-2013 face does not come with that person’s emergency text. TREC-IS also has event images; those are scenes from incidents, not expression portraits of the speaker.

Do not build a row by attaching a random face to a random post. A model can memorize that artificial pairing and the number will look real. It is not a study of a person, a message, or an event.

Use one public dataset in which each example already contains aligned language and a visible face or face track, under a license you have read.

## Dataset

Primary dataset: MELD (Multimodal EmotionLines Dataset).

- Site: https://affective-meld.github.io/
- Each example is a dialogue utterance with text, video of the speakers, and audio, labeled with an emotion: anger, disgust, sadness, joy, neutral, surprise, or fear.
- The video comes from a copyrighted television show. Use the files only as the MELD terms allow. Do not upload episodes to GitHub. Do not use the videos in a demo you do not have rights to show.
- Split by the dataset’s official train, development, and test dialogues. Also keep entire dialogues together. Do not put one utterance from a scene in train and the next utterance in test.

Backup dataset, if MELD’s terms block you: CMU-MOSEI, which has aligned language, visual features, and sentiment or emotion annotations. Read the CMU download terms the same way. Do not switch datasets after you have seen test numbers.

These datasets label emotion or sentiment, not your 1–10 urgency. Reuse the frozen face formula’s idea from Paper 2: publish a written map from this dataset’s labels to the 1–10 scale before test evaluation. A starting map, to freeze, not to quietly edit:

| MELD emotion | Urgency score |
| --- | --- |
| fear | 1 |
| anger | 3 |
| surprise | 4 |
| disgust | 5 |
| sadness | 6 |
| neutral | 8 |
| joy | 10 |

This map is author-defined. It is not the same object as TREC-IS priority and not the same object as FER+ vote weights. The paper has to say that in the first data paragraph. The useful claim is narrower: given this frozen map, does fusion beat text-only and face-only on paired examples?

Run a small human check, as in Paper 2, on 150 paired clips from the training or development split: two raters, scale 1–10, quadratic weighted kappa against this map. If agreement is poor, keep the fusion comparison but drop any sentence that calls the score real-world urgency.

## Methods to reuse, not to refit in secret

- Text branch: the same family as Paper 1. A fine-tuned small open encoder is the required text system. The small open LLM from Paper 1 is a second text system if you can run it on utterances. Record the checkpoint. You may train it on MELD text. You may not report Paper 1’s TREC-IS numbers as if they were MELD numbers.
- Face branch: the same family as Paper 2. Sample a face frame from the utterance video, or use an official visual feature if the release provides one and the paper says so. Train on MELD. Preprocessing must be written down, including what happens when a face is missing.
- Fusion, late, required: each branch outputs an urgency score in 1–10. Combine them with a rule fit only on the development split. Start with a weighted average. Weights are chosen on development mean absolute error, then frozen.
- Fusion, learned, optional: concatenate the two embeddings and train a small head. Same split rules.
- Missing face: the text-only score is the fallback. Count how often it happens. Do not drop those utterances from the test table without a row that shows the drop.

Audio can wait. This paper is the combination of the two papers you will already have published. Adding audio makes a fourth study.

## What the paper may claim

- On paired public examples, with a frozen label map, text-only versus face-only versus late fusion.
- Where the branches disagree, and which one is closer to the mapped label.
- The human-check agreement, including a weak agreement.

It may not claim a joint emergency-and-face detector, a deployment system, or that Paper 1 and Paper 2 tested this paired setting.

## Free tools

- The Python stack from Papers 1 and 2
- A video frame reader such as OpenCV or torchvision’s video utilities
- ffmpeg if you must extract frames locally
- CPU for frame samples and for the late-fusion rule
- Free GPU only if you fine-tune both branches on MELD
- No paid API and no paid labeling

## Learning path

### Phase A — Wait, then read the two papers

- [ ] Confirm Paper 1’s PDF and repository are public.
- [ ] Confirm Paper 2’s PDF and repository are public.
- [ ] Copy their scale, their “what we do not claim” sentences, and their model names into a one-page brief.
- [ ] Read one survey chapter or paper on early versus late multimodal fusion so the words in your method section match what you implement. The Baltrušaitis, Ahuja, and Morency survey on multimodal machine learning is a standard free starting point if you can obtain the PDF from the authors or from an open repository.

Check: you can explain late fusion without calling it early fusion.

### Phase B — Paired data only

- [ ] Read the MELD license and paper. Save a copy of the terms with your notes.
- [ ] Load the official utterance table. Print 20 rows with dialogue id, utterance id, text, emotion, and split.
- [ ] Extract one face frame for 20 training utterances. Record failures: no face, several faces, unreadable frame.
- [ ] Apply the frozen emotion-to-score map. Check 10 rows by hand.
- [ ] Prove the split has no dialogue id in more than one of train, development, and test.

Check: a rejected random-pair script is not in the repository. The loader errors if a row has text but you cannot state whether a face was found.

### Phase C — Single branches on this dataset

- [ ] Train the text encoder on MELD training text. Select on development mean absolute error of the mapped score.
- [ ] Train the face model on MELD training frames with the same target.
- [ ] Run the Paper 1 prompt, unchanged except for the label names in the instructions, as an extra text system if compute allows.
- [ ] Save text-only and face-only development scores before you fit any fusion weight.

Check: two development scores exist, and fusion weights are still unset.

### Phase D — Fusion

- [ ] On development predictions only, fit a weighted average `a * text_score + (1 - a) * face_score` for `a` in {0, 0.25, 0.5, 0.75, 1}.
- [ ] Freeze the best `a`.
- [ ] Optional: train an embedding fusion head with the same freeze rule.
- [ ] Inspect 40 development disagreements where the text score and face score differ by at least 3 points. Write the pattern you see before the test run.

Check: the weight `a` is written in the config, with the development metric that selected it.

### Phase E — Test once and write

- [ ] Freeze branches, weight, and map.
- [ ] Run the official test split once.
- [ ] Table: text-only, face-only, late fusion, and the majority or median baseline. Metric: mean absolute error on the 1–10 map, plus macro F1 on the original MELD emotions if you also predict emotion.
- [ ] Report a second cut: examples where a face was found versus examples where the text fallback was used.
- [ ] Report the human-check kappa next to the table so a reader sees whether the map holds.
- [ ] Write the paper: question, why unpaired TREC-IS and FER faces were refused, MELD license, map, branches cited back to Papers 1 and 2, fusion rule, test table, disagreements, limitations.
- [ ] Release code that downloads MELD through the official route and refuses to train if the split overlaps.
- [ ] Submit under the same fee rule as the earlier papers. Cite Paper 1 and Paper 2 as your own prior work, whether they are preprints or accepted papers. Say which.

Check: no test utterance was used to pick `a`, and the repository contains no joined TREC-IS/FER table.

## Experiment matrix

| System | Fit on |
| --- | --- |
| Median mapped score | Training labels only |
| Text encoder | MELD text, training split |
| Face model | MELD frames, training split |
| Late fusion of those two scores | Weight on development only |
| Optional LLM prompt from Paper 1 | Not fit, or few-shot from training utterances only |
| Optional embedding fusion | Training, selected on development |

## Metrics

- Mean absolute error on the frozen 1–10 map
- Mean absolute error inside each emotion, so joy and fear are not hidden by the average
- Rate of missing faces and the fallback error
- Quadratic weighted kappa for the human check
- A disagreement table: text closer, face closer, or tie, on the test set, computed after the freeze

## Paper outline

1. Introduction: Papers 1 and 2 studied text and faces separately because that is the public data they had.
2. Why not cross-pair those datasets.
3. Paired dataset, license, and the new frozen map.
4. Text branch and face branch, with citations to your earlier repositories.
5. Late fusion rule.
6. Human check.
7. Results against text-only and face-only.
8. Disagreement cases.
9. Limitations: television dialogue is not an emergency queue, the map is author-defined, copyright limits redistribution, no deployment claim.

## Stop rules

- Stop if you do not yet have public Papers 1 and 2.
- Stop if the only paired files you can find are scraped social-media portraits. Do not collect that crawl for this paper.
- Stop if a fusion gain appears only after a dialogue has leaked into both train and test.
- Stop if describing the work requires calling a sitcom utterance an emergency. Use the dataset’s real setting in every claim.

## Finish line

This paper is done when the public repository shows text-only, face-only, and fusion on the same paired test rows, the map is the one you froze, and the PDF cites the two earlier papers without changing their results.
