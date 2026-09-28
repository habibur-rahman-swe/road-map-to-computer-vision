# Paper 2 — Urgency from facial-expression images

Publish this paper second, after the text LLM paper is public. Do not start the combined paper’s experiments until this paper is public as well.

This file is only the guide for the face paper.

## What this paper is

The question is: using a public facial-expression dataset, how well can an open image model predict an urgency score that you define in advance from those expression labels, and does that score agree with a small set of fresh human urgency ratings?

Your reporting scale, used in all three papers:

| Score | Meaning |
| --- | --- |
| 1 | Highest urgency |
| 10 | Lowest urgency |

Public expression datasets do not contain this 1–10 urgency label. FER+ contains crowd votes for emotions, not urgency. This paper is allowed to derive a score from those votes only if the formula is frozen first, the paper calls it author-defined, and a separate human rating checks it. If the human check disagrees, publish that disagreement. Do not relabel the images by hand until the formula matches what you hoped.

## Dataset

Use FER+ annotations on the FER-2013 images.

- FER+ annotations and loader notes: https://github.com/microsoft/FERPlus
- The annotation repository explains that Microsoft does not host the images. Download the FER-2013 images from the source named in that repository, which is the facial-expression challenge data released through Kaggle.
- FER+ gives, for each image, vote counts from 10 taggers over: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, and NF (not a face).
- The FER+ code and label file are under the license in that repository (MIT at the time of writing). The images have their own terms. Read both before you train, and cite both.
- Images are small grayscale faces, about 48×48. That is a feature of the dataset, not a preprocessing accident.
- Use the official FER usage split already stored in the label file: Training, PublicTest for validation, PrivateTest for the final test. Do not reshuffle images across those splits.
- Drop NF and unknown-only images from urgency training, and report how many you dropped. Do not silently delete them from the emotion results; report emotion metrics with the same exclusion rule stated once.

There is no honest substitute that already contains “urgency 1–10” for faces and is fully public. AffectNet has valence and arousal, but access is requested and redistribution is restricted. Do not build the paper on AffectNet unless you have that permission in writing. Do not scrape faces from the web.

## Frozen urgency formula

Freeze this formula before you compute any test metric. It is a hypothesis, not a fact about human urgency.

Vote weights, higher meaning more urgent:

| Expression | Weight |
| --- | --- |
| fear | 1.0 |
| anger | 0.9 |
| surprise | 0.7 |
| sadness | 0.5 |
| disgust | 0.4 |
| neutral | 0.2 |
| contempt | 0.15 |
| happiness | 0.1 |

Let each weight be multiplied by that expression’s share of the 10 votes, ignoring unknown and NF in the share, and dropping the image if those two take the majority. Call the weighted sum `urgency_raw` (0 to 1). Convert it to your scale:

`urgency_score = round(1 + (1 - urgency_raw) * 9)`

A face dominated by fear maps near 1. A face dominated by happiness maps near 10. Mixed votes land in between. Write the formula in the paper and do not change it after seeing the private test set.

## What the paper may claim

- You can predict FER+ emotion distributions with a small open model, on the official split.
- You can predict the author-defined urgency score derived from those votes.
- On a small, separately rated subset, human 1–10 ratings agree or disagree with that score by a measured amount.

The paper may not claim that the model reads intent, detects lies, diagnoses anyone, or is fit to watch a crowd, a workplace, a school, or a street. The images are a public benchmark. Deployment on people who did not consent is out of scope.

## Free tools

- Python, Jupyter, Git, NumPy, Matplotlib, Pillow, PyTorch, torchvision
- CPU is enough for a tiny network and for evaluation of a small pretrained model
- Optional free GPU through Kaggle or Colab, with a CPU sample as fallback
- No paid annotation service. A second rater can be one person you can actually meet

## Learning path

Work in order. The text paper can teach you experimental hygiene. This path still includes the image skills, because the papers stay separate.

### Phase A — Images and the dataset

- [ ] Open one FER-2013 image and state its height, width, and channel count.
- [ ] Explain why a 48×48 grayscale face destroys identity detail and still leaves expression cues.
- [ ] Load `fer2013new.csv` and show the 10 vote columns for 15 images. Include at least one NF.
- [ ] Plot the majority emotion counts. Note which emotions are rare.
- [ ] Implement the frozen formula on 10 images and check the arithmetic by hand.
- [ ] Confirm the official Training / PublicTest / PrivateTest boundaries on a few row indexes.

Check: a script prints, for one image, the vote vector, `urgency_raw`, and `urgency_score`, and a second person gets the same numbers from the formula.

Free reading: the FER+ paper linked from the Microsoft repository (Barsoum et al.), plus a short convolution chapter from [Dive into Deep Learning](https://d2l.ai/) or the Stanford CS231n notes at https://cs231n.stanford.edu/.

### Phase B — Emotion model first

Predict the emotions that humans actually voted, before you predict urgency. If the emotion model is wrong, the urgency number is not evidence.

- [ ] Turn votes into a training target: the majority emotion, and also the vote distribution as a softer target if you are ready for that.
- [ ] Train a small convolutional network from scratch on CPU, on the Training split only.
- [ ] Fine-tune one small pretrained torchvision model (MobileNet or ResNet-18 are enough) if compute allows. Record the preprocessing that model expects. FER images are grayscale; repeat the channel or use a model input you have actually checked.
- [ ] Choose the checkpoint on PublicTest. Macro F1 is the selection metric.
- [ ] Look at mistakes: fear versus surprise, anger versus disgust, and any face the file marks poorly.

Check: PublicTest macro F1 is saved in a file you can reload, and the private test split has not been scored yet.

### Phase C — Urgency head

- [ ] Add a second target: the frozen `urgency_score` from training votes only.
- [ ] Train one model to regress that score, and one model to classify the integer scores that the formula actually produces. Compare them on PublicTest with mean absolute error.
- [ ] A useful extra baseline: ignore the image and predict the training-set median score. Your model has to beat that.
- [ ] Another baseline: compute the score from your predicted emotion distribution using the same formula. Compare “predict emotions, then apply the formula” with “predict the score directly.”
- [ ] Keep PrivateTest untouched while you choose between these.

Check: a validation table with median baseline, emotion-then-formula, and direct score model.

### Phase D — Human check of the formula

The formula is not ground truth until people rating urgency, without being shown the formula, land near it.

- [ ] From the Training or PublicTest images only, draw 200 images stratified across the score range. Do not draw them from PrivateTest.
- [ ] Write a one-page rater sheet. Urgency means “how quickly a bystander would believe this person needs help,” from 1 (immediately) to 10 (not urgent). Give three anchor examples you rated yourself and then do not include those three in the 200.
- [ ] Rate the 200 yourself. Ask a second person to rate the same 200 without seeing your scores or the formula.
- [ ] Report quadratic weighted kappa between the two raters, and mean absolute error between the average human score and the formula.
- [ ] If kappa is very low, say the 1–10 task is unstable for these images. Keep the emotion results. Narrow the urgency claim.
- [ ] If you cannot find a second rater, publish with one rater and write that limitation in the abstract, not only in a footnote.

Check: the rating sheet, the image ids, and both score columns are saved. The formula was not edited after the ratings.

### Phase E — One test run, then the paper

- [ ] Freeze the formula, the checkpoint, and the human-check writeup.
- [ ] Score PrivateTest once for emotion macro F1 and for urgency mean absolute error.
- [ ] Show a grid of successes and failures at several score levels.
- [ ] State demographic and image limits: 48×48, posed and unposed web faces collected years ago, uneven emotion counts, no consent trail you can audit here. Do not add a fairness claim you did not measure. If you report error by a group, only do it for labels that exist in the file.
- [ ] Write the report: question, FER and FER+ citations, formula, split, models, human check, test table, failures, limitations.
- [ ] Release code and a download script that fetches images from the official source. Do not commit the image archive if the image terms forbid redistribution.
- [ ] Release the PDF. Submit only to a venue you have checked for author fees. The text-paper guide’s submission rules apply here too: no promised acceptance, no pay-to-publish venue.
- [ ] Cite the text paper as a separate study. Do not mix its tweets into these tables.

Check: a reader can recompute `urgency_score` for one published image id and match your table.

## Experiment matrix

| System | Role |
| --- | --- |
| Training-median urgency score | Floor for the score |
| Small CNN from scratch | CPU reference for emotion |
| Pretrained small network, fine-tuned | Main emotion model if compute allows |
| Formula applied to predicted emotions | Tests whether emotion errors destroy the score |
| Direct urgency regression or classification | Tests predicting the score itself |

## Metrics

- Emotion: macro F1 and a confusion matrix on the official test split
- Urgency: mean absolute error against the formula score
- Human check: quadratic weighted kappa and mean absolute error against the formula, on the 200-image set only
- Counts of dropped NF / unknown images
- Parameter count and hardware

Do not lead with accuracy. Rare emotions and the ends of the 1–10 scale matter more than the middle.

## Paper outline

1. Introduction: expression labels exist; an urgency score does not. You define one and test it.
2. Related work: FER-2013, FER+, emotion recognition limits, and why expression is not the same thing as urgency.
3. Data and formula: votes, split, weights, rounding, exclusions.
4. Models.
5. Human rating protocol.
6. Results: emotion, formula prediction, human agreement.
7. Failures: low resolution, mixed votes, confusion between fear and surprise.
8. Limitations and ethics: benchmark only, no surveillance, no medical or safety deployment, possible dataset bias, single-dataset test.

## Stop rules

- Stop if the image license forbids the training you planned.
- Stop if you feel pressure to change the weights after seeing PrivateTest. Report the frozen formula.
- Stop if a result only looks good because NF images or the official split were discarded without a count.
- Do not pair these faces with emergency tweets. That is the third paper, and only with data that actually contains both.

## Hand-off to the third paper

When this paper is public, record:

- the formula and the vote weights
- the official split
- the chosen model and its preprocessing
- test emotion F1, urgency error, and the human-check kappa
- the repository URL

The third paper may reuse this model family only on examples that have a face and aligned text. It may not treat a random FER face plus a random tweet as one person.
