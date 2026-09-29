# Paper 1 — Urgency from text with an open LLM

Publish this paper first. Leave the face paper and the combined paper frozen until this study, its code, and its written report are public.

This file is only the guide for the text paper. It does not change the other two papers.

## What this paper is

The question is: how well can a small open language model estimate the urgency of a short public text, compared with a simple baseline and a fine-tuned encoder, when you have no paid API and little or no GPU?

Your reporting scale, used in all three papers:

| Score | Meaning |
| --- | --- |
| 1 | Highest urgency |
| 10 | Lowest urgency |

No large public text dataset is already labeled with that 1–10 scale. Do not invent 10-way labels. This paper uses the public priority labels that annotators actually assigned, then displays them on your scale through a mapping you freeze before you look at test results.

## Dataset

Use the TREC Incident Streams (TREC-IS) collection.

- Track page: https://www.dcs.gla.ac.uk/~richardm/TREC_IS/
- Download instructions: https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html
- Human labels cover emergency-related posts from events such as earthquakes, floods, fires, storms, and similar incidents.
- Each assessed post has information-type labels and a priority label. The priority labels are Critical, High, Medium, Low, and in some releases Irrelevant or Unknown.
- Official numeric mapping used by the track: Critical = 1.0, High = 0.75, Medium = 0.5, Low = 0.25. On that track scale, a larger number means more urgent. Your 1–10 scale runs in the opposite direction. Keep both tables in the paper so a reader cannot mix them up.

Frozen display mapping for this project:

| TREC-IS priority | Track score | Your urgency score |
| --- | --- | --- |
| Critical | 1.0 | 1 |
| High | 0.75 | 3 |
| Medium | 0.5 | 6 |
| Low | 0.25 | 9 |
| Irrelevant or Unknown | — | 10 |

Scores 2, 4, 5, 7, and 8 are not supervised classes. The model may output them only as an interpolation between these bins, and the paper must say they were not human labels.

Download and license rules:

- Read the track page before downloading. The client asks for a name, email, and institution. Use true contact details.
- Tweet text is not yours to republish. Share label files only if the track license allows, plus post identifiers and the official download command. Do not commit raw posts to GitHub.
- Start with a small edition (`trecis2018-A` or the events 1–75 label file) so a slow connection is not wasted. Expand only after the loader works.
- Split by event, never by random post. Train on earlier events (1–75). Keep later events (76–122) as the untouched test set. The track page says those later labels must not be used to submit a leaderboard run that was tuned on them. For this paper, use them only as a final test, after every model choice is finished.
- If the download server is offline, email the maintainers. Do not replace TREC-IS with a scraped social-media dump.

## Other dataset present in this folder

A second corpus sits in `papers/text-urgency-llm/Bengali Medical Dataset/doctor_qa_dataset.csv`. It is recorded here for provenance only. It is not the dataset of this paper and must not be substituted for TREC-IS.

| Field | Value |
| --- | --- |
| Hugging Face Hub ID | `shetumohanto/doctor_qa_bangla` |
| Maintainer | Shetu Mohanto (HF user `shetumohanto`) |
| Hosted | Directly on the Hugging Face Hub |
| Intended use | Instruction fine-tuning, Q&A training, and medical text classification for Bengali NLP models |
| Content | Medical questions asked by users paired with doctor responses, in Bengali and Banglish |
| Local file | `doctor_qa_dataset.csv`, one `text` column |
| Row format | `<s> [INST] <question> [/INST] <doctor answer> </s>` |
| Labels | None. Question and answer are concatenated into a single string |

Why it cannot serve Paper 1:

- It carries no priority annotation and no event identifiers, so there is no Critical, High, Medium, Low, or Irrelevant label to classify and no way to split by event.
- Priority labels would have to be inferred from the wording of the doctor answer. Those would be self-generated pseudo-labels, which the rule in the section above forbids. A study built on them would measure the author's annotation scheme, not human-judged priority.
- The domain is Bengali pediatric patient queries, not English emergency-related social posts. Results would not be comparable to the framing of this paper or to the later face paper.

Permitted use: as a smoke test for the Phase D harness, to confirm the chat template, generation settings, and parse-failure logging work end to end without opening the TREC-IS test events. Record nothing from it in the results table of this paper.

License for the corpus has not been recorded here. Check the dataset card on the Hub and write the license down before any use, including the smoke test.

### Second corpus, referenced but not downloaded

Not present in this folder. Recorded so the provenance is written down rather than recalled from memory.

| Field | Value |
| --- | --- |
| Kaggle URL | https://www.kaggle.com/datasets/shashwatwork/bengali-medical-dataset |
| Kaggle title | Bengali Medical Dataset |
| Kaggle publisher | `shashwatwork` |
| File of interest | `Specialist Classification.csv` |
| Companion file | `Bengali Medical NER.csv` |
| Language | Bengali |
| Content | Roughly six hundred patient chief-complaint statements, about eight thousand words, plus specialist labels such as medicine specialist, cardiologist, dentist, gynecologist |
| Second part | Manually annotated named entities: symptoms, body parts, colors, body fluids, blood, times, values, directions, effluents, adverbs |
| Size as reported | 658 utterances across 9 specialties, per the CrossStack-BERT report on this corpus |
| Upstream origin | A Kaggle re-upload of the Mendeley Data record below |
| Mendeley title | Bengali Colloquial Dataset of Primary Medical Issues for Improving Health System |
| Mendeley link supplied by the author | https://data.mendeley.com/datasets/4tt953xwk2/1, this points at version 1 |
| Mendeley landing page | https://data.mendeley.com/datasets/4tt953xwk2, this currently shows version 3 |
| Mendeley DOI | https://doi.org/10.17632/4tt953xwk2.3 for version 3, published 9 July 2021. The version 1 DOI is https://doi.org/10.17632/4tt953xwk2.1 |
| Mendeley contributors | Dr. M. F. Mridha, Tanvir Islam, Sakila Mahbin Zinat |
| Institution | Bangladesh University of Business and Technology, BUBT Medical Center |
| Mendeley license | CC BY 4.0 |
| Stated restriction | Health-systems research with AI, ML, DL, and NLP only. Explicitly not for treatment purposes |

Why it cannot serve Paper 1:

- The label is which specialist a patient should see, not how urgent the situation is. There is no Critical, High, Medium, Low, or Irrelevant class and no priority score to map onto the 1–10 display scale.
- There are no event identifiers. The 658 utterances are aggregated chief complaints, not posts from discrete incidents, so an event split cannot be built.
- Converting specialist label into urgency would be an invented labeling scheme. That is the same failure the rule in the Dataset section forbids.
- The stated Mendeley restriction also limits use to health-systems research, which an emergency-priority paper on social posts falls outside.

Permitted use: none for this paper. Do not train, tune, or validate on it. Cite it only if a related-work section discusses Bengali medical NLP resources, and cite the Mendeley DOI rather than the Kaggle mirror, because the mirror carries no license of its own.

## What the paper may claim

- A compute-limited comparison of a lexical baseline, a fine-tuned open encoder, and a small open instruction model on TREC-IS priority.
- How those systems behave on your declared 1–10 display scale.
- Which errors are worst, especially missed Critical posts.

The paper may not claim that a model is safe to route emergency services, that it measured a validated 10-level human scale, or that it was accepted anywhere before a venue accepts it.

## Free tools

- Python, Jupyter, Git, scikit-learn, PyTorch, Hugging Face `transformers`, `datasets`, and `evaluate`
- A CPU is enough for the baseline and for a small encoder on a sample
- Optional free GPU: Kaggle Notebooks or Google Colab. Quotas change. Every experiment needs a CPU fallback: fewer posts, a smaller model, or both
- OpenJDK only to run the official TREC-IS download client
- Do not depend on a paid LLM API. If you try a free-tier hosted model, record the date, model name, and that the tier can disappear. The paper’s main systems must be runnable from open weights

## Learning path

Work in order. Each phase ends when you can pass its check without looking at the notes.

### Phase A — Text, splits, and metrics

- [ ] Read a short post and say what a response officer would need from it: the event, the ask, and how soon it matters.
- [ ] Explain train, validation, and test. Explain why two posts from the same earthquake are not independent.
- [ ] Compute accuracy, macro precision, macro recall, macro F1, and a confusion matrix on a 4-class toy example.
- [ ] Compute mean absolute error on the scores 1, 3, 6, 9, 10. Show that accuracy can look fine while Critical recall is poor.
- [ ] Read the TREC-IS overview for the edition you use. Record the priority definition in your own words. Overview papers are linked from the track page.

Check: given 10 toy posts and labels, you can split them by event and compute macro F1 and Critical recall by hand.

Free reading: scikit-learn’s [common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) and the classification metrics guide in the same documentation.

### Phase B — A baseline you can train on a CPU

- [ ] Load only the label file and plot how many posts fall into Critical, High, Medium, Low, and other.
- [ ] Join labels to text for a small edition. Print 20 examples of each priority. Write down words that fooled you.
- [ ] Build TF-IDF plus a linear model in scikit-learn. Fit only on training events.
- [ ] Compare it with a majority-class baseline.
- [ ] Tune on validation events chosen from inside events 1–75. Do not touch events 76–122.
- [ ] Save the exact command, seed, and library versions.

Check: one command retrains the baseline and writes a metrics table. Critical recall is reported even if it is near zero.

### Phase C — Encoder fine-tuning

- [ ] Read the free [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/) chapters on tokenization, datasets, and fine-tuning a text classifier.
- [ ] Fine-tune a small open encoder. DistilBERT is a sound default. Read the model card and record the exact checkpoint.
- [ ] Use the official event split. Truncate long posts on purpose and state the max length.
- [ ] Select the checkpoint by validation macro F1, with Critical recall written beside it.
- [ ] If the full training set is too large for your machine, train on a stratified event-preserving sample and say so. Do not sample only easy posts.

Check: you can reload the saved encoder and reproduce validation metrics within a small tolerance.

### Phase D — The open LLM

- [ ] Learn the difference between an encoder classifier and an instruction model. The encoder is trained on your labels. The instruction model is asked, in text, to return a priority.
- [ ] Choose an open instruction model at or under about 3 billion parameters so a free GPU, or a slow CPU sample, can run it. Read the license and the model card before any experiment. Record the revision hash.
- [ ] Write one prompt that defines Critical, High, Medium, Low, and your 1–10 display mapping. Ask for a single priority word and nothing else.
- [ ] Run zero-shot on the validation events. Parse failures are errors, not items you may drop quietly. Report the parse-failure rate.
- [ ] Run a few-shot prompt that uses a fixed set of training-event examples, balanced across priorities. Do not pick the examples after seeing validation scores.
- [ ] If a free GPU is available, try one parameter-efficient fine-tune (LoRA or QLoRA) of the same small model. If it is not available, skip fine-tuning and report the prompted model only.
- [ ] Never send private or unpublished text to a hosted service.

Check: a table with majority baseline, TF-IDF, encoder, zero-shot LLM, and few-shot LLM, all on the same validation events.

### Phase E — Final test and error study

- [ ] Freeze the code, prompts, and checkpoint. Then run the test events once.
- [ ] Report, on the official priority labels: macro F1, per-class recall, and the track-style numeric error if you also predict 0.25–1.0.
- [ ] Report, on the frozen 1–10 display mapping: mean absolute error. State that intermediate scores were not annotated.
- [ ] Break results out by event type if the labels support it. A model that works on floods and fails on fires must not be averaged into one happy number.
- [ ] Read 30 Critical posts the model called Low, and 30 Low posts it called Critical. Group the failures.
- [ ] Run the test a second time only to measure seed or prompt variance, and label that run as a variance check rather than a new model hunt.

Check: the test numbers in the paper match the saved output file, and the file was produced after the freeze.

### Phase F — Write and release

- [ ] Write a report with: question, related work (include the TREC-IS overview), data and license, split, models, compute, results, failure cases, and limitations.
- [ ] Say clearly that your 1–10 numbers are a display mapping of four public bins.
- [ ] Put code, prompt text, config, and a data-download README in a public repository. Omit raw posts.
- [ ] Ask one other person to run the README on a clean machine, or to follow it until the first missing step.
- [ ] Release a preprint or a public PDF with the code link. arXiv is an archive, not peer review. New computer-vision or computation-and-language authors usually need an endorsement.
- [ ] Submit, if you submit, to a venue that charges authors nothing. Recheck fees in the month you submit. JMLR and TMLR have charged no author fee; TMLR limits how many submissions an author can make. Skip any venue that promises acceptance or pressures you to pay.
- [ ] After rejection, keep the reviews, fix what is actually wrong, and either resubmit or update the public report. Do not strengthen a claim the test does not support.

Check: a stranger can tell what was predicted, what was human-labeled, and how to rebuild the table without emailing you.

## Experiment matrix

Keep this small enough to finish.

| System | Role |
| --- | --- |
| Majority class | Floor |
| TF-IDF + linear model | Cheap baseline |
| Fine-tuned small encoder | Strong open baseline |
| Small open LLM, zero-shot | Prompted system |
| Same LLM, few-shot | Prompted system with fixed examples |
| Same LLM, LoRA | Only if a free GPU exists |

One ablation is enough: prompt with the priority definition versus prompt without it, on validation only.

## Metrics to put in the paper

- Macro F1 and per-class recall on Critical, High, Medium, Low
- Critical recall called out in the abstract if you discuss emergency use at all
- Mean absolute error on the frozen 1–10 display scores
- Parse-failure rate for the LLM
- Number of events in train, validation, and test
- Hardware, wall time, and whether you used a sample

Accuracy alone is not a result on this data. Critical posts are rare.

## Paper outline

1. Introduction: emergency posts are too many to read, and missing a critical post is the costly error.
2. Related work: TREC-IS, classical urgency or priority classification, encoder classifiers, and prompted language models.
3. Data: events, labels, the official score, your 1–10 display map, license, and event split.
4. Methods: baseline, encoder, prompted model, optional LoRA.
5. Setup: seeds, lengths, compute, what was frozen.
6. Results: validation choices, then the single test table.
7. Errors: missed Critical posts and false Critical posts.
8. Limitations: four bins rather than a true 1–10 annotation, Twitter terms, English-heavy text, no claim of operational deployment.
9. Ethics: the system is a research comparison, not a dispatcher.

## Stop rules

- Stop if the download terms forbid the use you planned. Pick another public priority corpus and rewrite the data section. Do not scrape a replacement.
- Stop tuning once the test events have been opened.
- Stop if the only way to “win” is a paid API. Report the open systems you actually ran.

## Hand-off to the later papers

When this paper is public, record:

- the exact checkpoint and prompt
- the frozen 1–10 mapping
- the test metrics and the failure groups
- the repository URL

The face paper does not reuse this dataset. The third paper may reuse this method only after both earlier papers are public, and only on examples that contain both text and a face.
