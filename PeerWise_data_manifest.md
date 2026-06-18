# PeerWise Data Manifest — what I currently hold vs. what's missing

_Prepared for the strict-discriminator filter (author answer ≠ modal student answer) and the Tier-B robustness ablation. Grain that matters: **per-student answer submissions**, from which per-option response counts → the modal student answer can be derived._

## Summary of the gap

The strict-discriminator variant needs, for each question, **how many students chose each option (A/B/C/D/E)**, so I can identify the *modal student answer* and keep only questions where it differs from the author's key.

- For **Biology / Law / Psychology** I have a per-student `Answers` table → I can compute this directly. ✅
- For **Cardiff / Sydney** I only have the **question-level** export (`*_all_questions.xlsx`). It carries `total_responses` (a single total) and the author's `answer`, but **no per-option breakdown and no per-student submission rows** → the modal student answer cannot be recovered. ❌  ← **this is what I'm asking for**
- For **Medicine (UK Y1/Y2)** I only have processed/generator JSONs locally, not the raw `Questions/Answers/Ratings` tables. ⚠️ (not blocking the current ablation, noted for completeness)

## What I have — raw tables

### Auckland corpora (`PeerWiseData/`)

| Corpus | Questions | Answers (per-student submissions) | Ratings | Per-option counts derivable? |
|---|---|---|---|---|
| Biology | ✅ 380 rows | ✅ **78,970 rows** — `AnswerID, UserID, Question_ID, Answer` | ✅ 36,644 | **Yes** |
| Law | ✅ 5,626 rows | ✅ **89,990 rows** — `Answer_ID, UserID, Question_ID, Answer` | ✅ 84,007 | **Yes** |
| Psychology | ✅ 10,933 rows | ✅ **24,854 rows** — `Answer_ID, User, Timestamp, Question_ID, Answer` | ✅ 17,581 | **Yes** |
| Medicine (UK Y1/Y2) | ⚠️ processed JSON only | ⚠️ processed JSON only | ⚠️ processed JSON only | No (raw tables absent locally) |

`Answers` schema (the table I need for Cardiff/Sydney too): one row = one student's submitted option for one question.

### Cardiff / Sydney (`Paul_new_data/`)

| Corpus | What I have | Per-option counts? | Per-student submissions? |
|---|---|---|---|
| Cardiff | `Cardiff_all_questions.xlsx` — **28,885 rows**, question-level | ❌ only `total_responses` (single total) | ❌ none |
| Sydney | `Sydney_all_questions.xlsx` — **22,121 rows**, question-level | ❌ only `total_responses` (single total) | ❌ none |

Question-level columns present: `id, course_id, timestamp, user, avg_rating, total_responses, total_ratings, top_rating_count, avg_difficulty, total_comments, deleted, answer, numAlts, question, altA–altE, explanation`.

_(For context: after the default filter the Cardiff training pool is 7,309 questions; Tier-B shrinks it to 1,041. Neither filter needs the submission table — only the strict-discriminator variant does.)_

## The ask, precisely

A **per-student answer-submission table for both Cardiff and Sydney** — same grain as the `Answers.xlsx` I already have for Biology/Law/Psychology:

> `Answer_ID, UserID, (Timestamp), Question_ID, Answer`

with `Question_ID` joinable to the `id` column of `Cardiff_all_questions.xlsx` / `Sydney_all_questions.xlsx`. From that I can group by question → per-option counts → modal student answer → run the strict-discriminator filter Paul suggested.

**Fallback if these can't be located:** the strict-discriminator ablation does not strictly require Cardiff/Sydney — it can run on **Biology / Law / Psychology**, which already carry the per-student `Answers` table. So if the Cardiff/Sydney submissions are hard to find, I can demonstrate the variant on one of those subjects today with no new data.
