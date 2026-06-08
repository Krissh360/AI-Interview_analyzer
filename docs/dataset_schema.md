# AI Interview Analyzer - Dataset Schema

## Dataset Name

interview_responses.csv

---

# Purpose

This dataset contains labeled responses for the interview question:

"Tell me about yourself"

The dataset will be used for:

* Baseline model training
* DistilBERT fine-tuning
* Model evaluation
* Scoring engine development

---

# Schema

| Column Name      | Data Type | Description                                        |
| ---------------- | --------- | -------------------------------------------------- |
| id               | Integer   | Unique identifier                                  |
| answer           | String    | Interview response text                            |
| content_score    | Integer   | Content Quality score (1-5)                        |
| relevance_score  | Integer   | Relevance score (1-5)                              |
| vocabulary_score | Integer   | Vocabulary & Communication score (1-5)             |
| structure_score  | Integer   | Structure score (1-5)                              |
| overall_score    | Float     | Weighted score calculated from the four dimensions |
| label            | String    | Poor, Average, Good, or Excellent                  |

---

# Example Record

id: 1

answer:
"I am a third-year Computer Science Engineering student with a strong interest in machine learning. I have built several projects using Python and participated in hackathons. My goal is to become an AI Engineer."

content_score: 5

relevance_score: 5

vocabulary_score: 4

structure_score: 5

overall_score: 4.85

label: Excellent

---

# Constraints

content_score must be between 1 and 5

relevance_score must be between 1 and 5

vocabulary_score must be between 1 and 5

structure_score must be between 1 and 5

overall_score must be calculated using the official scoring formula

label must be derived from overall_score using the official label mapping
