# AI Interview Analyzer - Scoring Rubric

## Objective

Evaluate responses to the interview question:

"Tell me about yourself"

The evaluation is based on four dimensions:

1. Content Quality
2. Relevance
3. Vocabulary & Communication
4. Structure

Each dimension is scored from 1 to 5.

---

# 1. Content Quality Score

Measures the amount and quality of professional information provided.

## Score 1

Very little useful information.

Examples:

* "My name is Rahul."
* "I am a student."

## Score 2

Basic information only.

Typically includes:

* Education OR interests

Example:

* "I am a college student who enjoys coding."

## Score 3

Moderate amount of information.

Typically includes:

* Education
* Some skills

Example:

* "I am a Computer Science student and I know Python and Java."

## Score 4

Strong amount of information.

Typically includes:

* Education
* Skills
* Projects

Example:

* "I am a Computer Science student and have built machine learning projects using Python."

## Score 5

Comprehensive professional introduction.

Typically includes:

* Education
* Skills
* Projects
* Achievements
* Career Goals

Example:

* "I am a third-year Computer Science student, have built multiple AI projects, participated in hackathons, and aim to become a Machine Learning Engineer."

---

# 2. Relevance Score

Measures how well the response answers the question.

## Score 1

Completely off-topic.

Example:

* "I enjoy cricket and watching movies."

## Score 2

Mostly irrelevant with very little professional information.

## Score 3

Partially relevant but incomplete.

## Score 4

Relevant and focused on professional background.

## Score 5

Highly relevant and directly answers the question with a professional self-introduction.

---

# 3. Vocabulary & Communication Score

Measures language quality, grammar, clarity, and professionalism.

## Score 1

Very poor communication.

Example:

* "Me student. Python good."

## Score 2

Weak communication with noticeable grammar issues.

## Score 3

Understandable but basic communication.

## Score 4

Clear and professional communication.

## Score 5

Highly polished, professional, and fluent communication.

---

# 4. Structure Score

Measures organization and logical flow.

Preferred structure:

Introduction
→ Education
→ Skills
→ Projects
→ Career Goals

## Score 1

No structure.

## Score 2

Weak organization.

## Score 3

Some logical flow.

## Score 4

Well-organized answer.

## Score 5

Excellent structure with smooth transitions and strong flow.

---

# Overall Score Formula

overall_score =
0.35 × content_score +
0.25 × relevance_score +
0.25 × vocabulary_score +
0.15 × structure_score

---

# Label Mapping

| Overall Score | Label     |
| ------------- | --------- |
| < 2.0         | Poor      |
| 2.0 - 3.0     | Average   |
| 3.0 - 4.0     | Good      |
| > 4.0         | Excellent |

---

# Notes

Content Quality carries the highest weight because interview introductions are primarily evaluated on the quality and completeness of information presented.

The scoring rubric should remain consistent across dataset generation, model training, evaluation, and inference.
