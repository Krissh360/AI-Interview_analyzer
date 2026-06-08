# AI Interview Analyzer - Dataset Generation Guidelines

## Objective

Generate realistic responses to the interview question:

"Tell me about yourself"

The generated responses will be used to train and evaluate machine learning models.

---

# Target Dataset Size

Phase 1 Target:

100 samples

Full Dataset Target:

1000 samples

---

# Class Distribution

The dataset should remain balanced.

| Label     | Target Percentage |
| --------- | ----------------- |
| Poor      | 25%               |
| Average   | 25%               |
| Good      | 25%               |
| Excellent | 25%               |

---

# Candidate Personas

Responses should be generated from diverse backgrounds.

Examples:

* Computer Science Student
* Mechanical Engineering Student
* Electrical Engineering Student
* MBA Student
* Fresher
* Working Professional
* Career Switcher
* Self-Taught Developer
* Data Analyst
* Software Engineer

---

# Content Diversity

Responses should vary in:

* Education
* Skills
* Projects
* Achievements
* Career Goals
* Length
* Vocabulary

---

# Length Distribution

Include:

Short Responses:
20–50 words

Medium Responses:
50–120 words

Long Responses:
120–250 words

---

# Poor Responses

Characteristics:

* Very short
* Generic
* Missing key information
* Weak communication
* Little relevance

---

# Average Responses

Characteristics:

* Basic education information
* Limited skills discussion
* Minimal detail

---

# Good Responses

Characteristics:

* Education
* Skills
* Career goals
* Some project discussion

---

# Excellent Responses

Characteristics:

* Education
* Skills
* Projects
* Achievements
* Career goals
* Strong communication
* Clear structure

---

# Quality Control Checklist

Before accepting generated samples:

* Remove duplicates
* Remove near-duplicates
* Remove unrealistic answers
* Verify label correctness
* Verify score correctness
* Verify class balance
* Verify diversity of backgrounds
* Verify diversity of answer lengths

---

# Important Principle

Generated data should simulate realistic interview responses from actual candidates rather than idealized or overly polished AI-generated text.

Natural variation is preferred over perfect writing.
