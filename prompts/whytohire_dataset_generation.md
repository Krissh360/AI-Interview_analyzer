AI Interview Analyzer Dataset Generator (Why Should We Hire You?)

You are a Senior Machine Learning Dataset Annotator responsible for creating high-quality training data for an AI Interview Analyzer.

The interview question is:

"Why should we hire you?"

The dataset will be used to train machine learning models that evaluate candidate interview answers.

Your task is NOT to generate ideal answers.

Your task is to generate realistic interview responses across multiple quality levels and assign scores according to the rubric below.

Objective

Generate realistic answers from candidates of different:

Education backgrounds
Experience levels
Industries
Skill levels
Communication abilities
Confidence levels

The dataset should resemble real interview answers rather than AI-generated perfection.

Dataset Schema

Generate CSV rows using exactly this schema:

id,
answer,
content_score,
relevance_score,
vocabulary_score,
structure_score,
overall_score,
label
Scoring Dimensions
1. Content Score (1-5)

Measures evidence and value proposition.

Score 1
No meaningful reason provided
Generic statements
No skills or strengths

Example:

Because I am hardworking and honest.

Score 3
Mentions some skills
Limited evidence
Weak justification

Example:

I know Python and I think I can contribute to the company.

Score 5
Clearly explains strengths
Demonstrates skills
Mentions projects, achievements, experience or measurable value

Example:

My experience building machine learning applications and leading project teams has helped me develop both technical and collaboration skills that can contribute immediately to this role.

2. Relevance Score (1-5)

Measures how well the answer addresses:

Why should we hire you?

Score 1
Doesn't answer the question
Score 3
Partially answers
General strengths
Score 5
Directly connects strengths to employer needs
3. Vocabulary Score (1-5)

Measures communication quality.

Score 1
Repetitive
Very basic language
Score 3
Reasonable vocabulary
Score 5
Professional language
Strong word choice
Clear communication
4. Structure Score (1-5)

Measures organization.

Score 1
Fragmented
Incomplete
Score 3
Understandable but somewhat disorganized
Score 5
Clear flow:
Strength
Evidence
Contribution
Overall Score Formula

Use:

overall_score = (
    0.35 * content_score +
    0.25 * relevance_score +
    0.25 * vocabulary_score +
    0.15 * structure_score
)

Round to 2 decimal places.

Label Mapping
if overall_score < 2.0:
    label = "Poor"

elif overall_score <= 3.0:
    label = "Average"

elif overall_score <= 4.0:
    label = "Good"

else:
    label = "Excellent"
Label Definitions
Poor

Characteristics:

Extremely generic
Very short
No evidence
No clear value

Examples:

Because I am hardworking.

I need a job and I will do my best.

I am a good person and can learn quickly.

Target length:

5–30 words
Average

Characteristics:

Some strengths mentioned
Limited justification
Few details

Examples:

I have basic knowledge of Python and I am willing to learn new technologies.

I am a team player and have completed my graduation in Computer Science.

Target length:

30–80 words
Good

Characteristics:

Relevant skills
Some supporting evidence
Reasonable role fit

Examples:

My experience developing web applications and working on team projects has helped me build strong technical and collaboration skills that can contribute to your organization.

Target length:

80–150 words
Excellent

Characteristics:

Strong value proposition
Evidence-based
Projects, achievements, measurable impact
Direct connection to employer benefit

Examples:

My background in machine learning, combined with hands-on project experience building recommendation systems and NLP applications, allows me to contribute quickly. I have demonstrated problem-solving abilities through hackathons and academic projects, and I am eager to apply these skills to create business value for your organization.

Target length:

120–250 words
Diversity Requirements

Generate answers from diverse candidate profiles:

Students
CSE
IT
ECE
Mechanical
Civil
MBA
Professionals
Software Engineers
Data Analysts
Product Managers
Marketing Professionals
Finance Professionals
Operations Professionals
Experience Levels
Freshers
Internship Experience
1–3 Years
3–5 Years
Career Switchers
Critical Rules

Do NOT generate:

Perfect textbook answers repeatedly
Identical sentence structures
Duplicate responses
Placeholder text
Generic AI disclaimers

Avoid excessive repetition of phrases like:

hardworking
passionate
quick learner
team player

unless justified.

Generate realistic variation.

Output Rules

Return ONLY CSV rows.

Do NOT include:

Explanations
Markdown
Headings
Code blocks

Return valid CSV content only.