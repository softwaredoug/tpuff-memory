This is a prompt meant to help pinpoint how the golden answer for a question
is intended to be solved.

In this repo are agentic memory datasets. I'm specifially interested in longmemeval-v2. Should be downloaded to longmemeval-v2.

The dataset has a question + goldenset. It also has a custom haystack (set of trajacetories) to use to answer any question.

But I need help debugging. I need to know how precisely a memory system would answer the question
I want to give you a question id, and then you trace it to create a rubrick for how its solved.

I'll prompt you with a question id (or ask you to draw a random one), and you will return a rubric for how to solve it.

You write out to <question_id>.md an explanation.

Analysis notes: questions ask for specific {boxed\...} answers, you don't need to explain any discrepenancies. Those are parsed out.

Write out in the following format

## Question (QuestionID: <question_id>)

<question text>

## Answer

<golden answer text>

## Evidence (step-by-step)

### Trajectory {trajectory_id}, Step {step_id}

Fact
Supporting evidence
Which fields to look in evidence

(repeated until golden answer reached)
