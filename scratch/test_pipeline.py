from app.ai_engine import generate_questions_from_chapter

sample_story = """
The Lighthouse of Moonbay stands on the northern cliff edge overlooking the stormy sea.
For generations, the master lightkeeper Anthony maintained the brass lanterns and mechanical gears that guided merchant vessels through the dangerous jagged rocks.

One chilly autumn evening, Anthony discovered an ancient brass key hidden behind the fog bell. The key unlocked a sealed compartment in the lower tower cellar.

Inside the cellar, Anthony found a dusty leather ledger detailing a forgotten warning: every seventy years, a silent storm called the Phantom Drift sweeps across Moonbay, extinguishing all fire and electrical lights.

To signal ships during the Phantom Drift, Anthony had to trigger a manual mirror array using sunlight reflections and moonlight prisms.

With the help of his granddaughter Clara, Anthony repaired the mirror alignment just as the sky turned pitch black and the town lights vanished.

Together, they flashed the beacon, saving three trading ships from crashing onto the Devil's Reef.
"""

result = generate_questions_from_chapter(
    chapter_title="The Lighthouse of Moonbay",
    chapter_text=sample_story,
    subject="English Literature",
    grade="Grade 8",
    board="General",
    language="English",
    difficulty="Medium"
)

vs = result.get("very_short_questions", [])
sq = result.get("short_questions", [])
lq = result.get("long_questions", [])

print(f"Total Questions: {len(vs)} VS, {len(sq)} S, {len(lq)} L")
assert len(vs) == 3, "Must have 3 Very Short questions"
assert len(sq) == 3, "Must have 3 Short questions"
assert len(lq) == 3, "Must have 3 Long questions"

all_q = vs + sq + lq
topics = [q["related_topic"] for q in all_q]
answers = [q["answer"] for q in all_q]
texts = [q["question_text"] for q in all_q]

print("\nGenerated Topics:")
for idx, t in enumerate(topics, 1):
    print(f" Q{idx} Topic:", t)

print("\nGenerated Questions & Answers Preview:")
for idx, q in enumerate(all_q, 1):
    print(f"\n--- Q{idx} ({q['question_type']}) ---")
    print(" Q:", q["question_text"])
    print(" A:", q["answer"])
    print(" Marks:", q["marks"], "| Marking Points:", q["marking_points"])

unique_answers = len(set(answers))
print(f"\nUnique Answers Count: {unique_answers} / 9")
assert unique_answers == 9, f"All 9 questions MUST have distinct answers! Got {unique_answers}"
assert not any("Overview" in t for t in topics), "Topics should not repeatedly be Overview!"

print("\nALL TEST ASSERTIONS PASSED PERFECTLY!")
