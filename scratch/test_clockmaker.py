import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from app.ai_engine import generate_questions_from_chapter

clockmaker_story = """
THE CLOCKMAKER OF RIVERTON

1. The Shop at the End of Bell Street
Every afternoon at four, the clock above Riverton's old railway station stopped for exactly one minute.
The stationmaster blamed the wiring, the mayor blamed the weather, and the commuters blamed the railway company. But seventeen-year-old Arjun Mehta, who repaired radios and watches in his uncle's shop, noticed something odd: the clock stopped only when the train from the eastern hills was late.
Riverton had once been famous for its clockmakers. Their clocks stood in schools, temples, offices, and homes across the district. Most workshops had closed, but Uncle Dev's tiny shop on Bell Street still smelled of brass, machine oil, and sandalwood. Arjun spent his evenings there learning how springs, gears, and pendulums turned small movements into reliable time.
One rainy day, a woman in a green coat entered the shop carrying a wooden clock with no hands. She placed it on the counter and asked Dev to repair it before the next full moon. When Dev saw the clock's carved back, his expression changed. He sent Arjun to fetch tea, then spoke quietly with the woman. By the time Arjun returned, she had vanished, leaving the clock and a folded railway map behind.
Inside the clock was a brass plate engraved with a date from thirty years earlier and a sentence: "When every clock disagrees, follow the one that has no hands." Arjun looked at the station clock through the shop window. It had just stopped again.

2. A Map with Missing Tracks
The railway map was old enough that its paper had turned the color of weak tea. Several routes had been crossed out in red ink, including a line that ran from Riverton into the eastern hills. At the edge of the map, someone had drawn a small circle around a station named Nandipur Junction. Arjun had never heard of it.
Uncle Dev finally admitted that Nandipur had been closed after a landslide decades earlier. The official report said the tracks were destroyed and the station abandoned. Yet Dev remembered a clockmaker named Leela Rao who had worked there before the closure. She had built an experimental clock designed to compare time signals from distant railway stations. Then, one winter night, she disappeared along with the station's records.
"Why did the woman bring this clock here?" Arjun asked. Dev shook his head. "Perhaps she wants us to finish something Leela began. But a mystery is not permission to take foolish risks." He promised to contact an old railway engineer while Arjun examined the mechanism.
Beneath the brass plate, Arjun found a narrow compartment containing three tiny gears and a strip of paper. It listed four times—4:00, 4:07, 4:19, and 4:31—and beside each time was a different station code. The last code had been circled twice.

3. The Clock Without Hands
Arjun cleaned the clock carefully and studied its gears. Unlike an ordinary clock, it did not measure hours or minutes. Its mechanism compared the tiny differences between signals arriving from separate places. When the signals matched, a pin would move and reveal a mark on the brass plate. Without the missing railway stations, the clock could not complete its cycle.
His uncle found a notebook in the shop's attic, written by Leela Rao. She had believed that a series of clocks could help railway workers detect a damaged signal cable before it caused a dangerous delay. Her notes described a test that had been interrupted when the eastern route closed. One page ended abruptly: "The discrepancy is not a fault in the clocks. Someone is changing the signal."
Arjun wondered whether the stopped station clock and the late train were connected. He visited the stationmaster, Mrs. Fernandes, who showed him maintenance records. The clock had been serviced repeatedly, but no repair explained why it stopped at four. The same minute appeared in reports of signal failures along the eastern line.
Before leaving, Arjun noticed that the station clock's minute hand trembled whenever the eastern train was delayed. It was not stopping randomly. Something in the signal system was interfering with its timing.

4. The Engineer's Warning
The old railway engineer, Mr. Banerjee, arrived at the shop the next morning carrying a canvas tool bag. He examined the clock without hands and grew quiet. He had worked with Leela and remembered her as a brilliant engineer who insisted that every measurement should be checked twice. He also remembered the night the line closed: the official explanation had never satisfied him.
Banerjee explained that the eastern route had been closed after a landslide damaged a bridge, but the signal network continued to operate for maintenance tests. If someone had altered the timing equipment, the errors might remain invisible until several signals failed together. The clock was designed to reveal those differences, but the railway had never completed the experiment.
He warned Arjun not to enter the abandoned route. The hillside was unstable, and old tunnels could contain loose rock, standing water, or damaged electrical equipment. Instead, they would request permission to inspect the surviving signal records at the depot. Arjun agreed, though he could not stop thinking about the map's circled station.
That evening, the woman in the green coat returned. She introduced herself as Nisha Rao, Leela's granddaughter. She had found her grandmother's notes among family papers and wanted to know why the investigation had been abandoned. She did not know what had happened to Leela, but she believed the missing records could explain it.

5. The Pattern in the Records
With Mrs. Fernandes's permission, Arjun, Nisha, and Banerjee compared archived signal logs. They spread the pages across a table and marked every delay that occurred at four in the afternoon. At first, the incidents looked unrelated. Then Arjun arranged them by station rather than by date, and a pattern emerged: the signal errors moved eastward, one station at a time.
The timing differences were too consistent to be random. A cable fault could cause irregular readings, but these errors followed a repeated sequence. Banerjee suspected that an old test circuit was still sending a signal through the closed route. If its timing had been changed, it could confuse the equipment still connected to the main line.
Nisha found a note in her grandmother's handwriting: "The final comparison must be made at Nandipur, where the old line divides." A second note warned that the original clock should not be connected to the railway network without supervision. Arjun realized that the clock was a measuring instrument, not a magical device. It could help locate the problem, but only if they used it safely.
They presented their findings to the railway operations manager. After reviewing the records, she approved a controlled inspection of the signal room at the depot and arranged for qualified technicians to isolate the old test circuit. The team would not enter the abandoned tunnel; they would begin with the equipment that could be reached safely.
"""

result = generate_questions_from_chapter(
    chapter_title="The Clockmaker of Riverton",
    chapter_text=clockmaker_story,
    subject="English Literature / Mystery",
    grade="Grade 8",
    board="General",
    language="English",
    difficulty="Medium"
)

vs = result.get("very_short_questions", [])
sq = result.get("short_questions", [])
lq = result.get("long_questions", [])

print("=== QUESTION PAPER GENERATION TEST RESULTS ===")
print(f"Total Generated: {len(vs)} Very Short, {len(sq)} Short, {len(lq)} Long Questions")

all_questions = vs + sq + lq

print("\n--- VERY SHORT ANSWER QUESTIONS ---")
for q in vs:
    print(f"Q{q['question_number']}. [{q['related_topic']}] ({q['marks']} Marks)")
    print(f"   Question: {q['question_text']}")
    print(f"   Model Answer: {q['answer']}")
    print(f"   Marking Scheme: {q['marking_points']}")
    print(f"   Expected Length: {q['expected_length']}\n")

print("--- SHORT ANSWER QUESTIONS ---")
for q in sq:
    print(f"Q{q['question_number']}. [{q['related_topic']}] ({q['marks']} Marks)")
    print(f"   Question: {q['question_text']}")
    print(f"   Model Answer: {q['answer']}")
    print(f"   Marking Scheme: {q['marking_points']}")
    print(f"   Expected Length: {q['expected_length']}\n")

print("--- LONG ANSWER QUESTIONS ---")
for q in lq:
    print(f"Q{q['question_number']}. [{q['related_topic']}] ({q['marks']} Marks)")
    print(f"   Question: {q['question_text']}")
    print(f"   Model Answer: {q['answer']}")
    print(f"   Marking Scheme: {q['marking_points']}")
    print(f"   Expected Length: {q['expected_length']}\n")

answers = [q['answer'] for q in all_questions]
topics = [q['related_topic'] for q in all_questions]

print(f"Total Unique Answers: {len(set(answers))} / 9")
print(f"Total Unique Topics: {len(set(topics))} / 9")
