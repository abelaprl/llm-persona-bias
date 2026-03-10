import sys; sys.path.insert(0, 'src')
from rebuild_results_csv_llama import extract_final_answer

# Test Bug 1: 'answers' (verb) should NOT trigger Answer marker
test1 = 'Amy correctly answers 80 of the 10 questions, so she answers 0.8 x 10 = 8 correctly.\nTherefore, Amy scores 41 points on the history test.'
r1 = extract_final_answer(test1)
print(f'Bug1 (expect 41): {r1}')

# Test Bug 2: setup phrase guard
test2 = 'To find the answer, we need to first calculate the number of rooms.\nTherefore, it would take 15 hours.'
r2 = extract_final_answer(test2)
print(f'Bug2 (expect 15): {r2}')

# Test Bug 3: P1 no infinitive fallback (needs "So," to trigger Rule 3 weak marker)
test3 = 'So, Bubbles should buy 6 stuffed goats to make up 30% of all her stuffed animals.'
r3 = extract_final_answer(test3)
print(f'Bug3 (expect 6): {r3}')

# Test Bug 4: Rule 5c >3 numbers with =
test4 = 'In total, Kimo spends 9 + 8 = 17 hours per week attending classes.\nSince there are 16 weeks in a semester, Kimo spends a total of 17 x 16 = 272 hours attending classes in one semester.'
r4 = extract_final_answer(test4)
print(f'Bug4 (expect 272): {r4}')

# Test Bug 5: P0c "which is" skip (needs "Therefore" to trigger Rule 3 strong marker)
test5 = 'Therefore, Sandra needs to babysit 5 times to get at least 45 eggs, which is enough to make 15 Spanish flans.'
r5 = extract_final_answer(test5)
print(f'Bug5 (expect 5): {r5}')

# Test Bug 6: P0e "for a total of N"
test6 = 'Kate needs to buy 3 cheese pizzas and 3 pepperoni pizzas, for a total of 6 pizza pies.'
r6 = extract_final_answer(test6)
print(f'Bug6 (expect 6): {r6}')
