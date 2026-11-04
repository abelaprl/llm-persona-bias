import unittest
from structure_cues import counts

class StructureCuesTest(unittest.TestCase):
    def test_preserved_cues_when_lines_are_removed(self):
        text='### Step 1: Count items\n- **Total items:** 20\n### Step 2: Add costs\n1. Count items.\n2. Add costs.'
        expected={'step_label':2,'numbered_sequence':2,'labeled_bullet':1,'hash_heading':2}
        self.assertEqual(counts(text),expected)
        self.assertEqual(counts(' '.join(text.split())),expected)

    def test_math_and_cross_reference_are_not_lists(self):
        for text in ['Profit = Selling price - Cost price',
                     'The price is 1.25 and the number is 2.5.',
                     'We use the amount found in step 1. Then add two.',
                     'Substitute the expression into the equation from step 4: R + Y = 37.',
                     'Using the equation from Step 3: x = y + 7.',
                     'The answer is **12**.', '#### 12',
                     'Multiply 3 * 4 and subtract 6 - 2.']:
            self.assertFalse(any(counts(text).values()),text)

    def test_single_number_is_not_sequence(self):
        self.assertEqual(counts('The result is 1. Therefore stop.')['numbered_sequence'],0)

    def test_restarted_numbered_lists(self):
        self.assertEqual(counts('1. First. 2. Second. 1. New list. 2. Next.')['numbered_sequence'],4)

    def test_plain_bold_is_not_heading(self):
        self.assertEqual(counts('This is **important** and **Answer: 12**.')['hash_heading'],0)

if __name__=='__main__':unittest.main()
