"""Bahan daftar kata kerja jawaban. Rule pack menyusunnya sendiri (K1)."""

BASE = (
    r'will\s+(?:take|cost|last|be|earn|save|spend|make|have|need|weigh|give|receive|pay|create|score)|'
    r'would\s+(?:take|cost|last|be|earn|save|spend|make|have|need|weigh|pay|create|score)|'
    r'is|are|was|were|costs?|earns?|saves?|takes?|makes?|needs?|weighs?|spends?|'
    r'pays?|paid|creates?|created|scores?|scored|spent|'
    r'remains?|add\s+up\s+to'
)

# Tambahan kata kerja naratif — dipakai model yang menutup dengan kalimat cerita
NARRATIVE = (
    r'has|have|had|earned?|got|gotten|collected?|picked?|won|bought|ran|walked?|watched?|'
    r'finished?|completed?|solved?|reached?|arrived?|found|learns?|learned?|grew|'
    r'planted?|painted?|wrote|written|drew|sang|cooked?|baked?|cleaned?|fixed?|built|'
    r'sold|hired?|owned?|holds?|held|visited?|drove|driven|flew|flown|swam|swum|'
    r'brought|kept|uses?|used|handles?|handled|drinks?|drank|receives?|received|'
    r'traveled|travels?|produced?|produces?|reads?|read|worked|works?|'
    r'consumed|consumes?|distributed|delivers?|delivered|printed|prints?|'
    r'carried|carries?|gathered|gathers?|processed|processes?|answered|answers?|'
    r'calculated|calculates?'
)

MODAL = r'can\s+\w+|could\s+\w+|must\s+\w+'
