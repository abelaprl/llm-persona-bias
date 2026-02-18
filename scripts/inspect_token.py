from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extractor34

tokens = ['180{,}000', '180,000', ' 180 000', '180\,000']
for t in tokens:
    s = t
    print('\nTOKEN:', repr(t))
    s1 = s.replace('\u2009', '').replace('\u202f', '').replace('\xa0', '')
    s2 = s1.replace('\\,', ',').replace('{,}', ',')
    s3 = s2.replace(' ', '')
    s4 = s3.replace('$', '')
    print('after unicode strip:', repr(s1))
    print('after escaped comma/brace replace:', repr(s2))
    print('after space remove:', repr(s3))
    # grouping detection
    import re
    grp = re.search(r"\d{1,3}(?:([,\\\u2009\\u202f])\d{3})+", s3)
    print('group match:', grp)
    s_num = re.sub(r'[,\\\u2009\\u202f]', '', s3)
    print('s_num:', s_num)
    m = re.search(r'([+-]?\d+\.?\d*)', s_num)
    print('first number match in s_num:', m.group(1) if m else None)
    try:
        print('float:', float(m.group(1)))
    except:
        pass
    # fallback
    s2fallback = re.sub(r'[^0-9\.-]', '', s4)
    m2 = re.search(r'([+-]?\d+\.?\d*)', s2fallback)
    print('fallback s2:', s2fallback, 'match:', m2.group(1) if m2 else None)
    try:
        print('fallback float:', float(m2.group(1)))
    except:
        pass
