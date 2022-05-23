"""Sort utilities"""
import re

__nums = re.compile('[0-9]+')


def key_str_with_int(orig: str, padding=5):
    """Pad every number inside orig.

    - orig: original string
    - padding: number of position to pad (default 5)"""
    parts = []
    last_pos = 0
    for match in __nums.finditer(orig):
        s_pos, end_pos = match.span()
        if s_pos > 0:
            parts.append(orig[last_pos:s_pos])
        parts.append(match.group().zfill(padding))
        last_pos = end_pos
    if last_pos < len(orig):
        parts.append(orig[last_pos:])
    return ''.join(parts)


if __name__ == '__main__':
    _strings = ['31_10', '31_1', '32_1', '31_20', '31_3', 'aa_4', 'aa_5', 'aa_40_bb', 'cc']
    for _s in _strings:
        print(key_str_with_int(_s))
    print(sorted(_strings, key=key_str_with_int))

