def condition_rock(ip, it, mp, mt, rp, rt, pp, pt):
    if (ip < it) and (mp < mt) and (rp < rt) and (pp < pt):
        return True
    return False
def condition_paper(ip, it, mp, mt, rp, rt, pp, pt):
    if (ip > it) and (mp > mt) and (rp > rt) and (pp > pt):
        return True
    return False
def condition_scissors(ip, it, mp, mt, rp, rt, pp, pt):
    if (ip > it) and (mp > mt) and (rp < rt) and (pp < pt):
        return True
    return False


WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8

MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12

RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20





