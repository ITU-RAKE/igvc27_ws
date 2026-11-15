"""camera_map.white_mask testleri: beyaz cizgi engel, asfalt / renkli / gorulmeyen degil."""
import cv2
import numpy as np

from igvc_vision.camera_map import white_mask

LO, HI = np.array([0, 0, 160], np.uint8), np.array([179, 60, 255], np.uint8)
K = (61, 61)                                   # 1.2 m / 2 cm


def _hsv(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _scene(asphalt=45, line=235, n=200):
    img = np.full((n, n, 3), asphalt, np.uint8)
    img[:, 98:102] = line                       # 4 px = 8 cm dikey serit
    return img


def test_line_on_dark_asphalt():
    m = white_mask(_hsv(_scene()), np.ones((200, 200), bool), LO, HI, 40, K)
    assert m[:, 98:102].min() == 255
    assert m[:, :95].max() == 0 and m[:, 105:].max() == 0


def test_bright_asphalt_rejected_by_contrast():
    """Gunesli asfalt (V 180) HSV esigini gecer ama zeminden farki yok: engel degil."""
    img = _scene(asphalt=180, line=250)
    valid = np.ones((200, 200), bool)
    assert white_mask(_hsv(img), valid, LO, HI, 0, K)[:, :90].min() == 255   # kontrast kapali: hepsi
    m = white_mask(_hsv(img), valid, LO, HI, 40, K)
    assert m[:, :90].max() == 0
    # serit 70 parlak: yine de bulunur
    assert m[:, 98:102].min() == 255


def test_colored_and_unseen_not_white():
    img = _scene()
    img[20:40, 20:40] = (0, 0, 230)              # kirmizi duba (doygun)
    valid = np.ones((200, 200), bool)
    valid[:, 150:] = False
    img[:, 150:] = 0                             # gorulmeyen: siyah
    m = white_mask(_hsv(img), valid, LO, HI, 40, K)
    assert m[20:40, 20:40].max() == 0
    assert m[:, 140:].max() == 0                 # gorulmeyen kenari asfalti parlak gostermez


def test_pothole_is_obstacle():
    """2 ft (30 px) beyaz daire, pencere 1.2 m: arka plan ortalamasi icinde kalsa da bulunur."""
    img = np.full((200, 200, 3), 45, np.uint8)
    cv2.circle(img, (100, 100), 15, (235, 235, 235), -1)
    m = white_mask(_hsv(img), np.ones((200, 200), bool), LO, HI, 40, K)
    assert m[100, 100] == 255 and m[100, 86] == 255
