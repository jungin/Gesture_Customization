import os



FRAMES = 37
EPOCHS = 20
BATCH_SIZE = 64

# JESTER_BASE_DIR = "/Users/koh/Research/Codes/Gesture_Customization_github/data/jester"
# CONGD_BASE_DIR = "/Users/koh/Research/Codes/Gesture_Customization_github/data/congd"
JESTER_BASE_DIR = "data/jester"
CONGD_BASE_DIR = "data/congd"

JESTER_INPUT_DIR = os.path.join(JESTER_BASE_DIR, "raw_data")
JESTER_OUTPUT_DIR = os.path.join(JESTER_BASE_DIR, "skeletons")
CONGD_INPUT_DIR = os.path.join(CONGD_BASE_DIR, "raw_data")
CONGD_OUTPUT_DIR = os.path.join(CONGD_BASE_DIR, "skeletons")

MP_HANDS_MODEL = "src/hand_landmarker.task"

JESTER_LABELS = {0: 'Doing other things',
                 1: 'Drumming Fingers',
                 2: 'No gesture',
                 3: 'Pulling Hand In',
                 4: 'Pulling Two Fingers In',
                 5: 'Pushing Hand Away',
                 6: 'Pushing Two Fingers Away',
                 7: 'Rolling Hand Backward',
                 8: 'Rolling Hand Forward',
                 9: 'Shaking Hand',
                 10: 'Sliding Two Fingers Down',
                 11: 'Sliding Two Fingers Left',
                 12: 'Sliding Two Fingers Right',
                 13: 'Sliding Two Fingers Up',
                 14: 'Stop Sign',
                 15: 'Swiping Down',
                 16: 'Swiping Left',
                 17: 'Swiping Right',
                 18: 'Swiping Up',
                 19: 'Thumb Down',
                 20: 'Thumb Up',
                 21: 'Turning Hand Clockwise',
                 22: 'Turning Hand Counterclockwise',
                 23: 'Zooming In With Full Hand',
                 24: 'Zooming In With Two Fingers',
                 25: 'Zooming Out With Full Hand',
                 26: 'Zooming Out With Two Fingers'}


