# utils/config.py

BATCH_SIZE = 128
LR = 1e-2
LOCAL_EPOCHS = 1
ROUNDS = 50

OPTIMIZER = "adam"   # or "sgd"
WEIGHT_DECAY = 1e-4

ATTACK_CLIENT_ID = 0          # which client is malicious
FLIP_FRACTION = 0.4        # 40% label flipping
NUM_CLASSES = 10
subsample_fraction = 0.1