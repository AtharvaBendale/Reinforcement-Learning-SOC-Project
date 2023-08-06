# Constants used for UCB (Upper Confidence Bound) Evaluation
C_base = 2000
C_init = 2

# Number of times the MCTS algorithm will run for any state encountered
MCTS_EPOCHS = 400

# Number of Residual Layers in the Neural Network
NUM_HIDDEN_LAYERS = 20

# The number of times program will perform selfplay and use it for training
SELFPLAY_BATCH_SIZE = 2

# The number of times the program will do [ selfplay + training ]
LEARNING_EPOCHS = 2

# Epsilon for exploratory behaviour (High because the model is still in initial stages of training)
EPSILON = 0.75

# Counters for storing the selfplay data
CURR_BATCH_NUMBER = 3
CURR_SINGLE_BATCH_NUMBER = 13