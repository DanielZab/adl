PREVIOUS_DATA_POINTS_AMOUNT = 5
MAX_BUY_LIMIT = 10
TRANSACTION_STEP = 1
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
EPSILON = 0.1 # exploration factor

START_BALANCE = 1000
MEAN_RUN_DURATION = 100 # Set to None to exhaust all data points
STD_RUN_DURATION = 10 # Set to None to exhaust all data points

TRUNCATION_PENALTY = 10

CONTINUOUS_MODEL = True # Determines whether the action space of the model is continuous or discrete

assert isinstance(TRANSACTION_STEP, int)