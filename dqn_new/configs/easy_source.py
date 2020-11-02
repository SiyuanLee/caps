"""
This is the example config file
"""

# More one-char representation will be added in order to support
# other objects.
# The following a=10 is an example although it does not work now
# as I have not included a '10' object yet.
a = 10

# This is the map array that represents the map
# You have to fill the array into a (m x n) matrix with all elements
# not None. A strange shape of the array may cause malfunction.
# Currently available object indices are # they can fill more than one element in the array.
# 0: nothing
# 1: wall
# 2: ladder
# 3: coin
# 4: spike
# 5: triangle -------source
# 6: square ------ source
# 7: coin -------- target
# 8: princess -------source
# 9: player # elements(possibly more than 1) filled will be selected randomly to place the player
# unsupported indices will work as 0: nothing

map_array = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 5, 1, 0, 0, 0, 6, 0, 1],
    [1, 0, 0, 0, 1, 9, 9, 9, 9, 9, 1],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1],
    [1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 2, 0, 7, 0, 1],
    [1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 0, 1, 0, 2, 0, 1, 0, 2, 1],
    [1, 2, 0, 1, 0, 2, 8, 1, 0, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# set to true -> win when touching the object
# 0, 1, 2, 3, 4, 9 are not possible
end_game = {
    6: True,
}

rewards = {
    "positive": 5,      # when collecting a coin
    "win": 50,          # endgame (win)
    "negative": -25,    # endgame (die)
    "tick": 0           # living
}

map_config = {
    'map_array': map_array,
    'rewards': rewards,
    'end_game': end_game,
    'init_score': 0,
    'init_lives': 1,  # please don't change, not going to work
}