from random import sample,choice
from collections import deque
from enum import Enum

class GridType(Enum):
    SPACE = 0
    FOOD = 1
    SNAKE = 2
    HEAD = 3

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake(object):
    def __init__(self,size,max_food,min_food):
        self.head_dir = Direction.UP
        self.game_size = size
        self.max_food = max_food
        self.min_food = min_food
        self.board = [[GridType.SPACE for j in range(size)] for i in range(size)]
        self.board[size/2][size/2] = GridType.HEAD
        self.board[size/2 + 1][size/2] = GridType.SNAKE
        for f in range(self.min_food): self.generate_food()

    def get_spaces(self):
        spaces = [[(i,j) for j in range(self.game_size) if self.board[i][j] == GridType.SPACE] for i in range(self.game_size)]
        return reduce(lambda a,b: a + b,spaces)

    def generate_food(self):
        spaces = self.get_spaces()
        if len(spaces) == 0: return False
        x,y = choice(spaces)
        self.board[x][y] = GridType.FOOD
        return True

    def get_head(self):
        for i in range(self.game_size):
            for j in range(self.game_size):
                if self.board[i][j] == GridType.HEAD:
                    return (i,j)
        assert False # should not happen

    def get_front(self,direction):
        next_loc = self.get_head()
        if direction == Direction.UP:
            next_loc[0] = next_loc[0] - 1
        elif direction == Direction.DOWN:
            next_loc[0] = next_loc[0] + 1
        elif direction == Direction.RIGHT:
            next_loc[1] = next_loc[1] + 1
        elif direction == Direction.LEFT:
            next_loc[1] = next_loc[1] - 1
        else:
            assert False
        x,y = next_loc
        return self.board[x][y]

    def run(self,direction):
        pass #TODO: this

