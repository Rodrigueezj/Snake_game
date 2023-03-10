import pygame
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


HEIGHT = 500
WIDTH = 500
ROWS = 20
COLS = 20
SQSIZE = WIDTH // ROWS
BLACK = 0, 0, 0
GREEN = 124, 252, 0
RED = 255, 160, 122
WHITE = 255, 255, 255

pygame.init()
surface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tricky')
surface.fill(BLACK)


class Snake:
    def __init__(self):
        self.body = [(8, 3 + square) for square in range(4)]
        self.direction = (0, 1)
        self.add = False
        self.head = self.body[len(self.body)-1]

    def draw(self):
        for square in self.body:
            col, row = square
            pygame.draw.rect(surface, GREEN, (row * SQSIZE,
                             col * SQSIZE, SQSIZE, SQSIZE))

    def erase(self):
        for square in self.body:
            col, row = square
            pygame.draw.rect(surface, BLACK, (row * SQSIZE,
                             col * SQSIZE, SQSIZE, SQSIZE))

    def move(self):
        # body_copy = [(self.direction[0]+a, self.direction[1]+b) for (a, b) in self.body]
        # self.body = body_copy
        if (not self.add):
            body_copy = self.body[1:]
        else:
            body_copy = self.body[:]
            self.add = False
        body_copy.append((body_copy[len(body_copy)-1][0]+self.direction[0],body_copy[len(body_copy)-1][1]+self.direction[1]))
        self.body = body_copy
        self.head = self.body[len(self.body)-1]


class Apple:
    def __init__(self):
        self.generate()

    def generate(self):

        
        self.x = random.randrange(1, ROWS - 1)
        self.y = random.randrange(1, COLS - 1)
        self.pos = (self.y, self.x)

    def draw(self):
        pygame.draw.rect(surface, RED, (self.x * SQSIZE,
                         self.y * SQSIZE, SQSIZE, SQSIZE))

    def collision(self, snake):
        if snake.head == self.pos:
            self.generate()
            snake.add = True

            for bloque in snake.body[1:]:
                if self.pos == bloque:
                    self.generate()

            return True


class Board:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLS))
        self.empty_squares = self.squares

class Game:
    def __init__(self):
        self.snake = Snake()
        self.board = Board()
        self.apple = Apple()
        self.score = 0
        self.draw_edges()
        self.graph = nx.Graph()
        self.draw_apple()
        self.draw_snake()
        

    def grid_to_graph(self):
        graph = nx.Graph()
        for row in range(self.board.squares.shape[0]):
            for col in range(self.board.squares.shape[1]):
                if self.board.squares[row, col] != 3:  # Ignore the edges of the self.board.squares
                    graph.add_node((row, col))
                    if row > 0 and self.board.squares[row - 1, col] != 3:
                        graph.add_edge((row, col), (row - 1, col))
                    if row < self.board.squares.shape[0] - 1 and self.board.squares[row + 1, col] != 3:
                        graph.add_edge((row, col), (row + 1, col))
                    if col > 0 and self.board.squares[row, col - 1] != 3:
                        graph.add_edge((row, col), (row, col - 1))
                    if col < self.board.squares.shape[1] - 1 and self.board.squares[row, col + 1] != 3:
                        graph.add_edge((row, col), (row, col + 1))
        return graph


    def draw_apple(self):
        self.board.squares[self.apple.x][self.apple.y] = 2

    def draw_snake(self):
        for segment in self.snake.body:
            row, col = segment
            self.board.squares[row][col] = 1
        """ print(self.board.squares) """

    def erase_snake(self):
        for segment in self.snake.body:
            row, col = segment
            self.board.squares[row][col] = 0

    def draw_edges(self):
        for row in range(ROWS):
            self.board.squares[row][0] = 3
            self.board.squares[row][COLS-1] = 3
            pygame.draw.rect(surface, WHITE, (row * SQSIZE,
                             0 * SQSIZE, SQSIZE, SQSIZE))
            pygame.draw.rect(surface, WHITE, (row * SQSIZE,
                             WIDTH - SQSIZE, SQSIZE, SQSIZE))
            
        for col in range(COLS):
            self.board.squares[0][col] = 3
            self.board.squares[ROWS-1][col] = 3
            pygame.draw.rect(surface, WHITE, (0 * SQSIZE,
                             col * SQSIZE, SQSIZE, SQSIZE))
            pygame.draw.rect(surface, WHITE, (HEIGHT - SQSIZE,
                             col * SQSIZE, SQSIZE, SQSIZE))

    def game_over(self):
        row, col = self.snake.head
        if self.board.squares[row][col] == 3: return True

        for bloque in self.snake.body[:-1]:
            if self.snake.head == bloque:
                return True

    def increaseScore(self):
        self.score += 1


def main():
    game = Game()
    fps = pygame.time.Clock()

    while True:
        fps.tick(5)
        snake = game.snake
        apple = game.apple
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != (1, 0):
                    snake.direction = (-1, 0)
                if event.key == pygame.K_DOWN and snake.direction != (-1, 0):
                    snake.direction = (1, 0)
                if event.key == pygame.K_LEFT and snake.direction != (0, 1):
                    snake.direction = (0, -1)
                if event.key == pygame.K_RIGHT and snake.direction != (0, -1):
                    snake.direction = (0, 1)

        game.erase_snake()
        snake.erase()
        snake.move()
        game.draw_snake()
        snake.draw()
        apple.draw()
        game.draw_edges()
        game.graph = game.grid_to_graph()
        print(game.board.squares)
        
        if apple.collision(game.snake):
            game.increaseScore()
            print('score: ', game.score)

        pygame.display.update()
        nx.astar_path_length
        if game.game_over():
            quit()

main()
