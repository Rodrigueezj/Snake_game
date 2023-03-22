import time
import pygame
from pygame import display, time, draw, QUIT, init, KEYDOWN, K_a, K_s, K_d, K_w
from pygame import time
import numpy as np
import cv2
from numpy import sqrt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

done = False
cols = 17
rows = 19
width = 600
height = 530
wr = width/rows
hr = height/cols
direction = 1
screen = display.set_mode([width, height])
display.set_caption("snake_self")
clock = time.Clock()

def get_path(food1, snake1):
    food1.camefrom = []
    for s in snake1:
        s.camefrom = []
    openset = [snake1[-1]]
    closedset = []
    dir_array1 = []
    while 1:
        current1 = min(openset, key=lambda x: x.f)
        openset = [openset[i] for i in range(len(openset)) if not openset[i] == current1]
        closedset.append(current1)
        for neighbor in current1.neighbors:
            if neighbor not in closedset and neighbor not in snake1:
                tempg = neighbor.g + 1
                if neighbor in openset:
                    if tempg < neighbor.g:
                        neighbor.g = tempg
                else:
                    neighbor.g = tempg
                    openset.append(neighbor)
                neighbor.h = sqrt((neighbor.x - food1.x) ** 2 + (neighbor.y - food1.y) ** 2)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.camefrom = current1
        if current1 == food1:
            break
    while current1.camefrom:
        if current1.x == current1.camefrom.x and current1.y < current1.camefrom.y:
            dir_array1.append(2)
        elif current1.x == current1.camefrom.x and current1.y > current1.camefrom.y:
            dir_array1.append(0)
        elif current1.x < current1.camefrom.x and current1.y == current1.camefrom.y:
            dir_array1.append(3)
        elif current1.x > current1.camefrom.x and current1.y == current1.camefrom.y:
            dir_array1.append(1)
        current1 = current1.camefrom
    for i in range(rows):
        for j in range(cols):
            grid[i][j].camefrom = []
            grid[i][j].f = 0
            grid[i][j].h = 0
            grid[i][j].g = 0
    return dir_array1

def get_head():
    # Capture screenshot of entire browser window
    screenshot_png = driver.get_screenshot_as_png()
    # Convert PNG data to NumPy array
    screenshot_array = np.frombuffer(screenshot_png, dtype=np.uint8)
    screenshot_img = cv2.imdecode(screenshot_array, cv2.IMREAD_COLOR)
    # Get location and dimensions of canvas element
    canvas_location = canvas.location
    canvas_size = canvas.size
    # Crop screenshot to just the canvas element
    canvas_img = screenshot_img[canvas_location['y']:canvas_location['y']+canvas_size['height'],
                                canvas_location['x']:canvas_location['x']+canvas_size['width']]
    hsv = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eye_centers = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        eye_center = (int(x + w/2), int(y + h/2))
        eye_centers.append(eye_center)
    eye1_center = eye_centers[0]
    eye2_center = eye_centers[1]
    distance = np.sqrt((eye2_center[0] - eye1_center[0])**2 + (eye2_center[1] - eye1_center[1])**2)
    head_center = int((eye1_center[0] + eye2_center[0])/2), int((eye1_center[1] + eye2_center[1])/2)
    #print("Head Coordinates: ({}, {})".format(int((head_center[0]+1)//wr) +1 , int((head_center[1]+1)//hr)+1))
    return int((head_center[0]+1)//wr) +1 , int((head_center[1]+1)//hr)+1
    cv2.imshow('Head Detection', canvas_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_apple():
    # Capture screenshot of entire browser window
    screenshot_png = driver.get_screenshot_as_png()
    # Convert PNG data to NumPy array
    screenshot_array = np.frombuffer(screenshot_png, dtype=np.uint8)
    screenshot_img = cv2.imdecode(screenshot_array, cv2.IMREAD_COLOR)
    # Get location and dimensions of canvas element
    canvas_location = canvas.location
    canvas_size = canvas.size
    # Crop screenshot to just the canvas element
    canvas_img = screenshot_img[canvas_location['y']:canvas_location['y']+canvas_size['height'],
                                canvas_location['x']:canvas_location['x']+canvas_size['width']]
    hsv = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas_img,"Apple",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        print("Apple Coordinates: ({}, {})".format(int((x+1)//wr)+1, int((y+1)//hr)+1))
    return int((x+1)//wr), int((y+1)//hr)
        # cv2.imshow('Apple Detection', canvas_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

class Spot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.camefrom = []

    def show(self, color):
        draw.rect(screen, color, [self.x*hr+2, self.y*wr+2, hr-4, wr-4])


    def add_neighbors(self):
        if self.x >= 1:
            self.neighbors.append(grid[self.x - 1][self.y])
        if self.y >= 1:
            self.neighbors.append(grid[self.x][self.y - 1])
        if self.x <= rows - 2:
            self.neighbors.append(grid[self.x + 1][self.y])
        if self.y <= cols - 2:
            self.neighbors.append(grid[self.x][self.y + 1])

grid = [[Spot(i, j) for j in range(cols)] for i in range(rows)]

for i in range(rows):
    for j in range(cols):
        grid[i][j].add_neighbors()

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get('https://www.google.com/fbx?fbx=snake_arcade')
driver.maximize_window()
HTML = driver.find_element(By.TAG_NAME, 'html')
HTML.send_keys(Keys.SPACE) # inicializar el juego
HTML.send_keys(Keys.ARROW_RIGHT) # la serpiente se empieza a mover
canvas = driver.find_element(By.CSS_SELECTOR, '.cer0Bd')

#snake = [grid[4][9],grid[5][9],grid[6][9]]
snake = [grid[6][9]]
food = grid[14][9] # crea una nueva comida una vez que ya se la comió
current = snake[-1] # cabeza de la serpiente
dir_array = get_path(food, snake)
food_array = [food]

while not done:
    clock.tick(6.8)
    screen.fill(BLACK)

    direction = dir_array.pop(-1) #devuelve el último valir de la lista y lo elimina
    if direction == 0:    # down
        snake.append(grid[current.x][current.y + 1])
        HTML.send_keys(Keys.ARROW_DOWN)
    elif direction == 1:  # right
        snake.append(grid[current.x + 1][current.y])
        HTML.send_keys(Keys.ARROW_RIGHT)
    elif direction == 2:  # up
        snake.append(grid[current.x][current.y - 1])
        HTML.send_keys(Keys.ARROW_UP)
    elif direction == 3:  # left
        HTML.send_keys(Keys.ARROW_LEFT)
        snake.append(grid[current.x - 1][current.y])

    current = snake[-1] # cabeza de la serpiente
    if current.x == food.x and current.y == food.y:
        #break
        food_cor = get_apple()
        food = grid[food_cor[0]][food_cor[1]] # crea una nueva comida una vez que ya se la comió
        food_array.append(food)
        dir_array = get_path(food, snake)
    else:
        snake.pop(0)# elimina la cola

    for spot in snake:
        spot.show(WHITE)


    food.show(GREEN)
    snake[-1].show(BLUE)
    display.flip()
    for event in pygame.event.get():
        if event.type == QUIT:
            done = True
        elif event.type == KEYDOWN:
            if event.key == K_w and not direction == 0:
                direction = 2
            elif event.key == K_a and not direction == 1:
                direction = 3
            elif event.key == K_s and not direction == 2:
                direction = 0
            elif event.key == K_d and not direction == 3:
                direction = 1