import time
import numpy as np
import cv2, copy
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

def get_coordinates(background, image):
    # Perform template matching for the image
    result = cv2.matchTemplate(background, image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    x = max_loc[0] + (image.shape[0] // 2)
    y = max_loc[1] + (image.shape[1] // 2)
    return x, y

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get('https://www.google.com/fbx?fbx=snake_arcade')
HTML = driver.find_element(By.TAG_NAME, 'html')
time.sleep(1)
HTML.send_keys(Keys.SPACE)
time.sleep(2)

#Height 530px 
#Width 600px 
canvas = driver.find_element(By.CSS_SELECTOR, '.cer0Bd')

# Capture screenshot of entire browser window
screenshot_png = driver.get_screenshot_as_png()

# Convert PNG data to NumPy array
screenshot_array = np.frombuffer(screenshot_png, dtype=np.uint8)
screenshot_img = cv2.imdecode(screenshot_array, cv2.IMREAD_GRAYSCALE)

# Get location and dimensions of canvas element
canvas_location = canvas.location
canvas_size = canvas.size

# Crop screenshot to just the canvas element
canvas_img = screenshot_img[canvas_location['y']:canvas_location['y']+canvas_size['height'],
                            canvas_location['x']:canvas_location['x']+canvas_size['width']]

# Load image of apple 32px
apple_img = cv2.imread('apple.png', cv2.IMREAD_GRAYSCALE)

# Load image of snake i guess 38x24px
snake_img = cv2.imread('snake.png', cv2.IMREAD_GRAYSCALE)

#----------------------------------------------------------------------------------------------------

apple_pos = get_coordinates(canvas_img, apple_img)
#head_pos = get_coordinates(canvas, snake_img)


cv2.destroyAllWindows()