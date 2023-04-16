import time
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys

cols = 15
rows = 17
done = False


class SnakeBoard:
    def cleanBoard(self):
        self.board = np.zeros((rows, cols))

    def printBoard(self):
        print(self.board)

    def updateHead(self, coords):
        self.board[coords[0]][coords[1]] = 1
        self.head = coords

    def updateApple(self, coords):
        self.board[coords[0]][coords[1]] = 2
        self.head = coords


# Esta funcion ejecuta chrome con el link de snake arcade e inicia el juego.
def initChromeLink():
    # Inicializar el controlador de Chrome
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Abrir la ventana de Chrome
    driver.get("https://www.google.com/fbx?fbx=snake_arcade")
    HTML = driver.find_element(By.TAG_NAME, 'html')
    HTML.send_keys(Keys.SPACE)  # inicializar el juego
    HTML.send_keys(Keys.ARROW_RIGHT)  # la serpiente se empieza a mover

    return driver


# Esta funcion nos permite obtener un screenshot del canva del juego.
def getCanvaData(driver):
    # Encontrar el elemento canvas
    canvas = driver.find_element(By.CSS_SELECTOR, '.cer0Bd')

    # Obtener la posición y tamaño del canvas
    location = canvas.location
    size = canvas.size

    # Capturar una captura de pantalla de la ventana de Chrome
    screenshot = driver.get_screenshot_as_png()

    # Convertir la captura de pantalla en una imagen de Pillow
    img = Image.open(io.BytesIO(screenshot))

    # Recortar la sección de la imagen que contiene el canvas
    left = location['x']
    top = location['y']
    right = left + size['width']
    bottom = top + size['height']
    canvas_img = img.crop((left, top, right, bottom))

    widthRow = size['width'] / 17
    heightRow = size['height'] / 15

    return canvas_img, widthRow, heightRow


def get_head(canvas_img):
    hsv_img = cv2.cvtColor(np.array(canvas_img), cv2.COLOR_RGB2HSV)

    # Definimos los rangos de color para el objeto que queremos detectar (en este caso, rojo)
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([255, 30, 255])

    # Creamos una máscara con los píxeles dentro del rango de color especificado
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Encontramos los contornos en la máscara de la imagen
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos rectángulos y etiquetas alrededor de los objetos detectados
    draw = ImageDraw.Draw(canvas_img)

    for contour in contours:
        x_head, y_head, w_head, h_head = cv2.boundingRect(contour)
        time.sleep(.1)

        draw.rectangle((x_head, y_head, x_head+w_head, y_head +
                       h_head), outline=(0, 255, 0), width=2)
        draw.text((x_head, y_head), 'Head', fill=(0, 0, 255))

    # Imprimimos las coordenadas de cada objeto detectado
    print(
        f'Head Coordinates: ({int((x_head-28) // 32)}, {int((y_head-25) // 32)})')

    # Devolvemos la última coordenada encontrada
    canvas_img.save("canvas_screenshot_head.png")
    return int((x_head-28) // 32), int((y_head-25) // 32)


def get_apple(canvas_img, head_coords):
    hsv_img = cv2.cvtColor(np.array(canvas_img), cv2.COLOR_RGB2HSV)

    # Definimos los rangos de color para el objeto que queremos detectar (en este caso, rojo)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Creamos una máscara con los píxeles dentro del rango de color especificado
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Encontramos los contornos en la máscara de la imagen
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos rectángulos y etiquetas alrededor de los objetos detectados
    draw = ImageDraw.Draw(canvas_img)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        time.sleep(.1)

        if (len(contours) > 1):
            if (head_coords == (x - 1, y) or head_coords == (x, y + 1) or head_coords == (x + 1, y) or head_coords == (x, y - 1)):
                print('tongue')
            else:
                x_apple = x
                y_apple = y
        else:
            x_apple = x
            y_apple = y

        draw.rectangle((x_apple, y_apple, x_apple+w, y_apple+h),
                       outline=(0, 255, 0), width=2)
        draw.text((x_apple, y_apple), 'Apple', fill=(0, 0, 255))

      # Imprimimos las coordenadas de cada objeto detectado
    print(
        f'Apple Coordinates: ({int((x_apple-28) // 32)}, {int((y_apple-25) // 32)})')

    # Devolvemos la última coordenada encontrada
    canvas_img.save("canvas_screenshot_apple.png")
    return int((x_apple-28) // 32), int((y_apple-25) // 32)


driver = initChromeLink()
canvas_img, widthRow, heightRow = getCanvaData(driver)
head_coords = get_head(canvas_img)
apple_coords = get_apple(canvas_img, head_coords)

snakeBodar = SnakeBoard()
snakeBodar.cleanBoard()
snakeBodar.updateHead(head_coords)
snakeBodar.updateApple(apple_coords)
snakeBodar.printBoard()
