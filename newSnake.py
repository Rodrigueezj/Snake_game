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

# Esta funcion ejecuta chrome con el link de snake arcade e inicia el juego.
def initChromeLink():
  # Inicializar el controlador de Chrome
  driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

  # Abrir la ventana de Chrome
  driver.get("https://www.google.com/fbx?fbx=snake_arcade")
  HTML = driver.find_element(By.TAG_NAME, 'html')
  HTML.send_keys(Keys.SPACE) # inicializar el juego
  HTML.send_keys(Keys.ARROW_RIGHT) # la serpiente se empieza a mover

  return driver


#Esta funcion nos permite obtener un screenshot del canva del juego.
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


# Esta es la funcion con A*
def get_path(food1, snake1):
    food1.camefrom = []
    for s in snake1:
      s.camefrom = []
    openset = [snake1[-1]]
    closedset = []
    dir_array1 = []
    while True:
      current1 = min(openset, key=lambda spot: spot.f)
      openset = [openset[i] for i in range(len(openset)) if openset[i] != current1]
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
          neighbor.h = np.sqrt((neighbor.x - food1.x) ** 2 + (neighbor.y - food1.y) ** 2)
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

    return dir_array1


def get_head(canvas_img, widthRow, heightRow):
    # Convertir la imagen a una matriz NumPy si no lo es
    if not isinstance(canvas_img, np.ndarray):
        canvas_img = np.array(canvas_img)
    
    # Asegurarse de que la imagen tenga el formato correcto
    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)
    
    # Definir el rango de colores para detectar la cabeza de la serpiente
    lower_color = np.array([0, 255, 0])
    upper_color = np.array([0, 255, 0])
    
    # Aplicar un filtro para resaltar los colores en el rango especificado
    mask = cv2.inRange(canvas_img, lower_color, upper_color)
    
    # Encontrar los contornos en la imagen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontrar el contorno más grande, que debería ser la cabeza de la serpiente
    max_contour = max(contours, key=cv2.contourArea)
    
    # Encontrar el centro del contorno
    moments = cv2.moments(max_contour)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    
    # Calcular la posición de la cabeza en la matriz del juego
    head_x = int(center_x // widthRow)
    head_y = int(center_y // heightRow)

    print(f'Head Coordinates: ({head_x}, {head_y})')
    cv2.imwrite("canvas_screenshot_head.png", canvas_img)
    return (head_x, head_y)


def get_apple(canvas_img):
  hsv_img = cv2.cvtColor(np.array(canvas_img), cv2.COLOR_RGB2HSV)

  # Definimos los rangos de color para el objeto que queremos detectar (en este caso, rojo)
  lower_red = np.array([0, 50, 50])
  upper_red = np.array([10, 255, 255])

  # Creamos una máscara con los píxeles dentro del rango de color especificado
  mask = cv2.inRange(hsv_img, lower_red, upper_red)

  # Encontramos los contornos en la máscara de la imagen
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Dibujamos rectángulos y etiquetas alrededor de los objetos detectados
  draw = ImageDraw.Draw(canvas_img)
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    draw.rectangle((x, y, x+w, y+h), outline=(0, 255, 0), width=2)
    draw.text((x, y), 'Apple', fill=(0, 0, 255))

    # Imprimimos las coordenadas de cada objeto detectado
    print(f'Apple Coordinates: ({int((x-28) // 32)}, {int((y-25) // 32)})')

  # Devolvemos la última coordenada encontrada
  canvas_img.save("canvas_screenshot_apple.png")
  return int((x-28) // 32), int((y-25) // 32)


driver = initChromeLink()
canvas_img, widthRow, heightRow = getCanvaData(driver)

apple_cords = get_apple(canvas_img)
head_cords = get_head(canvas_img, widthRow, heightRow)

#dir_array = get_path(apple_cords, head_cords)
