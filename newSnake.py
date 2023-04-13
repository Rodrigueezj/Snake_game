import io
import numpy as np
import pyautogui
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
import time


# Esta funcion ejecuta chrome con el link de snake arcade e inicia el juego.
def initChromeLink():
  # Inicializar el controlador de Chrome
  driver = webdriver.Chrome()

  # Abrir la ventana de Chrome
  driver.get("https://www.google.com/fbx?fbx=snake_arcade")
  time.sleep(5)  # Esperar a que la página cargue completamente

  pyautogui.press('space')
  pyautogui.press('right')

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


# Funcion que me devuelve las coordenadas de la cabeza de la serpiente
def get_head(canvas_img, wr, hr):

  # Guardar la imagen resultante
  canvas_img.save("canvas_screenshot.png")


  hsv = canvas_img.convert('HSV')

  lower_white = (0, 0, 200)
  upper_white = (255, 30, 255)
  white_mask = hsv.point(lambda x: 255 if (x >= lower_white and x <= upper_white) else 0, '1')

  """ contours = white_mask.find_contours()
  eye_centers = []
  for cnt in contours:
      x, y, _, _ = cnt.bbox
      eye_center = (int(x + cnt.width / 2), int(y + cnt.height / 2))
      eye_centers.append(eye_center)
  eye1_center = eye_centers[0]
  eye2_center = eye_centers[1]
  distance = np.sqrt((eye2_center[0] - eye1_center[0])**2 + (eye2_center[1] - eye1_center[1])**2)
  head_center = int((eye1_center[0] + eye2_center[0])/2), int((eye1_center[1] + eye2_center[1])/2)
  #print("Head Coordinates: ({}, {})".format(int((head_center[0]+1)//wr) +1 , int((head_center[1]+1)//hr)+1))
  return int((head_center[0]+1)//wr), int((head_center[1]+1)//hr) """

driver = initChromeLink()
canvas_img, widthRow, heightRow = getCanvaData(driver)

dir_array = get_head(canvas_img, widthRow, heightRow)

print('dir array:', dir_array)