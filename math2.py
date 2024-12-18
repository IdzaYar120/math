# Імпортуємо необхідні бібліотеки
import cvzone  # Для роботи з комп'ютерним зором і розпізнаванням рук
import cv2  # Основна бібліотека для обробки зображень
from cvzone.HandTrackingModule import HandDetector  # Модуль для відстеження рук
import numpy as np  # Робота з масивами
import google.generativeai as genai  # Інтеграція з Google Gemini AI
from PIL import Image  # Робота із зображеннями (перетворення формату)
import streamlit as st  # Для створення веб-інтерфейсу

# Налаштування сторінки Streamlit
st.set_page_config(layout="wide")
st.image('D:/Programs/math2/.venv/MathGestures.png')  # Відображення банера на сторінці

# Розподіл інтерфейсу на дві колонки
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)  # Checkbox для ввімкнення/вимкнення застосунку
    FRAME_WINDOW = st.image([])  # Вікно для відображення відео з вебкамери

with col2:
    st.title("Відповідь")  # Заголовок для результатів
    output_text_area = st.subheader("")  # Місце для текстового виводу від ШІ

# Ініціалізація Google Gemini AI
genai.configure(api_key="AIzaSyAozNCqiFI-P4SkPiLpM7bpBJ3M70IpRt4")  # API-ключ для доступу до сервісу Google Gemini
model = genai.GenerativeModel('gemini-1.5-flash')  # Використовуємо модель Gemini 1.5

# Ініціалізація вебкамери
cap = cv2.VideoCapture(0)  # Використання камери (зазвичай 0 — вбудована, 1 — зовнішня)
cap.set(3, 1920)  # Встановлення ширини кадру
cap.set(4, 1080)  # Встановлення висоти кадру

# Ініціалізація HandDetector для розпізнавання рук
detector = HandDetector(
    staticMode=False,  # Динамічний режим (оптимізовано для реального часу)
    maxHands=1,  # Максимум одна рука
    modelComplexity=1,  # Складність моделі
    detectionCon=0.7,  # Поріг впевненості для виявлення руки
    minTrackCon=0.5  # Мінімальна впевненість для відстеження
)


def getHandInfo(img):
    """
    Розпізнає руки в кадрі та повертає інформацію про положення пальців і координати.

    Параметри:
    - img: Кадр із вебкамери.

    Повертає:
    - fingers: Список станів пальців (1 — піднятий, 0 — опущений).
    - lmList: Список координат 21 точки руки.
    """
    # Знаходить руки в кадрі
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Якщо рука знайдена, повертає інформацію
    if hands:
        hand = hands[0]  # Беремо першу знайдену руку
        lmList = hand["lmList"]  # Список координат ключових точок
        fingers = detector.fingersUp(hand)  # Список станів пальців
        print(fingers)  # Виводимо для перевірки
        return fingers, lmList
    else:
        # Якщо руки не знайдено, повертає None
        return None


def draw(info, prev_pos, canvas):
    """
    Малює лінії на полотні залежно від жестів руки.

    Параметри:
    - info: Інформація про руку (стан пальців і координати).
    - prev_pos: Попередня позиція пальця (для малювання ліній).
    - canvas: Зображення-полотно для малювання.

    Повертає:
    - current_pos: Поточна позиція пальця.
    - canvas: Оновлене полотно.
    """
    fingers, lmList = info  # Отримуємо стан пальців і координати
    current_pos = None  # Початкова позиція

    # Якщо жест "вказівний палець піднятий" — малюємо
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]  # Координати кінчика вказівного пальця
        if prev_pos is None: prev_pos = current_pos  # Якщо немає попередньої позиції, використовуємо поточну
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (136, 0, 255), 10)  # Малюємо лінію

    # Якщо жест "великий палець піднятий" — очищаємо полотно
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)  # Очищення полотна

    return current_pos, canvas  # Повертаємо оновлені значення


def sendToAI(model, canvas, fingers):
    """
    Надсилає намальоване полотно до моделі ШІ для обробки.

    Параметри:
    - model: Модель штучного інтелекту.
    - canvas: Зображення-полотно.
    - fingers: Список станів пальців.

    Повертає:
    - response.text: Результат роботи ШІ.
    """
    # Якщо жест "всі пальці підняті, крім великого", надсилаємо до ШІ
    if fingers == [0, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)  # Перетворюємо полотно у формат PIL
        response = model.generate_content(["Дай результат українською мовою", pil_image])  # Відправляємо запит до моделі
        return response.text  # Повертаємо текстову відповідь


# Змінні для збереження стану
prev_pos = None  # Попередня позиція пальця
canvas = None  # Полотно для малювання
image_combined = None  # Поєднане зображення (кадр + полотно)
output_text = ""  # Текстовий результат від ШІ

# Головний цикл
while True:
    # Отримуємо кадр із вебкамери
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Дзеркальне відображення для зручності

    # Ініціалізуємо полотно, якщо воно не створене
    if canvas is None:
        canvas = np.zeros_like(img)

    # Отримуємо інформацію про руку
    info = getHandInfo(img)
    if info:
        fingers, lmList = info  # Розпізнаємо стан пальців і координати
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Малюємо на полотні
        output_text = sendToAI(model, canvas, fingers)  # Надсилаємо до ШІ, якщо відповідний жест

    # Поєднуємо кадр із полотном
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    # Відображення зображення у Streamlit
    FRAME_WINDOW.image(image_combined, channels="BGR")  # Відображаємо оновлене зображення

    # Виводимо результат роботи ШІ, якщо є
    if output_text:
        output_text_area.text(output_text)
