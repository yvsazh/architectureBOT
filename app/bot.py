import asyncio
from io import BytesIO
from PIL import Image
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from config import TOKEN, class_names_ua, class_names_en
from model import model, img_width, img_height
import tensorflow as tf

from keep_alive import keep_alive
keep_alive()

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Привіт!🤗\nНадішли малюнок будівлі і я скажу тобі, в якому стилі вона була побудована!⛩️\nЯ знаю цілих 48 різних стилів! Якщо ти хочеш конкретніше дізнатись про це, просто використай команду /help!🤫")

@dp.message(Command("help"))
async def help(message: Message):
    help_message = "Я - бот, який допоможе тобі визначити стиль лише по фото будівлі!🧐\nЯ можу розпізнати цілих 48 різних стилів, а конкретно:\n\n"
    for i in range(len(class_names_ua)):
        help_message += f"{i+1}. {class_names_ua[i]} ({class_names_en[i]})\n"
    help_message += "\nСподіваюсь я зміг тобі допомогти!😄"
    await message.answer(help_message)


@dp.message(F.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    
    # Отримуємо фото як байти
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    
    # Створюємо об'єкт BytesIO та відкриваємо його як зображення
    image_stream = BytesIO(file_bytes.read())
    img = Image.open(image_stream)
    img = img.resize((img_height, img_width))
    
    # Конвертуємо в масив
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    await message.answer(
        f"Будівля побудована у стилі: {class_names_ua[np.argmax(score)]}({class_names_en[np.argmax(score)]})"
    )

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())