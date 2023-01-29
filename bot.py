TOKEN = '5988078653:AAF45PYCHuBxwudeL6M8Lsp_Yc7PqmRR45Y'

import telebot
from telebot import types
from os import path
#is_content=False
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

from script import image_loader, ContentLoss, gram_matrix, StyleLoss, Normalization, get_style_model_and_losses, get_input_optimizer, run_style_transfer

bot = telebot.TeleBot(TOKEN, parse_mode=None)

@bot.message_handler(commands=['start'])
def send_welcome(message):
	kb = types.InlineKeyboardMarkup(row_width = 1)
	btn1 = types.InlineKeyboardButton(text='старт', callback_data='btn1')
	kb.add(btn1)
	bot.send_message(message.chat.id, "Бот применяет к первому изображению стиль второго изображения. Загрузите содержание(c подписью content), а затем стиль(с подписью style), нажмите start и дождитесь завершения переноса. Содержание должно иметь четко выраженный объект. Стиль должен иметь просто четкие линии, несколько однородных тонов.", reply_markup=kb)


@bot.message_handler(commands=['help'])
def send_welcome2(message):
	bot.send_message(message.chat.id, 'send me 2 photos. Image photo and style photo')


@bot.message_handler(content_types = ['photo'])
def handle_docs_document(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    #src = message.photo[1].file_id
    src = message.caption+'.jpg'
    if message.caption in ['content', 'style']:
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
            bot.reply_to(message, 'фотография '+message.caption+' сохранена')
    else:
        bot.reply_to(message, 'пожалуйста отметьте на фотографии content или style')


@bot.callback_query_handler(func=lambda callback: callback.data)
def check_callback_data(callback):
    if callback.data == 'btn1':
        if path.exists("content.jpg") and path.exists("style.jpg"):
            style_img = image_loader('style.jpg')
            content_img = image_loader('content.jpg')
            assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            imsize = 512 if torch.cuda.is_available() else 128

            cnn = models.vgg19(pretrained=True).features.to(device).eval()

            loader = transforms.Compose([
                transforms.Resize((imsize, imsize)),  # scale imported image
                transforms.ToTensor()])  # transform it into a torch tensor


            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


            content_layers_default = ['conv_4']
            style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

            input_img = content_img.clone()

            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

            with open('output.jpg', 'wb') as output_file:
                image = output.cpu().clone()  # we clone the tensor to not do changes on it
                image = image.squeeze(0)      # remove the fake batch dimension
                image = transforms.ToPILImage(output)
                output_file.write(output.transforms.ToPILImage())


            bot.send_photo(callback.message.chat.id, photo=open('output.jpg', 'rb'), caption='смотри, что получилось')




        else:
            bot.send_message(callback.message.chat.id, 'пришлите фотографии стиля и контента')




bot.infinity_polling()


