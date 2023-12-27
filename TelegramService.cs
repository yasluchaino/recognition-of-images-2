using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;

using System.Drawing;
using NeuralNetwork1;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        private readonly BaseNetwork net1;
        private readonly BaseNetwork net2;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }

        public TelegramService(string token, AIMLService aimlService)
        {
            aiml = aimlService;
            net1 = new AccordNet(new int[]{ 56, 300,8});
            net2 = new StudentNetwork(new int[] { 56, 28, 14, 8 });
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;
           //сразу обучим 
                Console.WriteLine("Обучение...");
                SamplesSet s = DataSet.GetDataSet();
                net1.TrainOnDataSet(s, 30, 0.0005, true);
                Console.WriteLine("Точность: " + net1.GetAccuracy(s));
                net2.TrainOnDataSet(s, 300, 0.0005, true);
                Console.WriteLine("Точность: " + net2.GetAccuracy(s));

                Console.WriteLine("Обучение завершено");

                    }

        async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            var message = update.Message;
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;
            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;          
                Console.WriteLine($"Received a '{messageText}' message in chat {chatId} with {username}.");

                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: aiml.Talk(chatId, username, messageText),
                    cancellationToken: cancellationToken);
                return;
            }
            // Загрузка изображений пригодится для соединения с нейросетью
            if (message.Type == MessageType.Photo)
            {
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = await client.GetFileAsync(photoId, cancellationToken: cancellationToken);
                var imageStream = new MemoryStream();
                await client.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);
                // Если бы мы хотели получить битмап, то могли бы использовать new Bitmap(Image.FromStream(imageStream))
                // Но вместо этого пошлём картинку назад
                // Стрим помнит последнее место записи, мы же хотим теперь прочитать с самого начала
                imageStream.Seek(0, 0);
                var bmp = new System.Drawing.Bitmap(System.Drawing.Image.FromStream(imageStream));
                bmp = ImageHelper.MakeBW(bmp);
                bmp = ImageHelper.Invert(bmp);
                bmp = ImageHelper.CutSquare(bmp);
                bmp = ImageHelper.CutContent(bmp);
                bmp = ImageHelper.Resize(bmp, new Size(28, 28));
                var sample1 = new Sample(ImageHelper.GetArray(bmp), 8);
                var sample2 = new Sample(ImageHelper.GetArray(bmp), 8);
                var res = net1.Predict(sample1);
                var res2 = net2.Predict(sample2);
                //Console.WriteLine("Accord result is " + res);
                //Console.WriteLine("Student result is " + res2);
                MemoryStream newImg = new MemoryStream();
                bmp.Save(newImg, System.Drawing.Imaging.ImageFormat.Jpeg);
                newImg.Seek(0, 0);

                string prediction1 = "undef";
                switch (res)
                {
                    case FigureType.Happy:
                        prediction1 = "happy";
                        break;
                    case FigureType.Angry:
                        prediction1 = "angry";
                        break;
                    case FigureType.Sad:
                        prediction1 = "sad";
                        break;
                    case FigureType.Tongue:
                        prediction1 = "tongue";
                        break;
                    case FigureType.PokerFace:
                        prediction1 = "pokerface";
                        break;
                    case FigureType.Smile:
                        prediction1 = "smile";
                        break;
                    case FigureType.Surprised:
                        prediction1 = "surprised";
                        break;
                    case FigureType.Wink:
                        prediction1 = "wink";
                        break;
                    case FigureType.Undef:
                        prediction1 = "undef";
                        break;
                }
                var answer1 = aiml.Talk(chatId, username, $"predicted {prediction1}");

                string prediction2 = "undef";
                switch (res2)
                {
                    case FigureType.Happy:
                        prediction2 = "happy";
                        break;
                    case FigureType.Angry:
                        prediction2 = "angry";
                        break;
                    case FigureType.Sad:
                        prediction2 = "sad";
                        break;
                    case FigureType.Tongue:
                        prediction2 = "tongue";
                        break;
                    case FigureType.PokerFace:
                        prediction2 = "pokerface";
                        break;
                    case FigureType.Smile:
                        prediction2 = "smile";
                        break;
                    case FigureType.Surprised:
                        prediction2 = "surprised";
                        break;
                    case FigureType.Wink:
                        prediction2 = "wink";
                        break;
                    case FigureType.Undef:
                        prediction2 = "undef";
                        break;
                }
                var answer2 = aiml.Talk(chatId, username, $"predicted {prediction2}");
                await client.SendPhotoAsync(
                    message.Chat.Id,
                    newImg,
                    "Accord: " + answer1 + '\n' +
                    "Student: " + answer2+'\n',
                    cancellationToken: cancellationToken
                );
                return;
            }
            // Можно обрабатывать разные виды сообщений, просто для примера пробросим реакцию на них в AIML
            if (message.Type == MessageType.Video)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Видео"), cancellationToken: cancellationToken);
                return;
            }
            if (message.Type == MessageType.Audio)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Аудио"), cancellationToken: cancellationToken);
                return;
            }
        }

        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Заканчиваем работу - корректно отменяем задачи в других потоках
            // Отменяем токен - завершатся все асинхронные таски
            cts.Cancel();
        }
    }
}
