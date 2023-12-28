using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using AForge;

namespace AIMLTGBot
{
    public class TelegramService : IDisposable
    {
        static string currentPath = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
        private readonly TelegramBotClient client;
        private readonly AIMLService aiml;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public string Username { get; }
        private DataSet dataHolder = new DataSet();
        private MagicEye processor = new MagicEye();
        private BaseNetwork network1 = new AccordNet(new int[] { 2304, 500, 100, 8 });
        private BaseNetwork network2 = new StudentNetwork(new int[] { 2304, 500, 100, 8 });
        public TelegramService(string token, AIMLService aimlService)
        {
            dataHolder.loadData(
                $"{currentPath}\\data\\train",
                $"{currentPath}\\data\\train",
                $"{currentPath}\\data\\train.txt",
                $"{currentPath}\\data\\train.txt"
            );
            network1.TrainOnDataSet(dataHolder.trainData, 30, 1e-9, true, 1e-1);
            network2.TrainOnDataSet(dataHolder.trainData, 300, 1e-9, true, 1e-1);
            double accuracy1 = dataHolder.testData.TestNeuralNetwork(network1);
            double accuracy2 = dataHolder.testData.TestNeuralNetwork(network2);
            Console.WriteLine(accuracy1.ToString());
            Console.WriteLine(accuracy2.ToString());
            aiml = aimlService;
            client = new TelegramBotClient(token);
            client.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
            {   // Подписываемся только на сообщения
                AllowedUpdates = new[] { UpdateType.Message }
            },
            cancellationToken: cts.Token);
            // Пробуем получить логин бота - тестируем соединение и токен
            Username = client.GetMeAsync().Result.Username;
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
                var qInd = messageText.LastIndexOf('?');
                string formattedMessageText = messageText.Replace('ё','е');
                if (qInd != -1)
                {
                    formattedMessageText = messageText.Insert(qInd, " ");
                }
                string answer = aiml.Talk(chatId, username, formattedMessageText);
                if (answer.Trim() == "")
                {
                    answer = "Не совсем уловил мысль";
                }
                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: answer,
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
                var image = new Bitmap(Image.FromStream(imageStream));
                processor.ProcessImage(image, false);
                var processedImage = processor.processed;
                var sample = bitmapToSample(processedImage);
                var res1 = network1.Predict(sample);
                var res2 = network2.Predict(sample);
                string prediction1 = "undef";
                string prediction2 = "undef";

                processedImage.Save($"{currentPath}//predicted.png", System.Drawing.Imaging.ImageFormat.Png);
                switch (res1)
                {
                  case SymbolType.Happy:
                        prediction1 = "happy";
                        break;
                    case SymbolType.Angry:
                        prediction1 = "angry";
                        break;  
                    case SymbolType.Sad:
                        prediction1 = "sad";
                        break;
                    case SymbolType.Tongue:
                        prediction1 = "tongue";
                        break;
                    case SymbolType.PokerFace:
                        prediction1 = "pokerface";
                        break;
                    case SymbolType.Smile:
                        prediction1 = "smile";
                        break;
                    case SymbolType.Surprised:
                        prediction1 = "surprised";
                        break;
                    case SymbolType.Wink:
                        prediction1 = "wink";
                        break;
                    case SymbolType.Undef:
                        prediction1 = "undef";
                        break;
                }
                var answer1 = aiml.Talk(chatId, username, $"predicted {prediction1}");
                switch (res2)
                {
                    case SymbolType.Happy:
                        prediction2 = "happy";
                        break;
                    case SymbolType.Angry:
                        prediction2 = "angry";
                        break;
                    case SymbolType.Sad:
                        prediction2 = "sad";
                        break;
                    case SymbolType.Tongue:
                        prediction2 = "tongue";
                        break;
                    case SymbolType.PokerFace:
                        prediction2 = "pokerface";
                        break;
                    case SymbolType.Smile:
                        prediction2 = "smile";
                        break;
                    case SymbolType.Surprised:
                        prediction2 = "surprised";
                        break;
                    case SymbolType.Wink:
                        prediction2 = "wink";
                        break;
                    case SymbolType.Undef:
                        prediction2 = "undef";
                        break;
                }
                var answer2 = aiml.Talk(chatId, username, $"predicted {prediction2}");
                await client.SendTextMessageAsync(
                    message.Chat.Id,
                       "Accord: " + answer1 + '\n' +
                   "Student: " + answer2 + '\n',
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
            if (message.Type == MessageType.Voice)
            {
                await client.SendTextMessageAsync(message.Chat.Id, aiml.Talk(chatId, username, "Войс"), cancellationToken: cancellationToken);
                return;
            }
        }

        Sample bitmapToSample(Bitmap processed)
        {
            Rectangle rect = new Rectangle(0, 0, processed.Width, processed.Height);
            BitmapData bmpData = processed.LockBits(rect, ImageLockMode.ReadOnly, processed.PixelFormat);
            double[] input = new double[48 * 48];
            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;
                int heightInPixels = bmpData.Height;
                int widthInBytes = bmpData.Stride;
                for (int y = 0; y < heightInPixels; y++)
                {
                    for (int x = 0; x < widthInBytes; x = x + 1)
                    {
                        double grayValue = ptr[(y * bmpData.Stride) + x] / 255.0;
                        input[(y * bmpData.Stride) + x] = grayValue;

                    }
                }
            }
            Sample sample = new Sample(input, 8);
            return sample;
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
