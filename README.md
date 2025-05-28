# Определение размера / локации повреждения авто по фото
Этот проект меняет подход к покупке автомобилей: с помощью компьютерного зрения можно увидеть то, что часто остаётся скрытым на фотографиях. Автоматическое обнаружение повреждений делает оценку машин понятной, быстрой и честной для всех, кто ищет надёжное авто.
## Команда
 - Название: Автоподстава
 - Состав: Александр Хасаметдинов
## Запуск сервиса и взаимодействие с ним через веб-интерфейс
 1. Склонируйте репозиторий
 - Если у вас не установлен `git lfs`, выполните следующие команды:  
 `sudo apt update`  
 `sudo apt install git-lfs`  
 `git lfs install`  
 `git lfs pull`  
 2. Соберите образ командой `docker build -t car_damage_service .`
 3. Запустите сервис:
     - Если вы используете CPU: `bash run.sh`
     - Если вы используете GPU:  
    `sudo apt update`  
    `sudo apt install -y nvidia-container-toolkit`  
    `sudo systemctl restart docker`  
    `docker run --gpus all -p 8000:8000 car_damage_service`  
 4. Перейдите в браузере по адресу `http://localhost:8000/`
 5. Загрузите фото автомобиля в форму для загрузки файлов
## Запуск сервиса и взаимодействие с ним через API
 1. Склонируйте репозиторий
 - Если у вас не установлен `git lfs`, выполните следующие команды:  
 `sudo apt update`  
 `sudo apt install git-lfs`  
 `git lfs install`  
 `git lfs pull`  
 2. Соберите образ командой `docker build -t car_damage_service .`
 3. Запустите сервис:
     - Если вы используете CPU: `bash run.sh`
     - Если вы используете GPU:  
    `sudo apt update`  
    `sudo apt install -y nvidia-container-toolkit`  
    `sudo systemctl restart docker`  
    `docker run --gpus all -p 8000:8000 car_damage_service`  
 4. В терминале выполните команду `python3 service/client.py`
 5. Введите путь к изображению на вашем устройстве