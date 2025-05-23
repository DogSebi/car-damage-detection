import requests
import time


class CarDamageFinderClient:
    def __init__(self, host):
        self.host = host

    def get_damage(self, image_path):
        with open(image_path, "rb") as f:
            files = {"file": ("image.jpg", f, "image/jpeg")}
            response = requests.post(
                f"http://{self.host}:8000/api", files=files)
            result = response.json()

        return result


if __name__ == '__main__':
    image_path = input('Введите путь к изображению: ')
    client = CarDamageFinderClient(host='localhost')
    result = client.get_damage(image_path)
    print(result['report'])
