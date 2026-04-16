from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open('test_pic.jpg').convert('RGB')
img_cropped = mtcnn(img)

embedding = resnet(img_cropped.unsqueeze(0))

T.ToPILImage()(img_cropped).save("cropped_test.jpg")
print(embedding)