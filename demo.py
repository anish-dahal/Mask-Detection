from facenet_pytorch import MTCNN
import cv2
import torch
import numpy as np
import torchvision.transforms as tt
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from Model.ResNet import MaskModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def play(model_path):
    """Demo using labtop webcam for mask detection

    Parameters
    ----------
    model_path : str
        path of the model
    """
    # load mask detection model
    model = torch.load(model_path, map_location = device)
    # tranformation used in image
    transform = tt.Compose([
                    tt.Resize(size=(256, 256)),
                    tt.CenterCrop(224),
                    tt.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    model.eval()

    # video capture using webcam
    cap = cv2.VideoCapture(0)
    # set FPS to 10
    cap.set(cv2.CAP_PROP_FPS, 10)
    # set height and width of the frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    while True:
        ret, image = cap.read()
        # convert BRG image to RGB image
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use mtcnn to detect human faces.
        faces, _ = mtcnn.detect(imageRGB)
        
        if faces is not None:
            for face in faces:
                try:
                    x1 = round(face[0])
                    y1 = round(face[1])
                    x2 = round(face[2])
                    y2 = round(face[3])
                    # get only array which contain face
                    face_image = imageRGB[y1 : y2, x1 : x2, :]
                    img = torch.tensor(face_image)
                    # img shape is (h, w, c) but required shape for model is (c, h, w) so change the shape of the image
                    img = img.permute(2, 0, 1)
                    # make each value of array between [0, 1] from [0, 255]
                    img = (img-0)/(255-0)
                    img = transform(img)
                    # model accept image of shape (b, c, h, w)
                    img = img[None, :]
                    # model.eval()
                    with torch.no_grad():
                        y = model(img.to(device))
                        y = F.softmax(y, dim = 1)
                        value, pred =torch.max(y, dim = 1)
                    
                    
                    if pred.item() == 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                    classes = ['with_mask', 'without_mask']
                    cv2.putText(
                                image,
                                classes[pred.item()]+f"({str(round(value.item(), 4))})",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                except Exception as e:
                    print(e)
        else:
            cv2.putText(
                    image,
                    "No face found",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play("../kaggle/model.pt")

