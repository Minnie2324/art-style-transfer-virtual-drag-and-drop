import time
start_import = float(f"{time.perf_counter():0.2f}")
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader
import utils.utils as utils
from models.definitions.transformer_net import TransformerNet
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

end_import = float(f"{time.perf_counter():0.2f}")
print(f"import run time: {end_import - start_import} seconds")


"""webcam capture"""
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
"""hand detection"""
detector = HandDetector(detectionCon=1)
"""Photo file"""
path1, path2, path3 = "Style", "Content", "Result"
styleList = os.listdir(path1)
contentList = os.listdir(path2)
resultList = os.listdir(path3)
stylePath = os.path.join(os.path.dirname(__file__), path1)
contentPath = os.path.join(os.path.dirname(__file__), path2)
resultPath = os.path.join(os.path.dirname(__file__), path3)
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
assert utils.dir_contains_only_models(model_binaries_path), f'Model directory should contain only model binaries.'
os.makedirs(resultPath, exist_ok=True)


def run_style_transfer(img_obj1, img_obj2):
    print(img_obj1.path)
    """trained style transfer model"""
    start_model = float(f"{time.perf_counter():0.2f}")
    outputTime = time.time()
    outputName = str(f'Image_{outputTime}.jpg')

    # Wrapping inference configuration into a dictionary
    inference_config = dict()
    inference_config['content_input'] = f"{img_obj2.path[8:img_obj2.path.rfind('.')]}.jpg"
    inference_config['batch_size'] = 5
    inference_config['img_width'] = 250
    inference_config['img_height'] = 300
    inference_config['model_name'] = f"{img_obj1.path[6:img_obj1.path.rfind('.')]}.pth"

    # Less frequently used arguments
    inference_config['should_not_display'] = 'store_false'
    inference_config['verbose'] = 'store_true'
    inference_config['redirected_output'] = resultPath
    inference_config['content_images_path'] = contentPath
    inference_config['output_images_path'] = resultPath
    inference_config['model_binaries_path'] = model_binaries_path
    if torch.cuda.is_available():
        device = torch.device("cuda")
        training_state = torch.load(
            os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]))
    else:
        device = torch.device("cpu")
        map_location = torch.device('cpu')
        training_state = torch.load(
            os.path.join(inference_config["model_binaries_path"], inference_config["model_name"]), map_location)
    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet().to(device)
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if inference_config['verbose']:
        utils.print_model_metadata(training_state)
    with torch.no_grad():
        if os.path.isdir(
            inference_config['content_input']):  # do a batch stylization (every image in the directory)
            img_dataset = utils.SimpleDataset(inference_config['content_input'], inference_config['img_width'])
            img_loader = DataLoader(img_dataset, batch_size=inference_config['batch_size'])
            try:
                processed_imgs_cnt = 0
                for batch_id, img_batch in enumerate(img_loader):
                    processed_imgs_cnt += len(img_batch)
                    if inference_config['verbose']:
                        print(
                            f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(img_dataset)} processed images).')
                    img_batch = img_batch.to(device)
                    stylized_imgs = stylization_model(img_batch).to('cpu').numpy()
                    for stylized_img in stylized_imgs:
                        utils.save_and_maybe_display_image(inference_config, stylized_img, should_display=False)
            except Exception as e:
                print(e)
                print(
                    f'Consider making the batch_size (current = {inference_config["batch_size"]} images) or img_width (current = {inference_config["img_width"]} px) smaller')
                exit(1)
        else:  # do stylization for a single image
            content_img_path = os.path.join(inference_config['content_images_path'],
                                            inference_config['content_input'])
            content_image = utils.prepare_img(content_img_path, inference_config['img_width'], device)
            stylized_img = stylization_model(content_image).to('cpu').numpy()[0]
            utils.save_and_maybe_display_image(inference_config, stylized_img, name=outputName,
                                                should_display=inference_config['should_not_display'])

    name_path = f'{path3}/{outputName}'
    print(name_path)
    input_img = cv2.imread(name_path)
    cv2.imwrite(name_path, cv2.resize(input_img, (250, 300)))
    end_model = float(f"{time.perf_counter():0.2f}")
    print(f"model run time: {end_model - start_model} seconds")
    return name_path


class DragImg:
    """image class"""
    def __init__(self, path, pos_center, img_type, number_img):
        """(path, position center point, type, img= image show on webcam, number, half size) of Photo"""
        self.path = path
        self.pos_center = pos_center
        self.img_type = img_type
        if self.img_type == "png":
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        elif self.img_type == "jpg":
            self.img = cv2.imread(path)
        self.number_img = number_img
        height, width = self.img.shape[:2]
        self.size = int(width / 2), int(height / 2)

    def update(self, cursor_point):
        """change position center point, set boundary, 145<=width<=1135, 170<=height<=550"""
        if cursor_point[0] <= 145:  # 145
            cursor_point[0] = 145
        elif cursor_point[0] >= 1135:  # 1135
            cursor_point[0] = 1135
        if cursor_point[1] <= 170:  # 170
            cursor_point[1] = 170
        elif cursor_point[1] >= 550:  # 550
            cursor_point[1] = 550
        self.pos_center = cursor_point


"""add Photo(class) in imgList (List)"""
style_imglist = []
for x, style_img in enumerate(styleList):
    if 'png' in style_img:
        imgtype = "png"
    else:
        imgtype = "jpg"
    style_imglist.append(DragImg(f'{path1}/{style_img}', [145, 170], imgtype, x))
content_imgList = []
for x, content_img in enumerate(contentList):
    if 'png' in content_img:
        imgtype = "png"
    else:
        imgtype = "jpg"
    content_imgList.append(DragImg(f"{path2}/{content_img}", [445, 170], imgtype, x))
result_imgList = []
for x, result in enumerate(resultList):
    if 'png' in result:
        imgtype = "png"
    else:
        imgtype = "jpg"
    result_imgList.append(DragImg(f"{path3}/{result}", [145, 550], imgtype, x))  # [445, 170], [145, 550]

"""add last Photo as default"""
style_img = style_imglist[(len(style_imglist) - 1)]
content_img = content_imgList[(len(content_imgList) - 1)]
result = result_imgList[(len(result_imgList) - 1)]
"""center (x,y), 1 for style painting, 2 for content Photo, 3 for result Photo"""
cx1, cy1 = style_img.pos_center
cx2, cy2 = content_img.pos_center
cx3, cy3 = result.pos_center
"""half (width, height)"""
w, h = content_img.size
update = False
Take_photo = 0
Change_photo = 0

"""run until error"""
while True:
    """the whole webcam = img"""
    success, img = cap.read()
    img = cv2.flip(img, 1)
    """not in taking Photo process"""
    if Take_photo == 0:
        """find hand, lmlist = fingertip+(fold aka joint)"""
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img)

        if lmList:
            """when detect hand 
            l1 = length of thumb & index fingertip, l2 = length of index & middle fingertip
            l3 = length of middle & ring fingertip, l4 = length of ring & pinky fingertip
            lindex = the whole index finger length
            cursor = index fingertip, position center point"""
            l1, _, _ = detector.findDistance(4, 8, img)
            l2, _, _ = detector.findDistance(8, 12, img)
            l3, _, _ = detector.findDistance(12, 16, img)
            l4, _, _ = detector.findDistance(16, 20, img)
            lpinky, _, _ = detector.findDistance(17, 20, img)
            ljoint, _, _ = detector.findDistance(19, 20, img)
            lindex, _, _ = detector.findDistance(5, 8, img)
            cursor = lmList[8]

            """✌, index & middle finger aka change Photo process"""
            if (l1 > lindex) and (l2 < 35) and (l3 > lindex) and (l4 < 35) and (lindex > 70) and (Change_photo == 0):
                """if index fingertip in within Photo """
                """change next Photo & if last Photo, change to 1st Photo"""
                if (cx1 - w) <= cursor[0] <= (cx1 + w) and (cy1 - h) <= cursor[1] <= (cy1 + h):
                    i = style_img.number_img
                    i = 0 if i == (len(style_imglist) - 1) else (i + 1)
                    style_imglist[i].pos_center = cx1, cy1
                    style_img = style_imglist[i]
                    Change_photo = 90
                elif (cx2 - w) <= cursor[0] <= (cx2 + w) and (cy2 - h) <= cursor[1] <= (cy2 + h):
                    i = content_img.number_img
                    i = 0 if i == (len(content_imgList) - 1) else (i + 1)
                    content_imgList[i].pos_center = cx2, cy2
                    content_img = content_imgList[i]
                    Change_photo = 90
                elif (cx3 - w) <= cursor[0] <= (cx3 + w) and (cy3 - h) <= cursor[1] <= (cy3 + h):
                    i = result.number_img
                    i = 0 if i == (len(result_imgList) - 1) else (i + 1)
                    result = result_imgList[i]
                    Change_photo = 90

            """☝, index finger aka move Photo process"""
            if l1 > lindex and l2 > lindex and l3 < 35 and l4 < 35 and lindex > 70:
                """if index fingertip within Photo"""
                """call function update, 1 for style, 2 for content Photo"""
                if (cx1 - w) <= cursor[0] <= (cx1 + w) and (cy1 - h) <= cursor[1] <= (cy1 + h) and not update:
                    style_img.update(cursor)
                    update = True
                if (cx2 - w) <= cursor[0] <= (cx2 + w) and (cy2 - h) <= cursor[1] <= (cy2 + h) and not update:
                    content_img.update(cursor)
                    update = True

            """🤟🏻, spiderman hand pose aka taking Photo process"""
            if (l1 > lindex) and (l2 > lindex) and (l3 < 35) and (l4 > lindex) and (lindex > 60):
                """give timer"""
                Take_photo = 330

            if Change_photo == 10 and (l1 < lpinky) and l2 < 35 and l3 < 35 and (l4 > lpinky + ljoint):
                cv2.destroyAllWindows()
                exit(0)

            if Change_photo == 0 and (l1 < lpinky) and l2 < 35 and l3 < 35 and (l4 > lpinky + ljoint):
                Change_photo = 50


    """photo 1 & 2 overlapped process"""
    if (cx1 - w < cx2 < cx1 + w) and (cy1 - h < cy2 < cy1 + h):
        print("fineeeeeeeeeeee")
        """move 1st Photo to front in x-axis or 2nd Photo to back if 1st Photo near boundary"""
        if cx1 < cx2:
            if cx1 < 395:
                cx2 += 260
                content_img.update([cx2, cy2])
            else:
                cx1 -= 260
                style_img.update([cx1, cy1])
        else:
            if cx2 < 395:
                cx1 += 260
                style_img.update([cx1, cy1])
            else:
                cx2 -= 260
                content_img.update([cx2, cy2])
        """start style transfer process, add Photo in List and default"""
        start_style = float(f"{time.perf_counter():0.2f}")
        name = run_style_transfer(style_img, content_img)
        end_style = float(f"{time.perf_counter():0.2f}")
        print(f"style run time: {end_style - start_style} seconds")
        result_imgList.append(DragImg(name, (cx3, cy3), "jpg", len(result_imgList)))
        result = result_imgList[(len(result_imgList) - 1)]

    """timer for change Photo process"""
    if Change_photo > 1:
        if Change_photo >= 60:
            cv2.putText(img, '2', (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        elif Change_photo >= 30:
            cv2.putText(img, '1', (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        Change_photo -= 1
    elif Change_photo == 1:
        Change_photo = 0

    """timer for taking Photo process"""
    if Take_photo > 1:
        imgcrop = img[100:700, 330:830]
        cv2.imshow("Selfie", imgcrop)
        if Take_photo >= 300:
            cv2.putText(imgcrop, '10', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 270:
            cv2.putText(imgcrop, '9', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 240:
            cv2.putText(imgcrop, '8', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 210:
            cv2.putText(imgcrop, '7', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 180:
            cv2.putText(imgcrop, '6', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 150:
            cv2.putText(imgcrop, '5', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 120:
            cv2.putText(imgcrop, '4', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 90:
            cv2.putText(imgcrop, '3', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 60:
            cv2.putText(imgcrop, '2', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        elif Take_photo >= 30:
            cv2.putText(imgcrop, '1', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        Take_photo -= 1
    elif Take_photo == 1:
        """after timer, take Photo and add it in List and default"""
        imgcrop = img[100:700, 330:830]
        name2 = f'{path2}/Image_{time.time()}.jpg'
        cv2.imwrite(name2, cv2.resize(imgcrop, (250, 300)))
        content_imgList.append(DragImg(name2, [cx2, cy2], "jpg", len(content_imgList)))
        content_img = content_imgList[(len(content_imgList) - 1)]
        cv2.destroyWindow("Selfie")
        Take_photo = 0

    """not in taking Photo process, add style, content, result Photo in webcam"""
    cx1, cy1 = style_img.pos_center
    cx2, cy2 = content_img.pos_center
    if Take_photo == 0:
        if style_img.img_type == "png":
            img = cvzone.overlayPNG(img, style_img.img, [cx1, cy1])
        else:
            img[cy1 - h:cy1 + h, cx1 - w:cx1 + w] = style_img.img
        img[cy2 - h:cy2 + h, cx2 - w:cx2 + w] = content_img.img
        img[cy3 - h:cy3 + h, cx3 - w:cx3 + w] = result.img  # 300,250
    update = False

    cv2.imshow("Image", img)
    cv2.waitKey(1)
























