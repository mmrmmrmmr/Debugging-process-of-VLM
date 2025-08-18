from ultralytics import YOLOWorld

from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world_my import WorldTrainerFromScratch

data = {
"train": 
{
"yolo_data": ["/home/mamingrui/sod/datasets/VHR.yaml"],
},
"val": 
{"yolo_data": ["/home/mamingrui/sod/datasets/VHR.yaml"]},
}

model = YOLOWorld("yolov8s-worldv2.yaml")
# model.train(data=data, batch=32, epochs=100, trainer=WorldTrainerFromScratch)

# Initialize a YOLO-World model
# model = YOLOWorld("yolov8s-worldv2.pt")
# model.set_classes([
# 'airplane',
# 'ship',
# 'storage tank',
# 'baseball diamond',
# 'tennis court',
# 'basketball court',
# 'ground track field',
# 'harbor',
# 'bridge',
# 'vehicle',
# ])
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VHR.yaml", epochs=2, imgsz=640)

# x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
