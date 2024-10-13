Automated pipe handling and up-and-down operations on drilling platforms currently face significant efficiency and safety challenges due to the low level of automation. Accurate detection of drill pipes and pipe joints is crucial for automatic pipe processing. This study proposes an enhanced YOLOv5s algorithm, PCD-YOLOv5s, for pipe column detection in oil drilling rigs. The algorithm introduces the EfficientNetv2 lightweight network and a Simplified Spatial Pyramid Pooling Fast (SimSPPF) module to reduce parameters and improve computational efficiency. The BiFormer attention mechanism is integrated to suppress background interference and enhance feature extraction. Additionally, the weighted bidirectional feature pyramid network (BiFPN) and CARAFE operator are employed to improve resolution and semantic information transmission. The model's convergence and generalization ability are further enhanced by replacing the loss function CIoU with SIOU and using the AdamW optimization algorithm. Experimental results on a self-built dataset demonstrate that PCD-YOLOv5s achieves an accuracy of 90.3%, which is 3.6% higher than the original YOLOv5s, and a mean Average Precision (mAP) of 94.2%, which is 4.1% higher. The proposed algorithm effectively addresses the issues of small target size, partial occlusion, and texture and shape variations in the drilling environment, showcasing its potential for real-time pipe column detection in automated drilling operations.
![image](https://github.com/user-attachments/assets/f8606758-24c7-497c-862f-5fe497ae9596)
Datasets: https://pan.baidu.com/s/1vJmtfj8Rut_onguxNy2KVw?pwd=w0ug
