from imageai.Classification import ImageClassification
import os
 
exec_path = os.getcwd()
 
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()
 
predictions, probabilities = prediction.classifyImage(os.path.join(exec_path,'vampire_bat.jpg'), result_count=5) #try out different jpgs!
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')