'''
usage: python gen_diff.py -h
'''
from __future__ import print_function
import argparse
from keras.layers import Input
from keras.models import load_model
import numpy as np
import random
import os

# Scipy.misc.imsave 대체 (최신 버전 호환성)
try:
    from imageio import imwrite
except ImportError:
    import scipy.misc
    imwrite = scipy.misc.imsave

from configs import bcolors
from utils import *

# 1. Argument Parsing
parser = argparse.ArgumentParser(description='DeepXplore for CIFAR-10 ResNet50')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm for differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm for neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations", type=int)
parser.add_argument('threshold', help="threshold for neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model index", choices=[0, 1], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)

args = parser.parse_args()

# 2. Setup
img_rows, img_cols = 224, 224 # 학습 시 224로 리사이즈했다면 유지
input_shape = (img_rows, img_cols, 3)
input_tensor = Input(shape=input_shape)

# 3. Load Models
K.set_learning_phase(0)
model1 = load_model('./models/resnet50_1.h5')
model2 = load_model('./models/resnet50_2.h5')

# 레이어 이름 자동 탐색 (보통 마지막 층은 'dense_1' 혹은 'predictions'임)
# 직접 확인하려면 model1.summary() 후 마지막 층 이름을 넣으세요.
final_layer_name = model1.layers[-1].name 

# 4. Init Coverage Table (2개 모델 위주로 처리)
model_layer_dict1, model_layer_dict2, _ = init_coverage_tables(model1, model2, model2)

# 결과 저장 폴더 생성
if not os.path.exists('./generated_inputs/'):
    os.makedirs('./generated_inputs/')

# 5. Start Generation
img_paths = image.list_pictures('./seeds/', ext='png') # .png 파일 읽기

for _ in range(args.seeds):
    gen_img = preprocess_image(random.choice(img_paths))
    orig_img = gen_img.copy()
    
    pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
    label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])
    
    if not label1 == label2:
        print(bcolors.OKGREEN + 'Difference found! Model1: {}, Model2: {}'.format(label1, label2) + bcolors.ENDC)
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        
        gen_img_deprocessed = deprocess_image(gen_img)
        imwrite('./generated_inputs/already_differ_{}_{}.png'.format(label1, label2), gen_img_deprocessed)
        continue

    # Joint Loss Function 구성
    orig_label = label1
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)

    # Differential behavior loss
    if args.target_model == 0:
        loss_diff = -args.weight_diff * K.mean(model1.output[..., orig_label]) + K.mean(model2.output[..., orig_label])
    else:
        loss_diff = K.mean(model1.output[..., orig_label]) - args.weight_diff * K.mean(model2.output[..., orig_label])

    # Neuron coverage loss
    loss_nc = args.weight_nc * (K.mean(model1.get_layer(layer_name1).output[..., index1]) + 
                                K.mean(model2.get_layer(layer_name2).output[..., index2]))
    
    final_loss = K.mean(loss_diff + loss_nc)
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    iterate = K.function([input_tensor], [grads])

    # Gradient Descent (Ascent)
    for iters in range(args.grad_iterations):
        grads_value = iterate([gen_img])[0]
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)

        gen_img += grads_value * args.step
        pred1, pred2 = model1.predict(gen_img), model2.predict(gen_img)
        label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])

        if not label1 == label2:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            
            nc1 = neuron_covered(model_layer_dict1)[2]
            nc2 = neuron_covered(model_layer_dict2)[2]
            print(bcolors.OKGREEN + 'NC: Model1: {:.3f}, Model2: {:.3f}'.format(nc1, nc2) + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            imwrite('./generated_inputs/{}_{}_{}.png'.format(args.transformation, label1, label2), gen_img_deprocessed)
            break