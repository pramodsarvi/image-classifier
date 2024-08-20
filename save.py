from torchvision.io import read_image
from torchvision.utils import make_grid
import cv2
import numpy as np
from config import *
import torch
import os
from torchvision import transforms
import matplotlib
import tqdm
import random 
from build_model import *
import matplotlib.pyplot as plt
import os
matplotlib.style.use('ggplot')



def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained,dest=os.getcwd()):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{dest}/efficient_net_acc{pretrained}.png")
    plt.close('all')
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Loss')
    plt.savefig(f"{dest}/efficient_net_loss{pretrained}.png")
    plt.close('all')




# def visualise_image(feats , op , image, model  , IMAGE_DIMS):

#     output = op.cpu().detach().numpy()

#     preds = np.argmax(output)

#     # print("This is preds las ", preds)

#     last_conv = feats.permute(0,3,2,1)

#     last_conv = last_conv.cpu().detach().numpy()

#     last_layer_weights = model.linear.weight[preds]

#     last_layer_weights = torch.unsqueeze(last_layer_weights,1).cpu().detach().numpy()

#     last_conv = np.squeeze(last_conv)
#     # print("!"*100)
#     # print(last_conv.shape)
#     h = IMAGE_DIMS[1]/last_conv.shape[0]
#     w = IMAGE_DIMS[0]/last_conv.shape[1]
#     # h,w=28,16

#     import scipy
#     upsamples_one = scipy.ndimage.zoom(last_conv, (h, w ,1), order = 1)

#     # print("this is the upsampled image las ", upsamples_one.shape)
#     # print("this is the last layer las ", last_layer_weights.shape)

#     heat_map = np.dot(upsamples_one.reshape((IMAGE_DIMS[0]*IMAGE_DIMS[1], 1280)), 
#                     last_layer_weights).reshape(IMAGE_DIMS[1],IMAGE_DIMS[0])
    
#     heat_map = cv2.applyColorMap(np.uint8(255 * heat_map), cv2.COLORMAP_JET)
#     # print("THis is the heatmap shape las ",heat_map.shape)
#     # print("Image shape",image.shape)

#     image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     heat_map=cv2.cvtColor(heat_map,cv2.COLOR_BGR2RGB)
#     image_heatmap=cv2.addWeighted(image,0.6,heat_map,0.4,0)
#     return image,image_heatmap





# from PIL import Image
# def combine_images(columns, space, images,width_max,height_max):
#     rows = len(images) // columns
#     if len(images) % columns:
#         rows += 1
#     background_width = width_max*columns + (space*columns)-space
#     background_height = height_max*rows + (space*rows)-space
#     print(background_width, background_height)
#     background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
#     x = 0
#     y = 0
#     for i, image in enumerate(images):
#         image=Image.fromarray(image)
#         img = Image.open(image)
#         x_offset = int((width_max-img.width)/2)
#         y_offset = int((height_max-img.height)/2)
#         background.paste(img, (x+x_offset, y+y_offset))
#         x += width_max + space
#         if (i+1) % columns == 0:
#             y += height_max + space
#             x = 0
#     background.save('image.png')
#     return background
# def image_grid(imgs, dims,is_best):
#     rows,cols=dims
#     total=rows*cols

#     old_imgs=imgs[0:total]
#     imgs=[]
#     for i in old_imgs:
#         imgs.append(Image.fromarray(i))

#     w, h = imgs[0].size
#     grid = Image.new('RGB', size=(cols*w, rows*h))
#     grid_w, grid_h = grid.size
    
#     for i, img in enumerate(imgs):
#         grid.paste(img, box=(i%cols*w, i//cols*h))
#     if is_best:
#         grid.save(dest+"/"+"best_gradcam.jpg")
#     else:
#         grid.save(dest+"/"+"last_gradcam.jpg")

# def save_gradcam(path):
#     grid_size=GRID_SIZE

#     total_images=grid_size[0]*grid_size[1]


#     class_names=os.listdir(VAL_DIR)
#     image_paths=[]
#     # print(path)
#     checkpoint = torch.load(path, map_location=DEVICE)
#     model=build_model_gradient(num_classes=2)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.train()

#     heat_maps=[]
    
    # for i,class_name in enumerate(class_names):
        
    #     # print(os.path.join(VAL_DIR,class_name))
    #     # add complete path
    #     temp=os.listdir(os.path.join(VAL_DIR,class_name))
    #     dir_temp=os.path.join(VAL_DIR,class_name)+"/"
    #     my_new_list = [dir_temp + x for x in temp]
    #     image_paths.extend(my_new_list)


    # image_paths=random.sample(image_paths,total_images)
    # print("[INFO]Creating Gradient images")

    # for image_path in tqdm(image_paths):
    #     # print(image_path)
    #     image = cv2.imread(image_path)
    #     orig_image = image.copy()
    #     orig_image =cv2.resize(orig_image, (IMAGE_DIMS[0], IMAGE_DIMS[1]),interpolation = cv2.INTER_LINEAR)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((IMAGE_DIMS[0], IMAGE_DIMS[1])),
    #         transforms.ToTensor(),
    #     ])
    #     image = transform(image)
    #     image = torch.unsqueeze(image, 0)
    #     image = image.to(DEVICE)
        
    #     op, feats = model(image)
    #     # print("Features size",feats.shape)
    #     # print("Image shape:",orig_image.shape)
    #     # print(feats.shape)
    #     # make_dot(op.mean(), params=dict(model.parameters())).render("attached", format="png")
    #     # make_dot(op).render("attached", format="png")
    #     heat_maps.extend(visualise_image(feats , op , orig_image , model , IMAGE_DIMS))

    # # heat_maps=torchvision.transforms.ToPILImage()
    # if "best" in path:
    #     image_grid(heat_maps,[GRID_SIZE[0],GRID_SIZE[1]*2],True)
    # else:
    #     image_grid(heat_maps,[GRID_SIZE[0],GRID_SIZE[1]*2],False)





        # save_gradcam(f"{dest}/efficient_net_last.pth")
        

