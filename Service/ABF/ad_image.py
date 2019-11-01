import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import PIL


# making ad image
def makeadfile(img_arr, img_path, img_type='_'): #img_type = '_ad','_det','_ad_resize'
    img = np.array(img_arr)
    
    mask_np = np.asarray(Image.open('mask_file.png','r'))
    ad = Image.open('ad_file.png','r')
    
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(g_channel.shape, dtype=b_channel.dtype) * 255 #creating a dummy alpha channel image.
    img_a = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    img_a[:,:,3] = mask_np[:,:,3]

    mask_img = Image.fromarray(np.uint8(img_a))

    final = Image.alpha_composite(ad, mask_img)
    #plt.imshow(final)
    final.save('./result/'+img_path.split('/')[-1].split('.')[0]+img_type+'.png')
    print('./result/'+img_path.split('/')[-1].split('.')[0]+img_type+'.png'+' 저장완료')
    


#preprocessing image
def PreProcessingImage(img_path, boxes, label, mask_x = 166, mask_y= 184):
    img = Image.open(img_path)
    
    img_w, img_h = img.size
    img_np = np.array(img)
    
    # (0) box check
    if len(boxes) == 0:
        return print('박스없음')
    
    b_mask_width, b_mask_height = 166,184
    s_mask_width, s_mask_height = 130,166
    
    mask_np = np.asarray(Image.open('mask_file.png','r'))
    ad_np = np.asarray(Image.open('ad_file.png','r'))
    
    # (1) Prediction plot
    
    x1,y1,x2,y2 = boxes[0]
    if x1 < 0:
        x1 = 1
    if y1 < 0:
        y1 = 1
    if x2 >= img_w:
        x2 = img_w-1
    if y2 >= img_h:
        y2 = img_h-1

    print(x1,y1,x2,y2)
    box_width, box_height = abs(x2-x1) , abs(y1-y2)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    rect = patches.Rectangle((x1, y1),box_width, box_height, linewidth=1.5 ,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    fig.savefig('./result/'+img_path.split('/')[-1].split('.')[0]+'_box.png', bbox_inches='tight', pad_inches=0)
    print('./result/'+img_path.split('/')[-1].split('.')[0]+'_box.png'+' 저장완료')
    
    # (2) Detected Box save
    
    img_box_crop = img.crop((x1,y1,box_width+x1, box_height+ y1))
    img_box_crop.save('./result/'+img_path.split('/')[-1].split('.')[0]+'_det.png')
    print('./result/'+img_path.split('/')[-1].split('.')[0]+'_det.png'+' 저장완료')
    
    # (3) AD save
    
    img_box_crop_b = img.crop((x1,y1,b_mask_width+x1, b_mask_height+ y1))
    
    makeadfile(img_box_crop_b, img_path, img_type = '_ad')
    
    # (4) AD-resize

    new_mask = np.zeros([b_mask_height, b_mask_width,3],dtype=np.int32)
    
    # mask 크기보다 작은경우
    
    det_image_ratio = box_width / box_height
    
    if box_width < s_mask_width or box_height < s_mask_height: 
        
        img_crop = img.crop((x1,y1,s_mask_width+x1, s_mask_height+ y1))
    
    # mask 크기보다 큰경우
    else: 
        if det_image_ratio < 1:

            base_width = s_mask_width
            scaling_ratio = base_width / box_width

            new_width = base_width
            new_height = int(box_height * scaling_ratio)

            resize_det_image_pil_BICUBIC = img_box_crop.resize((new_width, new_height),resample=PIL.Image.BICUBIC)

            img_crop = resize_det_image_pil_BICUBIC.crop((0,0, s_mask_width,s_mask_height))

        else:

            base_height = s_mask_height
            scaling_ratio = base_height / box_height

            new_width = int(box_width * scaling_ratio)
            new_height = base_height

            resize_det_image_pil_BICUBIC = img_box_crop.resize((new_width, new_height),resample=PIL.Image.BICUBIC)  

            img_crop = resize_det_image_pil_BICUBIC.crop((0,0, s_mask_width,s_mask_height))                      

    #new_mask.shape = (166, 186)        
    new_mask[:s_mask_height,:s_mask_width,:] = img_crop    
        
    makeadfile(new_mask,img_path, img_type = '_ad_resize')
    

def PreProcessingImage_v1(img_path, boxes, label, mask_x = 166, mask_y= 184):
    img = Image.open(img_path)
    
    img_w, img_h = img.size
    img_np = np.array(img)
    
    # (0) box check
    if len(boxes) == 0:
        return print('박스없음')
    
    b_mask_width, b_mask_height = 166,184
    s_mask_width, s_mask_height = 130,166
    
    resize_list = [95,100,105]
    
    mask_np = np.asarray(Image.open('mask_file.png','r'))
    ad_np = np.asarray(Image.open('ad_file.png','r'))
    
    # (1) Prediction plot
    
    x1,y1,x2,y2 = boxes[0]
    if x1 < 0:
        x1 = 1
    if y1 < 0:
        y1 = 1
    if x2 >= img_w:
        x2 = img_w-1
    if y2 >= img_h:
        y2 = img_h-1

    print(x1,y1,x2,y2)
    box_width, box_height = abs(x2-x1) , abs(y1-y2)
    
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    rect = patches.Rectangle((x1, y1),box_width, box_height, linewidth=1.5 ,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    fig.savefig('./result/'+img_path.split('/')[-1].split('.')[0]+'_box.png', bbox_inches='tight', pad_inches=0)
    print('./result/'+img_path.split('/')[-1].split('.')[0]+'_box.png'+' 저장완료')
    
    # (2) Detected Box save
    img_box_crop = img.crop((x1,y1,img_w-x1, img_h-y1))
    #img_box_crop = img.crop((x1,y1,box_width+x1, box_height+ y1))
    img_box_crop.save('./result/'+img_path.split('/')[-1].split('.')[0]+'_det.png')
    print('./result/'+img_path.split('/')[-1].split('.')[0]+'_det.png'+' 저장완료')
    
    # (3) AD save
    
    img_box_crop_b = img.crop((x1,y1,b_mask_width+x1, b_mask_height+ y1))
    
    makeadfile(img_box_crop_b, img_path, img_type = '_ad')
    
    # (4) AD-resize

    #new_mask = np.zeros([b_mask_height, b_mask_width,3],dtype=np.int32)
    new_mask = np.tile(255, (b_mask_height, b_mask_width, 3))
    # mask 크기보다 작은경우
    
    det_image_ratio = box_width / box_height
    
    for re in resize_list:
        
        s_mask_width = re
        
    
        if box_width < s_mask_width or box_height < s_mask_height: 
            s_mask_width = 130
            img_crop = img.crop((x1,y1,s_mask_width+x1, s_mask_height+ y1))

        # mask 크기보다 큰경우
        else: 
            if det_image_ratio < 1:

                base_width = s_mask_width
                scaling_ratio = base_width / box_width

                new_width = base_width
                new_height = int(box_height * scaling_ratio)

                resize_det_image_pil_BICUBIC = img_box_crop.resize((new_width, new_height),resample=PIL.Image.BICUBIC)

                img_crop = resize_det_image_pil_BICUBIC.crop((0,0, s_mask_width,s_mask_height))

            else:

                base_height = s_mask_height
                scaling_ratio = base_height / box_height

                new_width = int(box_width * scaling_ratio)
                new_height = base_height

                resize_det_image_pil_BICUBIC = img_box_crop.resize((new_width, new_height),resample=PIL.Image.BICUBIC)  
                
                
                img_crop = resize_det_image_pil_BICUBIC.crop((0,0, s_mask_width,s_mask_height))                      

        #new_mask.shape = (166, 186)        
        new_mask[:s_mask_height,:s_mask_width,:] = img_crop  
        
        img_t = str(re)

        makeadfile(new_mask,img_path, img_type = '_ad_resize_'+ img_t)    
    
