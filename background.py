from PIL import Image
import cv2
import os
import glob
import random
import math
import numpy as np
def Picture_Synthesis(mother_img,
                      son_img,
                      save_img,
                      coordinate,
                      f):
    """
    :param f:
    :param mother_img
    :param son_img
    :param save_img: new image name
    :param coordinate
    :return:
    """
    M_Img = Image.open(mother_img)
    S_Img = Image.open(son_img)
    r, g, b, a = S_Img.split()  # a is the mask to make background transparent
    #random factor
    #factor = random.uniform(1, 2.5)
    factor = f # factor resize the son image，2 means a half

    # color mode
    M_Img = M_Img.convert("RGBA")  # CMYK/RGBA

    # image size
    M_Img_w, M_Img_h = M_Img.size  # image size mother
    print("母图尺寸：", M_Img.size)
    S_Img_w, S_Img_h = S_Img.size  # image size son
    print("子图尺寸：", S_Img.size)

    size_w = int(S_Img_w * factor)
    size_h = int(S_Img_h * factor)

    # in case that son is larger than original
    if S_Img_w > size_w:
        S_Img_w = size_w
    if S_Img_h > size_h:
        S_Img_h = size_h

    #resize mother
    M_Img = M_Img.resize((300, 300), Image.ANTIALIAS)
    M_Img_w, M_Img_h = M_Img.size
    # in case son is larger than mother
    if S_Img_w > M_Img_w:
        S_Img_w = M_Img_w
    if S_Img_h > M_Img_h:
        S_Img_h = M_Img_h

    # # resize son
    # icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
    icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
    w = int((M_Img_w - S_Img_w) / 2)
    h = int((M_Img_h - S_Img_h) / 2)
    a = a.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)


    try:
        if coordinate == None or coordinate == "":
            # random coordinate
            #random_w = random.uniform(0, 2*w)
            #random_h = random.uniform(h, 2*h)
            #coordinate = (int(random_w), int(random_h))
            coordinate = (w, h)# do nothing will be centered
            print(coordinate)
            M_Img.paste(icon, coordinate, mask=a)
        else:
            print("coordinate is given")
            print(coordinate)
            M_Img.paste(icon, coordinate, mask=a)
    except:
        print("coordinate error ")
    # save image
    M_Img.save(save_img)

if __name__ == '__main__':
    # background mother
    bgdir = "D:\\deeplabcut\\blender\\background\\"
    bg_glob = os.path.join(bgdir, "*.jpg")
    bg_name_list = []
    bg_name_list.extend(glob.glob(bg_glob))
    print(len(bg_name_list))

    # image son
    imagedir = "D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model1\\merge\\"
    image_glob = os.path.join(imagedir, "*.png")
    image_name_list = []
    image_name_list.extend(glob.glob(image_glob))
    print(len(image_name_list))
    i = 0
    for image_i in image_name_list:
        savepath = "D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model1\\merge-withback\\" + str(i).zfill(4) + ".png"
        #random bg
        bg_i = random.randint(0, len(bg_name_list)-1)
        #random coordinate
        scalefactor = random.uniform(0.8, 1)
        coordinate_x = random.randint(int(math.fabs(scalefactor-1)*130)*(-1),int(math.fabs(scalefactor-1)*130))
        coordinate_y = random.randint(int(math.fabs(scalefactor-1)*130)*(-1),int(math.fabs(scalefactor-1)*130))

        Picture_Synthesis(mother_img=bg_name_list[bg_i], son_img=image_i, save_img=savepath,
                          coordinate=[coordinate_x, coordinate_y], f=scalefactor)  # coordinate=(50,50)
        data = [str(coordinate_x), ' ', str(coordinate_y), ' ', str(scalefactor)]
        print(data)
        with open("D:\\deeplabcut\\blender\\vae\\600edit_1.0\\model1\\data.txt", "a") as f:
            f.writelines(data)
            f.write('\n')
        i += 1
