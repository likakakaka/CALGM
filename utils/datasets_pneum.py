import random
import torch.utils.data as Data
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
import os
class JSRT_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(JSRT_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.Resize((512, 512)),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('JPCLN' in labelname)
        label.append('JPCNN' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, label, imgname
class JSRT_wmask_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(JSRT_wmask_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('JPCLN' in labelname)
        label.append('JPCNN' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname
class JSRT_w7masks_5Subregions_wsubroi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(JSRT_w7masks_5Subregions_wsubroi_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

        self.sub_img_size=sub_img_size
        self.csv = pd.read_csv(csvpath)
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        self.pil2tensor_transform = pil2tensor_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        sub_rois = []
        sub_rois2 = []
        imgs=[]
        masks=[]
        labels=[]
        mixed_imgs1 = []
        mixed_masks1 = []
        mixed_labels1 = []
        mixed_imgs2 = []
        mixed_masks2 = []
        mixed_labels2 = []
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask_image2)
        # plt.show()

        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["image_name"] == imgname)]

        if csv_line.size != 0:
            [x,y] = image.size
            masks_index = torch.tensor(2)
            degree_label = 5 - csv_line['degree_of_subtlety'].values[0]

            nodule_xmin = csv_line['x_coordinate_min'].values[0]
            nodule_ymin = csv_line['y_coordinate_min'].values[0]
            nodule_xmax = csv_line['x_coordinate_max'].values[0]
            nodule_ymax = csv_line['y_coordinate_max'].values[0]
            # plt.imshow(image)
            # rect = plt.Rectangle((nodule_xmin, nodule_ymin),
            #                      nodule_xmax - nodule_xmin,
            #                      nodule_ymax - nodule_ymin,
            #                      linewidth=1, edgecolor='red',
            #                      facecolor='none')
            # plt.gca().add_patch(rect)
            # plt.show()


            xmin = 0
            ymin = 0
            xmax = x//2
            ymax = y//3
            if nodule_xmin>xmin and nodule_xmin<xmax and nodule_ymin>ymin and nodule_ymin<ymax:
                left_top_label = np.array([int(degree_label)])
            else:
                left_top_label = np.array([5])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            # plt.imshow(left_top_img)
            # plt.show()
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            xmin = x // 2
            ymin = 0
            xmax = x
            ymax = y // 3
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_top_label = np.array([int(degree_label)])
            else:
                right_top_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_top_img)
            # plt.show()
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]
            xmin = 0
            ymin = y // 3
            xmax = x//2
            ymax = y // 3 *2
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                left_center_label = np.array([int(degree_label)])
            else:
                left_center_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(left_center_img)
            # plt.show()
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            xmin = x // 2
            ymin = y // 3
            xmax = x
            ymax = y// 3 * 2
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_center_label = np.array([int(degree_label)])
            else:
                right_center_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_center_img)
            # plt.show()
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            xmin = 0
            ymin = y // 3 * 2
            xmax = x// 2
            ymax = y
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                left_bottom_label = np.array([int(degree_label)])
            else:
                left_bottom_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(left_bottom_img)
            # plt.show()
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            xmin = x // 2
            ymin = y // 3 * 2
            xmax = x
            ymax = y
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_bottom_label = np.array([int(degree_label)])
            else:
                right_bottom_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_bottom_img)
            # plt.show()
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]


        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        mask_image2 = mask_all_images[0]
        # plt.imshow(image)
        # plt.show()
        label = []
        # label.append('Sick' in labelname)
        label.append('JPCNN' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)

        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage = F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            # plt.imshow(mask_new_subimage)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)
        return imgs, masks, labels, sub_rois_new, masks_index, imgname
class JSRT_w7masks_5Subregions_wsubroi_Mixednew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(JSRT_w7masks_5Subregions_wsubroi_Mixednew_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

        self.sub_img_size=sub_img_size
        self.csv = pd.read_csv(csvpath)
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        self.pil2tensor_transform = pil2tensor_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        sub_rois = []
        sub_rois2 = []
        imgs=[]
        masks=[]
        labels=[]
        mixed_imgs1 = []
        mixed_masks1 = []
        mixed_labels1 = []
        mixed_imgs2 = []
        mixed_masks2 = []
        mixed_labels2 = []
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask_image2)
        # plt.show()

        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["image_name"] == imgname)]

        if csv_line.size != 0:
            [x,y] = image.size
            masks_index = torch.tensor(2)
            degree_label = 5 - csv_line['degree_of_subtlety'].values[0]

            nodule_xmin = csv_line['x_coordinate_min'].values[0]
            nodule_ymin = csv_line['y_coordinate_min'].values[0]
            nodule_xmax = csv_line['x_coordinate_max'].values[0]
            nodule_ymax = csv_line['y_coordinate_max'].values[0]
            # plt.imshow(image)
            # rect = plt.Rectangle((nodule_xmin, nodule_ymin),
            #                      nodule_xmax - nodule_xmin,
            #                      nodule_ymax - nodule_ymin,
            #                      linewidth=1, edgecolor='red',
            #                      facecolor='none')
            # plt.gca().add_patch(rect)
            # plt.show()


            xmin = 0
            ymin = 0
            xmax = x//2
            ymax = y//3
            if nodule_xmin>xmin and nodule_xmin<xmax and nodule_ymin>ymin and nodule_ymin<ymax:
                left_top_label = np.array([int(degree_label)])
            else:
                left_top_label = np.array([5])

            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            # plt.imshow(left_top_img)
            # plt.show()
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            xmin = x // 2
            ymin = 0
            xmax = x
            ymax = y // 3
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_top_label = np.array([int(degree_label)])
            else:
                right_top_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_top_img)
            # plt.show()
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]
            xmin = 0
            ymin = y // 3
            xmax = x//2
            ymax = y // 3 *2
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                left_center_label = np.array([int(degree_label)])
            else:
                left_center_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(left_center_img)
            # plt.show()
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            xmin = x // 2
            ymin = y // 3
            xmax = x
            ymax = y// 3 * 2
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_center_label = np.array([int(degree_label)])
            else:
                right_center_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_center_img)
            # plt.show()
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            xmin = 0
            ymin = y // 3 * 2
            xmax = x// 2
            ymax = y
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                left_bottom_label = np.array([int(degree_label)])
            else:
                left_bottom_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(left_bottom_img)
            # plt.show()
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            xmin = x // 2
            ymin = y // 3 * 2
            xmax = x
            ymax = y
            if nodule_xmin > xmin and nodule_xmin < xmax and nodule_ymin > ymin and nodule_ymin < ymax:
                right_bottom_label = np.array([int(degree_label)])
            else:
                right_bottom_label = np.array([5])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_bottom_img)
            # plt.show()
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["image_name"] == imgname2)]
                if csv_line2.size != 0 and imgname2!=imgname:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    [x2,y2] = image2.size
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mask_image2_2)
                    # plt.show()

                    nodule_xmin2 = csv_line2['x_coordinate_min'].values[0]
                    nodule_ymin2 = csv_line2['y_coordinate_min'].values[0]
                    nodule_xmax2 = csv_line2['x_coordinate_max'].values[0]
                    nodule_ymax2 = csv_line2['y_coordinate_max'].values[0]
                    degree_label2 = 5 - csv_line2['degree_of_subtlety'].values[0]

                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()

                    xmin = 0
                    ymin = 0
                    xmax = x2 // 2
                    ymax = y2 // 3
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        left_top_label2 = np.array([int(degree_label2)])
                    else:
                        left_top_label2 = np.array([5])

                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_top_img2)
                    # plt.show()
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    xmin = x2 // 2
                    ymin = 0
                    xmax = x2
                    ymax = y2 // 3
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        right_top_label2 = np.array([int(degree_label2)])
                    else:
                        right_top_label2 = np.array([5])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(right_top_img)
                    # plt.show()
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label2)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]
                    xmin = 0
                    ymin = y2 // 3
                    xmax = x2 // 2
                    ymax = y2 // 3 * 2
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        left_center_label2 = np.array([int(degree_label2)])
                    else:
                        left_center_label2 = np.array([5])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_center_img)
                    # plt.show()
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label2)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    xmin = x2 // 2
                    ymin = y2 // 3
                    xmax = x2
                    ymax = y2 // 3 * 2
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        right_center_label2 = np.array([int(degree_label2)])
                    else:
                        right_center_label2 = np.array([5])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(right_center_img)
                    # plt.show()
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label2)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    xmin = 0
                    ymin = y2 // 3 * 2
                    xmax = x2 // 2
                    ymax = y2
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        left_bottom_label2 = np.array([int(degree_label2)])
                    else:
                        left_bottom_label2 = np.array([5])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_bottom_img)
                    # plt.show()
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label2)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    xmin = x2 // 2
                    ymin = y2 // 3 * 2
                    xmax = x2
                    ymax = y2
                    if nodule_xmin2 > xmin and nodule_xmin2 < xmax and nodule_ymin2 > ymin and nodule_ymin2 < ymax:
                        right_bottom_label2 = np.array([int(degree_label2)])
                    else:
                        right_bottom_label2 = np.array([5])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(right_bottom_img)
                    # plt.show()
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label2)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)

                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_subrois1=[sub_rois[0].clone(),sub_rois2[1].clone(),sub_rois[2].clone(),sub_rois2[3].clone(),sub_rois[4].clone(),sub_rois2[5].clone()]


                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np
                    mixed_subrois2 = [sub_rois2[0].clone(), sub_rois[1].clone(), sub_rois2[2].clone(), sub_rois[3].clone(), sub_rois2[4].clone(), sub_rois[5].clone()]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)

                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_mask1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    # plt.imshow(mixed_mask2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(values1)
                    if count1 != 30:
                        mixed_label1 = np.array([0])
                    else:
                        mixed_label1 = np.array([1])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(values2)
                    if count2 != 30:
                        mixed_label2 = np.array([0])
                    else:
                        mixed_label2 = np.array([1])

        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        mask_image2 = mask_all_images[0]
        # plt.imshow(image)
        # plt.show()
        label = []
        # label.append('Sick' in labelname)
        label.append('JPCNN' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)

        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage = F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            # plt.imshow(mask_new_subimage)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)


        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])
            # plt.imshow(mixed_img1)
            # plt.show()

            mixed_img1, mixed_mask1s, mixed_subrois1 = self.transforms(mixed_img1, mixed_mask1s, sub_rois=mixed_subrois1)
            mixed_mask_image = mixed_mask1s[0]
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask_image)
            # plt.show()
            for ii in range(len(mixed_subrois1)):
                [xmin, ymin, xmax, ymax] = mixed_subrois1[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img1, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                # plt.imshow(mask_new_subimage)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs1.append(imga)
                mixed_masks1.append(maska)
                mixed_labels1.append(mixed_sublabels[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois1_new = self.pil2tensor_transform(mixed_img1, mixed_mask1s,
                                                                                           sub_rois=mixed_subrois1)

            mixed_imgs1.insert(0, mixed_image_tensor)
            mixed_masks1.insert(0, mixed_mask_all_images_tensor)
            mixed_labels1.insert(0, mixed_label1)


            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])
            mixed_img2, mixed_mask2s, mixed_subrois2 = self.transforms(mixed_img2, mixed_mask2s,
                                                                       sub_rois=mixed_subrois2)
            # plt.imshow(mixed_img1)
            # plt.show()
            mixed_mask_image2 = mixed_mask2s[0]
            for ii in range(len(mixed_subrois2)):
                [xmin, ymin, xmax, ymax] = mixed_subrois2[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img2, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image2, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs2.append(imga)
                mixed_masks2.append(maska)
                mixed_labels2.append(mixed_sublabels2[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois2_new = self.pil2tensor_transform(mixed_img2,
                                                                                                             mixed_mask2s,
                                                                                                             sub_rois=mixed_subrois2)

            mixed_imgs2.insert(0, mixed_image_tensor)
            mixed_masks2.insert(0, mixed_mask_all_images_tensor)
            mixed_labels2.insert(0, mixed_label2)
        return [imgs, mixed_imgs1, mixed_imgs2], [masks, mixed_masks1, mixed_masks2], [labels, mixed_labels1, mixed_labels2], [sub_rois_new,mixed_subrois1_new,mixed_subrois2_new],masks_index, imgname


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img

import pandas as pd
from torchvision.transforms import functional as F

from PIL import Image

class Shanxi_wmask_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_wmask_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image = Image.open(mask_img_path)
        image, mask_image = self.transforms(image, mask_image)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label
        return image, mask_image, label, imgname



class Shanxi_w7masks_5Subregions_wsubroi_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

        self.sub_img_size=sub_img_size
        self.csv = pd.read_excel(csvpath)

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        self.pil2tensor_transform=pil2tensor_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        sub_rois = []
        imgs=[]
        masks=[]
        labels=[]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        aa=np.array(image)
        img_size = image.size[0]
        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)
        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]
        if csv_line.size != 0:
            masks_index=torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            elif 'unknown' in left_top_label:
                left_top_label=np.array([-1])
            else:
                left_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[int(ymin):int(ymax), int(xmin):int(xmax)]
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)



            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            elif 'unknown' in right_top_label:
                right_top_label = np.array([-1])
            else:
                right_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_center_label = values[0, 0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            elif 'unknown' in left_center_label:
                left_center_label = np.array([-1])
            else:
                left_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            elif 'unknown' in right_center_label:
                right_center_label = np.array([-1])
            else:
                right_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            elif 'unknown' in left_bottom_label:
                left_bottom_label = np.array([-1])
            else:
                left_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]


            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1]/1024*img_size
            ymin = values[0, 2]/1024*img_size
            xmax = values[0, 3]/1024*img_size
            ymax = values[0, 4]/1024*img_size
            right_bottom_label = values[0, 0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            elif 'unknown' in right_bottom_label:
                right_bottom_label = np.array([-1])
            else:
                right_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]
        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        ab = np.array(image)
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        ac = np.array(image)
        mask_image2=mask_all_images[0]
        # image, mask_all_images = self.transforms(image, mask_all_images)
        # print(imgname)

        label = []
        # label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        # random_noise=random.random()
        # if random_noise < self.label_noise_radio:
        #     label=1-label


        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage= F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        ad = np.array(image)
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)
        return imgs, masks, labels, sub_rois_new, masks_index, imgname

class Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 csvpath,
                 data_transform=None,
                 data_subregions_transform=None,
                 pil2tensor_transform=None,
                 sub_img_size=96,
                 seed=0,
                 label_noise_radio=0.
                 ):
        super(Shanxi_w7masks_5Subregions_wsubroi_Mixednew_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath=maskimgpath
        self.csvpath=csvpath

        self.sub_img_size=sub_img_size
        self.csv = pd.read_excel(csvpath)

        self.txtpath = txtpath
        self.label_noise_radio=label_noise_radio

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        self.data_subregions_transform=data_subregions_transform
        self.pil2tensor_transform = pil2tensor_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()


    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        subregions_imgs=[]
        subregions_labels=[]
        subregions_masks=[]
        subregions_imgs2=[]
        subregions_labels2 = []
        subregions_masks2 = []
        sub_rois = []
        sub_rois2 = []
        imgs=[]
        masks=[]
        labels=[]
        mixed_imgs1 = []
        mixed_masks1 = []
        mixed_labels1 = []
        mixed_imgs2 = []
        mixed_masks2 = []
        mixed_labels2 = []
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_img_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        mask_image2 = Image.open(mask_img_path)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(mask_image2)
        # plt.show()

        mask_new_subimage_org = np.asarray(mask_image2)
        mask_new_subimage_lefttop = np.zeros_like(mask_new_subimage_org)
        mask_new_subimage_leftcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_leftbottom = mask_new_subimage_lefttop.copy()
        mask_new_subimage_righttop = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightcenter = mask_new_subimage_lefttop.copy()
        mask_new_subimage_rightbottom = mask_new_subimage_lefttop.copy()
        csv_line = self.csv.loc[(self.csv["胸片名称"] == imgname)]

        if csv_line.size != 0:
            img_size = image.size[0]
            masks_index = torch.tensor(2)
            left_upper_index = csv_line.columns.get_loc('右上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_top_label = values[0]
            if '0/0' in left_top_label:
                left_top_label = np.array([4])
            elif '0/1' in left_top_label:
                left_top_label = np.array([3])
            elif '1/0' in left_top_label:
                left_top_label = np.array([2])
            elif '1/1' in left_top_label:
                left_top_label = np.array([1])
            elif 'unknown' in left_top_label:
                left_top_label=np.array([-1])
            else:
                left_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage_lefttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                  int(ymin):int(ymax),
                                                                                  int(xmin):int(xmax)]
            # plt.imshow(left_top_img)
            # plt.show()
            subregions_imgs.append(left_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_top_label)

            left_upper_index = csv_line.columns.get_loc('左上')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_top_label = values[0]
            if '0/0' in right_top_label:
                right_top_label = np.array([4])
            elif '0/1' in right_top_label:
                right_top_label = np.array([3])
            elif '1/0' in right_top_label:
                right_top_label = np.array([2])
            elif '1/1' in right_top_label:
                right_top_label = np.array([1])
            elif 'unknown' in right_top_label:
                right_top_label=np.array([-1])
            else:
                right_top_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_top_img = F.crop(image, ymin, xmin, height, width)
            # plt.imshow(right_top_img)
            # plt.show()
            subregions_imgs.append(right_top_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(right_top_label)
            mask_new_subimage_righttop[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                   int(ymin):int(ymax),
                                                                                   int(xmin):int(xmax)]

            # plt.imshow(F.crop(mask_image, ymin, xmin, height, width))
            # plt.show()
            left_upper_index = csv_line.columns.get_loc('右中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_center_label = values[0]
            if '0/0' in left_center_label:
                left_center_label = np.array([4])
            elif '0/1' in left_center_label:
                left_center_label = np.array([3])
            elif '1/0' in left_center_label:
                left_center_label = np.array([2])
            elif '1/1' in left_center_label:
                left_center_label = np.array([1])
            elif 'unknown' in left_center_label:
                left_center_label = np.array([-1])
            else:
                left_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(left_center_img)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_labels.append(left_center_label)
            mask_new_subimage_leftcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左中')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_center_label = values[0]
            if '0/0' in right_center_label:
                right_center_label = np.array([4])
            elif '0/1' in right_center_label:
                right_center_label = np.array([3])
            elif '1/0' in right_center_label:
                right_center_label = np.array([2])
            elif '1/1' in right_center_label:
                right_center_label = np.array([1])
            elif 'unknown' in right_center_label:
                right_center_label = np.array([-1])
            else:
                right_center_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            right_center_img = F.crop(image, ymin, xmin, height, width)
            subregions_imgs.append(right_center_img)
            subregions_labels.append(right_center_label)
            mask_new_subimage_rightcenter[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('右下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            left_bottom_label = values[0]
            if '0/0' in left_bottom_label:
                left_bottom_label = np.array([4])
            elif '0/1' in left_bottom_label:
                left_bottom_label = np.array([3])
            elif '1/0' in left_bottom_label:
                left_bottom_label = np.array([2])
            elif '1/1' in left_bottom_label:
                left_bottom_label = np.array([1])
            elif 'unknown' in left_bottom_label:
                left_bottom_label = np.array([-1])
            else:
                left_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            left_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(left_bottom_img)
            subregions_labels.append(left_bottom_label)
            mask_new_subimage_leftbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                     int(ymin):int(ymax),
                                                                                     int(xmin):int(xmax)]

            left_upper_index = csv_line.columns.get_loc('左下')
            next_three_columns = csv_line.columns[left_upper_index:left_upper_index + 5]
            values = csv_line[next_three_columns].values
            xmin = values[0, 1] / 1024 * img_size
            ymin = values[0, 2] / 1024 * img_size
            xmax = values[0, 3] / 1024 * img_size
            ymax = values[0, 4] / 1024 * img_size
            right_bottom_label = values[0]
            if '0/0' in right_bottom_label:
                right_bottom_label = np.array([4])
            elif '0/1' in right_bottom_label:
                right_bottom_label = np.array([3])
            elif '1/0' in right_bottom_label:
                right_bottom_label = np.array([2])
            elif '1/1' in right_bottom_label:
                right_bottom_label = np.array([1])
            elif 'unknown' in right_bottom_label:
                right_bottom_label = np.array([-1])
            else:
                right_bottom_label = np.array([0])
            sub_roi = torch.tensor([xmin, ymin, xmax, ymax])
            sub_rois.append(sub_roi)
            height = ymax - ymin
            width = xmax - xmin
            right_bottom_img = F.crop(image, ymin, xmin, height, width)
            subregions_masks.append(F.crop(mask_image2, ymin, xmin, height, width))
            subregions_imgs.append(right_bottom_img)
            subregions_labels.append(right_bottom_label)
            mask_new_subimage_rightbottom[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org[
                                                                                      int(ymin):int(ymax),
                                                                                      int(xmin):int(xmax)]

            choosen_index = 0
            while (choosen_index) == 0:
                randomidx = random.randint(0, len(self.lines) - 1)
                line2 = self.lines[randomidx]
                imgname2 = line2.split('\n')[0]
                csv_line2 = self.csv.loc[(self.csv["胸片名称"] == imgname2)]
                if csv_line2.size != 0:
                    choosen_index = 1
                    img_path2 = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                    image2 = Image.open(img_path2).convert('RGB')
                    img_size = image2.size[0]
                    mask_img_path2 = os.path.join(self.maskimgpath, imgname2.split('.png')[0] + '.png')
                    mask_image2_2 = Image.open(mask_img_path2)
                    # plt.imshow(image2)
                    # plt.show()
                    # plt.imshow(mask_image2_2)
                    # plt.show()
                    mask_new_subimage_org2 = np.asarray(mask_image2_2)
                    mask_new_subimage_lefttop2 = np.zeros_like(mask_new_subimage_org2)
                    mask_new_subimage_leftcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_leftbottom2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_righttop2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightcenter2 = mask_new_subimage_lefttop2.copy()
                    mask_new_subimage_rightbottom2 = mask_new_subimage_lefttop2.copy()
                    left_upper_index2 = csv_line2.columns.get_loc('右上')
                    next_three_columns2 = csv_line2.columns[left_upper_index2:left_upper_index2 + 5]
                    values = csv_line2[next_three_columns2].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_top_label2 = values[0]
                    if '0/0' in left_top_label2:
                        left_top_label2 = np.array([4])
                    elif '0/1' in left_top_label2:
                        left_top_label2 = np.array([3])
                    elif '1/0' in left_top_label2:
                        left_top_label2 = np.array([2])
                    elif '1/1' in left_top_label2:
                        left_top_label2 = np.array([1])
                    elif 'unknown' in left_top_label2:
                        left_top_label2 = np.array([-1])
                    else:
                        left_top_label2 = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_top_img2 = F.crop(image2, ymin, xmin, height, width)
                    # plt.imshow(left_top_img2)
                    # plt.show()
                    mask_new_subimage_lefttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                           int(ymin):int(ymax),
                                                                                           int(xmin):int(xmax)]
                    subregions_imgs2.append(left_top_img2)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_top_label2)

                    left_upper_index = csv_line2.columns.get_loc('左上')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_top_label = values[0]
                    if '0/0' in right_top_label:
                        right_top_label = np.array([4])
                    elif '0/1' in right_top_label:
                        right_top_label = np.array([3])
                    elif '1/0' in right_top_label:
                        right_top_label = np.array([2])
                    elif '1/1' in right_top_label:
                        right_top_label = np.array([1])
                    elif 'unknown' in right_top_label:
                        right_top_label = np.array([-1])
                    else:
                        right_top_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_top_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_top_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(right_top_label)
                    mask_new_subimage_righttop2[int(ymin):int(ymax), int(xmin):int(xmax)] = mask_new_subimage_org2[
                                                                                            int(ymin):int(ymax),
                                                                                            int(xmin):int(xmax)]


                    left_upper_index = csv_line2.columns.get_loc('右中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_center_label = values[0]
                    if '0/0' in left_center_label:
                        left_center_label = np.array([4])
                    elif '0/1' in left_center_label:
                        left_center_label = np.array([3])
                    elif '1/0' in left_center_label:
                        left_center_label = np.array([2])
                    elif '1/1' in left_center_label:
                        left_center_label = np.array([1])
                    elif 'unknown' in left_center_label:
                        left_center_label = np.array([-1])
                    else:
                        left_center_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(left_center_img)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_labels2.append(left_center_label)
                    mask_new_subimage_leftcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左中')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_center_label = values[0]
                    if '0/0' in right_center_label:
                        right_center_label = np.array([4])
                    elif '0/1' in right_center_label:
                        right_center_label = np.array([3])
                    elif '1/0' in right_center_label:
                        right_center_label = np.array([2])
                    elif '1/1' in right_center_label:
                        right_center_label = np.array([1])
                    elif 'unknown' in right_center_label:
                        right_center_label = np.array([-1])
                    else:
                        right_center_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    right_center_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_imgs2.append(right_center_img)
                    subregions_labels2.append(right_center_label)
                    mask_new_subimage_rightcenter2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('右下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    left_bottom_label = values[0]
                    if '0/0' in left_bottom_label:
                        left_bottom_label = np.array([4])
                    elif '0/1' in left_bottom_label:
                        left_bottom_label = np.array([3])
                    elif '1/0' in left_bottom_label:
                        left_bottom_label = np.array([2])
                    elif '1/1' in left_bottom_label:
                        left_bottom_label = np.array([1])
                    elif 'unknown' in left_bottom_label:
                        left_bottom_label = np.array([-1])
                    else:
                        left_bottom_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    left_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(left_bottom_img)
                    subregions_labels2.append(left_bottom_label)
                    mask_new_subimage_leftbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]

                    left_upper_index = csv_line2.columns.get_loc('左下')
                    next_three_columns = csv_line2.columns[left_upper_index:left_upper_index + 5]
                    values = csv_line2[next_three_columns].values
                    xmin = values[0, 1] / 1024 * img_size
                    ymin = values[0, 2] / 1024 * img_size
                    xmax = values[0, 3] / 1024 * img_size
                    ymax = values[0, 4] / 1024 * img_size
                    right_bottom_label = values[0]
                    if '0/0' in right_bottom_label:
                        right_bottom_label = np.array([4])
                    elif '0/1' in right_bottom_label:
                        right_bottom_label = np.array([3])
                    elif '1/0' in right_bottom_label:
                        right_bottom_label = np.array([2])
                    elif '1/1' in right_bottom_label:
                        right_bottom_label = np.array([1])
                    elif 'unknown' in right_bottom_label:
                        right_bottom_label = np.array([-1])
                    else:
                        right_bottom_label = np.array([0])
                    sub_roi2 = torch.tensor([xmin, ymin, xmax, ymax])
                    sub_rois2.append(sub_roi2)
                    height = ymax - ymin
                    width = xmax - xmin
                    right_bottom_img = F.crop(image2, ymin, xmin, height, width)
                    subregions_masks2.append(F.crop(mask_image2_2, ymin, xmin, height, width))
                    subregions_imgs2.append(right_bottom_img)
                    subregions_labels2.append(right_bottom_label)
                    mask_new_subimage_rightbottom2[int(ymin):int(ymax),
                    int(xmin):int(xmax)] = mask_new_subimage_org2[
                                           int(ymin):int(ymax),
                                           int(xmin):int(xmax)]
                    image2_np = np.asarray(image2)
                    image_np = np.asarray(image)

                    mixed_img1 = np.expand_dims((
                                                        mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2) // 255,
                                                axis=-1) * image2_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop) // 255,
                        axis=-1) * image_np
                    mixed_subrois1=[sub_rois[0].clone(),sub_rois2[1].clone(),sub_rois[2].clone(),sub_rois2[3].clone(),sub_rois[4].clone(),sub_rois2[5].clone()]


                    mixed_img2 = np.expand_dims(
                        (
                                mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop) // 255,
                        axis=-1) * image_np + np.expand_dims(
                        (
                                mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2) // 255,
                        axis=-1) * image2_np
                    mixed_subrois2 = [sub_rois2[0].clone(), sub_rois[1].clone(), sub_rois2[2].clone(), sub_rois[3].clone(), sub_rois2[4].clone(), sub_rois[5].clone()]

                    mixed_mask1 = mask_new_subimage_rightbottom2 + mask_new_subimage_rightcenter2 + mask_new_subimage_righttop2 + mask_new_subimage_leftbottom + mask_new_subimage_leftcenter + mask_new_subimage_lefttop
                    mixed_mask2 = mask_new_subimage_rightbottom + mask_new_subimage_rightcenter + mask_new_subimage_righttop + mask_new_subimage_leftbottom2 + mask_new_subimage_leftcenter2 + mask_new_subimage_lefttop2
                    mixed_mask1 = Image.fromarray(mixed_mask1, mode='L')
                    mixed_mask2 = Image.fromarray(mixed_mask2, mode='L')

                    mixed_img1 = Image.fromarray(mixed_img1)
                    mixed_img2 = Image.fromarray(mixed_img2)

                    # plt.imshow(mixed_img1)
                    # plt.show()
                    # plt.imshow(mixed_mask1)
                    # plt.show()
                    # plt.imshow(mixed_img2)
                    # plt.show()
                    # plt.imshow(mixed_mask2)
                    # plt.show()
                    values1 = [
                        subregions_labels2[1],  # 0
                        subregions_labels2[3],  # 1
                        subregions_labels2[5],  # 40
                        subregions_labels[0],  # 0
                        subregions_labels[2],  # 1
                        subregions_labels[4],  # 0
                    ]
                    count1 = sum(1 for x in values1 if x == 4 or x == 3)
                    if count1 <= 4:
                        mixed_label1 = np.array([0])
                    else:
                        mixed_label1 = np.array([1])

                    values2 = [
                        subregions_labels2[0],  # 0
                        subregions_labels2[2],  # 1
                        subregions_labels2[4],  # 40
                        subregions_labels[1],  # 0
                        subregions_labels[3],  # 1
                        subregions_labels[5],  # 0
                    ]
                    count2 = sum(1 for x in values2 if x == 4 or x == 3)
                    if count2 <= 4:
                        mixed_label2 = np.array([0])
                    else:
                        mixed_label2 = np.array([1])

        else:
            masks_index=torch.tensor(1)
        mask_all_images=[]
        mask_all_images.append(mask_image2)
        mask_all_images.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
        mask_all_images.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
        image, mask_all_images, sub_rois = self.transforms(image, mask_all_images, sub_rois=sub_rois)
        mask_image2 = mask_all_images[0]
        # plt.imshow(image)
        # plt.show()
        label = []
        # label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)

        for ii in range(len(sub_rois)):
            [xmin, ymin, xmax, ymax] = sub_rois[ii].numpy()
            height = ymax - ymin
            width = xmax - xmin
            left_top_img = F.crop(image, ymin, xmin, height, width)
            mask_new_subimage = F.crop(mask_image2, ymin, xmin, height, width)
            # rgb_img = np.array(denorm_batch)
            # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
            # plt.imshow(left_top_img)
            # plt.show()
            # plt.imshow(mask_new_subimage)
            # plt.show()
            imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
            imgs.append(imga)
            masks.append(maska)
            labels.append(subregions_labels[ii])
        image_tensor, mask_all_images_tensor, sub_rois_new = self.pil2tensor_transform(image, mask_all_images,
                                                                                       sub_rois=sub_rois)

        imgs.insert(0, image_tensor)
        masks.insert(0, mask_all_images_tensor)
        labels.insert(0, label)


        mixed_sublabels=[]
        mixed_sublabels2=[]
        if subregions_imgs!=[]:
            mixed_mask1s = []
            mixed_mask1s.append(mixed_mask1)
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_lefttop, mode='L'))
            mixed_sublabels.append(subregions_labels[0])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_righttop2, mode='L'))
            mixed_sublabels.append(subregions_labels2[1])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftcenter, mode='L'))
            mixed_sublabels.append(subregions_labels[2])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightcenter2, mode='L'))
            mixed_sublabels.append(subregions_labels2[3])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_leftbottom, mode='L'))
            mixed_sublabels.append(subregions_labels[4])
            mixed_mask1s.append(Image.fromarray(mask_new_subimage_rightbottom2, mode='L'))
            mixed_sublabels.append(subregions_labels2[5])
            # plt.imshow(mixed_img1)
            # plt.show()

            mixed_img1, mixed_mask1s, mixed_subrois1 = self.transforms(mixed_img1, mixed_mask1s, sub_rois=mixed_subrois1)
            mixed_mask_image = mixed_mask1s[0]
            # plt.imshow(mixed_img1)
            # plt.show()
            # plt.imshow(mixed_mask_image)
            # plt.show()
            for ii in range(len(mixed_subrois1)):
                [xmin, ymin, xmax, ymax] = mixed_subrois1[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img1, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                # plt.imshow(mask_new_subimage)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs1.append(imga)
                mixed_masks1.append(maska)
                mixed_labels1.append(mixed_sublabels[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois1_new = self.pil2tensor_transform(mixed_img1, mixed_mask1s,
                                                                                           sub_rois=mixed_subrois1)

            mixed_imgs1.insert(0, mixed_image_tensor)
            mixed_masks1.insert(0, mixed_mask_all_images_tensor)
            mixed_labels1.insert(0, mixed_label1)


            mixed_mask2s = []
            mixed_mask2s.append(mixed_mask2)
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_lefttop2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[0])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_righttop, mode='L'))
            mixed_sublabels2.append(subregions_labels[1])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftcenter2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[2])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightcenter, mode='L'))
            mixed_sublabels2.append(subregions_labels[3])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_leftbottom2, mode='L'))
            mixed_sublabels2.append(subregions_labels2[4])
            mixed_mask2s.append(Image.fromarray(mask_new_subimage_rightbottom, mode='L'))
            mixed_sublabels2.append(subregions_labels[5])
            mixed_img2, mixed_mask2s, mixed_subrois2 = self.transforms(mixed_img2, mixed_mask2s,
                                                                       sub_rois=mixed_subrois2)
            # plt.imshow(mixed_img1)
            # plt.show()
            mixed_mask_image2 = mixed_mask2s[0]
            for ii in range(len(mixed_subrois2)):
                [xmin, ymin, xmax, ymax] = mixed_subrois2[ii].numpy()
                height = ymax - ymin
                width = xmax - xmin
                left_top_img = F.crop(mixed_img2, ymin, xmin, height, width)
                mask_new_subimage = F.crop(mixed_mask_image2, ymin, xmin, height, width)
                # rgb_img = np.array(denorm_batch)
                # mixed_img1 = Image.fromarray((rgb_img * 255).astype(np.uint8))
                # plt.imshow(left_top_img)
                # plt.show()
                imga, maska = self.data_subregions_transform(left_top_img, mask_new_subimage)
                mixed_imgs2.append(imga)
                mixed_masks2.append(maska)
                mixed_labels2.append(mixed_sublabels2[ii])
            mixed_image_tensor, mixed_mask_all_images_tensor, mixed_subrois2_new = self.pil2tensor_transform(mixed_img2,
                                                                                                             mixed_mask2s,
                                                                                                             sub_rois=mixed_subrois2)

            mixed_imgs2.insert(0, mixed_image_tensor)
            mixed_masks2.insert(0, mixed_mask_all_images_tensor)
            mixed_labels2.insert(0, mixed_label2)
        return [imgs, mixed_imgs1, mixed_imgs2], [masks, mixed_masks1, mixed_masks2], [labels, mixed_labels1, mixed_labels2], [sub_rois_new,mixed_subrois1_new,mixed_subrois2_new],masks_index, imgname

