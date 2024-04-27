import os
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

def path_join(path,dir):
    path = os.path.join(path,dir)
    label = int(dir) - 1
    return path,label
class MyDataset(Dataset):
    def __init__(self,root):
        self.imgs = []
        for path in os.listdir(root):
            path,label = path_join(root,path)
            for img_path in os.listdir(path):
                self.imgs.append((os.path.join(path,img_path),label))     
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.106, 0.127, 0.076],std=[0.049, 0.064, 0.030])
        ])
    def __getitem__(self,index):
        img_path= self.imgs[index][0]
        img_label = self.imgs[index][1]
        data = Image.open(img_path)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = self.transform(data)
        return data,img_label
    def __len__(self):
        return len(self.imgs)  

def MyDataLoader(root,batch_size=64):
    datasets = MyDataset(root)
    dataLoader = DataLoader(datasets,batch_size=batch_size,shuffle=True,num_workers=4)
    return dataLoader


# if __name__ == '__main__':
#     root = 'data/train'
#     dataLoader = MyDataLoader(root,128)
#     for batch_idx, (data, labels) in enumerate(dataLoader):
#         print("Batch:", batch_idx)
#         print("Data shape:", data.shape)
#         print("Labels:", labels)
    # myDataset = MyDataset(root=root)
    # data,label = myDataset.__getitem__(1)
    # print(data)
    # print(label)
   
