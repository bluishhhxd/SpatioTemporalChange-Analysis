import os
import torch
import rasterio
import numpy as np
import time
import random
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

EXP_NAME="FinalFTL"
SAVE_DIR=os.path.join("experiments", EXP_NAME)
LOG_FILE=os.path.join(SAVE_DIR, "train_log.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

def get_key(filename):
    parts=filename.replace('.tif', '').split('-')
    return f"{parts[0]}-{parts[2]}-{parts[3]}-{parts[4]}"

class ChangeDataset(Dataset):
    def __init__(self,root):

        self.root=root
        self.img2006_root=os.path.join(root, "images_2006", "2006")
        self.img2012_root=os.path.join(root, "images_2012", "2012")
        self.lbl2006_root=os.path.join(root, "labels_land_cover_2006", "2006")
        self.lbl2012_root=os.path.join(root, "labels_land_cover_2012", "2012")
        self.files=[]
        self.src_cache = {}
        # self.cache={}
        
        # print(os.listdir(self.lbl2006_root + "/D14")[:5])
        # print(os.listdir(self.lbl2012_root + "/D14")[:5])
        for region in os.listdir(self.img2006_root):
            folder_2006=os.path.join(self.img2006_root, region)
            folder_2012=os.path.join(self.img2012_root, region)
            files_2006=os.listdir(folder_2006)
            files_2012=os.listdir(folder_2012)
            map_2012={}

            for f in files_2012:
                key=get_key(f)
                map_2012[key]=f

            for f2006 in files_2006:
                key=get_key(f2006)
                if key in map_2012:
                    f2012=map_2012[key]
                    self.files.append((region, f2006, f2012))
        # print("Total matched pairs:", len(self.files))
        # print(self.files[:5])

    def _src(self, path):
        src=self.src_cache.get(path)
        if src is None:
            src=rasterio.open(path)
            self.src_cache[path]=src
        return src

    def __del__(self):
        for src in self.src_cache.values():
            try:
                src.close()
            except:
                pass

    def __len__(self):
        return len(self.files)

    def read_image(self, path, i, j, patch_size):
        # if path in self.cache:
        #     return self.cache[path]
        with rasterio.open(path) as src:
            window=rasterio.windows.Window(j, i, patch_size, patch_size)
            img=src.read(window=window, out_dtype=np.float32)
        # self.cache[path]=img
        # if len(self.cache) > 50:
        #     self.cache.pop(next(iter(self.cache)))
        return img

    def read_label(self, path):
        # if path in self.cache:
        #     return self.cache[path]
        with rasterio.open(path) as src:
            lbl=src.read(1)
        # self.cache[path]=lbl
        # if len(self.cache) > 200:
        #     self.cache.clear()
        return lbl

    # def get_transition(self, a, b):
    #     valid=(a!=0) & (b!=0)
    #     transition=a*6+b
    #     transition[~valid]=255
    #     return transition

    def __getitem__(self, idx):
        idx=idx % len(self.files)
        region, file2006, file2012=self.files[idx]
        img2006_path=os.path.join(self.img2006_root, region,file2006)
        img2012_path=os.path.join(self.img2012_root, region,file2012)

        lbl2006_path=os.path.join(self.lbl2006_root, region,file2012)
        lbl2012_path=os.path.join(self.lbl2012_root, region,file2012)

        src0=self._src(img2006_path)
        H, W=src0.height, src0.width

        patch_size=256
        i=np.random.randint(0, H-patch_size+1)
        j=np.random.randint(0, W-patch_size+1)
        win=rasterio.windows.Window(j,i,patch_size,patch_size)

        img2006=self._src(img2006_path).read(window=win,out_dtype=np.float32)
        img2012=self._src(img2012_path).read(window=win, out_dtype=np.float32)

        a=self._src(lbl2006_path).read(1, window=win)
        b=self._src(lbl2012_path).read(1, window=win)

        valid=(a!=0)&(b!=0)
        y=a*6+b
        y[~valid]=255

        x=np.concatenate([img2006, img2012], axis=0).astype(np.float32)/255.0
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


#---------------------------------------------MODEL IMPLEMENTATION----------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=36):
        super().__init__()

        # Encoder
        self.enc1=DoubleConv(in_channels, 64)
        self.enc2=DoubleConv(64, 128)
        self.enc3=DoubleConv(128, 256)

        self.pool=nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck=DoubleConv(256, 512)

        # Decoder
        self.up3=nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3=DoubleConv(512, 256)

        self.up2=nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2=DoubleConv(256, 128)

        self.up1=nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1=DoubleConv(128, 64)

        # Output
        self.out=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1=self.enc1(x)      # [B,64,H,W]
        x2=self.pool(x1)

        x3=self.enc2(x2)     # [B,128,H/2,W/2]
        x4=self.pool(x3)

        x5=self.enc3(x4)     # [B,256,H/4,W/4]
        x6=self.pool(x5)

        # Bottleneck
        x7=self.bottleneck(x6)  # [B,512,H/8,W/8]

        # Decoder
        x=self.up3(x7)
        x=torch.cat([x, x5], dim=1)
        x=self.dec3(x)

        x=self.up2(x)
        x=torch.cat([x, x3], dim=1)
        x=self.dec2(x)

        x=self.up1(x)
        x=torch.cat([x, x1], dim=1)
        x=self.dec1(x)

        return self.out(x)


#-----------------------------------------DIFFERENT LOSSES-------------------------------------------------


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.ignore_index=ignore_index

    def forward(self, logits, targets):
        probs=torch.softmax(logits, dim=1)
        targets=targets.clone()

        mask=targets!=self.ignore_index
        targets[~mask]=0
        targets=targets.unsqueeze(1).long()

        pt=torch.gather(probs, 1, targets)
        log_pt=torch.log(pt + 1e-8)

        if self.alpha is not None:
            alpha=self.alpha.to(logits.device)
            alpha_t=alpha[targets.squeeze(1)]
            alpha_t=alpha_t.unsqueeze(1)
        else:
            alpha_t=1.0

        loss=-alpha_t*((1-pt)**self.gamma)*log_pt
        loss=loss[mask.unsqueeze(1)]
        return loss.mean()
    

class DiceLoss(nn.Module):
    def __init__(self, num_classes=36, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        self.smooth=smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        probs=torch.softmax(logits, dim=1)

        targets=targets.clone()
        mask=targets!=self.ignore_index

        targets[~mask]=0

        # One-hot encoding
        targets_onehot=F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot=targets_onehot.permute(0, 3, 1, 2).float()

        mask=mask.unsqueeze(1)

        probs=probs*mask
        targets_onehot=targets_onehot*mask

        # Dice calculation
        intersection=(probs*targets_onehot).sum(dim=(2, 3))
        union=probs.sum(dim=(2, 3))+targets_onehot.sum(dim=(2, 3))

        dice=(2*intersection+self.smooth)/(union+self.smooth)
        loss=1-dice.mean()

        return loss
    

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, weights, num_classes=36, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        self.smooth=smooth
        self.weights=weights

    def forward(self, logits, targets):
        probs=torch.softmax(logits, dim=1)

        targets=targets.clone()
        mask=targets!=self.ignore_index
        targets[~mask]=0

        targets_onehot=F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot=targets_onehot.permute(0, 3, 1, 2).float()

        mask=mask.unsqueeze(1)
        probs=probs*mask
        targets_onehot=targets_onehot*mask

        # w=targets_onehot.sum(dim=(2, 3))  # [B, C]

        w = self.weights.view(1, -1)

        intersection=(probs*targets_onehot).sum(dim=(2, 3))
        union=probs.sum(dim=(2, 3))+targets_onehot.sum(dim=(2, 3))

        dice=(2*intersection+self.smooth)/(union+self.smooth)

        # apply weights
        loss=1-(w*dice).sum(dim=1)/(w.sum(dim=1)+self.smooth)

        return loss.mean()

class JaccardLoss(nn.Module):
    def __init__(self, num_classes=36, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        self.smooth=smooth

    def forward(self, logits, targets):
        probs=torch.softmax(logits, dim=1)

        targets=targets.clone()
        mask=targets!=self.ignore_index
        targets[~mask]=0

        targets_onehot=F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot=targets_onehot.permute(0, 3, 1, 2).float()

        mask=mask.unsqueeze(1)
        probs=probs*mask
        targets_onehot=targets_onehot*mask

        intersection=(probs*targets_onehot).sum(dim=(2, 3))
        union=(probs.sum(dim=(2,3))+targets_onehot.sum(dim=(2,3))-intersection)

        iou=(intersection+self.smooth)/(union+self.smooth)
        loss=1-iou.mean()
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, num_classes=36, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        self.smooth=smooth

    def forward(self, logits, targets):
        probs=torch.softmax(logits, dim=1)

        targets=targets.clone()
        mask=targets!=self.ignore_index
        targets[~mask]=0

        targets_onehot=F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot=targets_onehot.permute(0, 3, 1, 2).float()

        mask=mask.unsqueeze(1)
        probs=probs*mask
        targets_onehot=targets_onehot*mask

        TP=(probs*targets_onehot).sum(dim=(2,3))
        FP=(probs*(1-targets_onehot)).sum(dim=(2,3))
        FN=((1-probs)*targets_onehot).sum(dim=(2,3))

        tversky=(TP+self.smooth)/(TP+self.alpha*FP+self.beta*FN+self.smooth)

        loss=1-tversky.mean()
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, num_classes=36, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        self.smooth=smooth

    def forward(self, logits, targets):
        probs=torch.softmax(logits, dim=1)

        targets=targets.clone()
        mask=targets!=self.ignore_index
        targets[~mask]=0

        targets_onehot=F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot=targets_onehot.permute(0, 3, 1, 2).float()

        mask=mask.unsqueeze(1)
        probs=probs*mask
        targets_onehot=targets_onehot*mask

        TP=(probs*targets_onehot).sum(dim=(2,3))
        FP=(probs*(1-targets_onehot)).sum(dim=(2,3))
        FN=((1-probs)*targets_onehot).sum(dim=(2,3))

        tversky=(TP+self.smooth)/(TP+self.alpha*FP+self.beta*FN+self.smooth)

        loss=(1-tversky)**self.gamma
        return loss.mean()
    
#-----------------------------DIFFERENT METRICS--------------------------------------
    
def compute_accuracy(pred, target):
    pred=torch.argmax(pred, dim=1)
    mask=target!=255
    correct=(pred[mask]==target[mask]).sum().item()
    total=mask.sum().item()
    return correct/total if total>0 else 0

def compute_iou(pred, target, num_classes=36):
    pred=torch.argmax(pred, dim=1)
    mask=target!=255
    pred=pred[mask]
    target=target[mask]

    ious=[]

    for cls in range(num_classes):
        pred_inds = pred==cls
        target_inds = target==cls

        intersection=(pred_inds&target_inds).sum().item()
        union=(pred_inds|target_inds).sum().item()

        if union==0:
            continue

        ious.append(intersection/union)

    return sum(ious)/len(ious) if len(ious)>0 else 0

def compute_f1(pred, target, num_classes=36):
    pred=torch.argmax(pred, dim=1)
    mask = target!=255
    pred=pred[mask]
    target=target[mask]

    f1s=[]

    for cls in range(num_classes):
        pred_inds = pred==cls
        target_inds = target==cls

        tp=(pred_inds&target_inds).sum().item()
        fp=(pred_inds & ~target_inds).sum().item()
        fn=(~pred_inds & target_inds).sum().item()

        denom=(2*tp+fp+fn)

        if denom==0:
            continue

        f1s.append((2 * tp) / denom)

    return sum(f1s)/len(f1s) if len(f1s)> 0 else 0

def compute_class_weights(dataset, indices, num_classes=36, samples=500):
    counts=np.zeros(num_classes, dtype=np.float64)
    z=0
    samples=min(samples,len(indices))
    chosen=random.sample(indices, samples)
    for i in chosen:
        if(z%10==0):
            print(z)
        region, file2006, file2012=dataset.files[i]
        lbl2006_path=os.path.join(dataset.lbl2006_root, region, file2012)
        lbl2012_path=os.path.join(dataset.lbl2012_root, region, file2012)

        with rasterio.open(lbl2006_path) as src:
            a=src.read(1)
        with rasterio.open(lbl2012_path) as src:
            b=src.read(1)

        valid=(a!=0)&(b!=0)
        y=a*6+b
        y[~valid]=255

        flat=y[y!=255].astype(np.int64)
        counts+=np.bincount(flat,minlength=num_classes)

        z+=1

    # avoid zero counts
    counts[counts==0]=1
    print(counts)
    weights=1.0/np.log(counts+1.1)
    print(weights)
    weights=weights/weights.mean()
    # weights=torch.clamp(weights,min=0.1,max=5)

    return torch.tensor(weights,dtype=torch.float32)

def compute_per_class_accuracy(model, dataset, indices, device, num_classes=36):
    model.eval()

    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)

    patch_size = 256

    with torch.no_grad():
        for count, idx in enumerate(indices):
            print(f"Image {count+1}/{len(indices)}")

            region, file2006, file2012 = dataset.files[idx]

            img2006_path = os.path.join(dataset.img2006_root, region, file2006)
            img2012_path = os.path.join(dataset.img2012_root, region, file2012)

            lbl2006_path = os.path.join(dataset.lbl2006_root, region, file2012)
            lbl2012_path = os.path.join(dataset.lbl2012_root, region, file2012)

            # img2006 = dataset.read_image(img2006_path)
            # img2012 = dataset.read_image(img2012_path)

            a = dataset.read_label(lbl2006_path)
            b = dataset.read_label(lbl2012_path)

            valid = (a != 0) & (b != 0)
            y = a * 6 + b
            y[~valid] = 255

            # x = np.concatenate([img2006, img2012], axis=0)

            H, W = y.shape

            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):

                    if i + patch_size > H or j + patch_size > W:
                        continue

                    img2006 = dataset.read_image(img2006_path, i, j, patch_size)
                    img2012 = dataset.read_image(img2012_path, i, j, patch_size)

                    x_patch = np.concatenate([img2006, img2012], axis=0)
                    y_patch = y[i:i+patch_size, j:j+patch_size]

                    x_patch = torch.tensor(x_patch).unsqueeze(0).to(device)
                    y_patch = torch.tensor(y_patch, dtype=torch.long).to(device)

                    pred = model(x_patch)
                    pred = torch.argmax(pred, dim=1).squeeze(0)

                    mask = y_patch != 255

                    pred = pred[mask]
                    y_patch = y_patch[mask]

                    for c in range(num_classes):
                        class_mask = (y_patch == c)
                        total[c] += class_mask.sum().item()
                        correct[c] += ((pred == c) & class_mask).sum().item()

    acc = correct / (total + 1e-6)
    return acc, total


#---------------------------------------------DATA DISTRIBUTION--------------------------------------------------


# def stratified_split(dataset, test_ratio=0.2, num_classes=36):
#     # chosen=random.sample(imgs, k)
#     class_to_images={i: [] for i in range(num_classes)}

#     print("Building class-image mapping...")

#     for idx in range(len(dataset.files)):
#         region, f2006, f2012=dataset.files[idx]

#         lbl2006_path=os.path.join(dataset.lbl2006_root, region, f2012)
#         lbl2012_path=os.path.join(dataset.lbl2012_root, region, f2012)

#         with rasterio.open(lbl2006_path) as src:
#             a=src.read(1)
#         with rasterio.open(lbl2012_path) as src:
#             b=src.read(1)

#         valid=(a!=0)&(b!=0)
#         y=a*6+b
#         y[~valid]=255
#         classes=np.unique(y[y!=255])

#         for c in classes:
#             class_to_images[int(c)].append(idx)

#     print("Selecting stratified test set...")

#     test_set=set()

#     for c in range(num_classes):
#         imgs=list(set(class_to_images[c]))
#         if len(imgs)==0:
#             continue
#         k=max(1, int(len(imgs)*test_ratio))
#         # chosen=random.sample(imgs, k)
#         test_set.update(random.sample(imgs, k))

#     test_idx=list(test_set)
#     train_idx=[i for i in range(len(dataset.files)) if i not in test_set]

#     print(f"Train images: {len(train_idx)}")
#     print(f"Test images: {len(test_idx)}")

#     return train_idx, test_idx

def split_train_test_with_coverage(dataset, train_ratio=0.8, num_classes=36, seed=42):
    rng=random.Random(seed)

    img_classes=[]
    class_to_images={i: [] for i in range(num_classes)}
    print("Building class-image mappingggggg...")
    for idx in range(len(dataset.files)):
        if(idx%10==0):
            print(idx)
        region, f2006, f2012 = dataset.files[idx]

        lbl2006_path=os.path.join(dataset.lbl2006_root, region, f2012)
        lbl2012_path=os.path.join(dataset.lbl2012_root, region, f2012)

        with rasterio.open(lbl2006_path) as src:
            a=src.read(1)
        with rasterio.open(lbl2012_path) as src:
            b=src.read(1)

        valid=(a!=0)&(b!=0)
        y=a*6+b
        y[~valid]=255

        classes=set(np.unique(y[y!=255]).astype(int).tolist())
        img_classes.append(classes)

        for c in classes:
            class_to_images[c].append(idx)

    order=list(range(len(dataset.files)))
    rng.shuffle(order)

    target_train=int(train_ratio * len(order))

    train=[]
    train_set=set()
    covered=set()

    # ensure class coverage in train
    for idx in order:
        if len(train)>=target_train:
            break
        gain=len(img_classes[idx]-covered)
        if gain>0:
            train.append(idx)
            train_set.add(idx)
            covered|=img_classes[idx]

    # fill remaining
    for idx in order:
        if len(train)>=target_train:
            break
        if idx not in train_set:
            train.append(idx)
            train_set.add(idx)

    test=[i for i in order if i not in train_set]

    return train, test

def build_probe_set(dataset, num_classes=36, max_per_class=2):
    class_to_images={i: [] for i in range(num_classes)}

    for idx in range(len(dataset.files)):
        if(idx%10==0):
            print(idx)
        region, f2006, f2012 = dataset.files[idx]

        lbl2006_path=os.path.join(dataset.lbl2006_root, region, f2012)
        lbl2012_path=os.path.join(dataset.lbl2012_root, region, f2012)

        with rasterio.open(lbl2006_path) as src:
            a=src.read(1)
        with rasterio.open(lbl2012_path) as src:
            b=src.read(1)

        valid=(a!=0)&(b!=0)
        y=a*6+b
        y[~valid]=255

        classes=np.unique(y[y!=255]).astype(int)

        for c in classes:
            class_to_images[c].append(idx)

    probe=set()
    for c in range(num_classes):
        imgs=class_to_images[c]
        random.shuffle(imgs)
        probe.update(imgs[:max_per_class])

    return list(probe)


#--------------------------------------------------------MAIN-------------------------------------------------------------


if __name__ == "__main__":

    ds=ChangeDataset("D:/SpatioTemporalChanges/dataset")
    # x, y = ds[0]

    # print(x.shape) 
    # print(y.shape) 
    # print(torch.unique(y))

    random.seed(42)
    train_idx,test_idx=split_train_test_with_coverage(ds, train_ratio=0.8)
    print(f"Train images: {len(train_idx)}")
    print(f"Test images: {len(test_idx)}")
    # indices=list(range(len(ds.files)))
    # random.seed(42)
    # random.shuffle(indices)

    # split=int(0.8 * len(indices))
    
    # train_idx=indices[:split]
    # test_idx=indices[split:]

    train_ds=Subset(ds, train_idx*400)
    test_ds=Subset(ds, test_idx*5)

    loader=DataLoader(
        train_ds,
        batch_size=12,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    eval_loader=DataLoader(
        test_ds,
        batch_size=12,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    
    print(len(loader))
    print(len(eval_loader))
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    # weights=compute_class_weights(ds, train_idx).to(device)
    # weights=torch.tensor([
    #     1.7341, 1.7341, 1.7341, 1.7341, 1.7341, 1.7341, 1.7341, 0.0594, 0.0882,
    #     1.7341, 0.1032, 0.1040, 1.7341, 0.0683, 0.0550, 0.0801, 0.0814, 0.0780,
    #     1.7341, 0.0856, 0.0981, 0.0603, 1.7341, 1.7341, 1.7341, 1.7341, 1.7341,
    #     1.7341, 1.7341, 1.7341, 1.7341, 0.1015, 0.0939, 1.7341, 0.0925, 0.0691
    # ],dtype=torch.float32).to(device)
    
    # loss_fn=nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    # loss_fn=nn.CrossEntropyLoss(ignore_index=255)
    # loss_fn=FocalLoss(gamma=2,alpha=weights)
    # loss_fn=FocalLoss(gamma=2)
    # loss_fn=DiceLoss(num_classes=36)
    # loss_fn=GeneralizedDiceLoss(weights, num_classes=36)
    # loss_fn=JaccardLoss(num_classes=36)
    # loss_fn=TverskyLoss(alpha=0.7, beta=0.3, num_classes=36)
    loss_fn=FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)

    model=UNet().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # print(weights)
    with open(LOG_FILE, "w") as f:
        f.write(f"Training Log - {EXP_NAME}\n\n")

    print("Training Started:")
    scaler=torch.amp.GradScaler("cuda")
    best_f1=-1.0    
    best_path=os.path.join(SAVE_DIR, "best_model.pth")

    for epoch in range(3):
        model.train()
        total_loss=0
        count=0
        for x,y in loader:
            x=x.to(device, non_blocking=True)
            y=y.to(device, non_blocking=True)
            # print(x.device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                pred=model(x)
                loss=loss_fn(pred, y)
            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss+=loss.item()
            count+=1
            if count%50==0:
                print(f"Loss {count}: {loss.item()}")

        avg_loss=total_loss/max(count,1)

        #EVALUATIONS-------------
        
        val_acc=0
        val_iou=0
        val_f1=0
        val_count=0
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x=x.to(device).float()
                y=y.to(device)

                pred=model(x)
                # pred_cpu=pred.cpu()
                # y_cpu=y.cpu()

                val_acc+=compute_accuracy(pred, y)
                val_iou+=compute_iou(pred, y)
                val_f1+=compute_f1(pred, y)
                val_count+=1


        val_acc/=max(val_count,1)
        val_iou/=max(val_count,1)
        val_f1/=max(val_count,1)

        if val_f1>best_f1:
            best_f1=val_f1
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved (F1={best_f1:.4f})")

        log_line=f"[{time.strftime('%H:%M:%S')}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f}"
        print(log_line)
        with open(LOG_FILE, "a") as f:
            f.write(log_line + "\n")
            f.flush()

    # torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_path))
    model.eval()

    probe_idx=build_probe_set(ds)
    print(f"Probe images: {len(probe_idx)}")
    per_class_acc,counts=compute_per_class_accuracy(model, ds, probe_idx, device)
    with open(LOG_FILE, "a") as f:
        f.write("\nPer-class Accuracy:\n")
        for i, val in enumerate(per_class_acc):
            f.write(f"Class {i}: Acc={val:.4f}, Count={int(counts[i])}\n")

    # torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))
    with open(os.path.join(SAVE_DIR, "metrics.txt"), "w") as f:
        f.write(f"Final Loss: {avg_loss}\n")
        f.write(f"Final Accuracy: {val_acc}\n")
        f.write(f"Final IoU: {val_iou}\n")
        f.write(f"Final F1: {val_f1}\n")
         
