# How to Organize LSUN dataset

## LSUN Church dataset 

### How the LSUN Churches Dataset should be organized

```
## train dataset
/lsun/church_outdoor_train.txt
/lsun/churches_train


## validation dataset
/lsun/church_outdoor_val.txt
lsun/churches_train
# validation uses images in the churches_train image, too

# train, LSUNChurchesTrain, 121227
# validation, LSUNChurchesValidation, 5000
```


LSUN Church dataset might be unavailable through official website. 
If that is the case, the best possible option is to download through academic torrent.

```
church_outdoor_train_lmdb.zip
church_outdoor_val_lmdb.zip
```

1. Unzip the dataset
```
unzip church_outdoor_train_lmdb.zip
unzip church_outdoor_val_lmdb.zip
```

This will create two directories: `church_outdoor_train_lmdb', 'church_outdoor_val_lmdb`.


2. Convert LMDB to `.webp` images.

The LSUN dataset is stored in LMDB format, which needs to be converted to `.webp` image files. 
The Python script for this can be downloaded at `https://github.com/fyu/lsun`.











