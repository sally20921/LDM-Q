# How to Organize LSUN dataset

## LSUN Church dataset 

For this dataset, in TDQ, we only use training images for both training and validation.
Set the validation data path to LSUN church train data as well.


### Necessary Script for LSUN Church dataset

```
git clone https://github.com/fyu/lsun.git
```


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

```
python data.py export church_outdoor_train lmdb --out_dir churches_train
```

3. (Additional) Ignoring Nested Subfolders and Making it a Flat Directory Structure

```bash

find churches_train -type f -name "*.webp" -exec mv -t ./train {} +
```



4. Create Text Files

```bash
ls ./churches_train > ./church_outdoor_train.txt
```
This is what `church_outdoor_train.txt` should look like. 
```
f6ce24f02a8a91bd0a0f60bb2cf04f4822cbe42d.webp
62f28196dad99a049fa20d6e4a3a27eb5ea6c0c7.webp
92c2d98f45c4cff7eec9d55193c5b97c72eb466a.webp
50df56048f09bb22ec032dedea6aaa529a086342.webp
bb4df84f7df2aba7bafb0fd20c0c9887978c0c27.webp
08b0126832c230122d9d3de99d7a5674bd48236c.webp
06b0c9faabb6dd3b6d04c2aec090eaf6b7fb9b39.webp
29301fbf43a61dcebcde606862ddb619fb4d17c5.webp
...
```

5. Final Structure 

Your directory structure for lsun should look like this:

```
lsun/
│
├── church_outdoor_train.txt
├── church_outdoor_val.txt
└── churches_train/
    ├── f6ce24f02a8a91bd0a0f60bb2cf04f4822cbe42d.webp
    ├── 29301fbf43a61dcebcde606862ddb619fb4d17c5.webp
    └── ...
```


















