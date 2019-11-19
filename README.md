# ResNet50-Classification
Can be used on your own dataset

## Step 1：Your own dataset
    You should put images under the data directory:
```
-dataset\
    -model\
    -train\
          -1\
          -2\
          ...
    -test\
          -1\
          -2\
          ...
```

If you have already put your images well, then look **Step 2**. If not, look the following optional or find your own method to put your images like the above.

**(optional)** or you can use data2img.m to convert imagedata to dataset folder
For example you should let your your data be in validation.mat, where the structure of validation is:
```
-validation
    -data
        -data{i,1} is image data 224×224×3
        -data{i,2} is image label id
    -train: training samples' id
    -test:  testing samples' id
```
```
Then run data2image.m to create your own dataset
```

 ## Step 2: Training
 ```
    run train_resnet.py to train the model
 ```

 ## Step 3: Testing
 ```
    run test_resnet.py to test the model
 ```
