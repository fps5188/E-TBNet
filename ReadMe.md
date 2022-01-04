### Instruction

Our model is trained on a public dataset, so original images cannot be uploaded directly. We use datasets.ImageFolder() to read all the images in the testing set to obtain the ImageFolder object, and then use the pickle.dump() to write the ImageFolder object to test_dataset.pkl. When predict.py is executed, the program reads test_dataset.pkl to achieve the data loading. We averaged the results of the 5-fold cross-validation to get the final indicators.

### Other Floder

others/predict.py: Model evaluation.

others/weight: Weight of 5-fold cross-validation.

others/model_all:  Other lightweight models.

others/test_dataset.pkl: ImageFloder Object.

others/train.py: Model training.

others/train_cross.py: 5-fold cross-validation training

### Ours Floder

our/predict.py: Model evaluation.

our/weight: Weight of 5-fold cross-validation.

our/our_model:  Other lightweight models.

our/test_dataset.pkl: ImageFloder Object.

our/train.py: Model training.

our/train_cross.py: 5-fold cross-validation training

### Evaluation Steps:

1. Set weights_path

   ```python
   weights_path = r"weight/MobileNetV2/mobilenet_v2_1Net.pth
   ```

2. Choose a model

   ```python
   model = get_mobilenet_v2().to(device)
   ```

3. Load dataset

   ```python
   output = open(r'test_dataset.pkl', 'rb')
   test_dataset = pickle.load(output)
   output.close()
   ```

   