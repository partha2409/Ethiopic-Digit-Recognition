<h3> Ethiopic Digit Recognition with Resnet </h3>

Download the data from https://www.kaggle.com/c/tau-ethiopic-digit-recognition/data <br>

This was part of a competition. Hence labels are available only for the training data. We are supposed to upload the predictied digits to Kaggle as a csv file.

<h3> Requirements: </h3> 
pytorch <br>
opencv <br>

<h3> Steps to train the model </h3>
1. Download and extract the data into the project folder. <br>
2. Run split_train_val.py to create the validation data. <br>
3. Run train.py to start training the model. <br>
4. Once the model is trained,in run_inference.py provide the path of the trained model and run it. <br>



