# Covid19-Case-Prediction in Malaysia

### 1. Project Description

  * This deep learning project is to make prediction on new Covid-19 cases in Malaysia in order to overcome the fast spread of this virus. This deep learning model can help government to make decisions such as whether they can allow travellers to come into Malaysia, can movement control order be lifted or not and many other decisions that can help to prevent Covid19 spread.  
  * I developed this project by using **_Long Short Term Memory(LSTM)_** neural
network because it is suited for modeling time series as here we want make prediction of the new Covid19 cases everyday.
  * Problem: while working on this project one of the problems that i faced was handling with the missing values in the dataset. About half of the dataset contains missing values in "cluster_import", "cluster_religious", "cluster_community", "cluster_highRisk", "cluster_education", "cluster_detentionCentre", "cluster_workplace" columns. At first i tried using interpolate method to handle the missing data but it doesn't work as the 1st half of dataset is missing. Hence, to overcome this issue i tried another method which is by using mean to replace the missing values in the dataset.
  * Suggestion: maybe we can include feature like: "cluster healthcare workers" specifically to know more how many covid case happen to this group of people as they dealing with covid patient.  



### 2. Software used, framework,how to run project


   * Software needed:
     * Visual Studio Code as my IDE. You can get here https://code.visualstudio.com/download
     * Anaconda. https://www.anaconda.com/download

   * Framework:
     * I use Tensorflow and Keras framework to develop this deep learning project, as for Keras it already a part of TensorFlow’s core API.
   
   * How to run project:
     * Download project in the github
     * In Visual Studio Code make sure install Python
     * Open Anaconda prompt : "(OPTIONAL) IS FOR GPU INSTALLATION IF YOU NEED FOR CPU THEN IGNORE OPTIONAL"
        * (base) conda create -n "name u want to have" python=3.8
        * (env) conda install -c anaconda ipykernel
        * conda install numpy,conda install pandas,conda install matplotlib (run each of this one by one)
        * (OPTIONAL) conda install -c anaconda cudatoolkit=11.3
        * (OPTIONAL) conda install -c anaconda cudnn=8.2
        * (OPTIONAL) conda install -c nvidia cuda-nvcc
        * conda install git
        * 1 (a) create a folder named TensorFlow inside the tensorflow environment. For example: “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow”
        * (b) type: cd “C:\Users\<USERNAME>\Anaconda3\envs\tensorflow\TensorFlow” (to change directory to the newly created TensorFlow folder) 
        * (c) type: git clone https://github.com/tensorflow/models.git
        * conda install -c anaconda protobuf
        * 2 (a) type: cd “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow\models\research” (into TensorFlow\models\research for example)
        * b) type: protoc object_detection/protos/*.proto --python_out=.
        * 3 a) pip install pycocotools-windows
        * b) cp object_detection/packages/tf2/setup.py .
        * c) python -m pip install .
      * Test your installation (RESTART TERMINAL BEFORE TESTING)  
         * Inside C:\Users\< USERNAME > \Anaconda3\envs\tensorflow\TensorFlow\models\research
         * python object_detection/builders/model_builder_tf2_test.py The terminal should show OK if it passes all the tests
         
      * Open Visual Studio Code:
         * Go to open new folder, open downloaded file that you download from my repository
         * Make sure downloaded dataset and the Malaysia_covid19.py file in same folder
         * ![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png) **ATTENTION!!! : root_path = "Please change the path according to your folder path" DON'T FOLLOW MY PATH IN Malaysia_covid19.py FILE SINCE THE PATH IS MY OWN FOLDER PATH**
         * Then you can run Malaysia_covid19.py file
         
          

 


 
 
### 3. Results
1. The architecture model that is used in this project is **_Long Short Term Memory(LSTM)_** ,number of nodes is 64 nodes & window size is 30 LSTM, Dense, and Dropout layers have been implemented in the
model.

![model_architecture](https://github.com/dalila28/covid19-case-prediction/blob/main/images/model_architecture.png)

                                                  Model Architecture



2. Below are the snapshot of the model performance under 100 epochs which **_Mean Squared Error(MSE)_** as loss & **_Mean Absolute Percentage Error (MAPE)_** as metrics. After epoch 100 model performance showing that value of training loss is 0.0039 while validation loss is 0.0101. There is slightly difference between the losses value. Based on the training and validation loss values, we can say that the model is performing quite well. The training loss of 0.0039 indicates that the model has achieved a low error on the training data, meaning it is fitting the training data quite closely. Similarly, the validation loss of 0.0101 suggests that the model is generalizing well to new data, as it also achieved a relatively low error on the validation set.



![model_performance1](https://github.com/dalila28/covid19-case-prediction/blob/main/images/model_performance1.png)
![model_performance2](https://github.com/dalila28/covid19-case-prediction/blob/main/images/model_performance2.png)


                                                     Model Performance
                                            

3. Tensorboard snapshot showing graph of Mean Squared Error as loss value for training and validation. It showing that the values of loss is decreasing for both training and validation. But a few last epoch showing slightly increasing in values of loss for training and validation. But the increment of loss values in training and validation is not too much. Hence the model is not being overfitted model and still perform quite well.




![tensorboard](https://github.com/dalila28/covid19-case-prediction/blob/main/images/tensorboard.png)


                                                       Tensorboard




4. Figure 1 showing the matplotlib graph comparison between actual & predicted result of covid-19 case in Malaysia based on my deep learning project. From the graph we can see that the predicted line is following the curve of actual line which I can say that the predicted result of Covid19 Cases in Malaysia is quite good.


![actual_vs_predicted](https://github.com/dalila28/covid19-case-prediction/blob/main/images/actual_vs_predicted.png)


                                                     Figure 1




### 4. Credits
1. The data on the COVID-19 epidemic in Malaysia is sourced from the official repository, https://github.com/MoH-Malaysia/covid19-public which is powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.
2. For creating tensorboard, I refer tutorial from https://www.tensorflow.org/tensorboard/get_started
3. Regarding the tensorflow API that I used in my project, I always refer to this documentation https://www.tensorflow.org/api_docs/python/tf/all_symbols

