# Covid19-Case-Prediction in Malaysia

### 1. Project Description
  * This is deep learning model using **_Long Short Term Memory(LSTM)_** neural
network to make prediction on new Covid-19 cases in Malaysia.
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
    * Open Visual Studio Code then open folder of downloaded file, search .py file then run it.

 
 
### 3. Results
1. The architecture model that is used in this project is **_Long Short Term Memory(LSTM)_** ,number of nodes is 64 nodes & window size is 30 LSTM, Dense, and Dropout layers have been implemented in the
model.

![model_architecture](https://github.com/dalila28/covid19-case-prediction/assets/135775153/a4e6edad-6be8-4bc9-b954-9bd18c550acb)


2. Below are the snapshot of the model performance under 100 epochs which **_Mean Squared Error(MSE)_** as loss & **_Mean Absolute Percentage Error (MAPE)_** as metrics
![model_performance1](https://github.com/dalila28/covid19-case-prediction/assets/135775153/90e47889-be52-4773-8119-a2bfc4915d0a)
![model_performance2](https://github.com/dalila28/covid19-case-prediction/assets/135775153/60274f15-60d8-4dc4-8818-e5f24ac688be)

3. Tensorboard snapshot showing graph of MSE
![tensorboard](https://github.com/dalila28/covid19-case-prediction/assets/135775153/43f140ac-c979-42e4-b436-66d4be2d1b0d)


4. Figure below showing the matplotlib graph comparison between actual & predicted result of covid-19 case in Malaysia based on my deep learning project. From the graph we can see that the predicted line is following the curve of actual line which as for my observation I can say that the result is good eventhough it not following correctly the spike of curve. If we want to improve the result, I think we can increase number of epochs so model has more opportunities to learn from the data and adjust its parameters to improve performance.
![actual_vs_predicted](https://github.com/dalila28/covid19-case-prediction/assets/135775153/b3da4626-c7c0-4a57-837c-b2e6221da0b2)


### 4. Credits
1. The original source of dataset is from Ministry of Health of Malaysia where you can get from here https://github.com/MoH-Malaysia/covid19-public
2. For creating tensorboard, I refer tutorial from https://www.tensorflow.org/tensorboard/get_started
3. Regarding the tensorflow API that I used in my project, I always refer to this documentation https://www.tensorflow.org/api_docs/python/tf/all_symbols
framework,method
