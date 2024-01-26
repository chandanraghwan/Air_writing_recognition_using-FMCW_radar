# Air Writing using UWB Radar

## Dataset
<pre>
  2D-Range-Doppler Data
  2D-Range-Doppler Images
</pre>

## CNN-LSTM Architecture for Radar Signal Classification
<pre>CNN-LSTM_2D.ipynb</pre>
This CNN-LSTM architecture for radar signal classification consists of the following components:

* **CNN model:** The CNN model extracts spatial features from the input 2D range-Doppler images. It consists of four convolutional layers, each followed by a batch normalization layer and a ReLU activation function. After each convolutional layer, there is a max pooling layer to downsample the feature maps and reduce the number of parameters.
* **LSTM model:** The LSTM model learns the temporal relationships between the extracted spatial features. It consists of two LSTM layers, each with 128 and 64 hidden units, respectively. The output of the LSTM model is a 10-dimensional vector, which is the predicted probability distribution over the different radar signal classes.

The combined model is a sequential model that consists of the CNN model followed by the LSTM model. The output of the CNN model is reshaped into a sequence format and fed to the LSTM model.

The training process for the combined model is as follows:

1. The CNN model is trained independently on the 2D range-Doppler images.
2. The features extracted by the CNN model are reshaped into a sequence format and fed to the LSTM model.
3. The LSTM model is trained to predict the correct radar signal class for each sequence.

The following metrics are used to evaluate the performance of the model:

* **Accuracy:** The percentage of correctly classified radar signals.
* **Loss:** The categorical cross-entropy loss between the predicted and actual probability distributions of the radar signal classes.


# PostProcessing(Conversion of radar raw data to Image)
<pre>postProcessing.py</pre>

**Step 1: Read the radar raw data**

The radar raw data is typically in a CSV file format, with each row representing a range bin and each column representing a Doppler bin.

**Step 2: Convert the radar raw data to complex numbers**

The radar raw data is typically in a string format, with each string representing a complex number. The complex number can be converted to a Python complex number using the `complex_str_to_complex()` function.

**Step 3: Compute the magnitude spectrogram**

The magnitude spectrogram is computed by taking the absolute value of the complex radar data.

**Step 4: Scale the magnitude spectrogram**
The magnitude spectrogram is scaled to the range of `0` to `255` for visualization purposes.

**Step 5: Compute the log-scaled magnitude spectrogram**

The log-scaled magnitude spectrogram is computed by taking the logarithm of the scaled magnitude spectrogram.

**Step 6: Normalize the log-scaled magnitude spectrogram**

The log-scaled magnitude spectrogram is normalized to the range of `0` to `1` for visualization purposes.

**Step 7: Convert the normalized log-scaled magnitude spectrogram to an image**

The normalized log-scaled magnitude spectrogram is converted to an image using the `cv2.imwrite()` function.

## Research Paper Citation

The radar dataset used in this project is sourced from the research paper titled "Deep Learning Approaches for Air-Writing Using Single UWB Radar".

By Authors:

Nermine Hendy
STEM College, RMIT University, Melbourne, VIC, Australia

Haytham M. Fayek
STEM College, RMIT University, Melbourne, VIC, Australia

Akram Al-Hourani
STEM College, RMIT University, Melbourne, VIC, Australia

The paper was published on 04 May 2022.

You can access the paper and find more details about the dataset by following this https://ieeexplore.ieee.org/document/9768813.

You can access the Dataset using https://figshare.com/articles/dataset/Dataset_for_Air-writing_recognition_based_on_Ultra-wide_band_Radar_Sensor/20225907



