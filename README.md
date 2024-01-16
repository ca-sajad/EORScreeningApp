# AI-Based EOR Screening Application
This application uses PyTorch's Artificial Neural Network to classify oilfields based on the Enhanced Oil Recovery (EOR) methods performed on them. <br>
This is a work in prorgess. <br>
So far, the followings have been performed: <br>
1. Rock and oil data from a database of oilfields where different EOR methods have been executed is gathered. This data has been cleaned and minimum and maximum values for each of 7 properties and 9 EOR methods have been calculated and saved in an Excel file.
2. The program reads this data and creates a sample of, for example, 100 data samples for each EOR method. This data is used as input for a NN.
3. There is also a set of test data collected from the initial database. As some of the entries of this database are missing some inputs, all of the entries cannot be used. The final database for testing has 364 samples for 9 EOR methods. However, each class has a different size, starting from less than 5 to more than 100.
4. NN is trained. There are several hyperparameters to be determined including number of epochs, batch size, and hidden layer size.
5. NN hyperparameters are optimized using a module called "optuna" where the objective function runs the model on the test set and tries to maximize the average multi-class f1-score.
6. The best model is selected and the following graphs are plotted. Currently, the average f1-score is 0.81.

#### &emsp;&emsp;&emsp;&emsp;f1-score for each class &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; precision-recall curve  &emsp;&emsp; &emsp;Receiver Operating Characteristic curve  <br>
<p float="left">
<img src="https://github.com/ca-sajad/EORScreeningApp/blob/main/saved_models/1/f1-score.png" alt="f1-score for each class" width="300"/>
<img src="https://github.com/ca-sajad/EORScreeningApp/blob/main/saved_models/1/precision-recall-curve.png" alt="precision-recall curve for each class" width="300"/>
<img src="https://github.com/ca-sajad/EORScreeningApp/blob/main/saved_models/1/roc-curve.png" alt="Receiver Operating Characteristic curve for each class" width="300"/>
</p>

### Modules used:
- torch
- torchmetrics
- optuna
- scikit-learn
- pandas

### TODO
- minor: using z-score instead of min-max to perform scaling (save a StandardScaler using joblib)
- minor: adjusting classification threshold to improve f1 score
- minor: checking the code to predict suitable EOR method for an oilfield based on its properties 
- major: assessing if probablities can be used instead of classification
- phase 2: creating a GUI using Next.js to deploy the model and plot the candidate oilfield on a graph of the first two principal components
