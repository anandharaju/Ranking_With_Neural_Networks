from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso

X=[]
y=[]
training_result=[]
# fix random seed for reproducibility
np.random.seed(7)
dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/training_data_preprocessed.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,2:]
qid = dataset[:,1]
y = dataset[:,:1]
print("\nDataset Dimensions : ",dataset.shape)

# LASSO SETUP
lasso = Lasso (alpha = 0.3,normalize=True)
lasso_coef = lasso.fit(X,y).coef_
lasso_coef_positive = lasso_coef[lasso_coef > 0]
plt.plot(range(len(lasso_coef_positive)),lasso_coef_positive)
plt.xticks(range(len(lasso_coef_positive)),range(0,58),rotation=60)
plt.ylabel('coefficients')
plt.show()

features_selected = np.where(np.array(lasso_coef) > 0)[0]
print("Features Selected [%d]:" %len(features_selected),features_selected)
X = X[:,features_selected]
print(X.shape)

#X = StandardScaler().fit_transform(X)
#pca = PCA(n_components=20)
#principalComponents = pca.fit_transform(X)
#pdf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11', 'pc12','pc13','pc14','pc15','pc16','pc17','pc18','pc19','pc20'])
#print(pdf.head())
#print(pdf.describe())
#X=pdf

# Save the number of columns in predictors: n_cols
n_cols = X.shape[1]
print(n_cols)
# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(n_cols, activation='relu',input_shape=(n_cols,)))

# Add the second layer
model.add(Dense((n_cols * n_cols), activation='relu'))
model.add(Dense(round((n_cols * n_cols)/2), activation='relu'))
model.add(Dense(round((n_cols * n_cols)/3), activation='relu'))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

#early stopping
es = EarlyStopping(patience=50)

#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=7)

# Fit the model
training_result = model.fit(X,y,validation_split=0.2,epochs=100,callbacks=[es])
print(np.mean(training_result.history['val_loss']))
print(np.std(training_result.history['loss']))
training_result = model.predict(X)


#######################################################################################################################################
dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/example_testing_data.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,2:]
X = X[:,features_selected]
qid = dataset[:,1]
y = dataset[:,:1]

y_pred = model.predict(X)

print("\n\n")
temp=[]
index=[]
old_qid = -1
count = 0
for i in range(0,len(y_pred)):
    temp.append(y_pred[i][0])
    if old_qid == -1 or old_qid != qid[i]:
        old_qid = qid[i]
        count=0
    index.append(count)
    count+=1
df = pd.DataFrame({'temp':temp,'qid':qid,'index':index})
df = df.sort_values('temp',ascending=False)
#df = df.groupby('qid').groups
grouped = (df.groupby('qid'))
for name,group in grouped:
    print (name)
    print (group)