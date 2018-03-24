
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import pandas as pd
import numpy as np
import h5py


# In[2]:


src_dir = './dataset'
dest_file = './dataset/data.h5'

names = ['abhishek', 'ahmad', 'akarsh', 'avatar', 'chaitanya', 'champ', 'harshith', 'ishan',
       'kalvik', 'manish', 'nishad', 'pavan', 'phani', 'prabhu', 'raghu', 'rahul', 'sanjay', 'shuang',
       'subramaniam', 'sushal', 'temesgen', 'vinay']

num_people = len(names)
num_rows = 90000
test_split = 0.20
num_obs = 15
num_cols = 6

mean_noise = 0
std_noise = 0.1


# In[3]:


print("merging csv")

data_x = []
data_y = []

for index, name in enumerate(names):
    print("Progress {:2.1%}".format(index / num_people), end="\r")
    
    for i in range(1,num_obs+1):
        df = np.array(pd.read_csv(src_dir + '/{0}/{0}{1}.csv'.format(name, i)))
        df = df[0:num_rows, :]
        if (df.shape[0] != 90000):
            print (filename)
        data_x.append(df)  
        data_y.append(index)  

data_x = np.array(data_x)
data_y = np.array(data_y)

print ("CSV data shape (x,y): ", data_x.shape, data_y.shape)


# In[4]:


print ("splitting data")

x_train, x_test, y_train, y_test = train_test_split(
									data_x, data_y, test_size=test_split, random_state=42)

del data_x, data_y
print ("\nSplit data shape")
print ("Train (x,y) : ", x_train.shape, y_train.shape)
print ("Test  (x,y) : ", x_test.shape, y_test.shape)


# In[7]:


def transformations(x, y):
    x_train_temp = []
    y_train_temp = []

    x_train_lr_mag = np.flip(x[:, :, :3], axis = 2)
    x_train_lr_phase = np.flip(x[:, :, 3:], axis = 2)
    x_train_lr = np.concatenate((x_train_lr_mag, x_train_lr_phase), axis = 2)

    x_train_ud = np.flip(x, axis = 1)
    
    x_train_combo = np.flip(x_train_lr, axis = 1)

    gaussian_noise = np.random.normal(mean_noise, std_noise, x.shape)
    data_corrupt = x + gaussian_noise

    x_train_temp.extend(x)
    x_train_temp.extend(x_train_lr)
    x_train_temp.extend(x_train_ud)
    x_train_temp.extend(x_train_combo)
    x_train_temp.extend(data_corrupt)

    for _ in range(5):
        y_train_temp.extend(y)
        
    return np.array(x_train_temp), np.array(y_train_temp)

print ("transforming training data")
x_train, y_train = transformations(x_train, y_train)


# In[8]:


print ("Shuffling data")
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)


# In[9]:


print ("Final data shape")
print ("Train (x,y) : ", x_train.shape, y_train.shape)
print ("Test  (x,y) : ", x_test.shape, y_test.shape)

print ("Generating h5 file")
hf = h5py.File(dest_file, 'w')
hf.create_dataset('x_train', data=x_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_test', data=x_test)
hf.create_dataset('y_test', data=y_test)
hf.close()

