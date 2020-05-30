#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:



df = pd.read_csv('wines.csv')


# In[ ]:




df.info()


# In[ ]:


y = df['Class']


# In[ ]:


y.value_counts()


# In[ ]:


y_cat = pd.get_dummies(y)


# In[ ]:


y


# In[ ]:


df.columns


# In[ ]:


df.columns


# In[ ]:


X = df.drop('Class' , axis=1)


# In[ ]:


X.info()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.scatterplot(x='Alcohol' , y=y , data=df)


# In[ ]:


from keras.models import Sequential


# In[ ]:


model  =  Sequential()


# In[ ]:


X.info()


# In[ ]:


X.shape


# In[ ]:






y_cat.shape


# In[ ]:


from keras.layers import Dense


# In[ ]:


model.add(Dense(units=5 , input_shape=(13,), 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[ ]:



model.summary()


# In[ ]:


model.add(Dense(units=8 , 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(units=2, 
                activation='relu', 
                kernel_initializer='he_normal' ))


# In[ ]:


model.summary()


# In[ ]:


model.add(Dense(units=3, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


from keras.optimizers import RMSprop


# In[ ]:


model.compile(optimizer=RMSprop(learning_rate=0.01),  
              loss='categorical_crossentropy',
             metrics=['accuracy']
             )


# In[ ]:


model.layers[0].input


# In[ ]:


model.layers[3].output


# In[ ]:


model.layers[2].output


# In[ ]:


model.fit(X,y_cat, epochs=100)


# In[ ]:


# import keras.backend as K


# In[ ]:


# K.clear_session()


# In[ ]:


model.get_weights()


# In[ ]:



model.save('modelsave.h5')


# In[ ]:





# In[ ]:




