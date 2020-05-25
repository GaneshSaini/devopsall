#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


a = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(a, 'o-')


# In[ ]:


a


# In[ ]:


b = np.array([1,-1])


# In[ ]:


b


# In[ ]:





# In[ ]:


c = np.convolve(a,b)


# In[ ]:


c


# In[ ]:


plt.plot(c)


# In[ ]:


from scipy import misc


# In[ ]:


img = misc.ascent()


# In[ ]:


img.shape


# In[ ]:


plt.imshow(img, cmap='gray')


# In[ ]:


kernel = np.array([
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]  ])


# In[ ]:


kernel = np.array([
            [0,0,0],
            [1,1,1],
            [0,0,0]  ])


# In[ ]:


plt.imshow(kernel, cmap='gray')


# In[ ]:


from scipy.signal import convolve2d


# In[ ]:


cimg = convolve2d(img, kernel)


# In[ ]:


cimg.shape


# In[ ]:


img.shape


# In[ ]:


plt.imshow(cimg, cmap='gray')


# In[ ]:




