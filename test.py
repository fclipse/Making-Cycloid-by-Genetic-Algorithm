#%%
import numpy as np 
import matplotlib.pylab as plt 
def sigmoid(x): 
    return 1/(1+np.exp(-x)) # 시그모이드 수식 
    
x = np.arange(-5.0, 5.0, 0.1) # -5.0 ~ 5.0까지 0.1씩 증가 
y = sigmoid(x) 
plt.plot(x,y) 
plt.ylim(-0.1,1.1) # y축 값의 범위 설정 plt.show()
print(y)