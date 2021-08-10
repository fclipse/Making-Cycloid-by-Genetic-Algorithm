from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt

def cycloid(r):
  x = [] #x좌표 리스트 만듦
  y = [] #y좌표 리스트 만듦

  for theta in np.linspace(0, np.pi, 100): #theta변수를 -2π 에서 2π 까지 반복함
    x.append(r*(theta - sin(theta))) #x 리스트에 매개변수함수값을 추가시킴
    y.append(10 -(r*(1 - cos(theta)))) #y 리스트에 매개변수함수값을 추가시킴

  plt.figure(figsize=(pi*2, 4)) # 그래프 비율 조정
  plt.plot(x,y)  #matplotlib.piplot을 이용해 그래프 그리기
  plt.title('Cycloid')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([0, 5*pi])
  plt.ylim([0, 10])
  plt.show()  #그래프 출력하기

cycloid(5)  #