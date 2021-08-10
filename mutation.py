from random import uniform
import copy
from math import tan, pi, sqrt
import matplotlib.pyplot as plt
x = list(range(11))
x_size = 10
y_size = 10
r = 5
G = 9.8
def rand(x, y): 
    return uniform(x, y)

def draw(arr):
    #arr : 기울기 배열 입력
    y = [10]
    for i in range(x_size):
        y.append(y[-1] + arr[i])
    
    plt.plot(x, y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Cycloid')
    #plt.show()

def sum_time(arr):
    sum = 0
    v0 = 0
    for m in arr:
        M = abs(m)
        batch = r * pi / x_size
        h = M * batch
        #가속도 정의, m <= 0이라는 가정 하에 만듦
        a = h * G / sqrt(batch**2 + h**2)
        #나중속도 산출
        if M == 0:
            after_v = v0
        else:
            #feat. 역학E보존
            after_v = sqrt(2*G*h + v0**2)
        #가속도 정의 이용, t 산출
        if a == 0:
            if v0 == 0:
                return 1000
            else :
                t = pi/v0
        else:
            t = (after_v - v0)/a
        sum += t
        v0 = after_v
    return sum

def generalization(arr0):
    arr = copy.deepcopy(arr0)#깊은 복사, 이래야 원래 값에 영향 x
    Msum = 0
    for i in arr:
        Msum += i
        if i > 0:
            arr[arr.index(i)] *= 1
    delta = y_size / Msum * -1

    for i in range(len(arr)): 
        arr[i] *= delta
        #arr[i] = round(arr[i], 4)
    return arr

mut = 0
mut_rate = 0.1
m = [-3.2603228493238032, -1.0332827037059389, -2.0145923667787486, -0.7167506475602782, -1.049655743301336, -0.7413646402396894, -0.3147298662258255, -0.3064669554012553, -0.4120247193942684, -0.1508095080688543]
#draw(m)
#plt.show()
t0 = sum_time(m)

while(mut < 2):
    m1 = []
    for i in range(10):
        point = int(rand(0, 10))
        m1.append(m[i])
        if point == i:
            mut += 1
            rate = rand(1 - mut_rate, 1 + mut_rate)
            m1[i] *= rate
            print('changed value :', m[i], '>', m1[i])
            print('mut rate :', rate)
            print(' ')
m1 = generalization(m1)
#draw(m1)
#plt.show()
print('mut :', mut)
t1 = sum_time(m1)
print('t0', t0, '> t1', t1)
print('delta', t1-t0)
        
