#유전 알고리즘을 이용한 최속강하곡선 그리기
#%%
import copy
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from math import sqrt, cos, sin, tan, atan, pi, exp

# According to PEP8, do not assign lambda
def rand(x, y):
    return uniform(x, y)
def intrand(x, y):
    return int(uniform(x, y))

#맵 크기
r = 5
x_len = 2*r     #그래프상 x범위
x_size = 20    #구간 개수

y_len = 2*r
y_size = 2*r

x = []
for i in range(x_size + 1):
    x.append(i*pi*r/x_size)
#중력가속도
G = 9.80665
#최단 시간
T = sqrt(r / G)*pi

def cycloid(r):
  x = [] #x좌표 리스트 만듦
  y = [] #y좌표 리스트 만듦

  for theta in np.linspace(0, np.pi, 100): #theta변수를 -2π 에서 2π 까지 반복함
    x.append(r*(theta - sin(theta))) #x 리스트에 매개변수함수값을 추가시킴
    y.append(10 -(r*(1 - cos(theta)))) #y 리스트에 매개변수함수값을 추가시킴

  #plt.figure(figsize=(pi*2, 4)) # 그래프 비율 조정
  plt.plot(x,y)  #matplotlib.piplot을 이용해 그래프 그리기
  plt.title('Cycloid')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([0, 5*pi])
  plt.ylim([0, 10])
  plt.show()  #그래프 출력하기

#정규화 함수, 배열의 기울기 합과 y_size가 같도록 만들어줌
#y_size와 m합의 비율만큼 각 항에 곱해줌
def generalization(arr0):
    arr = copy.deepcopy(arr0)#깊은 복사, 이래야 원래 값에 영향 x
    Msum = 0
    for i in arr:
        Msum += i
        if i > 0:
            arr[arr.index(i)] *= 1
    delta = y_len / Msum * -1

    for i in range(len(arr)): 
        arr[i] *= delta
    return arr

#시간 합 구하는 함수
def sum_time(arr):
    sum = 0
    v0 = 0
    for m in arr:
        M = abs(m)
        batch = r * pi / x_size
        h = M * batch
        #가속도 정의, m <= 0이라는 가정 하에 만듦
        #a = h * G / sqrt(batch**2 + h**2)
        a = G * M / sqrt(1+M**2)
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

#적합도 구하는 함수, 룰렛 휠 방식, 선택압 상수 이용.
#최대 적합도가 최소 적합도의 k배가 되도록 만듦
def fitness(t):
    #arr : sum_time입력받음
    #별로 차이가 나지 않아 제곱을 해 주는 것이 좋을 것 같음
    f = (T/t)**2 * 100
    f
    return f
#서로 다른 두 랜덤 정수를 반환하는 함수

def rand2num(x, y):
    num1 = intrand(x, y)
    num2 = intrand(x, y)
    while num1 == num2:
        num2 = intrand(x, y)
    #num2가 더 크도록 swap
    if num1 > num2:
        val = copy.deepcopy(num2)
        num2 =copy.deepcopy(num1)   
        num1 = copy.deepcopy(val)
    return num1, num2

#그래프 그려주는 함수
def draw_cycloid(arr):
    #arr : 기울기 배열 입력
    y = [y_len]
    for i in range(x_size):
        y.append(y[-1] + arr[i])
    
    plt.plot(x, y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('GA_Cycloid')
    plt.xlim([0, r*pi])
    plt.ylim([0, 10])
    plt.show()

#유전 알고리즘
def selection(num, min_m, max_gen, mut_chance, mut_rate):
    #초기 자녀 배열 생성, 자녀 수 num, 자녀 1명당 파라미터 수 x_size
    child = []
    parents = []
    #시간 합 배열
    time_table = []

    #0세대
    gen = 0
    #세대 변수
    generation = [0]
    #세대 최고 적합도
    gen_maxfitness = []
    #세대 평균 적합도
    gen_avefitness = []
    #세대 최저 적합도
    gen_minfitness = []
    sum_fitness = 0
    max_index = 0
    #적합도 배열, 부모의 수만큼 생성
    fitness_table = []
    for i in range(num):
        child.append([])
        child[i].append(0)      #fitness 자리
        child[i].append(0)      #time 자리
        child[i].append([])
        
        #무작위 생성
        for j in range(x_size):
            theta = rand(atan(min_m), 0)   #-2 이상 0 미만 실수값 입력, 단위 라디안
            #10개 숫자 평균적 합이 10이 되도록 맞춰줌 
            randM = tan(theta)
            #randM = rand(-2, 0)        # 기울기 자체를 랜덤값으로 결정
            child[i][2].append(randM)
        
        #정규화
        child[i][2] = generalization(child[i][2])
        #print(child[i])
    
    # 0세대 적합도 계산
    for i in range(num):
        time_table.append(sum_time(child[i][2]))
        child[i][1] = copy.deepcopy(time_table[i])

        fitness_table.append(fitness(time_table[-1]))
        child[i][0] = copy.deepcopy(fitness_table[i])
        sum_fitness += fitness_table[i]

    # 0세대 적합도 내림차순으로 정렬
    child.sort(reverse=True)
    fitness_table.sort(reverse=True)        #미리 정렬
    time_table.sort()

    gen_maxfitness.append(fitness_table[0])
    gen_avefitness.append(sum_fitness/num)
    gen_minfitness.append(fitness_table[-1])
    #print('0세대 best :',child[0])
    """
    #0세대 최대, 최소 그래프 그리기
    #max
    draw(child[0][2])
    #min
    draw(child[num-1][2])
    plt.legend(['max fitness', 'min fitness'])
    plt.show()
    print('max fitness :', fitness_table[0],'/ time :', time_table[0])
    print('min fitness :', fitness_table[-1],'/ time :', time_table[-1])

    #0세대 fitness 그래프 출력
    plt.plot(list(range(num)), fitness_table, marker = 'o')
    plt.title('Generation 1 fitness table')
    plt.xlim([0, num])
    plt.ylim([0, 100])
    plt.show()

    print(child[0])
    """    
    # 유전 시작=====================================================================
    
    while gen < max_gen: 
        #세대 카운팅
        gen += 1
        generation.append(gen)
        parents = []
        inherit = []
        # 0. 아이 유전자를 다음 세대로 전달
        #child.sort(reverse = True)
        parents = copy.deepcopy(child)
        child = []
        # 1. 상위 10%만 다음 세대로 전달
        for i in range(num//100*10):
            inherit.append(parents[i])
        child = copy.deepcopy(inherit)
        
        # 2. 50% 교차시킴
        #임시 저장 변수
        val = 0
        for i in range(num//100*50//2):
            #0, 1번째 인덱스는 사람을, 2, 3번째 인덱스는 교차 포인트를 결정함
            index = []
            for j in range(4):
                if j < 2:
                    index.append(int(rand(0, num//10)))
                else:
                    index.append((int(rand(0, x_size))))
                
                if j % 2 == 1:
                    #같을 경우 하나만 바꿔줌
                    while index[j-1] == index[j]:
                        if j == 1:
                            index[j] = int(rand(0, num//10))
                            index[j-1] = int(rand(0, num//10))
                        else:
                            index[j] = int(rand(0, x_size))
                            index[j-1] = int(rand(0, x_size))
                    
                    #0 1, 2 3은 각각 오름차순으로 정렬
                    if index[j-1] > index[j]:
                        val = copy.deepcopy(index[j])
                        index[j-1] = copy.deepcopy(index[j])
                        index[j] = copy.deepcopy(val)

            #child에 변화시킬 유전자 2개씩 추가
            child.append(inherit[index[0]])
            child.append(inherit[index[1]])
            #j번째 인덱스를 swap
            val = 0
            #2점교차
            for j in range(index[2], index[3]+1):
                val = copy.deepcopy(child[-1][2][j])
                child[-1][2][j] = copy.deepcopy(child[-2][2][j])
                child[-2][2][j] = copy.deepcopy(val)
        
        # 3. 50% 돌연변이
        inherit *= 3
        for i in inherit:
            child.append(i)
        
        for i in range(num//100*40, num//100*90):
            for j in range(x_size):
                point = rand(0, 1)
                if(point < mut_chance):
                    rate = rand(1-mut_rate, 1+mut_rate)
                    child[i][2][j] *= rate
        # 4. 무작위 10% 생성
        for i in range(num//10):
            child.append([])
            child[-1].append(0)
            child[-1].append(0)
            bucket = []
            b_fitness = 0
            #적합도 90 이상인 아이들만 유전시킴
            #while b_fitness < 90:
            for j in range(x_size):
                theta = rand(atan(min_m), 0)   #-2 이상 0 미만 실수값 입력, 단위 라디안
                #10개 숫자 평균적 합이 10이 되도록 맞춰줌 
                randM = tan(theta)
                #randM = rand(-2, 0)        # 기울기 자체를 랜덤값으로 결정
                bucket.append(randM)
                #b_fitness = fitness(sum_time(bucket))

            child[-1].append(bucket)
        # 5. 정규화
        for i in range(num):
            child[i][2] = generalization(child[i][2])
        
        #gen 출력
        if gen % 1000 == 0:
            print('gen :', gen)
        # 6. 적합도 계산
        fitness_table = []
        time_table = []
        sum_fitness = 0
        for i in range(num):
            time_table.append(sum_time(child[i][2]))
            child[i][1] = copy.deepcopy(time_table[i])

            fitness_table.append(fitness(time_table[-1]))
            child[i][0] = copy.deepcopy(fitness_table[i])
            sum_fitness += fitness_table[i]
        fitness_table.sort(reverse = True)
        time_table.sort()

        gen_maxfitness.append(fitness_table[0])
        if(gen_maxfitness[-1] > gen_maxfitness[-2]):
            max_index = gen
        gen_avefitness.append(sum_fitness/num)
        gen_minfitness.append(fitness_table[-1])
        """
        plt.plot(list(range(num)), fitness_table)
        plt.title('Fitness Table')
        plt.xlim([0, num])
        plt.ylim([0, 100])
        plt.show()
        """
        child.sort(reverse = True)
        

    # 7. 적합도 그래프 출력
    #plt.figure(figsize = (100, 100))
    plt.plot(generation, gen_maxfitness, marker = 'o')
    plt.plot(generation, gen_avefitness, marker = 'o')
    plt.plot(generation, gen_minfitness, marker = 'o')
    plt.title('Fitnesses of All Generations')
    plt.legend(['max', 'ave', 'min'])
    plt.show()
    #print('max', gen_maxfitness)
    #print('ave', gen_avefitness)
    #print('min', gen_minfitness)
    

    draw_cycloid(child[0][2])
    #세대출력
    print('gen :', max_index)
    t = sum_time(child[0][2])
    print('max fitness :', child[0][0])
    print('Solution :', t, 's')
    print('delta', t - T)
    print('child :', child[0])

"""알고리즘 개요"""
#1. 자식세대 유전자를 부모 세대로 전달(성숙)
#2. 적합도 상위 10% 자녀들만 다음 세대로 유전.
#3. child세대로 선택된 유전자 전달
#4. 전체의 50% 교차됨
#5. 전체의 50% 돌연변이를 일으킴
#6. 10%는 그대로 전달, 10%는 무작위 생성됨
#6. 정규화
#7. 세대 종료, 다음 세대로 이동


#최대 세대 수
max_gen = 2000
#자녀 수
num = 200
#최소 기울기값
min_m = -10
#돌연변이율
mut_chance = 0.3
#돌연변이시 바꾸는 최대 비율(양수 입력)
mut_rate = 0.6
print('Making Cycleroid by Genetic Algorithm by Han SJ')
print('Best Solution T :', T)
# 자녀 정보 출력
print('total child :', num, 'chromosom length : ', x_size)
print('min_m :', min_m, 'mut_chance :', mut_chance, 'mut_rate :', mut_rate, 'batch :', x_size)
#유전 알고리즘 시작
selection(num, min_m, max_gen, mut_chance, mut_rate)
cycloid(5)