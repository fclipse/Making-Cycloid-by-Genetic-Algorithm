from math import tan, pi, sqrt
r = 50
G = 9.8
T = sqrt(5/G)*pi
x_size = r*pi/100

def sum_time(m, v0):
    sum = 0
    M = abs(m)
    #가속도 정의, m <= 0이라는 가정 하에 만듦
    a = M * G / sqrt((r*pi/10)**2+m**2)
    #나중속도 산출
    if M == 0:
        after_v = v0
    else:
        #feat. 역학E보존
        after_v = sqrt(2*G*M + v0**2)
    #가속도 정의 이용, t 산출
    if a == 0:
        if v0 == 0:
            return 1000
        else :
            t = pi/v0
    else:
        t = (after_v - v0)/a
    sum += t
    return sum

print(sum_time(0.5/pi, 2*sqrt(10)))