
       
       
       
        
        """fitness 출력"""
        MFIT = max(fitness_table)
        print('max_fitness :', MFIT, '/ index :', fitness_table.index(MFIT))
        MINFIT = min(fitness_table)
        gen_minfitness.append(MINFIT)
        
        ### 룰렛 휠 배정
        fitness_sum = 0
        cnt = 0
        
        for i in fitness_table:
            fitness_sum += i
        
        """평균 fitness 출력"""
        AFIT = fitness_sum/num
        gen_avefitness.append(AFIT)
        print('average fitness :', AFIT)

        index_table = []
        while cnt < num:
            for i in fitness_table:
                point = rand(0, 100)
                index = fitness_table.index(i)
                if i > point:
                    """child에 parents 염색체 유전"""
                    child.append(parents[index])
                    cnt += 1
                    index_table.append(index)
                if cnt == num:
                    break
        #선택된 인덱스 출력
        #print('gen', gen, 'index_table', index_table)
        
        ### 교배
        turn = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #교배 순서 섞어주는 함수
        cross_num = num * (num - 1) / 2
        for i in range(intrand(cross_num, cross_num+20)):
            val = 0
            p1, p2 = rand2num(0, num)
            val = copy.deepcopy(turn[p1])
            turn[p1] = copy.deepcopy(turn[p2])
            turn[p2] = copy.deepcopy(val)
            
        ### 교차
        for i in range(0, num // 2):
            bucket = []
            d1, d2 = rand2num(0, x_size)
            """
            print('before : ', child[turn[i]], child[turn[i+1]])
            print('d1 :', d1, 'd2 :', d2)
            """
            for j in range(d1, d2+1):   #d1번째 인덱스부터 d2번째 인덱스까지
                bucket.append(child[turn[i]][j])
                child[turn[i]][j] = copy.deepcopy(child[turn[i+1]][j])
                child[turn[i+1]][j] = bucket[j-d1]
            
            #print('bucket :', bucket)
            #교차 유전자 출력
            #print(child[turn[i]], child[turn[i+1]])
            
            turn = turn[1:]
        
        ### 돌연변이
        #fitness 재산출
        time_table = []
        fitness_table = []
        for i in child:
            time_table.append(sum_time(i))
            fitness_table.append(time_table[-1])
            
        #돌연변이
        for i in range(num):
            rnum = intrand(0, 100)
            if fitness_table[i] < rnum:
                fit_to_num = round(fitness_table[i]/10)
                for j in range(fit_to_num):
                    rindex = intrand(0, 9)
                    rnum = intrand(-10, 0)
                    child[i][rindex] = rnum

        ### 정규화
        for i in range(num):
            child[i] = copy.deepcopy(generalization(child[i]))

        gen_maxfitness.append(max(fitness_table))

        MFIT = max(fitness_table)
        max_index = fitness_table.index(MFIT)
        MINFIT = min(fitness_table)
        min_index = fitness_table.index(MINFIT)
        #draw(child[max_index])
        #draw(child[min_index])

        fx = []
        for i in range(10):
            fx.append(i)
        #세대당 fitness 출력
        #plt.plot(fx, fitness_table)
        #plt.show()
        #plt.title('fitnesses of a generation')    
        
        #세대당 모든 그래프 출력
        for i in range(num):
            #draw(child[i])
            print(fitness_table[i])
        #print(child[index])
    """generations fitness 출력"""
    draw(child[max_index])

    plt.plot(generation, gen_maxfitness)
    plt.plot(generation, gen_avefitness)
    plt.plot(generation, gen_minfitness)
    plt.legend(['max_fitness', 'ave_fitness', 'min_fitness'])
    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.title('fitnesses per generations')
    plt.show()