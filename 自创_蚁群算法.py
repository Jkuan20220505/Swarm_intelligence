# 求50个城市之间最短距离
import random
import copy
import sys

# ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
# BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会加快，但是随机性不高，容易得到局部的相对最优
# RHO:旧信息素的衰减率
# Q:方便计算新的信息素，信息素与距离成反比
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# 城市数，蚁群
(city_num, ant_num) = (50, 50)
distance_x = [
    178, 272, 176, 171, 650, 499, 267, 703, 408, 437, 491, 74, 532,
    416, 626, 42, 271, 359, 163, 508, 229, 576, 147, 560, 35, 714,
    757, 517, 64, 314, 675, 690, 391, 628, 87, 240, 705, 699, 258,
    428, 614, 36, 360, 482, 666, 597, 209, 201, 492, 294]
distance_y = [
    170, 395, 198, 151, 242, 556, 57, 401, 305, 421, 267, 105, 525,
    381, 244, 330, 395, 169, 141, 380, 153, 442, 528, 329, 232, 48,
    498, 265, 343, 120, 165, 50, 433, 63, 491, 275, 348, 222, 288,
    490, 213, 524, 244, 114, 104, 552, 70, 425, 227, 331]
# 城市距离和信息素
distance_graph = [[0.0 for _ in range(city_num)] for _ in range(city_num)]
pheromone_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]

# 面向对象设计
# ----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self, ID):

        self.ID = ID  # ID
        self.__clean_data()  # 随机初始化出生点

    # 数据初始化
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径
        self.total_distance = 0.0  # 当前路径的总距离
        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)]  # 探索城市的状态

        city_index = random.randint(0, city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  # 存储去下个城市的概率
        total_prob = 0.0  #总的概率

        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,current=self.current_city, target=i))
                    sys.exit(1)
        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i  #选择出下一个城市
                        break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0
        start = 0
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i - 1]
            temp_distance += distance_graph[start][end]

        # 再加上构成回路的边的长度
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance

    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()

# 更新信息素
def __update_pheromone_gragh(Ant):
    # 获取每只蚂蚁在其路径上留下的信息素
    temp_pheromone = [[0.0 for _ in range(city_num)] for _ in range(city_num)]
    for ant in Ant:
        for i in range(1, city_num):
            start, end = ant.path[i - 1], ant.path[i]
            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
            temp_pheromone[start][end] += Q / ant.total_distance
            temp_pheromone[end][start] = temp_pheromone[start][end]  #无向图，记录每只蚂蚁在路径上留下的信息素

    # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
    for i in range(city_num):
        for j in range(city_num):
            pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]


def search_path():
    # 初始城市之间的距离和信息素
    for i in range(city_num):
        for j in range(city_num):
            pheromone_graph[i][j] = 1.0

    ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
    best_ant = Ant(-1)  # 初始最优解
    best_ant.total_distance = 1 << 31  # 初始最大距离

    # 计算城市之间的距离
    for i in range(city_num):
        for j in range(city_num):
            temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
            temp_distance = pow(temp_distance, 0.5)  # 平方和开根号
            distance_graph[i][j] = float(int(temp_distance + 0.5))  # 向上取整

    # 开启线程
    k = 1
    while k<500:
        # 遍历每一只蚂蚁
        for ant in ants:
            # 搜索一条路径
            ant.search_path()
            # 与当前最优蚂蚁比较
            if ant.total_distance < best_ant.total_distance:
                # 更新最优解
                best_ant = copy.deepcopy(ant)
        # 更新信息素
        __update_pheromone_gragh(ants)
        print(u"迭代次数：", k, u"最佳路径总距离：", int(best_ant.total_distance))
        k += 1


if __name__ == '__main__':

    search_path()
    # 迭代次数： 499   最佳路径总距离： 3717
    # 迭代次数： 499   最佳路径总距离： 3755
    # 迭代次数： 310   最佳路径总距离： 3687