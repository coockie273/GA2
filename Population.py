import random
import time
import numpy as np
import matplotlib.pyplot as plt

low_bound = -32.768
high_bound = 32.768
eta = 3


class Population:

    def __init__(self, power, iteration, p_cross, p_mut, function, n):
        self.population = []
        self.power = power
        self.iteration = iteration
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.func = function
        self.n = n

        # Создание начальной популяции
        for i in range(0, power):
            chromosome = []
            for j in range(0, n):
                chromosome.append(random.uniform(low_bound, high_bound))
            self.population.append(chromosome)

    def value(self, chromosome):
        return self.func(chromosome, self.n)

    def reproduction(self):

        values = [self.value(element) for element in self.population]

        delta = min(values) - 0.1
        if delta >= 0.1:
            delta = 0
        sum = 0

        for individual in self.population:  # Для поиска минимума получаем обратный результат каждой особи
            sum += (1 / (self.value(individual) - delta))  # delta - константа для сдвига отрицательных решений

        probabilities = []

        for individual in self.population:  # Формирование вероятностей
            probabilities.append((1 / (self.value(individual) - delta)) / sum)
        print(len(probabilities))
        new_population = random.choices(self.population, probabilities, k=self.power)  # Вращение колеса

        self.population = new_population

    def sbx_crossover(self):

        for i in range(0, self.power, 2):
            parent1 = self.population[i]
            parent2 = self.population[i + 1]

            child1, child2 = self.sbx_crossover_worker(parent1, parent2)

            self.population[i] = child1
            self.population[i + 1] = child2
    def sbx_crossover_worker(self, parent1, parent2):

        if np.random.rand() > self.p_cross:
            return parent1, parent2

        size = len(parent1)
        child1 = np.zeros(size)
        child2 = np.zeros(size)

        for i in range(size):
            if np.random.rand() < 0.5:
                beta = (2.0 * np.random.rand()) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - np.random.rand()))) ** (1.0 / (eta + 1.0))

            child1[i] = 0.5 * (((1.0 + beta) * parent1[i]) + ((1.0 - beta) * parent2[i]))
            child2[i] = 0.5 * (((1.0 - beta) * parent1[i]) + ((1.0 + beta) * parent2[i]))

        return child1, child2

    def mutation(self):
        for chromosome in self.population:
            random_number = random.random()
            if random_number < self.p_mut:
                k = random.randint(0, self.n - 1)
                chromosome[k] = random.uniform(low_bound, high_bound)

    def draw(self, value):
        x = np.linspace(-32.768, 32.768, 100)
        y = np.linspace(-32.768, 32.768, 100)

        # Создаем сетку из координат x и y
        x, y = np.meshgrid(x, y)

        # Создаем массив для z, используя функцию func
        coordinates = np.column_stack((x.flatten(), y.flatten()))
        z = np.apply_along_axis(lambda coord: self.func(coord, 2), 1, coordinates)

        # Изменяем форму массива z обратно на двумерный
        z = z.reshape(x.shape)

        # Создаем трехмерный график
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Строим поверхность
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

        # Добавляем метки
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Генерируем точки для отображения
        num_points = len(self.population)
        x_points = [ind[0] for ind in self.population]
        y_points = [ind[1] for ind in self.population]

        # Преобразуем координаты точек в массив
        coordinates_points = np.column_stack((x_points, y_points))
        # Рассчитываем значения z для точек
        z_points = np.vectorize(lambda x, y: self.func([x, y], 2))(x_points, y_points)

        # Отображаем точки на графике
        ax.scatter(x_points, y_points, z_points, color='blue', marker='o', s=50)

        ax.set_title("Минимальное найденное значение - " + str(value))

        # Показываем график
        plt.show()


    def start(self):
        #start_time = time.time()
        for i in range(0, self.iteration):
            if i == 5 or i == 10 or i == 20 or i == 100:
                self.draw(min(map(self.value, self.population)))
            print("Iteration: ", i)
            self.reproduction()
            self.sbx_crossover()
            self.mutation()
        #end_time = time.time()
        min_index, min_value = min(enumerate(map(self.value, self.population)), key=lambda x: x[1])

        return min_value, self.population[min_index]
