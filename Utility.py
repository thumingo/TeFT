import itertools
import numpy as np


def add_subtract(numbers):
    result = [(num - 0.1666, num - 0.0833, num, num + 0.0833, num + 0.1666) for num in numbers]
    return [item for sublist in result for item in sublist]


def remove_close_numbers(numbers):
    for pair in itertools.combinations(numbers, 2):
        if abs(pair[1] - pair[0]) < 5:
            numbers.remove(pair[1])
            return remove_close_numbers(numbers)
    return numbers


def index_number(li, defaultnumber):
    select = defaultnumber - li[0]
    index = 0
    for i in range(1, len(li) - 1):
        select2 = defaultnumber - li[i]
        if abs(select) > abs(select2):
            select = select2
            index = i
    if li[index] < defaultnumber:
        index = index + 1
    return index


def find_appltitude(ans2, amp):
    Idiff = np.convolve(ans2, np.array([1, -1]), 'full')
    idxPeak = []
    for i in range(ans2.size):
        if Idiff[i] > 0 >= Idiff[i + 1] and ans2[i] > amp:
            idxPeak.append(i)
    idxPeak = remove_close_numbers(idxPeak)
    return idxPeak


def remove_non_increasing_pairs(X, Y):
    i = 0
    while i < len(X) - 1:
        if X[i] >= X[i + 1]:
            X = np.delete(X, i + 1)
            Y = np.delete(Y, i + 1)
        else:
            i += 1
    return X, Y
