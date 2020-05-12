from functools import reduce

class ExtendedList(list):
    def __init__(self, seq=()):
        list.__init__(self, seq)

    def apply(self, func):
        return reduce(func, list(self))


x = ExtendedList([1,2,3])

print(x.apply(lambda x,y:x+y))