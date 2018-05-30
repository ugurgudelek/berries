
from tqdm import trange
from time import sleep

with trange(50,100) as t:
    for i in t:
        t.set_description('GEN %i' % i)
        t.set_postfix(loss=2.5, gen=1.1, str='h',
                      lst=[1, 2])
        sleep(1)