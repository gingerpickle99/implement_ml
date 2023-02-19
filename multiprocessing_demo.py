import multiprocessing as mp
import numpy as np
import time

# define a cost function
def cost(y):
    return -1*(y[0]*np.log(y[1])+(1-y[0])*np.log(1-y[1]))

def main():

    # create a pool of 10 processes
    pool=mp.Pool(processes=8)


    # create random examples of y_true and y_pred
    y_true=np.random.randint(0,2,size=100000)
    y_pred=np.random.random(size=100000)
    zipped=list(zip(y_true,y_pred))


    #without multiprocessing
    print("\n**********without multiprocessing**********")
    start=time.time()
    costs=[]
    for i in zipped:
        costs.append(cost(i))
    print(f"mean_cost={np.mean(costs)}")
    checkpoint_1=time.time()
    print(f"time_taken = {checkpoint_1-start}")


    #with multiprocessing
    print("\n**********with multiprocessing**********")
    checkpoint_2=time.time()
    costs=pool.map(cost,zipped)
    print(f"mean_cost={np.mean(costs)}")
    checkpoint_3=time.time()
    print(f"time_taken = {checkpoint_3-checkpoint_2}")

    print("\n******************************************")

    # percentage of time saved
    print(f"percentage of time saved = {((1-((checkpoint_3-checkpoint_2)/(checkpoint_1-start)))*100):.2f}%")

if __name__=='__main__':
    main()    