import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
from numpy import linalg as LA
import pickle
import time
from joblib import Parallel, delayed
import multiprocessing

def randsmpl(p):
    # from uniform 0-1 rv, find last bin it fits in, return index
    p = np.insert(p, 0, 0)  # 0 indexing fix
    cdf = np.cumsum(p)  # get cdf
    cdf = cdf / np.max(cdf)  # normalize
    x = np.flatnonzero(random.random() > cdf)
    x = x[-1]
    return x

def visualization(alldata, reps, T, probs):
    # visualize the tracks
    for iprob in range(len(alldata)):
        x = alldata[iprob][0]
        y = alldata[iprob][1]
        b = alldata[iprob][2]
        plt.figure()
        plt.title(str(probs[iprob]))

        for irep in range(reps+1):
            # x1 = [x[:T+3],x[T+3:2*(T+3)],x[2*(T+3):3*(T+3):],x[3*(T+3):]]
            # x1 = x[:(irep+1)*(T+3)]
            # y1 = y[:(irep+1)*(T+3)]
            # b1 = b[:(irep+1)*(T+3)]
            x1 = x[:(T + 2)]
            y1 = y[:(T + 2)]
            b1 = b[:(T + 2)]
            # for it in range(len(x1)):
            plt.plot(x1, y1)
                # plt.title(str(b1[it]))
            x = np.delete(x, range(0, (T + 2)))
            y = np.delete(y, range(0, (T + 2)))
            b = np.delete(b, range(0, (T + 2)))
    # plt.xlabel('time')
    # plt.ylabel('Temperature')
    return

def run_replicate(initial_point, find_point, map_data, T, p_behavior, alpha, LL):
    # fully run a single replicate for the LP model, returns final x y location and behavior used
    # define possible motions in body coordinates
    right = np.array([1, 0])
    left = np.array([-1, 0])
    front = np.array([0, 1])
    back = np.array([0, -1])
    stay = np.array([0, 0])
    frontright = np.array([1, 1])
    frontleft = np.array([-1, 1])
    backright = np.array([1, -1])
    backleft = np.array([-1, -1])

    # define possible motions
    possiblemotions = [frontleft, front, frontright, left, stay, right, backleft, back, backright]

    # initial BW_temp (0 or 1) for initial position
    bw_temp = np.empty((3, 3))
    bw_temp[:] = np.NaN

    # input first two positions
    icx, icy = np.empty((2, 1), dtype='int'), np.empty((2, 1), dtype='int')
    icx[0] = initial_point[0]
    icy[0] = initial_point[1]
    icx[1] = icx[0] + (math.floor(3 * random.random()) - 1)
    icy[1] = icy[0] + (math.floor(3 * random.random()) - 1)
    # icx[1] = icx[0] + (math.floor(3 * 1) - 1)         #### eliminate random number for debugging
    # icy[1] = icy[0] + (math.floor(3 * 1) - 1)

    # initialize position matrix
    # x, y, behavior = np.empty((1, T + 2), dtype='int'), np.empty((1, T + 2), dtype='int'), np.empty((1, T + 2),
    #                                                                                                 dtype='int')
    x, y, behavior = np.empty((1, T + 2)), np.empty((1, T + 2)), np.empty((1, T + 2))
    x[:], y[:], behavior[:] = np.NaN, np.NaN, np.NaN
    x[0, 0] = icx[0]
    y[0, 0] = icy[0]
    x[0, 1] = icx[1]
    y[0, 1] = icy[1]

    # initialize velocity and compute initial value in body coordinates
    u, v = np.empty((1, T + 2)), np.empty((1, T + 2))
    u[:], v[:] = np.nan, np.nan
    u[0, 0] = x[0, 1] - x[0, 0]
    v[0, 0] = y[0, 1] - y[0, 0]

    # initialize vx and vy
    vx, vy, xs, ys = np.empty((1, T + 2)), np.empty((1, T + 2)), np.empty((1, T + 2)), np.empty((1, T + 2))
    vx[:], vy[:], xs[:], ys[:] = np.nan, np.nan, np.nan, np.nan

    # initialize possible behavior matrix
    xy_temp = np.zeros((2, 1))

    # initialize flag for going off the map
    flag = 0

    counter = 1
    for ii in range(1, T+1):
        # check if it went off the map, break out for a failed rep if it did
        if flag == 1:
            break

        # Testing Temporal Decision Points
        # if counter % k == 1:
        #     # choose motion based on type of person from p_behavior
        #     behavior[0, ii] = randsmpl(p_behavior)
        # else:
        #     behavior[0, ii] = behavior[0, ii-1]
        #
        # counter = counter + 1

        # initialize staying put flag
        flag_sp = 0

        # #################  COMPUTE ALL POSSIBLE BEHAVIORS #################

        if behavior[0, ii] == 0:
            # ######### 1. random traveling (rw) #############
            px_rw = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) / 9  # random walk pdf
            inds = randsmpl(px_rw)
            motions = possiblemotions[inds]

            # provisional update for random traveling
            u, v = motions[0], motions[1]  # updated velocity in body coordinates

            if x[0, ii] - x[0, ii - 1] == 0 and y[0, ii] - y[0, ii - 1] == 0:  # prevents arctan2 from having (0/0)
                theta = 2 * math.pi * random.random()
            else:
                theta = -np.arctan2(x[0, ii] - x[0, ii - 1],
                                    y[0, ii] - y[0, ii - 1])  # angle of previous velocity in global coordinates

            m = np.array([[np.cos(theta), -np.sin(theta), x[0, ii]], [np.sin(theta), np.cos(theta), y[0, ii]],
                          [0, 0, 1]])  # transformation for rotation by theta and translation by previous position

            temp = np.matmul(m, np.array([u, v, 1]).reshape(-1, 1))  # updated position in global coord computed from transformed updated velocity in body coords
            x_rw = np.round(temp[0, 0])  # #### NOTE: ROUND IS DIFFERENT IN MATLAB (half->bigger integer)
            y_rw = np.round(temp[1, 0])

            xy_temp = np.array([x_rw, y_rw])
        elif behavior[0, ii] == 1:
            # ######### 2. route traveling (rt) #############
            px_rt = np.array([3, 3, 3, 0, 0, 0, 0, 0, 0]) / 9

            if x[0, ii] - x[0, ii - 1] == 0 and y[0, ii] - y[0, ii - 1] == 0:  # prevents arctan2 from having (0/0)
                theta = 2 * math.pi * random.random()
            else:
                theta = -np.arctan2(x[0, ii] - x[0, ii - 1],
                                    y[0, ii] - y[0, ii - 1])  # angle of previous velocity in global coordinates

            m = np.array([[np.cos(theta), -np.sin(theta), x[0, ii]], [np.sin(theta), np.cos(theta), y[0, ii]],
                          [0, 0, 1]])  # transformation for rotation by theta and translation by previous position

            #  create a new pdf for the prob of being on a linear feature using BW
            bw_temp = bw_temp.flatten('F')
            for jj in range(0, 9):
                aux = possiblemotions[jj]  # choose a motion
                aux1 = np.round(np.matmul(m, np.array([aux[0], aux[1], 1]).reshape(-1, 1)))  # put the updated position in world coordinates
                aux1 = np.delete(aux1,-1)

                if aux1[1] >= LL[1] or aux1[1] < LL[2] or aux1[0] >= LL[0] or aux1[0] < LL[2]:
                    flag = 1
                    break

                bw_temp[jj] = map_data[1][int(aux1[1]), int(aux1[0])]

            if flag == 1:
                break
            bw_temp = np.reshape(bw_temp, (3, 3), order='F')
            px_lin = px_rt.transpose() * bw_temp.flatten('F')  # pdf for linear feature using random trav px_rt
            px_lin = px_lin.transpose() / LA.norm(px_lin, 1) if LA.norm(px_lin, 1) != 0 else np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]) # normalizing the probability
            px_lin = np.where(np.isnan(px_lin), 1 / 9, px_lin)  # if no linear features, dist will be NaN, so set it back to random walk but it doesn't get used****

            inds_lin = randsmpl(px_lin)
            motions_lin = possiblemotions[inds_lin]

            # provisional update for route traveling
            u, v = motions_lin[0], motions_lin[1]  # updated velocity in body coordinates
            temp_lin = np.matmul(m, np.array([u, v, 1]).reshape(-1,
                                                                1))  # updated position in global coordinates computed from transformed updated velocity in body coords
            x_rt = np.round(temp_lin[0, 0])  # #### NOTE: ROUND IS DIFFERENT IN MATLAB (half->bigger integer)
            y_rt = np.round(temp_lin[1, 0])

            xy_temp = np.array([x_rt, y_rt])
        elif behavior[0, ii] == 2:
            # ################# 3. direction traveling (dt) ####################

            px_dt = np.array([0, 9, 0, 0, 0, 0, 0, 0, 0]) / 9  # pdf for traveling in forward direction

            inds = randsmpl(px_dt)
            motions = possiblemotions[inds]

            u, v = motions[0], motions[1]  # updated velocity in body coordinates
            if x[0, ii] - x[0, ii - 1] == 0 and y[0, ii] - y[0, ii - 1] == 0:  # prevents arctan2 from having (0/0)
                theta = 2 * math.pi * random.random()
            else:
                theta = -np.arctan2(x[0, ii] - x[0, ii - 1],
                                    y[0, ii] - y[0, ii - 1])  # angle of previous velocity in global coordinates

            m = np.array([[np.cos(theta), -np.sin(theta), x[0, ii]], [np.sin(theta), np.cos(theta), y[0, ii]],
                          [0, 0, 1]])  # transformation for rotation by theta and translation by previous position

            temp = np.matmul(m, np.array([u, v, 1]).reshape(-1,
                                                            1))  # updated position in global coord computed from transformed updated velocity in body coords
            x_dt = np.round(temp[0, 0])  # #### NOTE: ROUND IS DIFFERENT IN MATLAB (half->bigger integer)
            y_dt = np.round(temp[1, 0])

            xy_temp = np.array([x_dt, y_dt])
        elif behavior[0, ii] == 3:
            # ################# 4. staying put (sp) ##################
            x_sp = x[0, ii]
            y_sp = y[0, ii]

            flag_sp = 1

            xy_temp = np.array([x_sp, y_sp])
        elif behavior[0, ii] == 4:
            # ################# 5. view enhancing (ve) ##################
            if x[0, ii] - x[0, ii - 1] == 0 and y[0, ii] - y[0, ii - 1] == 0:  # prevents arctan2 from having (0/0)
                theta = 2 * math.pi * random.random()
            else:
                theta = -np.arctan2(x[0, ii] - x[0, ii - 1],
                                    y[0, ii] - y[0, ii - 1])  # angle of previous velocity in global coordinates

            m = np.array([[np.cos(theta), -np.sin(theta), x[0, ii]], [np.sin(theta), np.cos(theta), y[0, ii]],
                          [0, 0, 1]])  # transformation for rotation by theta and translation by previous position

            # create a new pdf based on intensity of gradient elevation (int)
            int_temp = np.zeros([9, 1])
            for kk in range(0, 9):
                auxv = possiblemotions[kk]  # choose a motion
                auxv1 = np.round(np.matmul(m, np.array([auxv[0], auxv[1], 1]).reshape(-1,
                                                                                      1)))  # put the updated position in world coordinates
                auxv1 = np.delete(auxv1,-1)

                if auxv1[1] >= LL[1] or auxv1[1] < LL[2] or auxv1[0] >= LL[0] or auxv1[0] < LL[2]:
                    flag = 1
                    break
                int_temp[kk] = map_data[2][int(auxv1[1]), int(auxv1[0])]  # find the elevation for each of the 9 possible movements

            if flag == 1:
                break

            int_temp = int_temp - int_temp[4]  # subtract the elevation of "stay put", i.e. the elevation of the current position

            aux_max = np.amax(int_temp)  # find the max elevation gain
            inds_ve = np.argwhere(int_temp.flatten() == np.amax(aux_max))  # find the position(s) where max occurs
            inds_ve = inds_ve[np.random.permutation(len(inds_ve))]  # permute the indicies of those positions uniformly
            motions_ve = possiblemotions[int(inds_ve[0])]  # choose the first one as the updated position

            # provisional update for view enhancing
            u, v = motions_ve[0], motions_ve[1]  # updated velocity in body coordinates
            temp_ve = np.matmul(m, np.array([u, v, 1]).reshape(-1,
                                                               1))  # updated position in global coordinates computed from transformed updated velocity in body coords
            x_ve = np.round(temp_ve[0, 0])  # #### NOTE: ROUND IS DIFFERENT IN MATLAB (half->bigger integer)
            y_ve = np.round(temp_ve[1, 0])

            xy_temp = np.array([x_ve, y_ve])
        else:
            # ############ 6. backtracking (bt) #################
            if behavior[0, ii - 1] != 5:  # if the last beh wasn't bt, go to the previous position
                x_bt = x[0, ii - 1]
                y_bt = y[0, ii - 1]
            elif behavior[0, ii - 1] == 5:  # if the last beh was bt, find the last non-bt (2 steps) and go to that position
                bt_steps = 1
                while behavior[0, ii - bt_steps - 1] == 5:
                    bt_steps = bt_steps + 1
                ind_bt = max(ii - 2 * bt_steps - 1, 1)
                x_bt = x[0, ind_bt]
                y_bt = y[0, ind_bt]
            xy_temp = np.array([x_bt, y_bt])

        # ########## choose update from provisional updates ############
        # smoothing with previous velocity
        vx[0, ii] = xy_temp[0] - x[0, ii]
        vy[0, ii] = xy_temp[1] - y[0, ii]
        xs[0, ii + 1] = round((2 - alpha) * x[0, ii] + (alpha - 1) * x[0, ii - 1] + alpha * vx[0, ii])
        ys[0, ii + 1] = round((2 - alpha) * y[0, ii] + (alpha - 1) * y[0, ii - 1] + alpha * vy[0, ii])

        x[0, ii + 1] = (1 - flag_sp) * xs[0, ii + 1] + flag_sp * xy_temp[0]
        y[0, ii + 1] = (1 - flag_sp) * ys[0, ii + 1] + flag_sp * xy_temp[1]

        #### check if it went off the map, break out for a failed rep if it did
        if x[0, ii + 1] >= LL[0] or x[0, ii + 1] < LL[2] or y[0, ii + 1] >= LL[1] or y[0, ii + 1] < LL[2]:
            flag = 1

        #### check if provisional update is inaccessible, stay put if it is
        if flag == 0 and map_data[0][int(y[0, ii + 1]), int(x[0, ii + 1])] == 1:
            x[0, ii + 1] = x[0, ii]
            y[0, ii + 1] = y[0, ii]

    # x = x';
    # y = y';
    # behavior = behavior';

    # find the end points not equal to nan
    # ind_end = np.argwhere(~np.isnan(x.flatten()))
    # endpts = [x(int(ind_end[-1])), y(int(ind_end[-1]))]

    # find points closest to find points
    # ind_closest = dsearchn([x, y],find_point)
    # closestpts = [x(ind_closest),y(ind_closest)]
    # time_closest = [ind_closest,ind_closest]

    return [x, y, behavior]


# Run the simulation
if __name__ == '__main__':
    # input in parameters for debugging

    nICs = 1                            # initial conditions
    icsFile = "InitialConditions.csv"   # file of ICs to run in lat/lon
    icsMFile = "ConvertedConditions.csv" # ICs converted to meters
    probsFile = 'test6beh.csv'          # file of probabilities to run 'beh_dist_6.csv'
    LLx = 3000                          # extent of map
    LLy = 3000                          # extent of map
    nBehaviors = 6                      # behavior combinations
    reps = 5                           # repetitions
    ts = 850                            # time steps - walking speed (850)
    simT = 100                            # simulation length in hours
    alpha = 0.55                        # smoothing parameter
    save_flag = True                    # save files
    plot_flag = True                   # plot files

    print("# ICs: %i " % nICs)
    print("Input Initial Conditions (lat/lon): %s" % icsFile)
    print("Input Initial Conditions (m): %s" % icsMFile)
    print("Input parameter file: %s" % probsFile)
    print("Map dimensions", LLx, LLy)
    print("# behaviors: %i " % nBehaviors)
    print("# reps: %i " % reps)
    print("# timesteps: %i " % ts)
    print("Simulation length in hrs: %i " % simT)
    print("Smoothing parameter: %f " % alpha)

    # load the initial conditions - in lat/lon and converted to meters
    icsname = np.loadtxt(icsMFile, delimiter=',', dtype='str', usecols=(0,))
    icsmeter = np.loadtxt(icsMFile, delimiter=',', usecols=(1, 2, 3, 4))
    icslatlon = np.loadtxt(icsFile, delimiter=',', usecols=(1, 2))

    # converted initial points and find points
    ics = icsmeter[:, 0:2]
    finds = icsmeter[:, 2:4]

    # limits of map
    ll = 1
    LL = [LLx, LLy, ll]

    # load behavior distribution and parameters
    probs = np.loadtxt(probsFile, delimiter=',')
    T = ts * simT  # length of simulations
    start_time = time.time()

    for iic in range(0, nICs):
        initial_point = ics[iic,]  # initial starting point
        find_point = finds[iic,]
        alldata = [[[], [], []] for _ in range(nBehaviors)]

        # map matrices for Elevation, Inac, LF
        fnInac = "./mapdata/BWInac_" + str(icsname[iic]) + ".csv"
        fnLF = "./mapdata/BWLF_" + str(icsname[iic]) + ".csv"
        fnElev = "./mapdata/Elev_" + str(icsname[iic]) + ".csv"
        BWInac = np.loadtxt(fnInac, delimiter=',')
        BWLF = np.loadtxt(fnLF, delimiter=',')
        Elev = np.loadtxt(fnElev, delimiter=',')
        map_data = (BWInac, BWLF, Elev)
        alltrajectories = {}

        for iprob in range(0, nBehaviors):
            p_behavior = probs[iprob,]
            allX, allY, allbeh = np.empty((1, 0), dtype='int'), np.empty((1, 0), dtype='int'), np.empty((1, 0),dtype='int')
            print("IC %d and prob %d" % (iic, iprob))
            repdict = {}

            for irep in range(0, reps + 1):
            # def parrep(irep):
                print(irep)
                [x, y, behavior] = run_replicate(initial_point, find_point, map_data, T, p_behavior, alpha, LL)
                allX = np.append(allX, x)
                allY = np.append(allY, y)
                allbeh = np.append(allbeh, behavior)
                repdict[("rep"+str(irep))] = {"x": x, "y": y, "behavior": behavior}


            # Parallel(n_jobs=-1, verbose=10)(delayed(parrep)(irep) for irep in range(0, reps + 1))
            alldata[iprob] = [allX, allY, allbeh]
            alltrajectories[("prob" + str(iprob))] = repdict

            # save each probability's trajectory to load into matlab
            if save_flag:
                fnprob = "sims/sim_" + str(icsname[iic]) + "_t" + str(simT) + "_p" + str(iprob) + ".csv"
                np.savetxt(fnprob, alldata[iprob])

        # save alldata for each IC
        if save_flag:
            fnic = "sims/all_" + str(icsname[iic]) + "_t" + str(simT) + ".pkl"
            fnic1 = "sims/alld_" + str(icsname[iic]) + "_t" + str(simT)
            with open(fnic, 'wb') as f:
                pickle.dump(alldata, f)
            with open(fnic,'rb') as f:
                loadalldata = pickle.load(f)
            # check that they're the same
            print(np.array_equal(alldata,loadalldata))
            np.save(fnic1,alltrajectories)

        # visualize the trajectories
        if plot_flag:
            visualization(alldata, reps, T, probs)
            plt.show()
    print("------- total time = {} seconds ------".format(time.time() - start_time))



# ######### TEST VISUALIZATION
# from main_hiker import *
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import random
# from numpy import linalg as LA
# import pickle
#
# fnic = "sims/all_AZ0060_t1.pkl"
# with open(fnic, 'rb') as f:
#     alldata = pickle.load(f)
#
# reps = 10
# T = 850
# probs = np.loadtxt('test6beh.csv', delimiter=',')
# visualization(alldata, reps, T, probs)
# plt.show()