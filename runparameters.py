import sys
from main_hiker import *

# python runparameters.py 1 "InitialConditions.csv" "ConvertedConditions.csv" "beh_dist_6.csv" 3000 3000 462 3 850 10 0.55 True True

nICs = int(sys.argv[1])         # number of initial conditions
icsFile = sys.argv[2]           # file of ICs to run in lat/lon
icsMFile = sys.argv[3]          # ICs converted to meters
probsFile = sys.argv[4]         # file of probabilities to run
LLx = int(sys.argv[5])          # extent of map
LLy = int(sys.argv[6])          # extent of map
nBehaviors = int(sys.argv[7])   # behavior combinations
reps = int(sys.argv[8])         # repetitions
ts = int(sys.argv[9])           # time steps
simT = int(sys.argv[10])        # simulation length (hrs)
alpha = float(sys.argv[11])     # smoothing parameter
save_flag = sys.argv[12]        # save all data
plot_flag = sys.argv[13]        # plot data

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
ics = icsmeter[:,0:2]
finds = icsmeter[:,2:4]

# limits of map
ll = 1
LL = [LLx, LLy, ll]

# load behavior distribution and parameters
probs = np.loadtxt(probsFile, delimiter=',')
T = ts * simT
start_time = time.time()

for iic in range(0, nICs):
    initial_point = ics[iic,]  # initial starting point
    find_point = finds[iic,]
    alldata = [[[],[],[]] for _ in range(nBehaviors)]

    # map matrices for Elevation, Inac, LF
    fnInac = "mapdata/BWInac_" + str(icsname[iic]) + ".csv"
    fnLF = "mapdata/BWLF_" + str(icsname[iic]) + ".csv"
    fnElev = "mapdata/Elev_" + str(icsname[iic]) + ".csv"
    BWInac = np.loadtxt(fnInac, delimiter=',')
    BWLF = np.loadtxt(fnLF, delimiter=',')
    Elev = np.loadtxt(fnElev, delimiter=',')
    map_data = (BWInac, BWLF, Elev)

    for iprob in range(0, nBehaviors):
        p_behavior = probs[iprob,]
        allX, allY, allbeh = np.empty((1,0), dtype='int'), np.empty((1,0), dtype='int'), np.empty((1,0), dtype='int')
        print("IC %d and prob %d" % (iic, iprob))
        for irep in range(0, reps + 1):
            print(irep)
            [x, y, behavior] = run_replicate(initial_point, find_point, map_data, T, p_behavior, alpha, LL)
            allX = np.append(allX, x)
            allY = np.append(allY, y)
            allbeh = np.append(allbeh, behavior)
        alldata[iprob] = [allX, allY, allbeh]
        # save each probability's trajectory to load into matlab
        if save_flag:
            fnprob = "sims/sim_" + str(icsname[iic]) + "_t" + str(T) + "_p" + str(iprob) + ".csv"
            np.savetxt(fnprob, alldata[iprob])

    # save alldata for each IC
    if save_flag:
        fnic = "sims/all_" + str(icsname[iic]) + "_t" + str(T) + ".pkl"
        with open(fnic, 'wb') as f:
            pickle.dump(alldata, f)
        with open(fnic,'rb') as f:
            loadalldata = pickle.load(f)
        # check that they're the same
        print(np.array_equal(alldata,loadalldata))

    # visualize the trajectories
    if plot_flag:
        visualization(alldata, reps, T, probs)
        # plt.show()

print("------- total time = {} seconds ------".format(time.time() - start_time))
print("done")
