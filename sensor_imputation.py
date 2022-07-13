import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_squared_error, r2_score
from FaultDetection import EnsembleSystem
from apply_fault import sensorfaultmodels


sensor_nodes = ['n54','n105','n114','n163','n188','n229','n288','n296','n332','n342','n410','n415','n429','n458','n469','n495','n506','n516','n519','n549','n613','n636','n644','n679','n722','n726','n740','n752','n769']
flow_nodes = []

failure_profile = {'c1': ['constant', 3], 'c2': ['constant', 2], 'c3': ['constant', 1], 'd1': ['drift', 1], 'd2': ['drift', 0.1], 'd3': ['drift', 0.01], 'n1': ['normal', 2], 'n2': ['normal', 1], 'n3': ['normal', 0.5], 'p1': ['percentage', 0.1], 'p2': ['percentage', 0.05], 'p3': ['percentage', 0.01], 's': ['stuckzero', 0]}

T0 = 8640 #startime of first fault
T1 = 12960 #starttime of second fault
T2 = 16991 #endtime of both faults

basic_imputation_fcts = {'mean':np.mean, 'median':np.median, 'min':np.min, 'max':np.max, 'zero':lambda x:0, 'one':lambda x:1}


def perform_basic_imputation(data, t_start, t_end, imputation_strategy):
    imputation = np.ones(t_end-t_start) * basic_imputation_fcts[imputation_strategy](data[0:t_start])
    sensor_imputed = np.hstack((data[0:t_start], imputation, data[t_end:]))
    return sensor_imputed


def virtual_sensor_imputation(data, t_start, t_end, sensor_id, flow_nodes, pressure_nodes, locally=False, threshold=1.2):
    X_all_train, X_all_test = data[:t_start,:], data[t_start:t_end,:]
    Y_all_train, Y_all_test = data[:t_start], data[t_start:t_end]
    ensemble_system = EnsembleSystem(flow_nodes, pressure_nodes, locally=locally, threshold=threshold)
    ensemble_system.fit(X_all_train, Y_all_train)

    imputation = ensemble_system.predict(X_all_test)[:, sensor_id]
    sensor_imputed = np.hstack((data[0:t_start, sensor_id], imputation, data[t_end:, sensor_id]))
    return sensor_imputed


def perform_neighbor_mean(data, t_start, t_end, sensor_name, with_offset=False):
    adj = np.load('l-town-sensor-graph.npz')['adj']
    res_graph = nx.from_numpy_array(adj, sensor_nodes)
    res_graph = nx.relabel_nodes(res_graph, {i:sensor_nodes[i] for i in range(len(sensor_nodes))})
    neighbors = list(res_graph.neighbors(sensor_name))
    imputation = np.zeros(t_end-t_start)
    sensor_id = sensor_nodes.index(sensor_name)
    for n in neighbors:
        imputation = imputation + data[t_start:t_end,sensor_nodes.index(n)]
    imputation = imputation/len(neighbors)
    if not with_offset:
        sensor_imputed = np.hstack((data[0:t_start, sensor_id], imputation, data[t_end:, sensor_id]))  
    if with_offset:
        historic_imputation = np.zeros(t_start)
        for n in neighbors:
            historic_imputation = historic_imputation + data[:t_start,sensor_nodes.index(n)]
        historic_imputation = historic_imputation/len(neighbors)
        offset = data[:t_start,sensor_id] - historic_imputation
        offset = np.ones(t_end-t_start) * np.mean(offset)
        sensor_imputed = np.hstack((data[0:t_start, sensor_id], imputation+offset, data[t_end:, sensor_id]))

    return sensor_imputed
    

def load_scenario(sensor_id, failure_id, t1=T1, t2=T2, second_sensor_id=None, second_failure_id=None, t0=-1):

    Y_GT = pd.read_csv('Measurements_Pressures.csv')
    times = Y_GT['Timestamp']
    del Y_GT['Timestamp']
    times = times.to_numpy()
    Y_GT = Y_GT.to_numpy()
    Y = np.copy(Y_GT)
    if second_failure_id is None:
        Y[:, sensor_id] = sensorfaultmodels(Y[:, sensor_id],t1,t2,failure_profile[failure_id][0],failure_profile[failure_id][1])
    else:
        Y[:, sensor_id] = sensorfaultmodels(Y[:, sensor_id],t0,t2,failure_profile['c1'][0],failure_profile['c1'][1])
        Y[:, second_sensor_id] = sensorfaultmodels(Y[:, second_sensor_id],t1,t2,failure_profile[second_failure_id][0],failure_profile[second_failure_id][1])

    y_sensor_fault = Y!=Y_GT

    return Y, Y_GT, y_sensor_fault


def evalute_imputations_dev(sensor_id, imputation_strategies, f_out, failure_id='c1', t0=T0, t1=T2, visualize=False, threshold=1.2):
    Y, Y_GT, _ = load_scenario(sensor_id, failure_id, t1=T0)

    for imputation_strategy in imputation_strategies:
        if imputation_strategy == 'baseline':
            Y_imputed = Y[:, sensor_id]
        elif imputation_strategy == 'virtual-sensor':
            Y_imputed = virtual_sensor_imputation(Y, t0, t1, sensor_id, flow_nodes, sensor_nodes, threshold=threshold) 
        elif imputation_strategy == 'virtual-sensor-local':
            Y_imputed = virtual_sensor_imputation(Y, t0, t1, sensor_id, flow_nodes, sensor_nodes, locally=True, threshold=threshold) 
        elif imputation_strategy == 'neighbor-mean':
            Y_imputed = perform_neighbor_mean(Y, t0, t1, sensor_nodes[sensor_id])
        elif imputation_strategy == 'neighbor-mean-offset':
            Y_imputed = perform_neighbor_mean(Y, t0, t1, sensor_nodes[sensor_id], with_offset=True)
        else:
            Y_imputed = perform_basic_imputation(Y[:, sensor_id], t0, t1, imputation_strategy)
        
        if visualize:
            plt.figure()
            plt.plot(Y_imputed, label='imputed')
            plt.plot(Y_GT[:, sensor_id], label='GT')
            plt.legend()
            plt.show()

        mse = mean_squared_error(Y_imputed[t0:t1], Y_GT[t0:t1, sensor_id])
        r2 = r2_score(Y_imputed[t0:t1], Y_GT[t0:t1, sensor_id])

        np.savez('{}-{}.npz'.format(f_out, imputation_strategy), mse=mse, r2=r2)


def evalute_sensor_isolation(sensor_id, failure_id, f_out, visualize=False, threshold=1.2):
    Y, Y_GT, _ = load_scenario(sensor_id, failure_id)

    ensemble_system = EnsembleSystem(flow_nodes, sensor_nodes, locally=True, threshold=threshold, isolation=True)
    t_train_split = 5000 
    ensemble_system.fit(Y_GT[:t_train_split,:], Y_GT[:t_train_split,:])

    _, _, _, isolated_sensors = ensemble_system.apply_detector(Y, Y)
    if len(isolated_sensors) == 1:
        if isolated_sensors[0]!=sensor_id:
            print(isolated_sensors[0])
        return isolated_sensors[0]==sensor_id
    else:
        print('multiple sensors', isolated_sensors)
        return False

def evalute_sensor_isolation_on_sensor(sensor_id, threshold, f_out):
    results = []
    for fault_id in failure_profile.keys():
        results.append(evalute_sensor_isolation(sensor_id, fault_id, 'test', visualize=False, threshold=threshold))
    results = np.array(results)
    np.savez(f_out, detection=results)
    print(sensor_id, np.sum(results)/len(results))


def evaluate_imputations_fault_detection(first_sensor_id, second_sensor_id, failure_id, imputation_strategies, f_out, t0=T0, t1=T1, t2=T2, locally=False, threshold=1.2, verbose=False):
    
    Y, Y_GT, y_sensor_fault = load_scenario(first_sensor_id, 'c1', t1=T1, t2=T2, second_sensor_id=second_sensor_id, second_failure_id=failure_id, t0=T0)
    
    y_sensor_fault_gt = np.zeros(len(y_sensor_fault))
    y_sensor_fault_gt[t1:t2] = np.ones(t2-t1)


    for imputation_strategy in imputation_strategies:
        Y_imputed = np.copy(Y)
        if imputation_strategy == 'virtual-sensor':
            Y_imputed[:, first_sensor_id] = virtual_sensor_imputation(Y, t0, t2, first_sensor_id, flow_nodes, sensor_nodes, threshold=threshold)
        elif imputation_strategy == 'virtual-sensor-local':
            Y_imputed[:, first_sensor_id] = virtual_sensor_imputation(Y, t0, t2, first_sensor_id, flow_nodes, sensor_nodes, locally=True, threshold=threshold)
        elif imputation_strategy == 'neighbor-mean':
            Y_imputed[:, first_sensor_id] = perform_neighbor_mean(Y, t0, t2, sensor_nodes[first_sensor_id])
        elif imputation_strategy == 'neighbor-mean-offset':
            Y_imputed[:, first_sensor_id] = perform_neighbor_mean(Y, t0, t2, sensor_nodes[first_sensor_id], with_offset=True)
        elif imputation_strategy == 'baseline-GT':
            Y_imputed[:, first_sensor_id] = Y_GT[:,first_sensor_id]
        # otherwise: baseline: do nothing

        ensemble_system = EnsembleSystem(flow_nodes, sensor_nodes, locally=locally, threshold=threshold)
        t_train_split = 5000
        ensemble_system.fit(Y_GT[:t_train_split,:], Y_GT[:t_train_split,:])

        suspicious_times, _, pred_faults, _ = ensemble_system.apply_detector(Y_imputed, Y_imputed)
      
        if verbose:
            print(imputation_strategy)
            print('FP',len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==0))[0]))
            print('FN',len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==1))[0]))
            print('TP',len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==1))[0]))
            print('TN',len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==0))[0]))

        np.savez('{}-{}.npz'.format(f_out, imputation_strategy), alarms=suspicious_times, failure_ids=np.array([first_sensor_id, second_sensor_id]), times=np.array([t0, t1]), fp=len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==0))[0]), tp=len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==1))[0]), fn=len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==1))[0]), tn=len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==0))[0]))
        

def evaluate_fault_detection(sensor_id, failure_id, f_out=None, t0=T0, t1=T1, t2=T2, locally=False, threshold=1.2, verbose=False):
    Y, Y_GT, y_sensor_fault = load_scenario(sensor_id, failure_id)
    
    y_sensor_fault_gt = np.zeros(len(y_sensor_fault))
    y_sensor_fault_gt[t1:t2] = np.ones(t2-t1)

    ensemble_system = EnsembleSystem(flow_nodes, sensor_nodes, locally=locally, threshold=threshold)
    t_train_split = 5000 
    ensemble_system.fit(Y_GT[:t_train_split,:], Y_GT[:t_train_split,:])

    suspicious_times, sensor_forecasting_errors, pred_faults, _ = ensemble_system.apply_detector(Y, Y)

    if verbose:
        print('FP',len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==0))[0]))
        print('FN',len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==1))[0]))
        print('TP',len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==1))[0]))
        print('TN',len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==0))[0]))
    if f_out is not None:
        np.savez(f_out, alarms=suspicious_times, failure_ids=np.array([failure_id]), times=np.array([t0, t1]), fp=len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==0))[0]), tp=len(np.where(np.logical_and(pred_faults==1, y_sensor_fault_gt==1))[0]), fn=len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==1))[0]), tn=len(np.where(np.logical_and(pred_faults==0, y_sensor_fault_gt==0))[0]))


def evaluate_fault_detection_on_sensor(sensor_id, locally, threshold):
    locally = 'local' if locally else 'global'
    for fault_id in failure_profile.keys():
        f_out = 'results/fault-detection-sensor-{}-{}-{}-{}.npz'.format(sensor_id, fault_id, locally, threshold)
        evaluate_fault_detection(sensor_id, fault_id, f_out, t0=T0, t1=T1, t2=T2, locally=False, threshold=1.2)

threshold = 1.2
# fault detection
print('experiments: fault detection')
for i in range(len(sensor_nodes)):
    print(i)
    evaluate_fault_detection_on_sensor(i, True, threshold)
    evaluate_fault_detection_on_sensor(i, False, threshold)

# fault isolation
print('fault isolation')
for i in range(len(sensor_nodes)):
    print(i)
    evalute_sensor_isolation_on_sensor(i, threshold, 'results/fault-isolation-sensor-{}-{}.npz'.format(i, threshold))

print('experiments: fault accommodation - deviation')
for i in range(len(sensor_nodes)):
    f_out = 'results/sensor-accommodation-sensor-{}'.format(i)
    evalute_imputations_dev(i, ['baseline', 'virtual-sensor', 'virtual-sensor-local','mean', 'neighbor-mean', 'neighbor-mean-offset', 'median'], f_out, visualize=False)

print('experiments: fault accommodation - detection')
for s1 in range(len(sensor_nodes)):
    for s2 in range(len(sensor_nodes)):
        if s1 != s2:
            print(s1, s2)
            for fault_id in failure_profile.keys():
                f_out = 'results/sensor-accommodation-detection-sensors-{}-{}-{}-global-{}'.format(s1,s2,fault_id,threshold)
                evaluate_imputations_fault_detection(s1, s2, fault_id, ['baseline-GT', 'virtual-sensor', 'virtual-sensor-local', 'mean', 'neighbor-mean', 'neighbor-mean-offset', 'median', 'baseline'], f_out, locally=False, threshold=1.2)
                f_out = 'results/sensor-accommodation-detection-sensors-{}-{}-{}-local-{}'.format(s1,s2,fault_id,threshold)
                evaluate_imputations_fault_detection(s1, s2, fault_id, ['baseline-GT', 'virtual-sensor', 'virtual-sensor-local', 'mean', 'neighbor-mean', 'neighbor-mean-offset', 'median', 'baseline'], f_out, locally=True, threshold=1.2)

