from copy import deepcopy
import numpy as np
from sklearn.linear_model import Ridge
import networkx as nx
from functools import reduce

sensor_nodes = ['n54','n105','n114','n163','n188','n229','n288','n296','n332','n342','n410','n415','n429','n458','n469','n495','n506','n516','n519','n549','n613','n636','n644','n679','n722','n726','n740','n752','n769']

def get_sensor_id(name):
    return sensor_nodes.index(name)

class MyModel():
    def __init__(self):
        self.model = Ridge()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_stat_abs_deviation(self, X, y):
        y_pred = self.predict(X)
        difference_abs = np.abs(y_pred - y)
        
        return np.max(difference_abs), np.min(difference_abs), np.mean(difference_abs)


class FaultDetection():
    def __init__(self, threshold, model=MyModel):
        self.model = model
        self.threshold = threshold
    
    def apply_to_labeled_stream(self, X, y):
        y_pred = self.model.predict(X)
        
        return list(np.where(np.abs(y_pred - y) > self.threshold)[0]), np.abs(y_pred - y) > self.threshold


def compute_indicator_function(time_points, y_leaks):
    return np.array([1. if t in time_points else 0. for t in range(len(y_leaks))])


class EnsembleSystem():
    def __init__(self, flow_nodes, pressure_nodes, model_class=MyModel, fault_detecotor_class=FaultDetection, locally=False, isolation=False, threshold=1.2):
        self.model_class = model_class
        self.fault_detecotor_class = fault_detecotor_class
        self.flow_nodes = flow_nodes
        self.pressure_nodes = pressure_nodes
        self.models = []
        self.locally = locally
        self.isolation = isolation
        self.threshold = threshold
        
        if self.locally:
            adj = np.load('l-town-sensor-graph.npz')['adj']
            graph = nx.from_numpy_array(adj, sensor_nodes)
            self.graph = nx.relabel_nodes(graph, {i:sensor_nodes[i] for i in range(len(sensor_nodes))})
            self.neighbors = {node : self.graph.neighbors(node) for node in pressure_nodes}


    def fit(self, X_train, Y_train):
        self.models = []
        # submodels only leaving out one feature
        if self.isolation:
            self.submodels = {i:dict() for i in range(len(self.pressure_nodes))}
        
        for i in range(len(self.pressure_nodes)):
            # Select inputs and output
            if self.locally:
                inputs_idx = self.neighbors[sensor_nodes[i]]
                inputs_idx = [get_sensor_id(neighbor) for neighbor in inputs_idx]
                if self.isolation:
                    if len(inputs_idx)>1:
                        for j in inputs_idx:
                            inputs_idx_ = deepcopy(inputs_idx)
                            inputs_idx_.remove(j)
                            x_train, y_train = X_train[:,inputs_idx_], Y_train[:,i]
                
                            # Fit regression
                            model = self.model_class()
                            model.fit(x_train, y_train)
                            
                            # Build fault detector
                            max_abs_error = self.threshold * model.get_stat_abs_deviation(x_train, y_train)[0] 
                            fault_detector = self.fault_detecotor_class(max_abs_error, model)
                            
                            # Store model
                            self.submodels[i][j] = {"model": model, "fault_detector": fault_detector, "input_idx": inputs_idx_, "target_idx": i}
                    else:
                        self.submodels[i] = None


            else:
                inputs_idx = list(range(X_train.shape[1]));inputs_idx.remove(i)
            x_train, y_train = X_train[:,inputs_idx], Y_train[:,i]
        
            # Fit regression
            model = self.model_class()
            model.fit(x_train, y_train)
            
            # Build fault detector
            max_abs_error = self.threshold * model.get_stat_abs_deviation(x_train, y_train)[0]
            fault_detector = self.fault_detecotor_class(max_abs_error, model)
            
            # Store model
            self.models.append({"model": model, "fault_detector": fault_detector, "input_idx": inputs_idx, "target_idx": i})
        

    def predict(self, X):
        Y = []
        
        for m in self.models:
            Y.append(m["model"].predict(X[:,m["input_idx"]]))
            
        Y = np.array(Y).T
        return Y
    
    def score(self, X, Y):
        scores = []
        
        for m in self.models:
            scores.append(m["model"].score(X[:,m["input_idx"]], Y[:,m["target_idx"]]))
        
        return scores
    
    def apply_detector(self, X, Y):
        isolated_sensors = []
        suspicious_time_points = []
        possible_faults = np.zeros(len(Y))
        sensor_forecasting_errors = []
        suspicious_sensors = []
        for m in self.models:
            x_in = X[:,m["input_idx"]]
            y_truth = Y[:,m["target_idx"]]
            
            sensor_forecasting_errors.append(np.square(m["model"].predict(x_in) - y_truth))# Squared error
            suspicious = m["fault_detector"].apply_to_labeled_stream(x_in, y_truth)
            suspicious_time_points += suspicious[0]
            possible_faults = possible_faults + suspicious[1]
            
            if self.isolation:
                if np.sum(suspicious[1]) > 10:
                    candidates = m['input_idx']
                    candidates.append(m['target_idx'])
                    suspicious_sensors.append(candidates)
        if self.isolation:
            suspicious_sensors = reduce(np.intersect1d, suspicious_sensors)
            isolated_sensors = []
            candidates_if_no_other = []
            for s in suspicious_sensors:
                alarm_count = 0
                for t in suspicious_sensors:
                    if s != t:
                        if self.submodels[s] is not None:
                            if t in self.submodels[s].keys():
                                m = self.submodels[s][t]
                                x_in = X[:,m["input_idx"]]
                                y_truth = Y[:,m["target_idx"]]
                                suspicious = m["fault_detector"].apply_to_labeled_stream(x_in, y_truth)
                                if len(suspicious[0]) > 10:
                                    alarm_count += 1
                        else:
                            candidates_if_no_other.append(s)
                if alarm_count == len(suspicious_sensors)-1:
                    isolated_sensors.append(s)
            if len(isolated_sensors) == 0 and len(candidates_if_no_other)==1:
                isolated_sensors.append(candidates_if_no_other[0])


        sensor_forecasting_errors = np.vstack(sensor_forecasting_errors).T
        suspicious_time_points = list(set(suspicious_time_points));suspicious_time_points.sort()
        possible_faults = np.clip(possible_faults, 0, 1)
        
        return suspicious_time_points, sensor_forecasting_errors, possible_faults, isolated_sensors
