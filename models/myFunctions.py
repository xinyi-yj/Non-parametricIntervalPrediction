import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import gc

# Scikit-learn imports
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Signal Processing imports
from PyEMD import EMD, EEMD, CEEMDAN
from vmdpy import VMD

import copy

class SymmetricUncertainty:
    """
    Contains functions for calculating Symmetric Uncertainty (SU)
    and a greedy algorithm for feature selection based on SU.
    """

    def calc_symmetric_uncertainty(self, x, y):
        """
        Calculates the Symmetric Uncertainty SU(X, Y) using normalized mutual information.
        
        Args:
            x: Feature vector X.
            y: Feature vector Y.
            
        Returns:
            float: SU(X, Y) score.
        """
        return normalized_mutual_info_score(x, y)

    def cal_symmetric_uncertainty_discrete(self, x, y, bins=8):
        """
        Calculates Symmetric Uncertainty SU(X, Y) after discretizing continuous variables.
        
        Args:
            x: Feature vector X.
            y: Feature vector Y.
            bins (int): Number of bins for discretization.
            
        Returns:
            float: SU(X, Y) score.
        """
        # Discretize data into bins
        x_discrete = pd.cut(x, bins=bins, labels=False)
        y_discrete = pd.cut(y, bins=bins, labels=False)
        
        return self.calc_symmetric_uncertainty(x_discrete, y_discrete)
    
    def __calc_SU_c(self, data, S, c):
        """
        Calculates the average SU between each selected feature in S and the target class c.
        Essentially: Mean(SU(s, c)) for s in S.
        """
        return np.mean([self.cal_symmetric_uncertainty_discrete(data[col], data[c]) for col in S])

    def __calc_SU_s(self, data, S):
        """
        Calculates the average SU between features within the selected set S (Redundancy).
        Essentially: Mean(SU(si, sj)) for si, sj in S.
        """
        m_s = len(S)
        su = np.zeros((m_s, m_s))
        for i in range(m_s):
            for j in range(m_s):
                su[i,j] = self.cal_symmetric_uncertainty_discrete(data[S[i]], data[S[j]])
        
        return np.sum(su)/(m_s**2)

    def __J(self, data, S, c, beta=1):
        """
        Calculates the objective function J(S, c).
        J = Relevance(S, c) - beta * Redundancy(S, S).
        """
        return self.__calc_SU_c(data, S, c) - beta * self.__calc_SU_s(data, S)

    def select_features(self, data, target_col):
        """
        Performs a greedy forward feature selection process based on the J value.
        
        Args:
            data (pd.DataFrame): The dataset containing features and target.
            target_col (str): The name of the target column.
            
        Returns:
            list: A list of selected feature names.
        """
        Q0 = -np.inf
        S = []
        unselected_features = data.columns[data.columns != target_col]
        c = target_col
        
        while True:
            Q1_max = -np.inf
            f_new = None
            
            # Iterate through unselected features to find the one that maximizes J
            for feature in unselected_features:
                S_new = S.copy()
                S_new.append(feature)
                Q1 = self.__J(data, S_new, c)
                if Q1 > Q1_max:
                    Q1_max = Q1
                    f_new = feature
            
            # If the new feature improves the objective function, add it to S
            if Q1_max > Q0:
                Q0 = Q1_max
                S.append(f_new)
                unselected_features = unselected_features[unselected_features != f_new]
            else:
                break
        
        return S


class ClusterAnalysis:
    """
    Cluster analysis using K-Means, utilizing Silhouette Score to find the optimal k.
    """
    
    def get_k_cluster(self, X, k_max=10, title=True):
        """
        Determines the optimal number of clusters (k) for K-Means using Silhouette Score.
        
        Args:
            X: Input data for clustering.
            k_max (int): Maximum number of clusters to test.
            title (bool): Whether to display the title on the plot.
            
        Returns:
            int: The optimal number of clusters (best_k).
        """
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        silhouette_scores = [] 
        K_range = range(2, k_max + 1)

        for k in K_range:
            # random_state is set for reproducibility
            kmeans = KMeans(n_clusters=k, random_state=10, n_init=10) 
            kmeans.fit(X)
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            # print(f"For n_clusters = {k}, the average silhouette_score is : {silhouette_avg}")

        # Plot Silhouette Score vs. k
        plt.figure(figsize=(12, 8), dpi=400)
        plt.plot(K_range, silhouette_scores, 'bo-')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel("Number of clusters, k", fontsize=16)
        plt.ylabel("Silhouette score", fontsize=16)
        if title:
            plt.title("Silhouette score vs number of clusters")
        plt.show()

        # Identify k with the highest score
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"The best number of clusters is: {best_k}")

        return best_k
    

class RegressionAnalysis:
    """
    Encapsulates various regression models and evaluation metrics.
    """

    def __mape(self, y_pred, y_true):
        """
        Calculates Mean Absolute Percentage Error (MAPE).
        Excludes data points where true value is close to 0 to avoid errors.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    
        # Filter out small values (close to 0) to prevent division by zero or extreme outliers
        mask = y_true > 10
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        return np.mean(np.abs((y_pred - y_true) / y_true))
    
    def __smape(self, y_pred, y_true):
        """
        Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
        Excludes small values similar to MAPE.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true > 10
        y_true, y_pred = y_true[mask], y_pred[mask]

        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    
    def LR_DT_SVR_BT_RF_finally_and_no_picture(self, X, y, shuffle=True):
        """
        Performs regression analysis using LR, DT, SVR, Bagging, and Random Forest.
        Prints evaluation metrics but does not generate plots.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            shuffle (bool): Whether to shuffle data during train/test split.
        """
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=shuffle, random_state=10)

        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=10),
            'Support Vector Regression': SVR(),
            'Bagging Tree': BaggingRegressor(estimator=DecisionTreeRegressor(), 
                                             n_estimators=10, random_state=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=10)
        }

        # Store results
        results = {}
        y_preds = {}

        # Train and evaluate models
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Post-processing correction: clip predictions to valid range [0, 1500]
            y_pred = np.clip(y_pred, a_min=0, a_max=1500)

            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            mape = self.__mape(y_pred=y_pred, y_true=y_test)
            smape = self.__smape(y_pred=y_pred, y_true=y_test)
            
            results[name] = {'R-squared':r2, 'RMSE':rmse, 'MAE':mae, 'MAPE':mape, 'SMAPE':smape}
            y_preds[name] = y_pred

        # Print metrics
        for name, metrics in results.items():
            print(f"{name}: R-squared = {metrics['R-squared']:.4f}, RMSE = {metrics['RMSE']:.4f}," 
                  f"MAE = {metrics['MAE']:.4f}, MAPE = {metrics['MAPE']:.2%}, SMAPE = {metrics['SMAPE']:.2%}")


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_weights(self, model):
        model.load_state_dict(self.best_model_state)

class LSTMModel(nn.Module):
    def __init__(self, input_size, pre_len, hidden_size1=64, hidden_size2=16, dropout=0.05):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, pre_len)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm1(x)          # (batch, seq_len, hidden_size1)
        out = self.dropout(out)
        out, _ = self.lstm2(out)        # (batch, seq_len, hidden_size2)
        out = out[:, -1, :]             
        out = self.fc(out)              # (batch, pre_len)
        return out
    
class LSTMAnalysis:
    """
    Performs LSTM-based time series forecasting, with support for EMD and VMD signal decomposition.
    """

    def __mape(self, y_pred, y_true):
        """
        Calculates MAPE, filtering out values close to 0.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    
        # Filter small values
        mask = y_true > 10
        y_true, y_pred = y_true[mask], y_pred[mask]
        
        return np.average(np.abs((y_true - y_pred) / y_true), axis=0)

    def __createXY(self, datasetX, datasetY, seq_len, pre_len, time_reflect=0):
        """
        Constructs the feature (X) and target (Y) sets for time series forecasting using a sliding window.

        Args:
            datasetX: The feature dataframe.
            datasetY: The target series.
            seq_len: The length of the input sequence (lookback window).
            pre_len: The prediction horizon (how many steps ahead to predict).
            time_reflect: Offset for data alignment (e.g., *96 for daily intervals).

        Returns:
            tuple: (length, features_array, targets_array)
        """
        X_arr = datasetX.values if isinstance(datasetX, pd.DataFrame) else datasetX
        Y_arr = datasetY.values if isinstance(datasetY, (pd.Series, pd.DataFrame)) else datasetY
        features = []
        targets = []

        length = len(X_arr) - seq_len + 1 - pre_len - time_reflect

        for i in range(length):
            features.append(X_arr[i : i + seq_len])
            targets.append(Y_arr[i + seq_len + time_reflect : i + seq_len + pre_len + time_reflect])
        
        return length, np.array(features), np.array(targets)

    def __emd_decompose(self, data, name):
        """Performs Empirical Mode Decomposition (EMD) on the specified column."""
        emd = EMD()
        imfs = emd(data[name].values)  # The last component is usually the Residual
        return imfs

    def __vmd_decompose(self, data, name):
        """Performs Variational Mode Decomposition (VMD) on the specified column."""
        # VMD Parameters
        alpha = 2000       # Bandwidth constraint
        tau = 0            # Noise-tolerance (0 = strict fidelity to signal)
        K = 5              # Number of modes
        DC = 0             # DC component
        init = 1           # Initialization parameter
        tol = 1e-7         # Convergence tolerance

        # Execute VMD
        u, _, _ = VMD(data[name], alpha, tau, K, DC, init, tol)
        return u
    
    def __eemd_decompose(self, data, name):
        """Performs Ensemble Empirical Mode Decomposition (EEMD) on the specified column."""
        eemd = EEMD()
        imfs = eemd(data[name].values)
        return imfs
    
    def __ceemdan_decompose(self, data, name):
        """Performs Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) on the specified column."""
        ceemdan = CEEMDAN()
        imfs = ceemdan(data[name].values)
        return imfs

    def run_time(self, data_with_time, time_name, name, seq_len, categorical_columns=[], epochs=50, batch_size=32, 
                    pre_len=1, emd=False, vmd=False, ceemdan=False, eemd=False, loss_method='mean_squared_error',
                    test_column='', test_range=[], time_shift=0,
                    time_reflect=0):
        """
        Main method to run the LSTM analysis pipeline.

        Args:
            data_with_time (pd.DataFrame): Dataset including the time column.
            time_name (str): Name of the time column.
            name (str): Name of the target column (e.g., price or load).
            seq_len (int): Length of input sequence.
            categorical_columns (list): List of categorical column names (excluded from scaling).
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            pre_len (int): Prediction length.
            emd (bool): Whether to apply EMD decomposition.
            vmd (bool): Whether to apply VMD decomposition.
            ceemdan (bool): Whether to apply CEEMDAN decomposition.
            eemd (bool): Whether to apply EEMD decomposition.
            loss_method (str): Loss function name.
            test_column (str): Column name used for splitting train/test sets based on time.
            test_range (list): [start_time, end_time] for the test set.
            time_shift (int): Days to shift the time for visualization purposes.
            time_reflect (int): Unit in days. If provided, assumes data is not shifted and handles offset logic (multiplied by 96).
        
        Returns:
            str: JSON string containing actual vs predicted values with timestamps.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        data = data_with_time.drop(time_name, axis=1)
        data_with_time[time_name] = pd.to_datetime(data_with_time[time_name], format='ISO8601')

        # --- Normalization ---
        scalers = {}
        for col in data.columns:
            if col not in categorical_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                scalers[col] = scaler

        # Construct initial X and Y (primarily to get length/structure)
        length, data_X, data_Y = self.__createXY(datasetX=data, datasetY=data[name], 
                                                 seq_len=seq_len, pre_len=pre_len, 
                                                 time_reflect=96*time_reflect)

        # --- Dataset Splitting ---
        if test_column == '':
            # Random split if no specific time column is provided
            train_data, test_data, train_label, test_label_all = train_test_split(data_X, data_Y, test_size=0.2, shuffle=False)
        else:
            # Time-based split
            test_mask = (data_with_time[test_column] >= (pd.to_datetime(test_range[0]) - pd.Timedelta(days=time_shift))) & \
                        (data_with_time[test_column] <= (pd.to_datetime(test_range[1]) - pd.Timedelta(days=time_shift)))
            
            # Select test set
            # Ensure indices align with seq_len offset
            test_data = data_X[test_mask[seq_len : length + seq_len]] 
            test_label_all = data_Y[test_mask[seq_len : length + seq_len]]
            # Select training set
            train_data, train_label = data_X[~test_mask[seq_len : length + seq_len]], data_Y[~test_mask[seq_len : length + seq_len]]

        # --- define loss function ---
        if loss_method == 'mean_squared_error':
            criterion = nn.MSELoss()
        elif loss_method == 'mean_absolute_error':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss method: {loss_method}")

        # define training function to avoid redundancy
        def train_and_predict(X_train, Y_train, X_test, Y_test, input_size):
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
            
            # shuffle=True for training set to improve convergence
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = LSTMModel(input_size=input_size, pre_len=pre_len).to(device)
            optimizer = optim.Adam(model.parameters())
            early_stopping = EarlyStopping(patience=10)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)
                train_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_loader.dataset)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            early_stopping.load_best_weights(model)
            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                preds = model(test_tensor).cpu().numpy()
            
            return preds

        # --- Logic for Decomposition (EMD, EEMD, CEEMDAN, or VMD) ---
        if emd or vmd or eemd or ceemdan:
            if emd:
                IMFs = self.__emd_decompose(data=data, name=name)
                method_name = "EMD"
            elif eemd:
                IMFs = self.__eemd_decompose(data=data, name=name)
                method_name = "EEMD"
            elif ceemdan:
                IMFs = self.__ceemdan_decompose(data=data, name=name)
                method_name = "CEEMDAN"
            else: # vmd
                IMFs = self.__vmd_decompose(data=data, name=name)
                method_name = "VMD"
            print(f'data shape: {data.shape}')
            print(f'{method_name} IMFs shape: {IMFs.shape}')

            predictions_sum = None  # Accumulator for predictions

            for i, imf in enumerate(IMFs):
                print(f'Processing IMF {i+1}/{len(IMFs)}')

                data[name] = imf  # Replace target column with specific IMF for training

                # Re-construct X/Y for the specific IMF
                _, data_X, data_Y = self.__createXY(datasetX=data, datasetY=data[name], 
                                                    seq_len=seq_len, pre_len=pre_len, 
                                                    time_reflect=96*time_reflect)
                
                # Split again for this specific IMF
                if test_column == '':
                    train_data_imf, test_data_imf, train_label_imf, test_label_imf = train_test_split(data_X, data_Y, test_size=0.2, shuffle=False)
                else:
                    test_data_imf, test_label_imf = data_X[test_mask[seq_len:length+seq_len]], data_Y[test_mask[seq_len:length+seq_len]]
                    train_data_imf, train_label_imf = data_X[~test_mask[seq_len:length+seq_len]], data_Y[~test_mask[seq_len:length+seq_len]]

                # Train and predict for this IMF
                predictions = train_and_predict(train_data_imf, train_label_imf, test_data_imf, test_label_imf, input_size=data_X.shape[2])

                # Visualize component prediction
                print(f'Prediction effect for IMF {i+1}/{len(IMFs)}:')
                plt.figure(figsize=(14, 7))
                plt.plot(test_label_imf, label='True Value')
                plt.plot(predictions, label='Predicted Value')
                plt.title(f'Processing IMF {i+1}/{len(IMFs)}')
                plt.legend()
                plt.show()

                # Accumulate results
                if predictions_sum is None:
                    predictions_sum = predictions
                else:
                    predictions_sum += predictions

        # --- Standard Logic (No Decomposition) ---
        else:
            predictions_sum = train_and_predict(train_data, train_label, test_data, test_label_all, input_size=data_X.shape[2])

        print(f"Prediction shape: {predictions_sum.shape}")

        # --- Evaluation & Inversion ---
        # Inverse transform to get original scale
        y_test_inv = scalers[name].inverse_transform(test_label_all.reshape(-1, 1)).flatten()
        predictions_inv = scalers[name].inverse_transform(predictions_sum.reshape(-1, 1)). flatten()

        # Calculate metrics
        r2 = r2_score(y_test_inv, predictions_inv)
        mae = mean_absolute_error(y_test_inv, predictions_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
        mape = self.__mape(y_pred=predictions_inv, y_true=y_test_inv)
            
        print(f'R-squared = {r2:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2%}, RMSE = {rmse:.4f}')

        # --- Format Output as JSON ---
        result = []
        # Get timestamps corresponding to the test set
        data_time = np.array(data_with_time[time_name][test_mask])[:len(predictions_inv)]

        for i in range(len(predictions_inv)):
            result.append({
                'actual_time': str(data_time[i]),                                   # Actual timestamp
                'virtual_time': str(data_time[i] + pd.Timedelta(days=time_shift)),  # Mapped timestamp (shifted)
                'true': float(y_test_inv[i]),                                       # True value
                'prediction': float(predictions_inv[i])                             # Predicted value
            })

        result_json = json.dumps(result, ensure_ascii=False, indent=4)

        # --- Final Visualization ---
        plt.figure(figsize=(14, 7))
        # Plot with shifted time for visual requirement
        plt.plot(data_time + pd.Timedelta(days=time_shift), y_test_inv, label='True Value')
        plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv, label='Predicted Value')

        title_str = f'LSTM + {method_name} Prediction for {name}' if emd or vmd or eemd or ceemdan else f'LSTM Prediction for {name}'
        
        plt.title(title_str)
        plt.xlabel('Time')
        plt.ylabel(name)
        plt.legend()
        plt.show()

        return result_json
    

class QuantileLoss(nn.Module):
    """Pinball Loss for Quantile Regression"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds shape: (batch_size, pre_len, num_quantiles)
        # target shape: (batch_size, pre_len)
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred_q = preds[:, :, i] 
            err = target - pred_q
            # Pinball loss: max(q * err, (q - 1) * err)
            loss += torch.max(q * err, (q - 1) * err).mean()
        return loss
    
class QuantileLSTMModel(nn.Module):
    """LSTM-QR model"""
    def __init__(self, input_size, pre_len, quantiles=[0.1, 0.5, 0.9], hidden_size1=64, hidden_size2=16, dropout=0.05):
        super(QuantileLSTMModel, self).__init__()
        self.pre_len = pre_len
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, pre_len * self.num_quantiles)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] 
        out = self.fc(out)
        out = out.view(-1, self.pre_len, self.num_quantiles)
        return out
    
class QuantileLSTMAnalysis:

    def __createXY(self, datasetX, datasetY, seq_len, pre_len, time_reflect=0):
        """
        Constructs the feature (X) and target (Y) sets for time series forecasting using a sliding window.

        Args:
            datasetX: The feature dataframe.
            datasetY: The target series.
            seq_len: The length of the input sequence (lookback window).
            pre_len: The prediction horizon (how many steps ahead to predict).
            time_reflect: Offset for data alignment (e.g., *96 for daily intervals).

        Returns:
            tuple: (length, features_array, targets_array)
        """
        X_arr = datasetX.values if isinstance(datasetX, pd.DataFrame) else datasetX
        Y_arr = datasetY.values if isinstance(datasetY, (pd.Series, pd.DataFrame)) else datasetY
        features = []
        targets = []

        length = len(X_arr) - seq_len + 1 - pre_len - time_reflect

        for i in range(length):
            features.append(X_arr[i : i + seq_len])
            targets.append(Y_arr[i + seq_len + time_reflect : i + seq_len + pre_len + time_reflect])
        
        return length, np.array(features), np.array(targets)

    def run_time(self, data_with_time, time_name, name, seq_len, categorical_columns=[], epochs=50, batch_size=32, 
                    pre_len=1, 
                    test_column='', test_range=[], time_shift=0,
                    time_reflect=0,
                    quantiles=[0.1, 0.5, 0.9]):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        data = data_with_time.drop(time_name, axis=1)
        data_with_time[time_name] = pd.to_datetime(data_with_time[time_name], format='ISO8601')

        # --- Normalization ---
        scalers = {}
        for col in data.columns:
            if col not in categorical_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                scalers[col] = scaler

        # Construct initial X and Y (primarily to get length/structure)
        length, data_X, data_Y = self.__createXY(datasetX=data, datasetY=data[name], 
                                                 seq_len=seq_len, pre_len=pre_len, 
                                                 time_reflect=96*time_reflect)

        # --- Dataset Splitting ---
        if test_column == '':
            # Random split if no specific time column is provided
            train_data, test_data, train_label, test_label_all = train_test_split(data_X, data_Y, test_size=0.2, shuffle=False)
        else:
            # Time-based split
            test_mask = (data_with_time[test_column] >= (pd.to_datetime(test_range[0]) - pd.Timedelta(days=time_shift))) & \
                        (data_with_time[test_column] <= (pd.to_datetime(test_range[1]) - pd.Timedelta(days=time_shift)))
            
            # Select test set
            # Ensure indices align with seq_len offset
            test_data = data_X[test_mask[seq_len : length + seq_len]] 
            test_label_all = data_Y[test_mask[seq_len : length + seq_len]]
            # Select training set
            train_data, train_label = data_X[~test_mask[seq_len : length + seq_len]], data_Y[~test_mask[seq_len : length + seq_len]]

        criterion = QuantileLoss(quantiles=quantiles)

        # define training function to avoid redundancy
        def train_and_predict(X_train, Y_train, X_test, Y_test, input_size, quantiles):
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
            
            # shuffle=True for training set to improve convergence
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = QuantileLSTMModel(input_size=input_size, pre_len=pre_len, quantiles=quantiles).to(device)
            optimizer = optim.Adam(model.parameters())
            early_stopping = EarlyStopping(patience=10)

            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)
                train_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_loader.dataset)
                
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            early_stopping.load_best_weights(model)
            model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                preds = model(test_tensor).cpu().numpy()
            
            return preds

        
        predictions_sum = train_and_predict(train_data, train_label, test_data, test_label_all, input_size=data_X.shape[2], quantiles=quantiles)

        print(f"Prediction shape: {predictions_sum.shape}")

        # --- Evaluation & Inversion ---
        # Inverse transform to get original scale
        y_test_inv = scalers[name].inverse_transform(test_label_all.reshape(-1, 1)).flatten()
        predictions_inv_low = scalers[name].inverse_transform(predictions_sum[:, 0, 0].reshape(-1, 1)).flatten()  
        # predictions_inv_mid = scalers[name].inverse_transform(predictions_sum[:, 0, 1].reshape(-1, 1)).flatten()  
        # predictions_inv_high = scalers[name].inverse_transform(predictions_sum[:, 0, 2].reshape(-1, 1)).flatten()
        predictions_inv_high = scalers[name].inverse_transform(predictions_sum[:, 0, -1].reshape(-1, 1)).flatten()

        # --- Format Output as JSON ---
        result = []
        # Get timestamps corresponding to the test set
        data_time = np.array(data_with_time[time_name][test_mask])[:len(predictions_inv_low)]

        for i in range(len(predictions_inv_low)):
            result.append({
                'actual_time': str(data_time[i]),                                   # Actual timestamp
                'virtual_time': str(data_time[i] + pd.Timedelta(days=time_shift)),  # Mapped timestamp (shifted)
                'true': float(y_test_inv[i]),                                       # True value
                'prediction_low': float(predictions_inv_low[i]),                     # Predicted value (10% quantile)
                # 'prediction_mid': float(predictions_inv_mid[i]),                     # Predicted value (50% quantile)
                'prediction_high': float(predictions_inv_high[i]),                   # Predicted value (90% quantile)
            })

        result_json = json.dumps(result, ensure_ascii=False, indent=4)

        # --- Final Visualization ---
        plt.figure(figsize=(14, 7))
        # Plot with shifted time for visual requirement
        plt.plot(data_time + pd.Timedelta(days=time_shift), y_test_inv, label='True Value')
        plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_low, label=f'Predicted Value ({100*quantiles[0]}%)')
        # plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_mid, label=f'Predicted Value ({100*quantiles[1]}%)')
        # plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_high, label=f'Predicted Value ({100*quantiles[2]}%)')
        plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_high, label=f'Predicted Value ({100*quantiles[-1]}%)')

        title_str = f'LSTM Prediction for {name}'
        
        plt.title(title_str)
        plt.xlabel('Time')
        plt.ylabel(name)
        plt.legend()
        plt.show()

        return result_json

class Chomp1d(nn.Module):
    """ Causal Convolution)"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """TCN Residual Block"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.05):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNQuantileRegressor(nn.Module):
    """TCN-QR model for quantile regression"""
    def __init__(self, num_inputs, num_channels, num_quantiles, kernel_size=3, dropout=0.05):
        super(TCNQuantileRegressor, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_quantiles)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.network(x)
        out = self.linear(y1[:, :, -1])
        return out

class QuantileTCNAnalysis:
    def __createXY(self, datasetX, datasetY, seq_len, pre_len, time_reflect=0):
        X_arr = datasetX.values if isinstance(datasetX, pd.DataFrame) else datasetX
        Y_arr = datasetY.values if isinstance(datasetY, (pd.Series, pd.DataFrame)) else datasetY
        features = []
        targets = []
        length = len(X_arr) - seq_len + 1 - pre_len - time_reflect
        for i in range(length):
            features.append(X_arr[i : i + seq_len])
            targets.append(Y_arr[i + seq_len + time_reflect : i + seq_len + pre_len + time_reflect])
        return length, np.array(features), np.array(targets)

    def __pinball_loss(self, y_pred, y_true, quantiles, device):
        """Pinball Loss"""
        loss = 0.0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i].unsqueeze(1)
            loss += torch.max(q * error, (q - 1) * error)
        return loss.mean()

    def run_time(self, data_with_time, time_name, name, seq_len, categorical_columns=[], epochs=50, batch_size=32, 
                 pre_len=1, quantiles=[0.1, 0.9], tcn_channels=[32, 64, 128], kernel_size=3,
                 test_column='', test_range=[], time_shift=0, time_reflect=0):
        """
        Args:
            quantiles (list): List of quantiles to predict, e.g., [0.1, 0.5, 0.9]
            tcn_channels (list): List of channel counts for TCN hidden layers, determining the network's depth and width
            kernel_size (int): Kernel size for convolutions, typically 2 or 3
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for TCN-QR: {device}")
        
        data = data_with_time.drop(time_name, axis=1)
        data_with_time[time_name] = pd.to_datetime(data_with_time[time_name], format='ISO8601')

        # --- normalization ---
        scalers = {}
        for col in data.columns:
            if col not in categorical_columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                scalers[col] = scaler

        length, data_X, data_Y = self.__createXY(datasetX=data, datasetY=data[name], 
                                                 seq_len=seq_len, pre_len=pre_len, 
                                                 time_reflect=96*time_reflect)

        # --- split train and test sets ---
        if test_column == '':
            train_data, test_data, train_label, test_label_all = train_test_split(data_X, data_Y, test_size=0.2, shuffle=False)
        else:
            test_mask = (data_with_time[test_column] >= (pd.to_datetime(test_range[0]) - pd.Timedelta(days=time_shift))) & \
                        (data_with_time[test_column] <= (pd.to_datetime(test_range[1]) - pd.Timedelta(days=time_shift)))
            test_data = data_X[test_mask[seq_len : length + seq_len]] 
            test_label_all = data_Y[test_mask[seq_len : length + seq_len]]
            train_data, train_label = data_X[~test_mask[seq_len : length + seq_len]], data_Y[~test_mask[seq_len : length + seq_len]]

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_label_all, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # --- TCN-QR model ---
        model = TCNQuantileRegressor(num_inputs=data_X.shape[2], 
                                     num_channels=tcn_channels, 
                                     num_quantiles=len(quantiles), 
                                     kernel_size=kernel_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=10)

        # --- training loop ---
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = self.__pinball_loss(outputs, batch_y, quantiles, device)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = self.__pinball_loss(outputs, batch_y, quantiles, device)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Pinball Loss: {train_loss:.6f}, Val Pinball Loss: {val_loss:.6f}')
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        early_stopping.load_best_weights(model)
        model.eval()

        # --- test set inference ---
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(test_tensor).cpu().numpy()  # shape: (batch_size, len(quantiles))

        # --- inverse normalization to restore physical space ---
        y_test_inv = scalers[name].inverse_transform(test_label_all.reshape(-1, 1)).flatten()
        predictions_inv_low = scalers[name].inverse_transform(preds[:, 0].reshape(-1, 1)).flatten()  
        predictions_inv_high = scalers[name].inverse_transform(preds[:, -1].reshape(-1, 1)).flatten()

        result = []
        data_time = np.array(data_with_time[time_name][test_mask])[:len(predictions_inv_low)]

        for i in range(len(predictions_inv_low)):
            result.append({
                'actual_time': str(data_time[i]),
                'virtual_time': str(data_time[i] + pd.Timedelta(days=time_shift)),
                'true': float(y_test_inv[i]),
                'prediction_low': float(predictions_inv_low[i]),
                'prediction_high': float(predictions_inv_high[i]),
            })

        result_json = json.dumps(result, ensure_ascii=False, indent=4)

        # --- visualization ---
        plt.figure(figsize=(14, 7))
        plt.plot(data_time + pd.Timedelta(days=time_shift), y_test_inv, label='True Value', color='black')
        plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_low, label=f'TCN-QR Bound ({100*quantiles[0]}%)', linestyle='--')
        plt.plot(data_time + pd.Timedelta(days=time_shift), predictions_inv_high, label=f'TCN-QR Bound ({100*quantiles[-1]}%)', linestyle='--')
        plt.fill_between(data_time + pd.Timedelta(days=time_shift), predictions_inv_low, predictions_inv_high, alpha=0.2, label='TCN-QR Prediction Interval')

        plt.title(f'TCN-QR Interval Prediction for {name}')
        plt.xlabel('Time')
        plt.ylabel(name)
        plt.legend()
        plt.show()

        return result_json