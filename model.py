import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler
import pickle
import matplotlib.pyplot as plt
import cloudpickle
from lightgbm import LGBMClassifier
import seaborn as sns


class ObecityPredictionModel:
    def __init__(self):
        self.model =  None
        self.rob_scaler = RobustScaler()
        self.std_scaler = StandardScaler()
        self.label_encoders = {}
        self.ohe_encoders = {}
        self.normal_cols = ['Age','Height','Weight','FCVC','CH2O','FAF','TUE']
        self.unnormal_cols = ['NCP']
        self.ohe_columns = ['CAEC','CALC']
        self.le_columns = ['Gender','FAVC','family_history_with_overweight','SCC']
        self.y_encoder = LabelEncoder()
        self.all_columns = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully with Shape {}".format(self.data.shape))
        return self.data

    def preprocess_data(self,test_size = 0.2):
        self.x_train, self.y_train, self.x_test, self.y_test = self.clean_data(test_size)
        self.scaling_data()
        self.one_hot_encode_column()
        self.label_encode()
        self.all_columns = self.x_train.columns
        
    def scaling_data(self):
        self.x_train[self.normal_cols] = self.std_scaler.fit_transform(self.x_train[self.normal_cols])
        self.x_test[self.normal_cols] = self.std_scaler.transform(self.x_test[self.normal_cols])
        self.x_train[self.unnormal_cols] = self.rob_scaler.fit_transform(self.x_train[self.unnormal_cols])
        self.x_test[self.unnormal_cols] = self.rob_scaler.transform(self.x_test[self.unnormal_cols])
    
    def one_hot_encode_column(self):
        for col in self.ohe_columns:
            self.ohe_encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_encoded = self.ohe_encoders[col].fit_transform(self.x_train[[col]])
            test_encoded = self.ohe_encoders[col].transform(self.x_test[[col]])

            col_names = [f'{col}_{cat}' for cat in self.ohe_encoders[col].categories_[0]]

            train_ohe_df = pd.DataFrame(train_encoded, columns=col_names, index=self.x_train.index)
            test_ohe_df = pd.DataFrame(test_encoded, columns=col_names, index=self.x_test.index)

            self.x_train = pd.concat([self.x_train.drop(columns=[col]), train_ohe_df], axis=1)
            self.x_test = pd.concat([self.x_test.drop(columns=[col]), test_ohe_df], axis=1)
        return

    def label_encode(self):
        for column in self.le_columns:
            self.label_encoders[column] = LabelEncoder()

            self.x_train[column] = self.label_encoders[column] .fit_transform(self.x_train[column])
            self.x_test[column] = self.label_encoders[column] .transform(self.x_test[column])
            
        self.y_train = self.y_encoder.fit_transform(self.y_train)
        self.y_test = self.y_encoder.transform(self.y_test)
        return
    
    def clean_data(self,test_size):
        if self.data is None:
            print("Data is not loaded. Please load the data first.")
            return
        self.data = self.data.drop_duplicates()
        self.data['Age'] = self.data['Age'].astype(str).str.extract('(\d+)').astype(int)
        
        
        x = self.data.drop(columns=["NObeyesdad","SMOKE","MTRANS"])
        y = self.data["NObeyesdad"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        FCVC_imputation = x_train["FCVC"].mean()
        x_train['FCVC'] = x_train['FCVC'].fillna(FCVC_imputation)
        x_test['FCVC'] = x_test['FCVC'].fillna(FCVC_imputation)
        return x_train, y_train, x_test, y_test

    def train_model(self):
        print("Training Model")
        self.model = LGBMClassifier(learning_rate = 0.1, n_estimators= 200,num_leaves= 15, verbose=-1, random_state = 42)
        self.model.fit(self.x_train, self.y_train)
        print("Training Model Done")
        return self.model
    
    def evaluate_model(self):
        print("Evaluating Model")
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred,average='macro')
        recall = recall_score(self.y_test, y_pred,average='macro')
        f1 = f1_score(self.y_test, y_pred,average='macro')
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred,target_names=self.y_encoder.classes_))
        print("Confusion Matrix : ")
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.y_encoder.classes_,
                    yticklabels=self.y_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        print("Evaluating Model Done")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_model(self, filename = 'model.pkl',cloud = False):
        model_data = {
            'model': self.model,
            'std_scaler': self.std_scaler,
            'rob_scaler': self.rob_scaler,
            'label_encoders': self.label_encoders,
            'ohe_encoders': self.ohe_encoders,
            'y_encoder': self.y_encoder,
            'normal_cols' : self.normal_cols,
            'unnormal_cols' : self.unnormal_cols,
            'ohe_columns' : self.ohe_columns,
            'le_columns' : self.le_columns,
            'all_columns' : self.x_train.columns
        }
        
        if cloud:
            with open(filename, 'wb') as f:
                cloudpickle.dump(self, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
        return
        
    
    @classmethod
    def load_model(cls, filename = 'model.pkl',cloud = False):
        if cloud:
            with open(filename, 'rb') as file:
                model = cloudpickle.load(file)
                return model
        else:
            with open(filename, 'rb') as file:
                model_data = pickle.load(file)
            
        predictor = cls()
        predictor.model = model_data['model']
        predictor.std_scaler = model_data['std_scaler']
        predictor.rob_scaler = model_data['rob_scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.ohe_encoders = model_data['ohe_encoders']
        predictor.y_encoder = model_data['y_encoder']
        predictor.normal_cols = model_data['normal_cols']
        predictor.unnormal_cols = model_data['unnormal_cols']
        predictor.ohe_columns = model_data['ohe_columns']
        predictor.le_columns = model_data['le_columns']
        predictor.all_columns = model_data['all_columns']
        return predictor
    
    def preprocess_predict_data(self,new_data):
        new_data = new_data.drop(columns=[col for col in ['NObeyesdad'] if col in new_data.columns])
        new_data['Age'] = new_data['Age'].astype(str).str.extract('(\d+)').astype(int)
        preprocess_data = new_data
        preprocess_data[self.normal_cols] = self.std_scaler.transform(preprocess_data[self.normal_cols])
        preprocess_data[self.unnormal_cols] = self.rob_scaler.transform(preprocess_data[self.unnormal_cols])
        for col in self.ohe_columns:
            test_encoded = self.ohe_encoders[col].transform(preprocess_data[[col]])
            col_names = [f'{col}_{cat}' for cat in self.ohe_encoders[col].categories_[0]]
            test_ohe_df = pd.DataFrame(test_encoded, columns=col_names, index=preprocess_data.index)
            preprocess_data = pd.concat([preprocess_data.drop(columns=[col]), test_ohe_df], axis=1)
        for col in self.le_columns:
            preprocess_data[col] = self.label_encoders[col] .transform(preprocess_data[col])
        preprocess_data = preprocess_data.reindex(columns=self.all_columns)
        if set(preprocess_data.columns) != set(self.all_columns):
            missing = set(self.all_columns) - set(preprocess_data.columns)
            extra = set(preprocess_data.columns) - set(self.all_columns)
            print("Mismatch in test data columns.")
            if missing:
                print(f"Missing columns: {missing}")
            if extra:
                print(f"Unexpected columns: {extra}")
            return None
        
        return preprocess_data
    
    def predict(self, test_data):
        preprocess_data = self.preprocess_predict_data(test_data)
        if self.model is None:
            print("Model is not loaded. Please load the model first.")
            return
        predictions = self.model.predict(preprocess_data)
        labels = self.y_encoder.inverse_transform(predictions)
        return predictions, labels



if __name__ == "__main__":
    model = ObecityPredictionModel()
    data = model.load_data('./ObesityDataSet2.csv')
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.save_model(filename='model_without_cloud.pkl',cloud = False)
    model.save_model(filename='model_with_cloud.pkl',cloud = True)

    # Prediction 
    new_data = pd.read_csv('./ObesityDataSet2.csv').dropna().head(1) # As Testing Data
    preds, labels = model.predict(new_data)
    print("Use Trained Model")
    print(preds)
    print(labels)
    # Use Saved Model (Cloud)
    print("Use Saved Model")
    model_with_cloud = ObecityPredictionModel.load_model('model_with_cloud.pkl',cloud=True)
    new_data = pd.read_csv('./ObesityDataSet2.csv').dropna().head(1) # As Testing Data
    preds, labels = model_with_cloud.predict(new_data)
    print("With Cloud")
    print(preds)
    print(labels)

    # Use Saved Model (Without Cloud)
    model_without_cloud = ObecityPredictionModel.load_model('model_without_cloud.pkl',cloud=False)
    new_data = pd.read_csv('./ObesityDataSet2.csv').dropna().head(1) # As Testing Data
    preds, labels = model_without_cloud.predict(new_data)
    print("Without Cloud")
    print(preds)
    print(labels)

