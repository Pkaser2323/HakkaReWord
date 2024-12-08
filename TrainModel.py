from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

def train_and_save_model():
    # 假設的訓練數據
    X = np.random.rand(100, 3)  # 特徵: [s_value, incorrect_count, days_elapsed]
    y = X[:, 0] * 10 + X[:, 1] * -5 + X[:, 2] * 3  # 模擬目標數據

    model = RandomForestRegressor()
    model.fit(X, y)

    # 保存模型
    with open("review_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("模型已成功訓練並保存！")
if __name__ == "__main__":
    train_and_save_model()