# streamlit run logistic_regression_demo.py
# C:\Users\孙冰\Desktop\AI助教\逻辑回归

# logistic_regression_demo.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
import io
from api_deepseek import client, ask_ai_assistant
import logistic_regression_step_by_step

# 设置页面
st.set_page_config(page_title="逻辑回归交互式学习平台", layout="wide")
st.title("📚 逻辑回归交互式学习平台")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def display_chat_interface(context=""):
    """显示聊天界面"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设问题快捷按钮
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是sigmoid函数?")
        btn2 = st.button("逻辑回归与线性回归的区别")
    
    with col2:
        btn3 = st.button("分类阈值如何选择")
        btn4 = st.button("交叉熵损失原理")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是sigmoid函数?它在逻辑回归中的作用是什么?"
    elif btn2:
        question = "逻辑回归与线性回归有什么主要区别?分别适用于什么场景?"
    elif btn3:
        question = "逻辑回归中分类阈值(阈值)如何选择?不同阈值有什么影响?"
    elif btn4:
        question = "请解释交叉熵损失函数的原理，为什么逻辑回归不用均方误差?"
    
    # 提问输入框
    user_input = st.sidebar.text_input("输入你的问题:", key="question_input")
    if user_input:
        question = user_input
    
    # 处理提问
    if question:
        # 显示当前问题
        st.sidebar.markdown(f"**你:** {question}")
        
        # 获取回答
        with st.spinner("助教思考中..."):
            answer = ask_ai_assistant(question, context)
        
        # 显示当前回答
        st.sidebar.markdown(f"**助教:** {answer}")
        st.sidebar.markdown("---")

# Sigmoid函数定义与可视化
def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def plot_sigmoid_function():
    """绘制sigmoid函数图像"""
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', label='阈值=0.5')
    ax.axvline(x=0, color='g', linestyle=':', label='x=0')
    ax.set_xlabel('x')
    ax.set_ylabel('sigmoid(x)')
    ax.set_title('Sigmoid函数图像')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    return fig

# 数据生成函数（二分类数据）
def generate_classification_data(data_type, n_samples, separation):
    """生成分类数据"""
    np.random.seed(42)
    
    if data_type == "线性可分":
        # 生成两个线性可分的类别
        X1 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([separation, separation])
        X2 = np.random.randn(n_samples//2, 2) * 0.8 - np.array([separation, separation])
        X = np.vstack((X1, X2))
        y = np.hstack((np.zeros(n_samples//2), np.ones(n_samples//2)))
    
    elif data_type == "线性不可分":
        # 生成线性不可分的数据
        X = np.random.randn(n_samples, 2) * 1.2
        # 基于二次函数生成标签，制造非线性边界
        y = (X[:, 0]**2 + X[:, 1]** 2 < 1.5).astype(int)
    
    elif data_type == "不平衡数据":
        # 生成不平衡数据
        n_majority = int(n_samples * 0.8)
        n_minority = n_samples - n_majority
        
        X_majority = np.random.randn(n_majority, 2) * 0.8 - np.array([separation/2, separation/2])
        X_minority = np.random.randn(n_minority, 2) * 0.8 + np.array([separation/2, separation/2])
        X = np.vstack((X_majority, X_minority))
        y = np.hstack((np.zeros(n_majority), np.ones(n_minority)))
    
    # 打乱数据顺序
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

# 绘制分类数据
def plot_classification_data(X, y, title):
    """绘制分类数据散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 逻辑回归梯度下降模拟
def logistic_regression_gradient_descent(X, y, learning_rate, n_iterations):
    """手动实现逻辑回归的梯度下降"""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    costs = []
    
    for _ in range(n_iterations):
        # 计算线性输出
        linear_model = np.dot(X, weights) + bias
        # 应用sigmoid函数
        y_pred = sigmoid(linear_model)
        
        # 计算交叉熵损失
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        costs.append(cost)
        
        # 计算梯度
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # 更新参数
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias, costs

# 绘制决策边界
def plot_decision_boundary(X, y, weights, bias, threshold=0.5, title="决策边界"):
    """绘制逻辑回归的决策边界"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点
    ax.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
    Z = (Z >= threshold).astype(int)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 绘制sigmoid曲线与分类阈值
def plot_sigmoid_threshold():
    """展示sigmoid函数与不同阈值的关系"""
    x = np.linspace(-5, 5, 100)
    y = sigmoid(x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='sigmoid函数')
    
    # 绘制不同阈值线
    thresholds = [0.3, 0.5, 0.7]
    colors = ['g', 'r', 'purple']
    for threshold, color in zip(thresholds, colors):
        # 找到对应阈值的x值
        x_threshold = np.log(threshold / (1 - threshold))
        ax.axhline(y=threshold, color=color, linestyle='--', 
                  label=f'阈值={threshold} (x={x_threshold:.2f})')
        ax.axvline(x=x_threshold, color=color, linestyle=':')
    
    ax.set_xlabel('线性输出 (z = wx + b)')
    ax.set_ylabel('概率 p(y=1)')
    ax.set_title('Sigmoid函数与不同分类阈值')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    return fig

# 数据生成与探索模块
def data_generation_section():
    st.header("📊 分类数据生成与探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("选择数据类型", 
                               ["线性可分", "线性不可分", "不平衡数据"])
        n_samples = st.slider("样本数量", 50, 500, 200)
        separation = st.slider("类别分离程度", 0.5, 5.0, 2.0, 0.5)
        
        X, y = generate_classification_data(data_type, n_samples, separation)
        
        st.write(f"数据统计:")
        st.write(f"- 类别0数量: {np.sum(y == 0)}")
        st.write(f"- 类别1数量: {np.sum(y == 1)}")
        st.write(f"- 特征1均值: {np.mean(X[:, 0]):.2f}, 标准差: {np.std(X[:, 0]):.2f}")
        st.write(f"- 特征2均值: {np.mean(X[:, 1]):.2f}, 标准差: {np.std(X[:, 1]):.2f}")
    
    with col2:
        fig = plot_classification_data(X, y, f'{data_type}数据分布')
        st.pyplot(fig)
    
    st.info("""
    **分类数据探索要点:**
    - 线性可分数据: 可以用一条直线完美分隔两个类别
    - 线性不可分数据: 无法用一条直线完美分隔
    - 不平衡数据: 一个类别的样本数量远多于另一个类别
    - 类别分离程度影响分类难度，分离越好越容易分类
    """)
    
    # 存储数据供后续模块使用
    st.session_state.X = X
    st.session_state.y = y
    
    return f"数据生成模块: 创建了{data_type}数据，样本数={n_samples}，分离程度={separation}"

# Sigmoid函数交互模块
def sigmoid_interactive_section():
    st.header("🔄 Sigmoid函数交互演示")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Sigmoid函数公式:**
        $$\sigma(z) = \\frac{1}{1 + e^{-z}}$$
        
        其中 $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$ 是线性组合
        
        **Sigmoid函数的特点:**
        - 输出值范围在(0, 1)之间，可解释为概率
        - 当z→+∞时，σ(z)→1
        - 当z→-∞时，σ(z)→0
        - 当z=0时，σ(z)=0.5
        - 函数光滑且可导，适合梯度下降优化
        """)
        
        z_value = st.slider("选择z值", -10.0, 10.0, 0.0, 0.1)
        sigmoid_value = sigmoid(z_value)
        st.metric("sigmoid(z)值", f"{sigmoid_value:.4f}")
        
        if sigmoid_value >= 0.5:
            st.success(f"当z={z_value:.1f}时，预测为类别1（概率={sigmoid_value:.4f}）")
        else:
            st.info(f"当z={z_value:.1f}时，预测为类别0（概率={1-sigmoid_value:.4f}）")
    
    with col2:
        # 绘制sigmoid函数
        fig1 = plot_sigmoid_function()
        st.pyplot(fig1)
        
        # 绘制不同阈值的影响
        fig2 = plot_sigmoid_threshold()
        st.pyplot(fig2)
    
    return f"Sigmoid函数模块: 探索了z={z_value:.1f}时的函数值"

# 手动调整参数模块
def manual_tuning_section():
    st.header("🎛️ 逻辑回归参数手动调整")
    
    # 检查是否已有数据，没有则生成默认数据
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.session_state.X, st.session_state.y = generate_classification_data("线性可分", 200, 2.0)
    
    X = st.session_state.X
    y = st.session_state.y
    
    # 只使用第一个特征进行简单可视化
    X_single = X[:, 0].reshape(-1, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("调整模型参数")
        weight = st.slider("权重 (w)", -5.0, 5.0, 1.0, 0.1)
        bias = st.slider("偏置 (b)", -5.0, 5.0, 0.0, 0.1)
        threshold = st.slider("分类阈值", 0.1, 0.9, 0.5, 0.05)
        
        # 计算预测值
        z = weight * X_single.flatten() + bias
        y_prob = sigmoid(z)
        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        st.metric("分类准确率", f"{accuracy:.4f}")
        
        # 显示混淆矩阵
        st.subheader("混淆矩阵")
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=['实际0', '实际1'], columns=['预测0', '预测1'])
        st.dataframe(cm_df)
    
    with col2:
        # 绘制sigmoid曲线与分类结果
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 第一个图：sigmoid曲线与数据点
        x_range = np.linspace(X_single.min() - 1, X_single.max() + 1, 100)
        ax1.scatter(X_single[y==0], np.zeros_like(X_single[y==0]), alpha=0.7, label='类别0')
        ax1.scatter(X_single[y==1], np.ones_like(X_single[y==1]), alpha=0.7, label='类别1')
        ax1.plot(x_range, sigmoid(weight * x_range + bias), 'r-', linewidth=2, label='sigmoid曲线')
        ax1.axhline(y=threshold, color='g', linestyle='--', label=f'阈值={threshold}')
        ax1.set_xlabel('特征值')
        ax1.set_ylabel('预测概率')
        ax1.set_title('Sigmoid曲线与分类点')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 第二个图：决策边界（使用两个特征）
        if X.shape[1] >= 2:
            # 为了可视化，使用前两个特征和当前权重偏置
            weights_2d = np.array([weight, 0.5])  # 固定第二个特征的权重
            fig_db = plot_decision_boundary(X, y, weights_2d, bias, threshold, "决策边界可视化")
            ax2 = fig_db.gca()
            plt.close(fig_db)  # 避免重复显示
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.info("""
    **参数调整指南:**
    - 权重(w): 控制特征对输出的影响强度和方向
    - 偏置(b): 控制sigmoid曲线的左右移动
    - 阈值: 控制分类的严格程度，高阈值会减少假阳性但可能增加假阴性
    
    尝试调整参数使准确率尽可能高！
    """)
    
    return f"手动调整模块: 权重={weight:.1f}, 偏置={bias:.1f}, 阈值={threshold:.2f}, 准确率={accuracy:.4f}"

# 梯度下降可视化模块
def gradient_descent_section():
    st.header("📉 逻辑回归梯度下降可视化")
    
    # 检查是否已有数据，没有则生成默认数据
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.session_state.X, st.session_state.y = generate_classification_data("线性可分", 200, 2.0)
    
    X = st.session_state.X
    y = st.session_state.y
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率", 0.001, 0.5, 0.1, 0.001)
        n_iterations = st.slider("迭代次数", 50, 2000, 500)
        
        if st.button("开始梯度下降演示"):
            # 运行梯度下降
            weights, bias, costs = logistic_regression_gradient_descent(
                X_scaled, y, learning_rate, n_iterations
            )
            
            # 显示过程
            placeholder = st.empty()
            # 只显示部分迭代步骤，避免太慢
            step = max(1, n_iterations // 20)
            for i in range(0, n_iterations + 1, step):
                with placeholder.container():
                    # 计算当前迭代的参数（如果超出范围则用最后一组）
                    current_weights = weights if i == n_iterations else \
                                     logistic_regression_gradient_descent(
                                         X_scaled, y, learning_rate, i)[0]
                    current_bias = bias if i == n_iterations else \
                                  logistic_regression_gradient_descent(
                                      X_scaled, y, learning_rate, i)[1]
                    
                    # 绘制决策边界和损失曲线
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # 决策边界
                    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                         np.arange(y_min, y_max, 0.01))
                    
                    # 标准化网格点
                    grid_points = np.c_[xx.ravel(), yy.ravel()]
                    grid_points_scaled = scaler.transform(grid_points)
                    
                    Z = sigmoid(np.dot(grid_points_scaled, current_weights) + current_bias)
                    Z = (Z >= 0.5).astype(int)
                    Z = Z.reshape(xx.shape)
                    
                    ax1.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
                    ax1.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
                    ax1.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
                    ax1.set_title(f'迭代 {i}/{n_iterations}')
                    ax1.legend()
                    
                    # 损失曲线
                    ax2.plot(range(min(i+1, len(costs))), costs[:min(i+1, len(costs))])
                    ax2.set_xlabel('迭代次数')
                    ax2.set_ylabel('交叉熵损失')
                    ax2.set_title(f'损失: {costs[min(i, len(costs)-1)]:.4f}')
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    time.sleep(0.2)
            
            st.success(f"梯度下降完成! 最终损失: {costs[-1]:.4f}")
    
    with col2:
        st.markdown("""
        **逻辑回归梯度下降原理:**
        
        1. **初始化**权重和偏置为0
        2. 计算**线性输出** $z = wx + b$
        3. 应用**sigmoid函数**得到概率预测 $\\hat{y} = \\sigma(z)$
        4. 计算**交叉熵损失**:
           $$L = -\\frac{1}{n}\\sum(y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y}))$$
        5. 计算损失对权重和偏置的**梯度**
        6. 沿梯度反方向**更新参数**:
           $$w = w - \\alpha \\cdot \\frac{\\partial L}{\\partial w}$$
           $$b = b - \\alpha \\cdot \\frac{\\partial L}{\\partial b}$$
        7. 重复步骤2-6直到收敛
        
        **学习率选择建议:**
        - 太小: 收敛速度慢，需要更多迭代
        - 太大: 可能导致不收敛，损失波动甚至增大
        """)
    
    return f"梯度下降模块: 学习率={learning_rate}, 迭代次数={n_iterations}"

# 模型评估模块
# 模型评估模块（专注于混淆矩阵和评估指标解释）
def model_evaluation_section():
    st.header("📊 模型评估与指标解释")
    
    # 生成一个更具挑战性的二分类数据集（两个特征）
    np.random.seed(42)
    n_samples = 200
    
    # 生成两个重叠的高斯分布，确保有错误分类
    X0 = np.random.multivariate_normal(
        mean=[-1, -1], 
        cov=[[2, 1.5], [1.5, 2]], 
        size=n_samples//2
    )
    X1 = np.random.multivariate_normal(
        mean=[1, 1], 
        cov=[[2, -1.5], [-1.5, 2]], 
        size=n_samples//2
    )
    
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples//2), np.ones(n_samples//2)))
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 显示数据散点图和决策边界
        st.subheader("数据分布与决策边界")
        fig1 = plt.figure(figsize=(8, 6))
        
        # 绘制数据点
        plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', alpha=0.7, label='实际类别 0')
        plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', alpha=0.7, label='实际类别 1')
        
        # 绘制错误分类的点
        misclassified = (y_test != y_pred)
        plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1], 
                   c='black', marker='x', s=100, label='错误分类')
        
        # 绘制决策边界
        x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
        y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('决策边界与分类结果')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)
    
    with col2:
        # 显示混淆矩阵和评估指标
        st.subheader("混淆矩阵")
        
        # 创建混淆矩阵可视化
        fig2, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 设置标签
        classes = ['0', '1']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='混淆矩阵',
               ylabel='真实标签',
               xlabel='预测标签')
        
        # 在矩阵中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        st.pyplot(fig2)

    
    # 解释混淆矩阵
    st.subheader("混淆矩阵解读")
    st.markdown("""
    混淆矩阵是评估分类模型性能的重要工具，它显示了模型预测结果与真实标签的对比：
    
    - **真正例 (TP)**: 实际为正类，预测为正类 (右下角)
    - **真反例 (TN)**: 实际为负类，预测为负类 (左上角)
    - **假正例 (FP)**: 实际为负类，预测为正类 (右上角) - 第一类错误
    - **假反例 (FN)**: 实际为正类，预测为负类 (左下角) - 第二类错误
    
    **不同场景下的指标选择:**
    - 高精确率: 适合需要减少假阳性(FP)的场景，如垃圾邮件检测
    - 高召回率: 适合需要减少假阴性(FN)的场景，如疾病诊断
    - F1分数: 在精确率和召回率之间寻求平衡
    
    在实际应用中，需要根据具体业务需求选择合适的评估指标。
    """)
        
    # 显示评估指标
    st.subheader("评估指标")
    st.markdown(f"""
        - **准确率 (Accuracy)**: {accuracy:.4f}  ——  所有预测中正确预测的比例
        
        - **精确率 (Precision)**: {precision:.4f}  ——  预测为正类的样本中真正为正类的比例    (真正例 / (真正例 + 假正例))
        
        - **召回率 (Recall)**: {recall:.4f}  ——  真正为正类的样本中被正确预测的比例    (真正例 / (真正例 + 假反例))
        
        - **F1分数**: {f1:.4f}  ——  精确率和召回率的调和平均    (2 × Precision × Recall) / (Precision + Recall)
    """)
    
    return f"模型评估模块: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}"


# 概念测验模块
def quiz_section():
    st.header("🎯 逻辑回归概念测验")
    
    question = st.selectbox(
        "选择一个问题测试你的理解:",
        [
            "逻辑回归的输出是什么?",
            "sigmoid函数的作用是什么?",
            "逻辑回归为什么使用交叉熵损失?",
            "分类阈值如何影响模型性能?",
            "逻辑回归可以处理非线性问题吗?"
        ]
    )
    
    if question == "逻辑回归的输出是什么?":
        answer = st.radio("选择正确答案:", [
            "连续的预测值",
            "0或1的分类结果",
            "属于某个类别的概率"
        ], key="q1")
        if st.button("检查答案", key="b1"):
            if answer == "属于某个类别的概率":
                st.success("✅ 正确! 逻辑回归输出的是样本属于正类的概率，范围在0到1之间。")
            else:
                st.error("❌ 不正确，逻辑回归的核心输出是概率值，而非直接的分类结果或连续值。")
    
    elif question == "sigmoid函数的作用是什么?":
        answer = st.radio("选择正确答案:", [
            "增加模型复杂度",
            "将线性输出转换为概率",
            "加速模型训练"
        ], key="q2")
        if st.button("检查答案", key="b2"):
            if answer == "将线性输出转换为概率":
                st.success("✅ 正确! Sigmoid函数能将任意实数映射到(0,1)区间，适合表示概率。")
            else:
                st.error("❌ 不正确，sigmoid函数的主要作用是将线性输出转换为概率。")
    
    elif question == "逻辑回归为什么使用交叉熵损失?":
        answer = st.radio("选择正确答案:", [
            "交叉熵损失计算更简单",
            "交叉熵损失是凸函数，更容易优化",
            "没有特别原因，只是传统习惯"
        ], key="q3")
        if st.button("检查答案", key="b3"):
            if answer == "交叉熵损失是凸函数，更容易优化":
                st.success("✅ 正确! 对于逻辑回归，交叉熵损失是凸函数，存在唯一最小值，而均方误差是non-convex的。")
            else:
                st.error("❌ 不正确，主要原因是交叉熵损失对于逻辑回归是凸函数，更容易通过梯度下降找到最优解。")
    
    elif question == "分类阈值如何影响模型性能?":
        answer = st.radio("选择正确答案:", [
            "阈值不影响模型性能",
            "高阈值会提高精确率但降低召回率",
            "高阈值会同时提高精确率和召回率"
        ], key="q4")
        if st.button("检查答案", key="b4"):
            if answer == "高阈值会提高精确率但降低召回率":
                st.success("✅ 正确! 高阈值意味着更严格的正类判断标准，减少误报但可能增加漏报。")
            else:
                st.error("❌ 不正确，高阈值会提高精确率（更少误报）但降低召回率（更多漏报）。")
    
    elif question == "逻辑回归可以处理非线性问题吗?":
        answer = st.radio("选择正确答案:", [
            "不能，逻辑回归只能处理线性可分问题",
            "可以，通过特征工程引入非线性特征",
            "可以，逻辑回归本身是非线性模型"
        ], key="q5")
        if st.button("检查答案", key="b5"):
            if answer == "可以，通过特征工程引入非线性特征":
                st.success("✅ 正确! 逻辑回归的决策边界本身是线性的，但通过添加多项式特征等方式，可以处理非线性问题。")
            else:
                st.error("❌ 不正确，逻辑回归可以通过特征工程（如添加多项式特征）处理非线性问题。")
    
    return f"概念测验模块: 当前问题='{question}'"

# 实际应用案例模块
def real_world_example_section():
    st.header("🌍 逻辑回归实际应用案例")
    
    example = st.selectbox(
        "选择实际应用案例:",
        ["信用卡欺诈检测", "客户流失预测", "疾病风险预测", "上传自己的数据"]
    )
    
    if example == "上传自己的数据":
        uploaded_file = st.file_uploader("上传CSV文件", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("数据预览:", data.head())
            
            # 检查是否有分类变量
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.warning("检测到分类变量，本演示将自动忽略这些列。")
                data = data.select_dtypes(exclude=['object'])
            
            # 选择目标列
            if len(data.columns) < 2:
                st.error("数据至少需要包含一个特征列和一个目标列!")
                return
            
            target_col = st.selectbox("选择目标列(应包含0和1)", data.columns)
            
            # 检查目标列是否为二分类
            unique_vals = data[target_col].unique()
            if len(unique_vals) != 2 or not set(unique_vals).issubset({0, 1}):
                st.error("目标列必须是二分类(只包含0和1)!")
                return
            
            # 选择特征列
            feature_cols = [col for col in data.columns if col != target_col]
            if not feature_cols:
                st.error("没有可用的特征列!")
                return
            
            X = data[feature_cols].values
            y = data[target_col].values
            
            analyze_custom_data(X, y, feature_cols, target_col)
            return f"实际应用模块: 上传自定义数据, 目标列={target_col}"
    else:
        # 生成示例数据
        X, y, description = load_example_dataset(example)
        st.write(description)
        
        analyze_custom_data(X, y, ["特征1", "特征2", "特征3"], "目标变量")
        return f"实际应用模块: 使用{example}数据集"

# 加载示例数据集
def load_example_dataset(example_name):
    np.random.seed(42)
    
    if example_name == "信用卡欺诈检测":
        # 生成欺诈检测数据：大多数是正常交易，少数是欺诈
        n_samples = 500
        n_fraud = int(n_samples * 0.1)  # 10%欺诈率
        
        # 正常交易特征
        normal_amount = np.random.normal(500, 300, n_samples - n_fraud)
        normal_time = np.random.normal(12, 6, n_samples - n_fraud)
        normal_freq = np.random.normal(2, 1, n_samples - n_fraud)
        
        # 欺诈交易特征（金额更大，时间更晚，频率更低）
        fraud_amount = np.random.normal(2000, 800, n_fraud)
        fraud_time = np.random.normal(20, 4, n_fraud)
        fraud_freq = np.random.normal(0.5, 0.3, n_fraud)
        
        # 合并数据
        X = np.vstack([
            np.column_stack((normal_amount, normal_time, normal_freq)),
            np.column_stack((fraud_amount, fraud_time, fraud_freq))
        ])
        y = np.hstack([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])
        
        # 打乱顺序
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        description = "信用卡欺诈检测数据: 包含交易金额、时间和频率特征，预测交易是否为欺诈(1=欺诈)"
        return X, y, description
    
    elif example_name == "客户流失预测":
        # 生成客户流失数据
        n_samples = 500
        
        # 特征：使用时长(月)、月消费、客服联系次数
        tenure = np.random.normal(30, 20, n_samples)
        monthly_charge = np.random.normal(50, 30, n_samples)
        support_calls = np.random.randint(0, 10, n_samples)
        
        X = np.column_stack((tenure, monthly_charge, support_calls))
        
        # 流失概率：使用时长越短、月消费越高、客服联系越多，流失概率越大
        z = -0.05*tenure + 0.03*monthly_charge + 0.3*support_calls - 2
        prob = sigmoid(z)
        y = np.random.binomial(1, prob)
        
        description = "客户流失预测数据: 包含使用时长、月消费和客服联系次数，预测客户是否会流失(1=流失)"
        return X, y, description
    
    elif example_name == "疾病风险预测":
        # 生成疾病风险预测数据
        n_samples = 500
        
        # 特征：年龄、BMI、血压
        age = np.random.normal(50, 15, n_samples)
        bmi = np.random.normal(25, 5, n_samples)
        blood_pressure = np.random.normal(120, 15, n_samples)
        
        X = np.column_stack((age, bmi, blood_pressure))
        
        # 患病概率：年龄越大、BMI越高、血压越高，患病概率越大
        z = 0.04*age + 0.1*bmi + 0.03*blood_pressure - 10
        prob = sigmoid(z)
        y = np.random.binomial(1, prob)
        
        description = "疾病风险预测数据: 包含年龄、BMI和血压，预测患病风险(1=患病)"
        return X, y, description
    
    return None, None, ""

# 分析自定义数据（不使用标准化）
def analyze_custom_data(X, y, feature_names, target_name):
    if len(X) != len(y):
        st.error("特征和目标的长度不匹配!")
        return
    
    if len(X) < 10:
        st.error("数据点太少，至少需要10个样本!")
        return
    
    # 训练模型（不使用标准化）
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # 预测概率
    y_prob = model.predict_proba(X)[:, 1]
    
    # 评估模型
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    st.subheader("模型性能")
    st.text(report)
    
    # 显示系数
    st.subheader("特征重要性（系数）")
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': model.coef_[0]
    }).sort_values('系数', ascending=False)
    st.dataframe(coef_df)
    
    st.info("""
    **系数解释:**
    - 正系数: 该特征值越大，属于正类的概率越高
    - 负系数: 该特征值越大，属于正类的概率越低
    - 系数绝对值越大，特征对预测的影响越大
    
    注意：系数大小受特征尺度影响，这里使用的是原始数据，未进行标准化处理。
    """)
    
    # 如果是二维数据，绘制决策边界
    if X.shape[1] == 2:
        fig = plot_decision_boundary(X, y, model.coef_[0], model.intercept_[0])
        st.pyplot(fig)

        
# 主程序
def main():
    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "数据生成与探索"
    
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "数据生成与探索",
        "Sigmoid函数交互演示",
        "参数手动调整",
        "梯度下降可视化",
        "模型评估",
        "概念测验",
        "实际应用案例",
        "编程实例（乳腺癌数据集）" 
    ])
  
    # 更新会话状态
    st.session_state.section = section
    
    context = ""
    if section == "数据生成与探索":
        context = data_generation_section()
    elif section == "Sigmoid函数交互演示":
        context = sigmoid_interactive_section()
    elif section == "参数手动调整":
        context = manual_tuning_section()
    elif section == "梯度下降可视化":
        context = gradient_descent_section()
    elif section == "模型评估":
        context = model_evaluation_section()
    elif section == "概念测验":
        context = quiz_section()
    elif section == "实际应用案例":
        context = real_world_example_section()
    elif section == "编程实例（乳腺癌数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        logistic_regression_step_by_step.main()
        context = "编程实例模块: 乳腺癌数据集逻辑回归分步练习"
    
    # 显示聊天界面
    display_chat_interface(context)
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **逻辑回归交互式学习平台**
    
    设计用于机器学习教学，帮助学生理解:
    - 逻辑回归基本原理
    - Sigmoid函数的作用与特性
    - 分类阈值的选择策略
    - 模型评估指标与解释
    """)


if __name__ == "__main__":
    main()
