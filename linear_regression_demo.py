# linear_regression_demo.py
# C:\Users\孙冰\Desktop\AI助教\linear_regression
# streamlit run linear_regression_demo.py

# linear_regression_demo.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import time

from api_deepseek import client, ask_ai_assistant
import linear_regression_step_by_step

# 设置页面
st.set_page_config(page_title="线性回归交互式学习平台", layout="wide")
st.title("📚 线性回归交互式学习平台")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def display_chat_interface(context=""):
    """显示聊天界面（不保存历史记录）"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设问题快捷按钮
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是过拟合?")
        btn2 = st.button("梯度下降原理")
    
    with col2:
        btn3 = st.button("R²的意义")
        btn4 = st.button("残差分析")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是过拟合?如何识别和避免?"
    elif btn2:
        question = "请用简单易懂的方式解释梯度下降的原理"
    elif btn3:
        question = "决定系数R²的含义是什么?如何解释它的值?"
    elif btn4:
        question = "为什么要进行残差分析?它能告诉我们什么?"
    
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

# 数据生成函数
def generate_data(data_type, n_samples, noise_level):
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples)
    
    if data_type == "线性关系":
        y = 2 * X + 1 + np.random.normal(0, noise_level, n_samples)
    elif data_type == "非线性关系":
        y = 0.5 * X**2 + np.random.normal(0, noise_level, n_samples)
    elif data_type == "带有异常值":
        y = 2 * X + 1 + np.random.normal(0, noise_level, n_samples)
        outlier_indices = np.random.choice(n_samples, size=5, replace=False)
        y[outlier_indices] += 10 * noise_level
    elif data_type == "不同噪声水平":
        y = 2 * X + 1 + np.random.normal(0, noise_level*2, n_samples)
    
    return X, y

# 绘制数据函数
def plot_data(X, y, data_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.7)
    ax.set_xlabel('X (自变量)')
    ax.set_ylabel('y (因变量)')
    ax.set_title(f'数据分布: {data_type}')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 梯度下降模拟函数
def simulate_gradient_descent(X, y, learning_rate, n_iterations):
    slope = 0
    intercept = 0
    n = len(X)
    
    slopes = [slope]
    intercepts = [intercept]
    losses = [mean_squared_error(y, slope * X + intercept)]
    
    for i in range(n_iterations):
        y_pred = slope * X + intercept
        slope_gradient = (-2/n) * np.sum(X * (y - y_pred))
        intercept_gradient = (-2/n) * np.sum(y - y_pred)
        
        slope = slope - learning_rate * slope_gradient
        intercept = intercept - learning_rate * intercept_gradient
        
        slopes.append(slope)
        intercepts.append(intercept)
        losses.append(mean_squared_error(y, slope * X + intercept))
    
    return losses, slopes, intercepts

# 绘制梯度下降步骤函数
def plot_gradient_descent_step(X, y, slope, intercept, loss, iteration, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.scatter(X, y, alpha=0.7, label='数据点')
    y_pred = slope * X + intercept
    ax1.plot(X, y_pred, color='red', linewidth=2, label='当前拟合')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title(f'迭代 {iteration}: 斜率={slope:.2f}, 截距={intercept:.2f}')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(range(iteration+1), losses[:iteration+1], color='blue')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失 (MSE)')
    ax2.set_title(f'当前损失: {loss:.2f}')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# 训练模型函数
def train_model(X, y, model_type):
    X_reshaped = X.reshape(-1, 1)
    
    if model_type == "线性回归":
        model = LinearRegression()
        model.fit(X_reshaped, y)
    elif model_type == "多项式回归(2次)":
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X_reshaped)
        model = LinearRegression()
        model.fit(X_poly, y)
    elif model_type == "多项式回归(3次)":
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X_reshaped)
        model = LinearRegression()
        model.fit(X_poly, y)
    
    return model

# 绘制模型比较图函数
def plot_model_comparison(X, y, y_pred, model_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.7, label='数据点')
    
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='拟合曲线')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'模型拟合: {model_type}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 加载示例数据集函数
def load_example_dataset(example_name):
    if example_name == "房价预测":
        np.random.seed(42)
        sizes = np.random.randint(50, 200, 100)
        prices = 5000 * sizes + 100000 + np.random.normal(0, 20000, 100)
        description = "模拟房价数据: 房屋面积(平方米) vs 价格(元)"
        return sizes, prices, description
    
    elif example_name == "学生成绩预测":
        np.random.seed(42)
        study_hours = np.random.uniform(1, 10, 100)
        grades = 5 * study_hours + 50 + np.random.normal(0, 10, 100)
        description = "模拟学生数据: 学习时间(小时) vs 考试成绩(分)"
        return study_hours, grades, description
    
    elif example_name == "销售额预测":
        np.random.seed(42)
        ad_budget = np.random.uniform(10, 100, 100)
        sales = 50 * ad_budget + 200 + np.random.normal(0, 100, 100)
        description = "模拟销售数据: 广告预算(千元) vs 销售额(万元)"
        return ad_budget, sales, description
    
    return None, None, ""

# 数据生成与探索模块
def data_generation_section():
    st.header("📈 数据生成与探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("选择数据分布类型", 
                               ["线性关系", "非线性关系", "带有异常值", "不同噪声水平"])
        n_samples = st.slider("样本数量", 50, 500, 100)
        noise_level = st.slider("噪声水平", 0.1, 5.0, 1.0, 0.1)
        
        X, y = generate_data(data_type, n_samples, noise_level)
        
        st.write(f"数据统计:")
        st.write(f"- X均值: {np.mean(X):.2f}, X标准差: {np.std(X):.2f}")
        st.write(f"- y均值: {np.mean(y):.2f}, y标准差: {np.std(y):.2f}")
        st.write(f"- X和y的相关系数: {np.corrcoef(X, y)[0, 1]:.2f}")
    
    with col2:
        fig = plot_data(X, y, data_type)
        st.pyplot(fig)
    
    st.info("""
    **数据探索要点:**
    - 观察数据的分布模式和关系
    - 注意异常值对整体模式的影响
    - 噪声水平影响数据的分散程度
    - 相关系数衡量X和y之间的线性关系强度
    """)
    
    return f"数据生成模块: 创建了{data_type}数据，样本数={n_samples}，噪声水平={noise_level}"

# 手动拟合体验模块
def manual_fitting_section():
    st.header("✋ 手动拟合体验")
    
    X, y = generate_data("线性关系", 100, 1.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        slope = st.slider("斜率", -5.0, 5.0, 1.0, 0.1)
        intercept = st.slider("截距", -10.0, 10.0, 0.0, 0.5)
        
        y_pred = slope * X + intercept
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        st.metric("当前均方误差(MSE)", f"{mse:.2f}")
        st.metric("当前决定系数(R²)", f"{r2:.2f}")
        
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        st.write(f"最佳拟合参考: 斜率={model.coef_[0]:.2f}, 截距={model.intercept_:.2f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, alpha=0.7, label='数据点')
        ax.plot(X, y_pred, color='red', linewidth=2, label='手动拟合')
        
        y_best = model.predict(X.reshape(-1, 1))
        ax.plot(X, y_best, color='green', linestyle='--', linewidth=2, label='最佳拟合')
        
        ax.set_title("手动拟合 vs 最佳拟合")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    
    if st.button("挑战: 找到最佳拟合参数"):
        st.info("尝试调整斜率和截距，使均方误差最小化，R²分数最大化!")
        
        with st.expander("需要提示?"):
            st.write("""
            1. 观察数据点的整体趋势
            2. 先调整斜率使直线方向与数据趋势一致
            3. 再调整截距使直线通过数据点的中心区域
            4. 微调两个参数使误差最小化
            """)
    
    return f"手动拟合模块: 当前斜率={slope:.2f}, 截距={intercept:.2f}, MSE={mse:.2f}, R²={r2:.2f}"

# 梯度下降可视化模块
def gradient_descent_section():
    st.header("📉 梯度下降过程可视化")
    
    X, y = generate_data("线性关系", 50, 1.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001)
        n_iterations = st.slider("迭代次数", 10, 1000, 100)
        
        if st.button("开始梯度下降演示"):
            losses, slopes, intercepts = simulate_gradient_descent(
                X, y, learning_rate, n_iterations
            )
            
            placeholder = st.empty()
            for i in range(0, n_iterations + 1, max(1, n_iterations // 20)):
                with placeholder.container():
                    fig = plot_gradient_descent_step(X, y, slopes[i], intercepts[i], losses[i], i, losses)
                    st.pyplot(fig)
                    time.sleep(0.2)
            
            st.success(f"梯度下降完成! 最终参数: 斜率={slopes[-1]:.2f}, 截距={intercepts[-1]:.2f}")
    
    with col2:
        st.markdown("""
        **梯度下降原理:**
        
        梯度下降是一种优化算法，用于找到使损失函数最小化的参数值。
        
        1. 初始化参数(斜率和截距)
        2. 计算损失函数的梯度(导数)
        3. 沿梯度反方向更新参数
        4. 重复直到收敛
        
        **参数更新公式:**
        - 斜率 = 斜率 - 学习率 × ∂损失/∂斜率
        - 截距 = 截距 - 学习率 × ∂损失/∂截距
        
        **学习率的影响:**
        - 太小: 收敛慢
        - 太大: 可能无法收敛，甚至发散
        """)
    
    return f"梯度下降模块: 学习率={learning_rate}, 迭代次数={n_iterations}"

# 模型比较模块
def model_comparison_section():
    st.header("🔍 模型比较")
    
    X, y = generate_data("线性关系", 100, 1.0)
    
    model_type = st.radio("选择模型类型", 
                         ["线性回归", "多项式回归(2次)", "多项式回归(3次)"],
                         horizontal=True)
    
    model = train_model(X, y, model_type)
    
    X_reshaped = X.reshape(-1, 1)
    if model_type != "线性回归":
        degree = 2 if "2次" in model_type else 3
        poly = PolynomialFeatures(degree=degree)
        X_reshaped = poly.fit_transform(X_reshaped)
    
    y_pred = model.predict(X_reshaped)
    
    fig = plot_model_comparison(X, y, y_pred, model_type)
    st.pyplot(fig)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("均方误差(MSE)", f"{mse:.2f}")
    with col2:
        st.metric("决定系数(R²)", f"{r2:.2f}")
    
    st.info("""
    **模型复杂度与过拟合:**
    - 线性回归: 简单模型，可能欠拟合
    - 多项式回归(2次): 中等复杂度，可能适合
    - 多项式回归(3次): 高复杂度，可能过拟合
    
    **过拟合**指模型过于复杂，过度适应训练数据中的噪声，导致在新数据上表现不佳。
    """)
    
    return f"模型比较模块: 当前模型={model_type}, MSE={mse:.2f}, R²={r2:.2f}"

# 残差分析模块
def residual_analysis_section():
    st.header("📊 残差分析")
    
    X, y = generate_data("线性关系", 100, 1.0)
    
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.scatter(y_pred, residuals, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel("预测值")
        ax1.set_ylabel("残差")
        ax1.set_title("残差 vs 预测值")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.hist(residuals, bins=20, alpha=0.7)
        ax2.set_xlabel("残差")
        ax2.set_ylabel("频数")
        ax2.set_title("残差分布")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("残差统计分析:")
        st.write(f"- 均值: {np.mean(residuals):.4f} (应接近0)")
        st.write(f"- 标准差: {np.std(residuals):.4f}")
        st.write(f"- 最小值: {np.min(residuals):.4f}")
        st.write(f"- 最大值: {np.max(residuals):.4f}")
        
        from scipy import stats
        _, p_value = stats.normaltest(residuals)
        st.write(f"- 正态性检验p值: {p_value:.4f}")
        if p_value > 0.05:
            st.write("  ✅ 残差近似正态分布")
        else:
            st.write("  ⚠️ 残差可能不是正态分布")
        
        st.info("""
        **残差分析帮助检查模型假设:**
        
        1. 残差应该随机分布在0附近
        2. 不应该有明显的模式或趋势
        3. 理想情况下应该接近正态分布
        
        如果残差图显示模式(如曲线形状)，可能意味着线性模型不合适。
        """)
    
    return f"残差分析模块: 残差均值={np.mean(residuals):.4f}, 标准差={np.std(residuals):.4f}"

# 概念测验模块
def quiz_section():
    st.header("🎯 概念测验")
    
    question = st.selectbox(
        "选择一个问题测试你的理解:",
        [
            "线性回归的目标是什么?",
            "什么是梯度下降?",
            "过拟合是什么意思?",
            "R² score的取值范围是什么?",
            "残差分析的作用是什么?"
        ]
    )
    
    if question == "线性回归的目标是什么?":
        answer = st.radio("选择正确答案:", [
            "最小化预测值与实际值的绝对差",
            "最小化预测值与实际值的平方差",
            "最大化预测值与实际值的相关系数"
        ], key="q1")
        if st.button("检查答案", key="b1"):
            if answer == "最小化预测值与实际值的平方差":
                st.success("✅ 正确! 线性回归使用最小二乘法最小化平方误差。")
            else:
                st.error("❌ 不正确，请再思考一下。")
    
    elif question == "什么是梯度下降?":
        answer = st.radio("选择正确答案:", [
            "一种数据可视化技术",
            "一种优化算法，用于最小化损失函数",
            "一种数据预处理方法"
        ], key="q2")
        if st.button("检查答案", key="b2"):
            if answer == "一种优化算法，用于最小化损失函数":
                st.success("✅ 正确! 梯度下降通过迭代更新参数来最小化损失函数。")
            else:
                st.error("❌ 不正确，请再思考一下。")
    
    elif question == "过拟合是什么意思?":
        answer = st.radio("选择正确答案:", [
            "模型过于简单，无法捕捉数据中的模式",
            "模型过于复杂，过度适应训练数据中的噪声",
            "模型在训练和测试数据上都表现良好"
        ], key="q3")
        if st.button("检查答案", key="b3"):
            if answer == "模型过于复杂，过度适应训练数据中的噪声":
                st.success("✅ 正确! 过拟合的模型在训练数据上表现很好，但在新数据上表现不佳。")
            else:
                st.error("❌ 不正确，请再思考一下。")
    
    elif question == "R² score的取值范围是什么?":
        answer = st.radio("选择正确答案:", [
            "0到1之间",
            "-∞到+∞",
            "-1到1之间"
        ], key="q4")
        if st.button("检查答案", key="b4"):
            if answer == "-∞到+∞":
                st.success("✅ 正确! R² score理论上可以取任何值，但通常接近1表示好拟合。")
            else:
                st.error("❌ 不正确，请再思考一下。")
    
    elif question == "残差分析的作用是什么?":
        answer = st.radio("选择正确答案:", [
            "检查模型假设是否成立",
            "增加模型复杂度",
            "减少计算时间"
        ], key="q5")
        if st.button("检查答案", key="b5"):
            if answer == "检查模型假设是否成立":
                st.success("✅ 正确! 残差分析帮助验证线性回归模型的假设。")
            else:
                st.error("❌ 不正确，请再思考一下。")
    
    return f"概念测验模块: 当前问题='{question}'"

# 实际应用案例模块
def real_world_example_section():
    st.header("🌍 实际应用案例")
    
    example = st.selectbox(
        "选择实际应用案例:",
        ["房价预测", "学生成绩预测", "销售额预测", "上传自己的数据"]
    )
    
    if example == "上传自己的数据":
        uploaded_file = st.file_uploader("上传CSV文件", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("数据预览:", data.head())
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("选择自变量(X)", data.columns)
            with col2:
                y_col = st.selectbox("选择因变量(y)", data.columns)
            
            X = data[x_col].values
            y = data[y_col].values
            
            analyze_custom_data(X, y, x_col, y_col)
            return f"实际应用模块: 上传自定义数据, X={x_col}, y={y_col}"
    else:
        X, y, description = load_example_dataset(example)
        st.write(description)
        
        analyze_custom_data(X, y, "X", "y")
        return f"实际应用模块: 使用{example}数据集"

# 分析自定义数据函数
def analyze_custom_data(X, y, x_name, y_name):
    if len(X) != len(y):
        st.error("X和y的长度不匹配!")
        return
    
    if len(X) < 2:
        st.error("数据点太少!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.7)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{y_name} vs {x_name}")
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("斜率(系数)", f"{model.coef_[0]:.4f}")
        st.metric("截距", f"{model.intercept_:.4f}")
    with col2:
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.metric("均方误差(MSE)", f"{mse:.4f}")
        st.metric("决定系数(R²)", f"{r2:.4f}")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(X, y, alpha=0.7, label='数据点')
    
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    ax2.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='拟合线')
    ax2.set_xlabel(x_name)
    ax2.set_ylabel(y_name)
    ax2.set_title(f"线性回归拟合: {y_name} = {model.coef_[0]:.2f}×{x_name} + {model.intercept_:.2f}")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)
    
    st.info(f"""
    **结果解释:**
    
    线性回归模型为: {y_name} = {model.coef_[0]:.2f}×{x_name} + {model.intercept_:.2f}
    
    - 斜率({model.coef_[0]:.2f}): {x_name}每增加1个单位，{y_name}平均变化{model.coef_[0]:.2f}个单位
    - 截距({model.intercept_:.2f}): 当{x_name}=0时，{y_name}的预测值
    - R²({r2:.2f}): 模型解释了{y_name}变异性的{r2*100:.1f}%
    """)

# 主程序
def main():
    # 初始化会话状态（新增：确保section变量存在）
    if 'section' not in st.session_state:
        st.session_state.section = "数据生成与探索"
    
    st.sidebar.title("导航菜单")
    # 用会话状态的section作为当前选中项
    section = st.sidebar.radio("选择学习模块", [
        "数据生成与探索",
        "手动拟合体验",
        "梯度下降可视化",
        "模型比较",
        "残差分析",
        "概念测验",
        "实际应用案例",
        "编程实例（糖尿病数据集）"  # 保留新增选项
    ])
  
    # 更新会话状态的section
    st.session_state.section = section
    
    context = ""
    if section == "数据生成与探索":
        context = data_generation_section()
    elif section == "手动拟合体验":
        context = manual_fitting_section()
    elif section == "梯度下降可视化":
        context = gradient_descent_section()
    elif section == "模型比较":
        context = model_comparison_section()
    elif section == "残差分析":
        context = residual_analysis_section()
    elif section == "概念测验":
        context = quiz_section()
    elif section == "实际应用案例":
        context = real_world_example_section()
    elif section == "编程实例（糖尿病数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        linear_regression_step_by_step.main()
        context = "编程实例模块: 糖尿病数据集线性回归分步练习"
    
    # 只有在非编程实例页面时显示聊天界面
    if section != "编程实例（糖尿病数据集）":
        display_chat_interface(context)
    
    # 侧边栏信息（保持不变）
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **线性回归交互式学习平台**
    
    设计用于机器学习教学，帮助学生理解:
    - 线性回归基本原理
    - 模型训练与评估
    - 梯度下降优化
    - 模型诊断与改进
    """)


if __name__ == "__main__":
    main()
