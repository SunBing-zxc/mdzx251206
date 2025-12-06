# streamlit run neural_network_step_by_step.py
# 加州房价预测 - 线性回归与神经网络对比

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing  # 加州房价数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.neural_network import MLPRegressor  # 神经网络回归模型
from sklearn.metrics import mean_squared_error, r2_score
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 特征名称中英文映射
FEATURE_NAME_MAP = {
    'MedInc': '收入中位数',
    'HouseAge': '房屋平均年龄',
    'AveRooms': '平均房间数',
    'AveBedrms': '平均卧室数',
    'Population': '人口数',
    'AveOccup': '平均住户人数',
    'Latitude': '纬度',
    'Longitude': '经度'
}

# 初始化会话状态
if 'step' not in st.session_state:
    st.session_state.step = 0  # 从0开始
if 'data' not in st.session_state:
    st.session_state.data = None  # 数据集
if 'X' not in st.session_state:
    st.session_state.X = None  # 特征
if 'y' not in st.session_state:
    st.session_state.y = None  # 目标变量
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'linear_model' not in st.session_state:
    st.session_state.linear_model = None
if 'nn_model' not in st.session_state:
    st.session_state.nn_model = None
if 'y_pred_linear' not in st.session_state:
    st.session_state.y_pred_linear = None
if 'y_pred_nn' not in st.session_state:
    st.session_state.y_pred_nn = None


# AI代码检查函数（适配回归模型）
def ai_code_checker(step, user_code):
    try:
        if step == 1:
            errors = []
            if 'fetch_california_housing' not in user_code:
                errors.append("❌ 请加载加州房价数据集（使用fetch_california_housing）")
            if 'shape' not in user_code:
                errors.append("❌ 请查看数据形状（使用.shape）")
            if 'mean' not in user_code or 'std' not in user_code:
                errors.append("❌ 请查看数据统计信息（均值、标准差等）")
            if 'corrcoef' not in user_code:
                errors.append("❌ 请计算特征相关性（使用np.corrcoef）")
            return "✅ 步骤1通过！" if not errors else "\n".join(errors)
        
        elif step == 2:
            errors = []
            if 'train_test_split' not in user_code:
                errors.append("❌ 请使用train_test_split划分训练集和测试集")
            if 'test_size=0.2' not in user_code:
                errors.append("❌ 请设置test_size=0.2")
            if 'random_state=42' not in user_code:
                errors.append("❌ 请设置random_state=42保证结果可复现")
            return "✅ 步骤2通过！" if not errors else "\n".join(errors)
        
        elif step == 3:
            errors = []
            if 'StandardScaler' not in user_code:
                errors.append("❌ 请用StandardScaler进行特征标准化")
            if 'fit_transform' not in user_code or 'transform' not in user_code:
                errors.append("❌ 应使用fit_transform处理训练集，transform处理测试集")
            return "✅ 步骤3通过！" if not errors else "\n".join(errors)
        
        elif step == 4:
            errors = []
            if 'LinearRegression' not in user_code or 'linear_model = LinearRegression()' not in user_code:
                errors.append("❌ 请实例化线性回归模型（linear_model = LinearRegression()）")
            if 'fit' not in user_code:
                errors.append("❌ 请训练模型（使用.fit()方法）")
            if 'predict' not in user_code:
                errors.append("❌ 请预测测试集结果（使用.predict()方法）")
            return "✅ 步骤4通过！" if not errors else "\n".join(errors)
        
        elif step == 5:
            errors = []
            if 'MLPRegressor' not in user_code or 'nn_model = MLPRegressor' not in user_code:
                errors.append("❌ 请实例化神经网络回归模型（nn_model = MLPRegressor()）")
            if 'hidden_layer_sizes' not in user_code:
                errors.append("❌ 请设置hidden_layer_sizes参数")
            if 'max_iter=200' not in user_code:
                errors.append("❌ 请设置max_iter=200")
            if 'random_state=42' not in user_code:
                errors.append("❌ 请设置random_state=42")
            return "✅ 步骤5通过！" if not errors else "\n".join(errors)
        
        elif step == 6:
            errors = []
            if 'mean_squared_error' not in user_code:
                errors.append("❌ 请用mean_squared_error计算均方误差")
            if 'r2_score' not in user_code:
                errors.append("❌ 请用r2_score计算R²分数")
            return "✅ 步骤6通过！" if not errors else "\n".join(errors)
    
    except Exception as e:
        return f"⚠️ 代码错误：{str(e)}"


# 步骤0：项目说明与数据展示
def step0():
    st.header("步骤0：项目说明")
    st.subheader("加州房价预测 - 线性回归与神经网络对比")
    
    # 项目目标
    st.info("""
    **数据集说明**：
    加州房价数据集包含加州地区的房价数据，共20640个样本，8个特征。
    目标是根据这些特征预测该地区的房屋中位数价格（单位：10万美元）。
    
    特征说明：
    - MedInc（收入中位数）：该地区住户的收入中位数
    - HouseAge（房屋平均年龄）：该地区房屋的平均年龄
    - AveRooms（平均房间数）：平均每个住户的房间数
    - AveBedrms（平均卧室数）：平均每个住户的卧室数
    - Population（人口数）：该地区的人口数
    - AveOccup（平均住户人数）：平均每个住户的人数
    - Latitude（纬度）：该地区的纬度
    - Longitude（经度）：该地区的经度
    
    **项目目标**：
    通过两种回归方法（线性回归和神经网络）对加州房价进行预测，
    对比两种方法的预测效果，理解不同模型的特点和适用场景。
    """)
    
    # 加载数据集
    housing = fetch_california_housing()
    st.session_state.data = housing
    
    # 数据集展示
    st.subheader("数据集介绍")
    st.write("""
    该数据集包含20640个样本，8个特征，目标变量为房屋中位数价格。
    以下是部分样本数据：
    """)
    
    # 构建特征数据DataFrame，使用中文列名
    df = pd.DataFrame(
        data=housing.data,
        columns=[FEATURE_NAME_MAP[name] for name in housing.feature_names]  # 使用中文特征名称
    )
    # 添加目标值列（房价）
    df['房价（10万美元）'] = housing.target
    
    # 显示前10条数据，隐藏索引列
    st.set_page_config(layout="wide")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    
    st.button("进入步骤1：数据观察与理解",
             on_click=lambda: setattr(st.session_state, 'step', 1))


# 步骤1：数据观察与理解
def step1():
    st.header("步骤1：数据观察与理解")
    st.subheader("目标：加载数据集，观察数据基本信息和特征相关性")
    
    st.info("""
    **任务说明**：
    1. 加载加州房价数据集
    2. 观察数据形状、基本统计信息
    3. 分析特征之间的相关性，特别是与目标变量的相关性
    """)
    
    # 代码骨架
    code_skeleton = """
# 1. 加载数据
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()

# 特征名称中英文映射
feature_name_map = {
    'MedInc': '收入中位数',
    'HouseAge': '房屋平均年龄',
    'AveRooms': '平均房间数',
    'AveBedrms': '平均卧室数',
    'Population': '人口数',
    'AveOccup': '平均住户人数',
    'Latitude': '纬度',
    'Longitude': '经度'
}

# 提取特征和目标变量
X = housing.data  # 特征数据
y = housing.target  # 目标变量（房价）
feature_names = housing.feature_names  # 英文特征名称

# 中文特征名称
chinese_feature_names = []
for name in feature_names:
    chinese_feature_names.append(feature_name_map[name])

# 2. 查看数据基本信息
print("数据形状：", X.shape)  # 样本数和特征数
print("特征名称（中文）：", chinese_feature_names)
print("样本数量：", X.shape[0])
print("特征数量：", X.shape[1])

# 3. 查看数据统计信息
print("数据统计信息：")
for i, name in enumerate(chinese_feature_names):
    print(f"{name}:")
    print(f"  均值: {X[:, i].mean():.4f}")
    print(f"  标准差: {X[:, i].std():.4f}")
    print(f"  最小值: {X[:, i].min():.4f}")
    print(f"  最大值: {X[:, i].max():.4f}")
    print(f"  中位数: {np.median(X[:, i]):.4f}")

# 目标变量统计信息
print("目标变量（房价）统计信息：")
print(f"  均值: {y.mean():.4f}")
print(f"  标准差: {y.std():.4f}")
print(f"  最小值: {y.min():.4f}")
print(f"  最大值: {y.max():.4f}")
print(f"  中位数: {np.median(y):.4f}")

# 4. 分析特征相关性
# 将目标变量添加到数据中以便计算相关性
data_with_target = np.column_stack((X, y))
correlation = np.corrcoef(data_with_target, rowvar=False)

# 与目标变量的相关性（最后一行/列）
target_corr = correlation[-1, :-1]  # 排除与自身的相关性

print("与目标变量的相关性：")
for name, corr in zip(chinese_feature_names, target_corr):
    print(f"{name}: {corr:.4f}")

# 绘制相关性热力图
plt.figure(figsize=(10, 8))
im = plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='相关系数')

# 添加特征名称（使用中文）
names = chinese_feature_names + ['房价']
plt.xticks(range(len(names)), names, rotation=45)
plt.yticks(range(len(names)), names)

# 在热力图上添加数值
for i in range(len(names)):
    for j in range(len(names)):
        plt.text(j, i, f"{correlation[i, j]:.2f}", 
                 ha='center', va='center', color='white')

plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=2000, key="step1_code")
    
    if st.button("运行代码", key="run_step1"):
        try:
            locals_dict = {
                'np': np, 'plt': plt,
                'fetch_california_housing': fetch_california_housing
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存数据
            st.session_state.X = locals_dict['X']
            st.session_state.y = locals_dict['y']
            st.session_state.feature_names = locals_dict['feature_names']
            st.session_state.chinese_feature_names = locals_dict['chinese_feature_names']
            
            # 展示结果
            st.success("代码运行成功！")
            with st.expander("查看输出"):
                st.write(f"数据形状：{locals_dict['X'].shape}")
                st.write("特征名称（中文）：", locals_dict['chinese_feature_names'])
                
                st.write("数据统计信息：")
                for i, name in enumerate(locals_dict['chinese_feature_names']):
                    st.write(f"{name}:")
                    st.write(f"  均值: {locals_dict['X'][:, i].mean():.4f}")
                    st.write(f"  标准差: {locals_dict['X'][:, i].std():.4f}")
                    st.write(f"  最小值: {locals_dict['X'][:, i].min():.4f}")
                    st.write(f"  最大值: {locals_dict['X'][:, i].max():.4f}")
                    st.write(f"  中位数: {np.median(locals_dict['X'][:, i]):.4f}")
                
                st.write("与目标变量的相关性：")
                data_with_target = np.column_stack((locals_dict['X'], locals_dict['y']))
                correlation = np.corrcoef(data_with_target, rowvar=False)
                target_corr = correlation[-1, :-1]
                for name, corr in zip(locals_dict['chinese_feature_names'], target_corr):
                    st.write(f"{name}: {corr:.4f}")
                
                st.pyplot(locals_dict['plt'])

                st.info("""可见，收入中位数与房价呈现最强的正相关（相关系数通常在 0.6 左右），说明该
                        地区居民收入水平是影响房价的核心因素，收入越高的区域房价普遍越高。
                            """)
            
            # AI反馈
            ai_feedback = ai_code_checker(1, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤2：数据集划分",
                         on_click=lambda: setattr(st.session_state, 'step', 2))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤2：数据集划分
def step2():
    st.header("步骤2：数据集划分")
    st.subheader("目标：将数据集划分为训练集和测试集")
    
    if st.session_state.X is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：
    1. 将数据集划分为训练集（80%）和测试集（20%）
    2. 训练集用于模型训练，测试集用于评估模型泛化能力
    3. 设置random_state保证结果可复现
    """)
    
    # 代码骨架
    code_skeleton = """
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
import numpy as np

# 划分训练集和测试集（测试集占20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 测试集比例
    random_state=42  # 随机种子，保证结果可复现
)

# 查看划分后的数据集大小
print("训练集样本数：", X_train.shape[0])
print("测试集样本数：", X_test.shape[0])
print("特征数：", X_train.shape[1])
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=400, key="step2_code")
    
    if st.button("运行代码", key="run_step2"):
        try:
            locals_dict = {
                'X': st.session_state.X, 
                'y': st.session_state.y,
                'train_test_split': train_test_split,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存划分后的数据集
            st.session_state.X_train = locals_dict['X_train']
            st.session_state.X_test = locals_dict['X_test']
            st.session_state.y_train = locals_dict['y_train']
            st.session_state.y_test = locals_dict['y_test']
            
            st.success("数据集划分结果：")
            st.write(f"训练集样本数：{locals_dict['X_train'].shape[0]}")
            st.write(f"测试集样本数：{locals_dict['X_test'].shape[0]}")
            st.write(f"特征数：{locals_dict['X_train'].shape[1]}")
            
            ai_feedback = ai_code_checker(2, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤3：特征标准化",
                         on_click=lambda: setattr(st.session_state, 'step', 3))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤3：特征标准化
def step3():
    st.header("步骤3：特征标准化")
    st.subheader("目标：对特征进行标准化处理（尤其对神经网络重要）")
    
    if st.session_state.X_train is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：
    1. 特征标准化可以使不同量级的特征具有相同的尺度
    2. 对线性回归影响较小，但对神经网络模型非常重要
    3. 使用StandardScaler将特征转换为均值为0，方差为1的分布
    4. 注意：只用训练集拟合标准化器，再分别转换训练集和测试集
    """)
    
    # 代码骨架
    code_skeleton = """
# 导入标准化工具
from sklearn.preprocessing import StandardScaler
import numpy as np

# 初始化标准化器
scaler = StandardScaler()

# 用训练集拟合标准化器，并转换训练集
X_train_scaled = scaler.fit_transform(X_train)

# 用同样的标准化器转换测试集（不要重新拟合）
X_test_scaled = scaler.transform(X_test)

# 查看标准化效果（以第一个特征为例）
print(f"原始训练集{chinese_feature_names[0]}的均值：", X_train[:, 0].mean())
print(f"标准化后训练集{chinese_feature_names[0]}的均值：", X_train_scaled[:, 0].mean().round(4))
print(f"原始训练集{chinese_feature_names[0]}的方差：", X_train[:, 0].var())
print(f"标准化后训练集{chinese_feature_names[0]}的方差：", X_train_scaled[:, 0].var().round(4))
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=460, key="step3_code")
    
    if st.button("运行代码", key="run_step3"):
        try:
            locals_dict = {
                'X_train': st.session_state.X_train,
                'X_test': st.session_state.X_test,
                'chinese_feature_names': st.session_state.chinese_feature_names,
                'StandardScaler': StandardScaler,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存标准化后的数据
            st.session_state.X_train_scaled = locals_dict['X_train_scaled']
            st.session_state.X_test_scaled = locals_dict['X_test_scaled']
            st.session_state.scaler = locals_dict['scaler']
            
            st.success("特征标准化完成！")
            st.write(f"标准化后训练集{st.session_state.chinese_feature_names[0]}的均值：{locals_dict['X_train_scaled'][:, 0].mean().round(4)}")
            st.write(f"标准化后训练集{st.session_state.chinese_feature_names[0]}的方差：{locals_dict['X_train_scaled'][:, 0].var().round(4)}")
            
            ai_feedback = ai_code_checker(3, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤4：线性回归模型",
                         on_click=lambda: setattr(st.session_state, 'step', 4))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤4：线性回归模型
def step4():
    st.header("步骤4：线性回归模型")
    st.subheader("目标：构建并训练线性回归模型")
    
    if 'X_train_scaled' not in st.session_state:
        st.warning("请先完成步骤3！")
        st.button("返回步骤3", on_click=lambda: setattr(st.session_state, 'step', 3))
        return
    
    st.info("""
    **任务说明**：
    1. 线性回归是最简单的回归模型，假设特征与目标变量之间存在线性关系
    2. 模型形式：y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
    3. 训练线性回归模型并在测试集上进行预测
    """)
    
    code_skeleton = """
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
import numpy as np

# 实例化线性回归模型
linear_model = LinearRegression()

# 训练模型（使用标准化后的特征）
linear_model.fit(X_train_scaled, y_train)

# 查看模型系数
print("线性回归系数（权重）：")
for i, (name, coef) in enumerate(zip(chinese_feature_names, linear_model.coef_)):
    print(f"{name}: {coef:.4f}")
print(f"截距：{linear_model.intercept_:.4f}")

# 在测试集上进行预测
y_pred_linear = linear_model.predict(X_test_scaled)

# 查看部分预测结果
print("部分预测结果（实际值 vs 预测值）：")
for i in range(5):
    print(f"实际值: {y_test[i]:.4f}, 预测值: {y_pred_linear[i]:.4f}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=580, key="step4_code")
    
    if st.button("运行代码", key="run_step4"):
        try:
            locals_dict = {
                'X_train_scaled': st.session_state.X_train_scaled,
                'X_test_scaled': st.session_state.X_test_scaled,
                'y_train': st.session_state.y_train,
                'y_test': st.session_state.y_test,
                'chinese_feature_names': st.session_state.chinese_feature_names,
                'LinearRegression': LinearRegression,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存模型和预测结果
            st.session_state.linear_model = locals_dict['linear_model']
            st.session_state.y_pred_linear = locals_dict['y_pred_linear']
            
            st.success("线性回归模型训练完成！")
            # 显示模型系数
            st.write("线性回归系数（权重）：")
            for i, (name, coef) in enumerate(zip(
                st.session_state.chinese_feature_names, locals_dict['linear_model'].coef_
            )):
                st.write(f"{name}: {coef:.4f}")
            st.write(f"截距：{locals_dict['linear_model'].intercept_:.4f}")
            
            # 显示部分预测结果
            st.write("部分预测结果：")
            for i in range(10):
                st.write(f"实际值: {locals_dict['y_test'][i]:.4f}, 预测值: {locals_dict['y_pred_linear'][i]:.4f}")
            
            ai_feedback = ai_code_checker(4, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤5：神经网络模型",
                         on_click=lambda: setattr(st.session_state, 'step', 5))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤5：神经网络模型
def step5():
    st.header("步骤5：神经网络模型")
    st.subheader("目标：构建并训练神经网络回归模型")
    
    if 'linear_model' not in st.session_state:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：
    1. 神经网络可以捕捉特征与目标变量之间的非线性关系
    2. 使用MLPRegressor（多层感知器回归器）构建神经网络
    3. 设置隐藏层结构，训练模型并在测试集上进行预测
    """)
    
    code_skeleton = """
# 导入神经网络回归模型
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# 实例化神经网络模型
# hidden_layer_sizes指定隐藏层结构，例如(64, 32)表示两个隐藏层，分别有64和32个神经元
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),  # 隐藏层结构
    activation='relu',  # 激活函数
    solver='adam',  # 优化器
    max_iter=200,  # 最大迭代次数
    random_state=42,  # 随机种子
    verbose=False  # 是否打印训练过程
)

# 训练模型（使用标准化后的特征）
nn_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred_nn = nn_model.predict(X_test_scaled)

# 查看部分预测结果
print("部分预测结果（实际值 vs 神经网络预测值）：")
for i in range(5):
    print(f"实际值: {y_test[i]:.4f}, 预测值: {y_pred_nn[i]:.4f}")

# 绘制神经网络的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(nn_model.loss_curve_)
plt.title('神经网络训练损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.grid(True)
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=850, key="step5_code")
    
    if st.button("运行代码", key="run_step5"):
        try:
            locals_dict = {
                'X_train_scaled': st.session_state.X_train_scaled,
                'X_test_scaled': st.session_state.X_test_scaled,
                'y_train': st.session_state.y_train,
                'y_test': st.session_state.y_test,
                'MLPRegressor': MLPRegressor,
                'plt': plt,
                'np': np
            }

            # 显示训练中提示
            with st.spinner("神经网络训练中，请稍候..."):
                exec(user_code, globals(), locals_dict)
            
            # 保存模型和预测结果
            st.session_state.nn_model = locals_dict['nn_model']
            st.session_state.y_pred_nn = locals_dict['y_pred_nn']
            
            st.success("神经网络模型训练完成！")
            
            # 显示部分预测结果
            st.write("部分预测结果对比：")
            for i in range(10):
                st.write(f"实际值: {locals_dict['y_test'][i]:.4f}, "
                         f"神经网络预测值: {locals_dict['y_pred_nn'][i]:.4f}, "
                         f"线性回归预测值: {st.session_state.y_pred_linear[i]:.4f}")
            
            # 显示损失曲线
            st.pyplot(locals_dict['plt'])
            
            ai_feedback = ai_code_checker(5, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤6：模型评估与对比",
                         on_click=lambda: setattr(st.session_state, 'step', 6))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤6：模型评估与对比
def step6():
    st.header("步骤6：模型评估与对比")
    st.subheader("目标：评估两种模型的性能并进行对比分析")
    
    if 'nn_model' not in st.session_state:
        st.warning("请先完成步骤5！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
    st.info("""
    **任务说明**：
    1. 使用均方误差（MSE）和R²分数评估模型性能
    2. 均方误差越小越好，R²分数越接近1越好
    3. 对比线性回归和神经网络的性能差异
    4. 可视化预测结果与实际值的关系
    """)
    
    code_skeleton = """
# 导入评估指标
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 评估线性回归模型
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

# 评估神经网络模型
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_r2 = r2_score(y_test, y_pred_nn)

# 打印评估结果
print("线性回归模型评估：")
print(f"均方误差（MSE）：{linear_mse:.4f}")
print(f"R²分数：{linear_r2:.4f}")

print("神经网络模型评估：")
print(f"均方误差（MSE）：{nn_mse:.4f}")
print(f"R²分数：{nn_r2:.4f}")

# 可视化预测结果
plt.figure(figsize=(12, 5))

# 线性回归预测 vs 实际值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('线性回归：预测值 vs 实际值')
plt.xlabel('实际房价')
plt.ylabel('预测房价')

# 神经网络预测 vs 实际值
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_nn, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('神经网络：预测值 vs 实际值')
plt.xlabel('实际房价')
plt.ylabel('预测房价')

plt.tight_layout()
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=1050, key="step6_code")
    
    if st.button("运行代码", key="run_step6"):
        try:
            locals_dict = {
                'y_test': st.session_state.y_test,
                'y_pred_linear': st.session_state.y_pred_linear,
                'y_pred_nn': st.session_state.y_pred_nn,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'plt': plt,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存评估结果
            st.session_state.linear_mse = locals_dict['linear_mse']
            st.session_state.linear_r2 = locals_dict['linear_r2']
            st.session_state.nn_mse = locals_dict['nn_mse']
            st.session_state.nn_r2 = locals_dict['nn_r2']
            
            st.success("模型评估完成！")
            
            # 显示评估指标对比

            st.write("模型评估指标对比：")
            col1,col2=st.columns(2)
            with col1:
                st.write(f"线性回归 - 均方误差（MSE）：{locals_dict['linear_mse']:.4f}")
                st.write(f"线性回归 - R²分数：{locals_dict['linear_r2']:.4f}")
            with col2:
                st.write(f"神经网络 - 均方误差（MSE）：{locals_dict['nn_mse']:.4f}")
                st.write(f"神经网络 - R²分数：{locals_dict['nn_r2']:.4f}")
            
            # 显示可视化结果
            st.pyplot(locals_dict['plt'])
            
            ai_feedback = ai_code_checker(6, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.subheader("恭喜！已用sklearn库完成加州房价数据集的神经网络回归全流程")
                st.button("进入步骤7：总结与思考",
                         on_click=lambda: setattr(st.session_state, 'step', 7))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤7：总结与思考
def step7():
    st.header("步骤7：总结与思考")
    st.subheader("目标：总结两种回归方法的特点，理解神经网络的优势与局限")
    
    # 检查前置条件
    if st.session_state.step < 6:
        st.warning("请先完成前面所有步骤再进行总结！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return

    st.info("""
    **任务说明**：
    1. 对比线性回归和神经网络在房价预测任务上的表现
    2. 分析两种模型的优缺点和适用场景
    3. 思考如何进一步改进模型性能
    """)

    # 显示步骤6的评估指标
    st.subheader("📊 模型评估结果回顾")
    metrics_available = all(key in st.session_state for key in 
                          ['linear_mse', 'linear_r2', 'nn_mse', 'nn_r2'])
    if metrics_available:
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("线性回归 - 均方误差", f"{st.session_state.linear_mse:.4f}")
            st.metric("线性回归 - R²分数", f"{st.session_state.linear_r2:.4f}")
        with metrics_cols[1]:
            st.metric("神经网络 - 均方误差", f"{st.session_state.nn_mse:.4f}")
            st.metric("神经网络 - R²分数", f"{st.session_state.nn_r2:.4f}")
    else:
        st.warning("未找到完整的模型评估数据，请确保已完成步骤6")
    


    # 学生思考输入（关键问题）
    st.write("### 思考与分析")
    student_answer = st.text_area(
        """请思考以下问题：
            对比两种模型的性能，哪种模型在房价预测任务中表现更好？为什么？
        """,
        height=150,
        key="student_answer"
    )
    
    # 调用AI评价
    if st.button("获取AI评价", key="get_ai_feedback"):
        if not student_answer.strip():
            st.warning("请先输入你的思考内容")
            return
        
        if not metrics_available:
            st.warning("未找到评估数据，请先完成步骤6的模型评估")
            return
        
        # 构建包含评估指标和学生回答的上下文
        context = f"""
        模型评估指标数据：
        - 线性回归 - 均方误差：{st.session_state.linear_mse:.4f}
        - 线性回归 - R²分数：{st.session_state.linear_r2:.4f}
        - 神经网络 - 均方误差：{st.session_state.nn_mse:.4f}
        - 神经网络 - R²分数：{st.session_state.nn_r2:.4f}
        
        学生的分析回答：
        {student_answer}
        """
        
        # 调用AI评价的代码
        from neural_network_demo import ask_ai_assistant  # 导入复用的AI助手函数
        response = ask_ai_assistant(
            question="请结合提供的两种评估指标和学生的分析，先解读评估指标含义，重点评价学生分析的合理性，并补充专业建议。",
            context=context.strip()
        )
        
        # 显示AI评价
        st.write("### AI评价与分析")
        st.info(response)

    # 重新开始按钮
    st.button("重新开始全部流程", on_click=lambda: setattr(st.session_state, 'step', 0))


# 主程序
def main():
    st.title("📝 神经网络 vs 线性回归")
    st.title("（加州房价预测）")
    st.write("基于加州房价数据集，对比线性回归与神经网络的预测效果，理解不同模型的特点")

    # 初始化会话状态变量（新增）
    if 'ai_feedback' not in st.session_state:
        st.session_state.ai_feedback = ""  # 初始化为空字符串
    
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 数据集划分", "3. 特征标准化",
        "4. 线性回归模型", "5. 神经网络模型", "6. 模型评估", "7. 总结与思考"
    ]
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✅ {step}")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"▶️ **{step}**")
        else:
            st.sidebar.markdown(f"◻️ {step}")
    
    # 核心：根据当前步骤显示对应主内容
    if st.session_state.step == 0:
        step0()
    elif st.session_state.step == 1:
        step1()
    elif st.session_state.step == 2:
        step2()
    elif st.session_state.step == 3:
        step3()
    elif st.session_state.step == 4:
        step4()
    elif st.session_state.step == 5:
        step5()
    elif st.session_state.step == 6:
        step6()
    elif st.session_state.step == 7:
        step7()


if __name__ == "__main__":
    main()
