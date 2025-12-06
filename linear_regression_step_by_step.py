# streamlit run linear_regression_step_by_step.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 新增：用于表格展示
from sklearn.datasets import load_diabetes    # 替换为糖尿病数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化会话状态（修改：初始步骤改为0）
if 'step' not in st.session_state:
    st.session_state.step = 0  # 从0开始
if 'data' not in st.session_state:
    st.session_state.data = None    # 特征数据（numpy数组）
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None    # 特征名称
if 'X' not in st.session_state:
    st.session_state.X = None    # 特征
if 'y' not in st.session_state:
    st.session_state.y = None    # 目标变量（疾病进展）
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = {}    # 存储各步骤代码
# 新增：存储原始数据集供步骤0展示
if 'raw_dataset' not in st.session_state:
    st.session_state.raw_dataset = None


# 改进后的AI代码检查函数（精确检查填空内容）
def ai_code_checker(step, user_code):
    try:
        if step == 1:
            errors = []
           
            # 检查shape填空
            if 'X_raw.shape' not in user_code and 'X_raw. shape' not in user_code:  # 允许空格误差
                errors.append("❌ 数据形状应使用X_raw.shape（提示：.shape）")
            
            # 检查切片填空
            if 'X_raw[0:5]' not in user_code or '0:' not in user_code:
                errors.append("❌ 前5行特征应使用X_raw[0:5]（提示：切片0:5）")
            
            # 检查均值填空
            if 'np.mean(y_raw)' not in user_code:
                errors.append("❌ 目标变量均值应使用np.mean(y_raw)（提示：参数是y_raw）")
            
            # 检查标准差填空
            if 'X_raw[:, 0]' not in user_code:
                errors.append("❌ 第一个特征标准差应使用X_raw[:, 0]（提示：[:,0]取第一列）")
            
            if not errors:
                return "✅ 步骤1通过！成功观察数据"
            return "\n".join(errors)
        
        elif step == 2:
            errors = []
            if not errors:
                return "✅ 步骤2通过！特征与目标划分正确"
            return "\n".join(errors)
        
        elif step == 3:
            errors = []
            if 'train_test_split' not in user_code:
                errors.append("❌ 请用train_test_split划分训练集和测试集")
            if 'test_size=0.2' not in user_code:
                errors.append("❌ train_test_split的test_size参数应为0.2")
            if 'random_state=42' not in user_code:
                errors.append("❌ train_test_split的random_state参数应为42")
            if 'scaler.fit_transform(X_train)' not in user_code:
                errors.append("❌ 训练集标准化应使用scaler.fit_transform(X_train)")
            if 'scaler.transform(X_test)' not in user_code:
                errors.append("❌ 测试集标准化应使用scaler.transform(X_test)")
            
            if not errors:
                return "✅ 步骤3通过！数据预处理完成"
            return "\n".join(errors)
        
        elif step == 4:
            if 'LinearRegression()' not in user_code:
                return "❌ 请实例化LinearRegression模型（如model = LinearRegression()）"
            if 'model = LinearRegression()' not in user_code:
                return "❌ 模型实例化应为model = LinearRegression()"
            return "✅ 步骤4通过！模型构建正确"
        
        elif step == 5:
            errors = []
            if '.fit(' not in user_code or 'X_train' not in user_code:
                errors.append("❌ 请用model.fit(X_train, y_train)训练模型")
            if 'model.fit(X_train_scaled, y_train)' not in user_code:
                errors.append("❌ 训练模型应为model.fit(X_train_scaled, y_train)")
            if '.predict(' not in user_code or 'X_test' not in user_code:
                errors.append("❌ 请用model.predict(X_test)生成预测结果")
            if 'model.predict(X_test_scaled)' not in user_code:
                errors.append("❌ 预测应为model.predict(X_test_scaled)")
            
            if not errors:
                return "✅ 步骤5通过！训练和预测完成"
            return "\n".join(errors)
        
        elif step == 6:
            errors = []
            if 'mean_squared_error' not in user_code and 'r2_score' not in user_code:
                errors.append("❌ 请用MSE或R²评估模型")
            if 'mean_squared_error(y_test, y_pred)' not in user_code:
                errors.append("❌ MSE计算应为mean_squared_error(y_test, y_pred)")
            if 'r2_score(y_test, y_pred)' not in user_code:
                errors.append("❌ R²计算应为r2_score(y_test, y_pred)")
            
            if not errors:
                return "✅ 步骤6通过！模型评估完成"
            return "\n".join(errors)
    
    except Exception as e:
        return f"⚠️ 代码错误：{str(e)}。提示：检查numpy数组索引是否正确"


# 新增：步骤0：项目说明
def step0():
    st.header("步骤0：项目说明")
    st.subheader("糖尿病疾病进展预测分析")
    
    # 项目目标
    st.info("""
    **项目目标**：  
    通过患者的生理特征数据，建立线性回归模型预测糖尿病患者一年后的疾病进展情况。
    本项目将引导你完成从数据加载、预处理到模型构建、评估的完整机器学习流程，
    帮助你理解如何用线性回归解决实际医疗预测问题。
    """)
    
    # 加载数据集用于展示
    diabetes = load_diabetes()
    st.session_state.raw_dataset = diabetes  # 保存数据集供后续使用
    
    # 数据集介绍
    st.subheader("数据集介绍")
    st.write("""
    该数据集包含442名糖尿病患者的临床数据，包含10项生理特征和1项目标变量。
    所有特征均已进行预处理（均值为0，标准差为1），适合直接用于线性回归模型。
    以下是数据集的部分样本展示：
    """)
    
    # 转换为DataFrame展示表格
    df = pd.DataFrame(
        data=diabetes.data,
        columns=diabetes.feature_names
    )
    # 添加目标变量列
    df['疾病进展（目标变量）'] = diabetes.target
    
    # 展示前10行数据
    st.dataframe(df.head(10), use_container_width=True)
    
    # 字段说明
    st.subheader("字段说明")
    st.write("**生理特征字段解释：**")
    field_explanations = {
        'age': '年龄（岁）',
        'sex': '性别（1=男性，2=女性）',
        'bmi': '体重指数（Body Mass Index）',
        'bp': '平均血压（Blood Pressure）',
        's1': '血清总胆固醇（TC）',
        's2': '低密度脂蛋白（LDL）',
        's3': '高密度脂蛋白（HDL）',
        's4': '总胆固醇/高密度脂蛋白比值',
        's5': '血清甘油三酯（TG）',
        's6': '血糖水平'
    }
    
    # 以表格形式展示字段说明
    explanation_df = pd.DataFrame(
        list(field_explanations.items()),
        columns=['特征字段', '说明']
    )
    st.dataframe(explanation_df, use_container_width=True)
    
    st.info("""
    **目标变量说明**：  
    疾病进展（目标变量）：表示患者从基线时间点开始，一年后的糖尿病病情进展指标，
    数值越高表示病情进展越明显。
    """)
    
    # 进入下一步按钮
    st.button("进入步骤1：数据观察",
             on_click=lambda: setattr(st.session_state, 'step', 1))


# 步骤1：数据观察与理解（糖尿病数据集版）
def step1():
    st.header("步骤1：数据观察与理解")
    st.subheader("目标：加载糖尿病数据集，用numpy观察基本信息")
    
      # 数据集说明单独展示（上下布局）
    st.info("""
    **数据集说明**：  
    糖尿病数据集包含442名患者的10项生理特征（如年龄、体重、血压等），目标变量是“一年后的疾病进展指标”。  
    特征已被标准化处理，适合直接用于线性回归。
    """)
    
      # 代码骨架（糖尿病数据集版）
    code_skeleton = """
# 加载糖尿病数据集
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_raw = diabetes.data      # 特征数据（numpy数组，形状：(442, 10)）
y_raw = diabetes.target    # 目标变量（疾病进展，形状：(442,)）
feature_names = diabetes.feature_names    # 特征名称（10个生理指标）

# 观察数据（补充代码）
print("数据形状：", X_raw.________)    # 提示：.shape
print("前5行特征：\\n", X_raw[________:________])    # 提示：切片取前5行（0:5）
print("特征名称：", feature_names)    # 如['age', 'sex', 'bmi'等]

# 计算统计量（选做）
print("目标变量（疾病进展）的均值：", np.mean(________))    # 提示：目标变量是y_raw
print("第一个特征（年龄）的标准差：", np.std(X_raw[:, ________]))    # 提示：[:,0]取第一列
    """.strip()
    
      # 代码区域加宽，高度增加以提供更大填空空间
    user_code = st.text_area(
        "请补充代码（numpy操作）：", 
        code_skeleton, 
        height=400,    # 增加高度
        key="step1_code"
    )
    
    if st.button("运行代码", key="run_step1"):
        try:
            locals_dict = {'np': np, 'load_diabetes': load_diabetes}
            exec(user_code, globals(), locals_dict)
            
              # 保存数据供后续步骤
            st.session_state.data = locals_dict['X_raw']
            st.session_state.y_raw = locals_dict['y_raw']
            st.session_state.feature_names = locals_dict['feature_names']
            
              # 展示结果（numpy数组转列表便于阅读）
            st.success("代码运行成功！输出结果：")
            with st.expander("查看输出"):
                st.write(f"数据形状：{locals_dict['X_raw'].shape}（442个样本，10个特征）")
                st.write("前5行特征：")
                st.write(locals_dict['X_raw'][:5].tolist())    # 转列表展示
                st.write(f"特征名称：{locals_dict['feature_names']}")
            
              # AI反馈
            ai_feedback = ai_code_checker(1, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[1] = user_code
            if "✅" in ai_feedback:
                st.button("进入步骤2：特征与目标变量划分", 
                         on_click=lambda: setattr(st.session_state, 'step', 2))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(1, user_code)}")


# 步骤2：特征与目标变量划分（糖尿病数据集版）
def step2():
    st.header("步骤2：特征与目标变量划分")
    st.subheader("目标：用numpy切片提取特征（X）和目标变量（y）")
    
    if st.session_state.data is None:
        st.warning("请先完成步骤1加载数据！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
      # 任务说明单独展示（上下布局）
    st.info("""
    **任务说明**：  
    1. 特征（X）：直接使用数据集的特征数据（X_raw，10个生理指标）  
    2. 目标变量（y）：疾病进展指标（y_raw）  
    提示：numpy中用变量直接赋值即可（无需切片，因X_raw已包含所有特征）
    """)
    
      # 展示特征名称和数据形状
    st.write(f"特征名称：{st.session_state.feature_names}")
    st.write(f"数据形状：{st.session_state.data.shape}（样本数, 特征数）")
    
      # 代码骨架（糖尿病数据集版）
    code_skeleton = f"""
# 划分特征（X）和目标变量（y）
# 用numpy变量直接赋值（无需切片，因X_raw已包含所有特征）
X = X_raw    # 特征（10个生理指标）
y = y_raw    # 目标变量（疾病进展）

# 查看形状
print("X形状：", X.shape)    # 应是(442, 10)
print("y形状：", y.shape)    # 应是(442,)
    """.strip()
    
      # 加宽代码区域
    user_code = st.text_area(
        "请补充代码（numpy操作）：", 
        code_skeleton, 
        height=300,    # 增加高度
        key="step2_code"
    )
    
    if st.button("运行代码", key="run_step2"):
        try:
            locals_dict = {
                'X_raw': st.session_state.data,
                'y_raw': st.session_state.y_raw
            }
            exec(user_code, globals(), locals_dict)
            
            st.session_state.X = locals_dict['X']
            st.session_state.y = locals_dict['y']
            
            st.success("划分结果：")
            st.write(f"X形状：{locals_dict['X'].shape}，y形状：{locals_dict['y'].shape}")
            st.write("前3个样本的特征：", locals_dict['X'][:3].tolist())
            st.write("前3个样本的目标值：", locals_dict['y'][:3].tolist())
            
            ai_feedback = ai_code_checker(2, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[2] = user_code
            if "✅" in ai_feedback:
                st.button("进入步骤3：数据预处理", 
                         on_click=lambda: setattr(st.session_state, 'step', 3))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(2, user_code)}")


# 步骤3：数据预处理
def step3():
    st.header("步骤3：数据预处理")
    st.subheader("目标：划分训练集/测试集，用StandardScaler标准化")
    
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("请先完成步骤2划分X和y！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
      # 任务说明单独展示
    st.info("""
    **任务说明**：  
    1. 用train_test_split划分训练集（X_train, y_train）和测试集（X_test, y_test）  
    2. 用StandardScaler对特征标准化（训练集fit_transform，测试集transform）  
    提示：糖尿病数据集特征已初步标准化，但实践中仍需统一尺度
    """)
    
      # 代码骨架（通用）
    code_skeleton = """
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 补充参数（测试集数据占20%，随机数种子为42）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=________, random_state=________)

# 特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.________(X_train)   # 提示：训练集用fit_transform
X_test_scaled = scaler.________(X_test)    # 提示：测试集用transform

print("训练集特征形状：", X_train_scaled.shape)
print("测试集特征形状：", X_test_scaled.shape)
    """.strip()
    
      # 加宽代码区域
    user_code = st.text_area(
        "请补充代码：", 
        code_skeleton, 
        height=400,    # 增加高度
        key="step3_code"
    )
    
    if st.button("运行代码", key="run_step3"):
        try:
            locals_dict = {
                'X': st.session_state.X,
                'y': st.session_state.y,
                'train_test_split': train_test_split,
                'StandardScaler': StandardScaler
            }
            exec(user_code, globals(), locals_dict)
            
              # 保存预处理后的数据
            st.session_state.X_train = locals_dict['X_train_scaled']
            st.session_state.X_test = locals_dict['X_test_scaled']
            st.session_state.y_train = locals_dict['y_train']
            st.session_state.y_test = locals_dict['y_test']
            
            st.success("预处理完成：")
            st.write(f"训练集样本数：{locals_dict['X_train_scaled'].shape[0]}，测试集样本数：{locals_dict['X_test_scaled'].shape[0]}")
            st.write("标准化后训练集前2行：", locals_dict['X_train_scaled'][:2].tolist())
            
            ai_feedback = ai_code_checker(3, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[3] = user_code
            if "✅" in ai_feedback:
                st.button("进入步骤4：构建线性回归模型", 
                         on_click=lambda: setattr(st.session_state, 'step', 4))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(3, user_code)}")


def step4():
    st.header("步骤4：构建线性回归模型")
    st.subheader("目标：实例化sklearn的LinearRegression模型")
    
      # 任务说明单独展示
    st.info("""
    **任务说明**：  
    1. 从sklearn.linear_model导入LinearRegression  
    2. 实例化模型（默认参数即可，无需修改）  
    提示：模型将处理numpy数组作为输入
    """)
    
    code_skeleton = """
# 导入线性回归模型
from sklearn.linear_model import LinearRegression

# 实例化模型
model = LinearRegression()

# 查看模型参数
print("模型参数：", model.get_params())
    """.strip()
    
      # 加宽代码区域
    user_code = st.text_area(
        "请补充代码：", 
        code_skeleton, 
        height=250,    # 增加高度
        key="step4_code"
    )
    
    if st.button("运行代码", key="run_step4"):
        try:
            locals_dict = {'LinearRegression': LinearRegression}
            exec(user_code, globals(), locals_dict)
            st.session_state.model = locals_dict['model']
            
            st.success("模型构建成功！")
            st.write("模型参数：", locals_dict['model'].get_params())
            
            ai_feedback = ai_code_checker(4, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[4] = user_code
            if "✅" in ai_feedback:
                st.button("进入步骤5：模型训练与预测", 
                         on_click=lambda: setattr(st.session_state, 'step', 5))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(4, user_code)}")


def step5():
    st.header("步骤5：模型训练与预测")
    st.subheader("目标：用numpy数组训练模型并生成预测结果")
    
    if 'model' not in st.session_state:
        st.warning("请先完成步骤4构建模型！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
      # 任务说明单独展示
    st.info("""
    **任务说明**：  
    1. 用训练集（X_train_scaled, y_train）训练模型（fit方法）  
    2. 用测试集（X_test_scaled）生成预测结果（predict方法）  
    提示：模型参数（系数和截距）可通过model.coef_和model.intercept_查看  
    系数表示“特征每变化1单位，疾病进展的变化量”
    """)
    
    code_skeleton = """
# 用训练集训练模型
# 补充训练代码
model.fit(________, ________)    # 提示：参数为X_train_scaled,  y_train

# 查看模型参数（关注特征系数与疾病进展的关系）
print("特征系数（权重）：", model.coef_)    # 对应10个特征的影响程度
print("截距：", model.intercept_)

# 用测试集预测
# 补充预测代码
y_pred = model.predict(________)    # 提示：参数为X_test_scaled

# 查看前5个预测结果
print("前5个预测值（疾病进展）：", y_pred[:5])
print("前5个实际值：", y_test[:5])
    """.strip()
    
      # 加宽代码区域
    user_code = st.text_area(
        "请补充代码：", 
        code_skeleton, 
        height=400,    # 增加高度
        key="step5_code"
    )
    
    if st.button("运行代码", key="run_step5"):
        try:
            locals_dict = {
                'model': st.session_state.model,
                'X_train_scaled': st.session_state.X_train,
                'y_train': st.session_state.y_train,
                'X_test_scaled': st.session_state.X_test,
                'y_test': st.session_state.y_test
            }
            exec(user_code, globals(), locals_dict)
            st.session_state.y_pred = locals_dict['y_pred']
            
            st.success("训练和预测完成！")
              # 保持结果展示的上下布局
            st.write("特征系数（各生理指标对疾病的影响）：", locals_dict['model'].coef_.tolist())
            st.write("截距：", locals_dict['model'].intercept_)
            st.write("预测值 vs 实际值（前5个）：")
            st.write([(locals_dict['y_pred'][i], locals_dict['y_test'][i]) for i in range(5)])
            
            ai_feedback = ai_code_checker(5, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[5] = user_code
            if "✅" in ai_feedback:
                st.button("进入步骤6：模型评估", 
                         on_click=lambda: setattr(st.session_state, 'step', 6))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(5, user_code)}")


def step6():
    st.header("步骤6：模型评估")
    st.subheader("目标：用numpy数组计算MSE和R²评估模型")
    
    if 'y_pred' not in st.session_state:
        st.warning("请先完成步骤5生成预测结果！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
      # 任务说明单独展示
    st.info("""
    **任务说明**：  
    1. 计算均方误差（MSE）：mean_squared_error(y_test, y_pred)  
    2. 计算决定系数（R²）：r2_score(y_test, y_pred)  
    提示：R²越接近1，说明模型能更好地解释疾病进展的变化
    """)
    
    code_skeleton = """
# 导入评估指标
from sklearn.metrics import mean_squared_error, r2_score

# 计算评估指标（补充参数）
mse = mean_squared_error(________, ________)    # 提示：实际值y_test，预测值y_pred
r2 = r2_score(________, ________)

print(f"均方误差（MSE）：{mse:.2f}")
print(f"决定系数（R²）：{r2:.2f}")
    """.strip()
    
      # 加宽代码区域
    user_code = st.text_area(
        "请补充代码：", 
        code_skeleton, 
        height=300,    # 增加高度
        key="step6_code"
    )
    
    if st.button("运行代码", key="run_step6"):
        try:
            locals_dict = {
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'y_test': st.session_state.y_test,
                'y_pred': st.session_state.y_pred
            }
            exec(user_code, globals(), locals_dict)
            mse = locals_dict['mse']
            r2 = locals_dict['r2']
            
            st.success("评估结果：")
              # 保持结果展示的上下布局
            st.metric("均方误差（MSE）", f"{mse:.2f}")
            st.metric("决定系数（R²）", f"{r2:.2f}")
            # 在step6函数的执行成功部分添加
            st.session_state.mse = mse
            st.session_state.r2 = r2
              # AI优化建议（结合糖尿病数据集特点）
            ai_suggestion = "AI优化建议：\n"
            if r2 < 0.5:    # 糖尿病数据集线性回归R²通常在0.5左右
                ai_suggestion += "- 可尝试筛选关键特征（如'bmi'体重指数对疾病影响较大）\n"
                ai_suggestion += "- 考虑添加特征交互项（如bmi与血压的乘积）"
            else:
                ai_suggestion += "- 模型效果较好，可对比Lasso回归（自动筛选特征）的结果"
            st.info(ai_suggestion)
            
            ai_feedback = ai_code_checker(6, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            st.session_state.code_snippets[6] = user_code
            if "✅" in ai_feedback:
                st.subheader("恭喜！已用numpy完成糖尿病数据集的线性回归全流程")
                st.button("进入步骤7：总结与思考", 
                         on_click=lambda: setattr(st.session_state, 'step', 7))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            st.info(f"AI提示：{ai_code_checker(6, user_code)}")
   
# 步骤7：总结与思考
def step7():
    st.header("步骤7：总结与思考")
    st.subheader("模型评估指标分析与改进思考")
    
    # 检查是否有评估指标数据
    if 'y_pred' not in st.session_state or 'mse' not in st.session_state or 'r2' not in st.session_state:
        st.warning("请先完成步骤6的模型评估！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return
    
    # 显示评估指标
    st.write("### 模型评估结果")
    st.metric("均方误差（MSE）", f"{st.session_state.mse:.2f}")
    st.metric("决定系数（R²）", f"{st.session_state.r2:.2f}")
    
    # 学生思考输入
    st.write("### 思考与分析")
    student_answer = st.text_area(
        "请回答以下问题：\n1. 你对上述MSE和R²指标有何看法？\n2. 如果指标不理想，可能是什么原因导致的？",
        height=150,
        key="student_answer"
    )
    
    # 调用AI评价
    if st.button("获取AI评价", key="get_ai_feedback"):
        if not student_answer.strip():
            st.warning("请先输入你的思考内容")
            return
        
        # 准备上下文
        context = f"""
        学生正在分析糖尿病预测模型的评估指标：
        - MSE: {st.session_state.mse:.2f}
        - R²: {st.session_state.r2:.2f}
        学生回答：{student_answer}
        """
        
        # 调用deepseek模型（复用demo中的函数）
        from linear_regression_demo import ask_ai_assistant
        response = ask_ai_assistant(
            question="请简单解读得到的评估指标，并评价我的分析是否合理。如果指标不理想，结合糖尿病数据集分析可能的原因",
            context=context
        )
        
        # 显示AI评价
        st.write("### AI评价与分析")
        st.info(response)
    
    # 完成按钮
    st.button("重新开始全部流程", on_click=lambda: setattr(st.session_state, 'step', 0))

# 主程序

def main():
    # 仅在本模块单独运行时显示标题，嵌入时不显示
    #if 'section' not in st.session_state or st.session_state.section != "编程实例（糖尿病数据集）":
    st.title("📝 线性回归分步编程训练")
    st.title("（糖尿病数据集版）")
    st.write("基于糖尿病数据集，用sklearn完成线性回归全流程，AI辅助检查代码")
    
    # 原有步骤进度显示（保持不变）
    st.sidebar.title("步骤进度")
    steps = [
    "0. 项目说明",
    "1. 数据观察", "2. 特征划分", "3. 数据预处理",
    "4. 模型构建", "5. 训练预测", "6. 模型评估", "7. 总结与思考"  
    ]
    for i, step in enumerate(steps):
        if st.session_state.step > i:
            st.sidebar.markdown(f"✅ {step}")
        elif st.session_state.step == i:
            st.sidebar.markdown(f"▶️ **{step}**")
        else:
            st.sidebar.markdown(f"◻️ {step}")
    
    # 步骤处理逻辑（保持不变）
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
#    elif st.session_state.step == 8:
#        step8()
        
if __name__ == "__main__":
    main()
