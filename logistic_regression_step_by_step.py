# streamlit run logistic_regression_step_by_step.py
# 乳腺癌良恶性预测分析 - 逻辑回归完整流程

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer  # 乳腺癌分类数据集
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from api_deepseek import ask_ai_assistant

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化会话状态
if 'step' not in st.session_state:
    st.session_state.step = 0  # 从0开始
if 'data' not in st.session_state:
    st.session_state.data = None  # 特征数据
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None  # 特征名称
if 'X' not in st.session_state:
    st.session_state.X = None  # 特征
if 'y' not in st.session_state:
    st.session_state.y = None  # 目标变量（分类标签）
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = {}  # 存储各步骤代码
if 'raw_dataset' not in st.session_state:
    st.session_state.raw_dataset = None  # 原始数据集
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None  # 预测结果


# AI代码检查函数（适配逻辑回归）
def ai_code_checker(step, user_code):
    try:
        if step == 1:
            errors = []
            if 'load_breast_cancer' not in user_code:
                errors.append("❌ 请加载乳腺癌数据集（使用load_breast_cancer）")
            if 'X_raw.shape' not in user_code:
                errors.append("❌ 数据形状应使用X_raw.shape（提示：.shape）")
            if 'X_raw[:3]' not in user_code:
                errors.append("❌ 前3行特征应使用X_raw[:3]（提示：切片[:3]）")
            if 'np.mean(X_raw, axis=0)' not in user_code:
                errors.append("❌ 特征均值应使用np.mean(X_raw, axis=0)（提示：计算列均值）")
            if 'np.var(X_raw, axis=0)' not in user_code:
                errors.append("❌ 特征方差应使用np.var(X_raw, axis=0)（提示：计算列方差）")
            return "✅ 步骤1通过！" if not errors else "\n".join(errors)
        
        elif step == 2:
            errors = []
            if 'X = X_raw' not in user_code or 'y = y_raw' not in user_code:
                errors.append("❌ 请用X = X_raw和y = y_raw划分特征与目标")
            if 'shape' not in user_code:
                errors.append("💡 建议打印X.shape和y.shape查看维度")
            return "✅ 步骤2通过！" if not errors else "\n".join(errors)
        
        elif step == 3:
            errors = []
            if 'train_test_split' not in user_code:
                errors.append("❌ 请用train_test_split划分训练集和测试集")
            if 'test_size=0.' not in user_code or 'random_state=' not in user_code:
                errors.append("❌ 请设置参数test_size 和 random_state")
            if 'scaler.fit_transform(X_train)' not in user_code or 'scaler.transform(X_test)' not in user_code:
                errors.append("❌ 训练集用fit_transform，测试集用transform")
            return "✅ 步骤3通过！" if not errors else "\n".join(errors)
        
        elif step == 4:
            if 'LogisticRegression()' not in user_code or 'model = LogisticRegression()' not in user_code:
                return "❌ 请实例化逻辑回归模型（model = LogisticRegression()）"
            return "✅ 步骤4通过！"
        
        elif step == 5:
            errors = []
            if 'model.fit(X_train_scaled, y_train)' not in user_code:
                errors.append("❌ 训练模型应为model.fit(X_train_scaled, y_train)")
            if 'model.predict(X_test_scaled)' not in user_code:
                errors.append("❌ 预测应为model.predict(X_test_scaled)")
            return "✅ 步骤5通过！" if not errors else "\n".join(errors)
        
        elif step == 6:
            errors = []
            if 'accuracy_score' not in user_code:
                errors.append("❌ 请用accuracy_score计算准确率")
            if 'precision_score' not in user_code:
                errors.append("❌ 请用precision_score计算精确率")
            if 'recall_score' not in user_code:
                errors.append("❌ 请用recall_score计算召回率")
            if 'f1_score' not in user_code:
                errors.append("❌ 请用f1_score计算F1分数")
            if 'confusion_matrix' not in user_code:
                errors.append("❌ 请用confusion_matrix生成混淆矩阵")
            if 'classification_report' not in user_code:
                errors.append("❌ 请用classification_report生成详细分类报告")
            return "✅ 步骤6通过！" if not errors else "\n".join(errors)
    
    except Exception as e:
        return f"⚠️ 代码错误：{str(e)}"


# 步骤0：项目说明与数据展示
def step0():
    st.header("步骤0：项目说明")
    st.subheader("乳腺癌良恶性预测分析")
    
    # 项目目标
    st.info("""
    **项目目标**：  
    通过乳腺肿块的特征数据（如半径、纹理等），建立逻辑回归模型预测肿块为良性（0）或恶性（1），
    理解分类问题的完整机器学习流程。
    """)
    
    # 加载数据集
    cancer = load_breast_cancer()
    st.session_state.raw_dataset = cancer
    
    # 数据集展示
    st.subheader("数据集介绍")
    st.write("""
    该数据集包含569个样本，30个特征（均为数值型），目标变量为二分类标签（0=恶性，1=良性）。
    以下是部分样本数据：
    """)
    
    # 数据表格展示
    df = pd.DataFrame(
        data=cancer.data,
        columns=cancer.feature_names
    )
    df['诊断结果（0=恶性，1=良性）'] = cancer.target  # 目标变量列
    st.dataframe(df.head(10), use_container_width=True)
    
    # 特征说明（前5个特征示例）
    st.subheader("特征字段说明（部分）")
    field_explanations = {
        'mean radius': '平均半径（肿块的平均半径）',
        'mean texture': '平均纹理（灰度值标准差）',
        'mean perimeter': '平均周长',
        'mean area': '平均面积',
        'mean smoothness': '平均平滑度（半径变化的陡峭度）'
    }
    explanation_df = pd.DataFrame(
        list(field_explanations.items()),
        columns=['特征字段', '说明']
    )
    st.dataframe(explanation_df, use_container_width=True)
    
    st.button("进入步骤1：数据观察与理解",
             on_click=lambda: setattr(st.session_state, 'step', 1))


# 步骤1：数据观察与理解
def step1():
    st.header("步骤1：数据观察与理解")
    st.subheader("目标：加载数据集，用numpy观察基本信息")
    
    st.info("""
    **数据集说明**：  
    乳腺癌数据集包含569个样本，30个特征，目标变量为二分类标签（0=恶性，1=良性）。
    """)
    
    # 代码骨架（填空式）
    code_skeleton = """
# 1. 加载数据并定义特征中文名称
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_raw = cancer.data
y_raw = cancer.target
feature_names_en = cancer.feature_names  # 英文特征名

# 乳腺癌数据集特征的中文翻译
feature_names_cn = [
    "平均半径", "平均纹理", "平均周长", "平均面积", "平均平滑度",
    "平均紧凑度", "平均凹度", "平均凹点", "平均对称性", "平均分形维数",
    "半径标准差", "纹理标准差", "周长标准差", "面积标准差", "平滑度标准差",
    "紧凑度标准差", "凹度标准差", "凹点标准差", "对称性标准差", "分形维数标准差",
    "最大半径", "最大纹理", "最大周长", "最大面积", "最大平滑度",
    "最大紧凑度", "最大凹度", "最大凹点", "最大对称性", "最大分形维数"
]

# 目标变量中文含义（0：恶性，1：良性）
target_names_cn = ["恶性", "良性"]

print("数据形状：", X_raw.shape)  # 提示：使用.shape获取数据维度
print("前3行特征（英文）：", X_raw[:3])  # 提示：使用[:3]获取前3行

# 显示每个特征的均值和方差（使用中文名称）
feature_means = np.mean(X_raw, axis=0)  # 提示：计算X_raw的列均值（axis=0）
feature_vars = np.var(X_raw, axis=0)  # 提示：计算X_raw的列方差（axis=0）

print("每个特征的均值和方差：")
for i in range(len(feature_names_cn)):
    print(f"特征 {i+1} [{feature_names_cn[i]}]:")
    print(f"  均值: {feature_means[i]:.4f}")
    print(f"  方差: {feature_vars[i]:.4f}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=800, key="step1_code")
    
    if st.button("运行代码", key="run_step1"):
        try:
            locals_dict = {'np': np, 'load_breast_cancer': load_breast_cancer}
            exec(user_code, globals(), locals_dict)
            
            # 保存数据
            st.session_state.data = locals_dict['X_raw']
            st.session_state.y_raw = locals_dict['y_raw']
            st.session_state.feature_names = locals_dict['feature_names_cn']  # 保存中文特征名
            
            # 展示结果
            st.success("代码运行成功！")
            
            with st.expander("查看输出"):
                st.write(f"数据形状：{locals_dict['X_raw'].shape}")
                st.write("前3行特征：", locals_dict['X_raw'][:3].tolist())
                st.write("前3个特征的均值：", [f"{v:.4f}" for v in locals_dict['feature_means'][:3]])
            
            # AI反馈
            ai_feedback = ai_code_checker(1, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤2：特征与目标划分",
                         on_click=lambda: setattr(st.session_state, 'step', 2))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤2：特征与目标变量划分
def step2():
    st.header("步骤2：特征与目标变量划分")
    st.subheader("目标：提取特征（X）和分类标签（y）")
    
    if st.session_state.data is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：  
    1. 特征（X）：使用所有30个生理特征（X_raw）  
    2. 目标变量（y）：良恶性标签（y_raw，0=恶性，1=良性）
    """)
    
    # 代码骨架
    code_skeleton = """
# 划分特征（X）和目标变量（y）
X = X_raw    # 特征（30个肿块特征）
y = y_raw    # 目标变量（0=恶性，1=良性）

# 查看形状
print("X形状：", X.shape)    # 应是(569, 30)
print("y形状：", y.shape)    # 应是(569,)
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=230, key="step2_code")
    
    if st.button("运行代码", key="run_step2"):
        try:
            locals_dict = {'X_raw': st.session_state.data, 'y_raw': st.session_state.y_raw}
            exec(user_code, globals(), locals_dict)
            
            st.session_state.X = locals_dict['X']
            st.session_state.y = locals_dict['y']
            
            st.success("划分结果：")
            st.write(f"X形状：{locals_dict['X'].shape}，y形状：{locals_dict['y'].shape}")
            
            ai_feedback = ai_code_checker(2, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤3：数据预处理",
                         on_click=lambda: setattr(st.session_state, 'step', 3))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤3：数据预处理
def step3():
    st.header("步骤3：数据预处理")
    st.subheader("目标：划分训练集/测试集并标准化")
    
    if st.session_state.X is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：  
    1. 用train_test_split划分数据集（测试集占20%，随机种子42）  
    2. 用StandardScaler标准化特征（训练集拟合，测试集转换）
    """)
    
    # 代码骨架
    code_skeleton = """
# 划分训练集和测试集
from sklearn.model_selection import train_test_split

# 补充参数（你可以尝试修改 测试集数据占比 和 随机数种子）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # 提示：训练集用 fit_transform
X_test_scaled = scaler.transform(X_test)  # 提示：测试集用 transform

print("训练集样本数：", X_train_scaled.shape[0])
print("测试集样本数：", X_test_scaled.shape[0])
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=400, key="step3_code")
    
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
            
            st.success("预处理完成！")
            st.write(f"训练集：{locals_dict['X_train_scaled'].shape[0]}样本，测试集：{locals_dict['X_test_scaled'].shape[0]}样本")
            
            ai_feedback = ai_code_checker(3, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤4：构建逻辑回归模型",
                         on_click=lambda: setattr(st.session_state, 'step', 4))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤4：构建逻辑回归模型
def step4():
    st.header("步骤4：构建逻辑回归模型")
    st.subheader("目标：实例化LogisticRegression模型")
    
    st.info("""
    **任务说明**：  
    从sklearn.linear_model导入LogisticRegression并实例化（默认参数）
    """)
    
    code_skeleton = """
# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression

# 实例化模型
model = LogisticRegression()

# 查看模型参数
print("模型参数：", model.get_params())
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=250, key="step4_code")
    
    if st.button("运行代码", key="run_step4"):
        try:
            locals_dict = {'LogisticRegression': LogisticRegression}
            exec(user_code, globals(), locals_dict)
            st.session_state.model = locals_dict['model']
            
            st.success("模型构建成功！")
            st.write("模型参数：", locals_dict['model'].get_params())
            
            ai_feedback = ai_code_checker(4, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤5：模型训练与预测",
                         on_click=lambda: setattr(st.session_state, 'step', 5))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤5：模型训练与预测
def step5():
    st.header("步骤5：模型训练与预测")
    st.subheader("目标：训练模型并生成测试集预测结果")
    
    if 'model' not in st.session_state:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：  
    1. 用训练集（X_train_scaled, y_train）训练模型  
    2. 用测试集（X_test_scaled）生成预测标签
    """)
    
    code_skeleton = """
# 训练模型
# 用标准化的训练集训练
model.fit(X_train_scaled, y_train)    # 提示：参数为X_train_scaled, y_train

# 预测测试集
# 用标准化的测试集预测
y_pred = model.predict(X_test_scaled)    # 提示：参数为X_test_scaled

print("前10个预测结果：", y_pred[:10])
print("前10个真实标签：", y_test[:10])
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=300, key="step5_code")
    
    if st.button("运行代码", key="run_step5"):
        try:
            locals_dict = {
                'model': st.session_state.model,
                'X_train_scaled': st.session_state.X_train,
                'X_test_scaled': st.session_state.X_test,
                'y_train': st.session_state.y_train,
                'y_test': st.session_state.y_test
            }
            exec(user_code, globals(), locals_dict)
            st.session_state.y_pred = locals_dict['y_pred']
            
            st.success("训练与预测完成！")
            st.write("前10个预测结果：", locals_dict['y_pred'][:10])
            
            ai_feedback = ai_code_checker(5, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤6：模型评估",
                         on_click=lambda: setattr(st.session_state, 'step', 6))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤6：模型评估
def step6():
    st.header("步骤6：模型评估")
    st.subheader("目标：用分类指标评估模型性能")
    
    if 'y_pred' not in st.session_state:
        st.warning("请先完成步骤5！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
    st.info("""
    **任务说明**：  
    1. 计算准确率（Accuracy）、精确率（Precision）、召回率（Recall）和F1分数  
    2. 生成混淆矩阵和详细分类报告
    """)
    
    code_skeleton = """
# 导入评估指标
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)    # 提示：y_test为真实值，y_pred为预测值
print(f"准确率（Accuracy）：{accuracy:.4f}")

# 计算精确率
precision = precision_score(y_test, y_pred)    
print(f"精确率（Precision）：{precision:.4f}")

# 计算召回率
recall = recall_score(y_test, y_pred)    # 提示：计算召回率 recall_score
print(f"召回率（Recall）：{recall:.4f}")

# 计算F1分数
f1 = f1_score(y_test, y_pred)    # 提示：计算F1分数 f1_score
print(f"F1分数（F1-Score）：{f1:.4f}")

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)    
print("混淆矩阵：", cm)

# 输出详细分类报告

report=classification_report(y_test, y_pred, target_names=target_names_cn)
print("详细分类报告：",report)
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=680, key="step6_code")
    
    if st.button("运行代码", key="run_step6"):
        try:
            # 确保target_names_cn可用
            target_names_cn = ["恶性", "良性"]
            locals_dict = {
                'y_test': st.session_state.y_test,
                'y_pred': st.session_state.y_pred,
                'accuracy_score': accuracy_score,
                'confusion_matrix': confusion_matrix,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'f1_score': f1_score,
                'classification_report': classification_report,
                'target_names_cn': target_names_cn
            }
            exec(user_code, globals(), locals_dict)
            
            # 新增：保存评估数据到session_state
            st.session_state.accuracy = locals_dict.get('accuracy', 0)
            st.session_state.precision = locals_dict.get('precision', 0)
            st.session_state.recall = locals_dict.get('recall', 0)
            st.session_state.f1 = locals_dict.get('f1', 0)
            st.session_state.cm = locals_dict.get('cm', [])
            st.session_state.report = locals_dict.get('report', '')            
            st.success("评估完成！")
            st.write(f"准确率：{locals_dict.get('accuracy', 0):.4f}")
            st.write(f"精确率：{locals_dict.get('precision', 0):.4f}")
            st.write(f"召回率：{locals_dict.get('recall', 0):.4f}")
            st.write(f"F1分数：{locals_dict.get('f1', 0):.4f}")
            st.write("混淆矩阵：")
#            st.write(locals_dict.get('cm', []))
            cm_df = pd.DataFrame(
                st.session_state.cm,
                index=["实际恶性", "实际良性"],
                columns=["预测恶性", "预测良性"]
            )
            st.dataframe(cm_df, use_container_width=True)
        # ################## 新增代码开始 ##################
            cm = st.session_state.cm
            context = f"这是逻辑回归模型的混淆矩阵：{cm}，行是实际标签（0=恶性，1=良性），列是预测标签（0=恶性，1=良性）。"
            question = "请用300字简单且非常生动易懂的语言解释这个混淆矩阵的含义，模型表现如何？重点说明错误分类的情况（比如假阳性、假阴性有多少，对医疗场景的影响）。"
            with st.spinner("AI助教正在分析评估指标..."):
                st.info(ask_ai_assistant(question, context))
        # ################## 新增代码结束 ##################           
            st.write("详细分类报告：")
            st.text(locals_dict.get('report', ''))
       
            # AI反馈
            ai_feedback = ai_code_checker(6, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤7：特征影响力分析",
                         on_click=lambda: setattr(st.session_state, 'step', 7))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")

# 步骤7：特征影响力分析
def step7():
    st.header("步骤7：特征影响力分析")
    st.subheader("目标：分析各特征对模型预测结果的影响强度")
    
    # 检查前置条件
    if 'model' not in st.session_state or 'feature_names' not in st.session_state:
        st.warning("请先完成前面的模型训练步骤！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return
    
    st.info("""
    **任务说明**：  
    逻辑回归模型的特征系数（coef_）可反映特征对预测结果的影响：  
    - 系数为正：特征值越大，越倾向于预测为良性（1）  
    - 系数为负：特征值越大，越倾向于预测为恶性（0）  
    - 绝对值越大：特征对预测的影响越强  
    """)
    
    # 代码骨架
    code_skeleton = """
# 获取模型特征系数（每个特征对应一个系数）
feature_coef = model.coef_[0]  # model为训练好的逻辑回归模型

# 按系数绝对值从大到小排序（影响力从强到弱）
sorted_indices = np.argsort(np.abs(feature_coef))[::-1]  # 倒序排列

# 排序后的特征名称（中文）
sorted_names=[]
for i in sorted_indices:
    sorted_names.append(feature_names[i])
    
# 排序后的系数值
sorted_coef=[]
for i in sorted_indices:
    sorted_coef.append(feature_coef[i])
    
# 绘制水平条形图展示特征影响力
plt.figure(figsize=(12, 10))

# 负数系数（倾向恶性）用红色，正数系数（倾向良性）用青色
bars = plt.barh(sorted_names, sorted_coef, 
                color=['#ff6b6b' if c < 0 else '#4ecdc4' for c in sorted_coef])

# 添加图表标题和标签
plt.title('各特征对乳腺癌良恶性判断的影响力', fontsize=16)
plt.xlabel('特征系数（绝对值越大，影响力越强）', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.yticks(fontsize=11)  # 调整特征名称字体大小

# 在条形上标注系数具体值
for bar in bars:
    width = bar.get_width()  # 获取条形长度（系数值）
    # 在条形右侧标注数值，居中对齐
    plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             f'{width:.3f}', va='center', fontsize=10)

plt.tight_layout()  # 自动调整布局，避免文字重叠
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=830, key="step7_code")
    
    if st.button("运行代码", key="run_step7"):
        
        try:
            locals_dict = {
                'model': st.session_state.model,
                'feature_names': st.session_state.feature_names,
                'np': np,
                'plt': plt
            }
            exec(user_code, globals(), locals_dict)

            st.success("特征影响力分析完成！")
            
            # 展示特征系数排序结果（前5名）
            with st.expander("查看前5名影响力特征"):
                top5_names = locals_dict['sorted_names'][:5]
                top5_coef = [f"{c:.3f}" for c in locals_dict['sorted_coef'][:5]]
                top5_tendency = ["恶性" if float(c) < 0 else "良性" for c in top5_coef]
                
                # 用纯文本表格展示
                st.write("| 排名 | 特征名称 | 系数值 | 倾向判断 |")
                st.write("|------|----------|--------|----------|")
                for i in range(5):
                    st.write(f"| {i+1} | {top5_names[i]} | {top5_coef[i]} | {top5_tendency[i]} |")
            
            # 展示特征影响力条形图
            with st.expander("查看特征影响力可视化结果"):
                st.pyplot(locals_dict['plt'])  # 显示matplotlib图表
            
            # 分析解读提示
            st.info("""
            **结果解读**：  
            - 红色条形：特征系数为负，该特征值越大，模型越可能判断为恶性（0）  
            - 青色条形：特征系数为正，该特征值越大，模型越可能判断为良性（1）  
            例如：若"最大凹度"系数为-2.5，说明该特征值越大，越倾向于预测为恶性。
            """)
            st.subheader("恭喜！已用sklearn库完成乳腺癌数据集的逻辑回归全流程")
            st.button("进入步骤8：总结与思考",on_click=lambda: setattr(st.session_state, 'step', 8))
                
            
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")
            
# 步骤8：总结与思考
def step8():
    st.header("步骤8：总结与思考")
    st.subheader("目标：梳理全流程并深入理解模型评估的场景意义")
    
    # 检查前置条件
    if st.session_state.step < 7:
        st.warning("请先完成前面所有步骤再进行总结！")
        st.button("返回步骤7", on_click=lambda: setattr(st.session_state, 'step', 7))
        return

    st.info("""
    **任务说明**：  
    1. 回顾逻辑回归解决乳腺癌分类问题的完整流程  
    2. 结合医疗场景特点，深入思考模型评估指标的选择逻辑  
    3. 提出可能的模型优化方向
    """)

    # 显示步骤6的评估指标
    st.subheader("📊 模型评估结果回顾")
    metrics_available = all(key in st.session_state for key in ['accuracy', 'precision', 'recall', 'f1', 'cm', 'report'])
    if metrics_available:
        # 显示关键指标
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("准确率（Accuracy）", f"{st.session_state.accuracy:.4f}")
            st.metric("精确率（Precision）", f"{st.session_state.precision:.4f}")
        with metrics_cols[1]:
            st.metric("召回率（Recall）", f"{st.session_state.recall:.4f}")
            st.metric("F1分数", f"{st.session_state.f1:.4f}")
        
        # 显示混淆矩阵
        st.write("### 混淆矩阵")
        cm_df = pd.DataFrame(
            st.session_state.cm,
            index=["实际恶性", "实际良性"],
            columns=["预测恶性", "预测良性"]
        )
        st.dataframe(cm_df, use_container_width=True)
        
        # 显示分类报告
        st.write("### 详细分类报告")
        st.text(st.session_state.report)
    else:
        st.warning("未找到完整的模型评估数据，请确保已完成步骤6")
    

    
    # 学生思考输入
    st.write("### 思考与分析")
    student_answer = st.text_area(
        "请结合案例分析，评估数据有什么意义，我们应该更关注模型的精确率还是召回率？",
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
        - 准确率（Accuracy）：{st.session_state.accuracy:.4f}
        - 精确率（Precision）：{st.session_state.precision:.4f}
        - 召回率（Recall）：{st.session_state.recall:.4f}
        - F1分数：{st.session_state.f1:.4f}
        - 混淆矩阵：
        {st.session_state.cm}
        - 详细分类报告：
        {st.session_state.report}
        
        学生的分析回答：
        {student_answer}
        """
        
        # 调用deepseek模型（复用demo中的函数）

        response = ask_ai_assistant(
            question="请结合提供的模型评估指标和学生的分析，先简单解读评估指标的含义，重点评价学生分析中的合理与不合理的地方，并给出补充建议。",
            context=context.strip()  # 传递完整上下文
        )
        
        # 显示AI评价
        st.write("### AI评价与分析")
        st.info(response)
    
    # 完成按钮
    st.button("重新开始全部流程", on_click=lambda: setattr(st.session_state, 'step', 0))

# 主程序
def main():
    st.title("📝 逻辑回归分步编程训练")
    st.title("（乳腺癌数据集版）")
    st.write("基于乳腺癌数据集，用sklearn完成逻辑回归全流程，AI辅助检查代码")
    
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 特征划分", "3. 数据预处理",
        "4. 模型构建", "5. 训练预测", "6. 模型评估", "7. 特征影响力分析","8. 总结与思考"
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
        step0()  # 显示步骤0内容
    elif st.session_state.step == 1:
        step1()  # 显示步骤1内容
    elif st.session_state.step == 2:
        step2()  # 显示步骤2内容
    elif st.session_state.step == 3:
        step3()  # 显示步骤3内容
    elif st.session_state.step == 4:
        step4()  # 显示步骤4内容
    elif st.session_state.step == 5:
        step5()  # 显示步骤5内容
    elif st.session_state.step == 6:
        step6()  # 显示步骤6内容
    elif st.session_state.step == 7:
        step7()  # 显示步骤7内容
    elif st.session_state.step == 8:
        step8()  # 显示步骤8内容


if __name__ == "__main__":
    main()
