# streamlit run KMeans_step_by_step.py
# 葡萄酒聚类分析 - KMeans完整流程

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  # 葡萄酒数据集
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # KMeans聚类模型
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA  # 用于降维可视化
from api_deepseek import ask_ai_assistant  # 导入复用的AI助手函数

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
if 'true_labels' not in st.session_state:
    st.session_state.true_labels = None  # 原始标签（用于后续对比）
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = {}  # 存储各步骤代码
if 'raw_dataset' not in st.session_state:
    st.session_state.raw_dataset = None  # 原始数据集
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None  # 聚类结果


# AI代码检查函数（适配KMeans聚类）
def ai_code_checker(step, user_code):
    try:
        if step == 1:
            errors = []
            if 'load_wine' not in user_code:
                errors.append("❌ 请加载葡萄酒数据集（使用load_wine）")
            if 'X_raw.shape' not in user_code:
                errors.append("❌ 数据形状应使用X_raw.shape（提示：.shape）")
            if 'X_raw[:3]' not in user_code:
                errors.append("❌ 前3行特征应使用X_raw[:3]（提示：切片[:3]）")
            if 'np.mean(X_raw, axis=0)' not in user_code:
                errors.append("❌ 特征均值应使用np.mean(X_raw, axis=0)")
            if 'np.var(X_raw, axis=0)' not in user_code:
                errors.append("❌ 特征方差应使用np.var(X_raw, axis=0)")
            return "✅ 步骤1通过！" if not errors else "\n".join(errors)
        
        elif step == 2:
            errors = []
            if 'X = X_raw' not in user_code:
                errors.append("❌ 请用X = X_raw定义特征数据")
            if 'shape' not in user_code:
                errors.append("💡 建议打印X.shape查看维度")
            return "✅ 步骤2通过！" if not errors else "\n".join(errors)
        
        elif step == 3:
            errors = []
            if 'StandardScaler' not in user_code:
                errors.append("❌ 请用StandardScaler进行特征标准化")
            if 'scaler.fit_transform(X)' not in user_code:
                errors.append("❌ 应使用scaler.fit_transform(X)标准化特征")
            return "✅ 步骤3通过！" if not errors else "\n".join(errors)
        
        elif step == 4:
            errors = []
            if 'KMeans' not in user_code or 'model = KMeans' not in user_code:
                errors.append("❌ 请实例化KMeans模型（model = KMeans()）")
            if 'n_clusters=3' not in user_code:
                errors.append("❌ 请设置n_clusters=3（葡萄酒数据集原始有3类）")
            if 'random_state=42' not in user_code:
                errors.append("❌ 请设置random_state=42保证结果可复现")
            return "✅ 步骤4通过！" if not errors else "\n".join(errors)
        
        elif step == 5:
            errors = []
            if 'model.fit' not in user_code:
                errors.append("❌ 训练模型应为model.fit(X_scaled)")
            if 'model.predict(X_scaled)' not in user_code and 'model.fit_predict' not in user_code:
                errors.append("❌ 请使用model.predict或fit_predict获取聚类标签")
            return "✅ 步骤5通过！" if not errors else "\n".join(errors)
        
        elif step == 6:
            errors = []
            if 'silhouette_score' not in user_code:
                errors.append("❌ 请用silhouette_score计算轮廓系数")
            if 'calinski_harabasz_score' not in user_code:
                errors.append("❌ 请用calinski_harabasz_score计算CH指数")
            if 'PCA' not in user_code:
                errors.append("❌ 请用PCA进行降维可视化")
            return "✅ 步骤6通过！" if not errors else "\n".join(errors)
    
    except Exception as e:
        return f"⚠️ 代码错误：{str(e)}"


# 步骤0：项目说明与数据展示
def step0():
    st.header("步骤0：项目说明")
    st.subheader("葡萄酒聚类分析")
    
    # 项目目标
    st.info("""
    **数据集说明**：
    葡萄酒数据集本质上源于UCI 葡萄酒数据集，其划分的 3 类并非抽象标签，而是对应意大利同一地区
    3 种不同品种的葡萄酿造的葡萄酒。这三类葡萄酒的实际差异主要体现在化学成分、感官特性
    （口感 / 风味 / 色泽）和酿造定位上，可结合数据集的 13 个特征（如酒精含量、脯氨酸、类黄酮等）
    具体分析。

        类别1：“高端浓郁”，靠品种的高脯氨酸、高类黄酮，支撑复杂风味和陈年能力。
        类别2：“轻量易饮”，靠品种的低成分积累，主打清新、平价；
        类别3：“高酸果香”，靠品种的高色素和酸度，平衡口感与性价比。
    
    **项目目标**：  
    通过葡萄酒的化学成分特征（如酒精含量、苹果酸含量等），使用KMeans聚类算法对葡萄酒进行分组，
    理解无监督学习中聚类问题的完整流程。
    """)
    
    # 加载数据集
    wine = load_wine()
    st.session_state.raw_dataset = wine
    
    # 数据集展示
    st.subheader("数据集介绍")
    st.write("""
    该数据集包含178个样本，13个特征（均为数值型），原始数据包含3种不同类型的葡萄酒。
    以下是部分样本数据：
    """)
    
    # 数据表格展示
    df = pd.DataFrame(
        data=wine.data,
        columns=wine.feature_names
    )
    df['原始类别'] = wine.target  # 原始类别列
    st.dataframe(df.head(10), use_container_width=True)
    
    # 特征说明（前5个特征示例）
    st.subheader("特征字段说明（部分）")
    field_explanations = {
        'alcohol': '酒精含量',
        'malic_acid': '苹果酸含量',
        'ash': '灰分含量',
        'alcalinity_of_ash': '灰分碱度',
        'magnesium': '镁含量'
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
    葡萄酒数据集包含178个样本，13个特征，原始数据分为3类（但聚类时不使用标签）。
    """)
    
    # 代码骨架
    code_skeleton = """
# 1. 加载数据并定义特征中文名称
from sklearn.datasets import load_wine
wine = load_wine()
X_raw = wine.data  # 特征数据
true_labels = wine.target  # 原始标签（聚类时不使用，仅用于后续对比）
feature_names_en = wine.feature_names  # 英文特征名

# 葡萄酒数据集特征的中文翻译
feature_names_cn = [
    "酒精含量", "苹果酸含量", "灰分含量", "灰分碱度", "镁含量",
    "总酚含量", "类黄酮含量", "非黄烷类酚类", "原花青素", "颜色强度",
    "色调", "稀释葡萄酒的OD280/OD315", "脯氨酸含量"
]

print("数据形状：", X_raw.shape)  # 提示：使用.shape获取数据维度
print("前3行特征：", X_raw[:3])  # 提示：使用[:3]获取前3行

# 显示每个特征的均值和方差（使用中文名称）
feature_means = np.mean(X_raw, axis=0)  # numpy中使用mean计算均值
feature_vars = np.var(X_raw, axis=0)  # 计算列方差（axis=0）

print("每个特征的均值和方差：")
for i in range(len(feature_names_cn)):
    print(f"特征 {i+1} [{feature_names_cn[i]}]:")
    print(f"  均值: {feature_means[i]:.4f}")
    print(f"  方差: {feature_vars[i]:.4f}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=650, key="step1_code")
    
    if st.button("运行代码", key="run_step1"):
        try:
            locals_dict = {'np': np, 'load_wine': load_wine}
            exec(user_code, globals(), locals_dict)
            
            # 保存数据
            st.session_state.data = locals_dict['X_raw']
            st.session_state.true_labels = locals_dict['true_labels']
            st.session_state.feature_names = locals_dict['feature_names_cn']
            
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
                st.button("进入步骤2：特征数据准备",
                         on_click=lambda: setattr(st.session_state, 'step', 2))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤2：特征数据准备
def step2():
    st.header("步骤2：特征数据准备")
    st.subheader("目标：提取特征数据并查看原始标签分布")
    
    if st.session_state.data is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：  
    1. 特征（X）：使用所有13个化学成分特征（X_raw）  
    2. 原始标签（true_labels）：数据集自带的3类标签（0、1、2），仅用于后续对比分析
    """)
    
    # 代码骨架
    code_skeleton = """
# 定义特征数据
X = X_raw  # 特征（13个化学成分特征）

# 查看原始标签分布（了解数据本来的类别数量）
print("原始标签值：", np.unique(true_labels))  # 应输出[0 1 2]
print("各类别样本数：", np.bincount(true_labels))  # 统计每个类别的样本数量

# 查看特征形状
print("X形状：", X.shape)  # 应是(178, 13)
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=280, key="step2_code")
    
    if st.button("运行代码", key="run_step2"):
        try:
            locals_dict = {'X_raw': st.session_state.data, 'true_labels': st.session_state.true_labels, 'np': np}
            exec(user_code, globals(), locals_dict)
            
            st.session_state.X = locals_dict['X']
            
            st.success("数据准备结果：")
            st.write(f"X形状：{locals_dict['X'].shape}")
            st.write(f"原始标签值：{locals_dict['np'].unique(locals_dict['true_labels'])}")
            st.write(f"各类别样本数：{locals_dict['np'].bincount(locals_dict['true_labels'])}")
            
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
    st.subheader("目标：标准化特征（KMeans对特征尺度敏感）")
    
    if st.session_state.X is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：  
    1. KMeans基于距离计算，需对特征进行标准化（均值为0，方差为1）  
    2. 使用StandardScaler完成标准化处理
    """)
    
    # 代码骨架
    code_skeleton = """
# 特征标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 对特征数据进行标准化
X_scaled = scaler.fit_transform(X)  # 提示：使用fit_transform

# 查看标准化后的均值和方差（应接近0和1）
print("标准化后各特征的均值（应接近0）：", np.mean(X_scaled, axis=0).round(4))
print("标准化后各特征的方差（应接近1）：", np.var(X_scaled, axis=0).round(4))
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=300, key="step3_code")
    
    if st.button("运行代码", key="run_step3"):
        try:
            locals_dict = {
                'X': st.session_state.X,
                'StandardScaler': StandardScaler,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存预处理后的数据
            st.session_state.X_scaled = locals_dict['X_scaled']
            
            st.success("预处理完成！")

            st.write("标准化后前3个特征的均值：", [f"{v:.4f}" for v in locals_dict['np'].mean(locals_dict['X_scaled'], axis=0)[:3]])
            st.write("标准化后前3个特征的方差：", [f"{v:.4f}" for v in locals_dict['np'].var(locals_dict['X_scaled'], axis=0)[:3]])           

            ai_feedback = ai_code_checker(3, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤4：构建KMeans模型",
                         on_click=lambda: setattr(st.session_state, 'step', 4))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤4：构建KMeans模型
def step4():
    st.header("步骤4：构建KMeans模型")
    st.subheader("目标：实例化KMeans聚类模型")
    
    st.info("""
    **任务说明**：  
    1. 从sklearn.cluster导入KMeans  
    2. 实例化模型，设置聚类数n_clusters=3（与原始数据类别数一致）
    """)
    
    code_skeleton = """
# 导入KMeans模型
from sklearn.cluster import KMeans

# 实例化模型（设置3个聚类，随机种子42保证结果可复现）
model = KMeans(n_clusters=3, random_state=42)

# 查看模型参数
print("模型参数：", model.get_params())
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=250, key="step4_code")
    
    if st.button("运行代码", key="run_step4"):
        try:
            locals_dict = {'KMeans': KMeans}
            exec(user_code, globals(), locals_dict)
            st.session_state.model = locals_dict['model']
            
            st.success("模型构建成功！")
            st.write("模型参数：", locals_dict['model'].get_params())
            
            ai_feedback = ai_code_checker(4, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤5：模型训练与聚类",
                         on_click=lambda: setattr(st.session_state, 'step', 5))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤5：模型训练与聚类
def step5():
    st.header("步骤5：模型训练与聚类")
    st.subheader("目标：训练模型并获取聚类结果")
    
    if 'model' not in st.session_state:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：  
    1. 用标准化的特征数据训练KMeans模型  
    2. 获取每个样本的聚类标签（0、1、2）
    """)
    
    code_skeleton = """
# 训练模型并获取聚类标签
cluster_labels = model.fit_predict(X_scaled)  # 同时完成训练和预测

# 查看聚类结果分布
print("聚类标签值：", np.unique(cluster_labels))  # 应输出[0 1 2]
print("各聚类的样本数：", np.bincount(cluster_labels))  # 统计每个聚类的样本数量

# 对比原始标签与聚类标签的分布差异
print("原始标签分布：", np.bincount(true_labels))
print("聚类标签分布：", np.bincount(cluster_labels))
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=300, key="step5_code")
    
    if st.button("运行代码", key="run_step5"):
        try:
            locals_dict = {
                'model': st.session_state.model,
                'X_scaled': st.session_state.X_scaled,
                'true_labels': st.session_state.true_labels,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            st.session_state.cluster_labels = locals_dict['cluster_labels']
            
            st.success("聚类完成！")
            # 正确：从locals_dict中获取cluster_labels和true_labels
            st.write("各聚类的样本数：", locals_dict['np'].bincount(locals_dict['cluster_labels']))
            st.write("原始标签分布：", locals_dict['np'].bincount(locals_dict['true_labels']))
            
            ai_feedback = ai_code_checker(5, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤6：聚类结果评估与可视化",
                         on_click=lambda: setattr(st.session_state, 'step', 6))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤6：聚类结果评估与可视化
def step6():
    st.header("步骤6：聚类结果评估与可视化")
    st.subheader("目标：用评估指标和降维可视化分析聚类效果")
    
    if 'cluster_labels' not in st.session_state:
        st.warning("请先完成步骤5！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
    st.info("""
    **任务说明**：  
    1. 计算轮廓系数（越接近1越好）和Calinski-Harabasz指数（越大越好）  
    2. 用PCA降维到2D，可视化聚类结果与原始标签的对比
    """)
    
    code_skeleton = """
# 导入评估指标和PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 计算聚类评估指标
silhouette = silhouette_score(X_scaled, cluster_labels)  # 轮廓系数
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)  # CH指数

print(f"轮廓系数（越接近1越好）：{silhouette:.4f}")
print(f"Calinski-Harabasz指数（越大越好）：{calinski_harabasz:.4f}")

# PCA降维用于可视化（降到2维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)  # 对标准化数据进行降维

# 绘制聚类结果与原始标签的对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 聚类结果可视化
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
ax1.set_title('KMeans聚类结果（PCA降维）', fontsize=14)
ax1.set_xlabel('PCA维度1')
ax1.set_ylabel('PCA维度2')

# 原始标签可视化
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.8)
ax2.set_title('原始标签分布（PCA降维）', fontsize=14)
ax2.set_xlabel('PCA维度1')
ax2.set_ylabel('PCA维度2')

plt.tight_layout()
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=800, key="step6_code")
    
    if st.button("运行代码", key="run_step6"):
        try:
            locals_dict = {
                'X_scaled': st.session_state.X_scaled,
                'cluster_labels': st.session_state.cluster_labels,
                'true_labels': st.session_state.true_labels,
                'silhouette_score': silhouette_score,
                'calinski_harabasz_score': calinski_harabasz_score,
                'PCA': PCA,
                'plt': plt,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存评估数据
            st.session_state.silhouette = locals_dict.get('silhouette', 0)
            st.session_state.calinski_harabasz = locals_dict.get('calinski_harabasz', 0)
            st.session_state.X_pca = locals_dict.get('X_pca', None)
            
            st.success("评估与可视化完成！")
            st.write(f"轮廓系数：{locals_dict.get('silhouette', 0):.4f}")
            st.write(f"Calinski-Harabasz指数：{locals_dict.get('calinski_harabasz', 0):.4f}")
            st.pyplot(locals_dict['plt'])

        # ################## 新增代码开始 ##################
            silhouette = st.session_state.silhouette
            calinski_harabasz=st.session_state.calinski_harabasz
            context = f"这是KMeans模型的轮廓系数：{silhouette}，Calinski-Harabasz指数：{calinski_harabasz}。"
            question = "请用300字简单且非常生动易懂的语言解释廓系数和Calinski-Harabasz指数的含义，模型表现如何？"
            with st.spinner("AI助教正在分析评估指标..."):
                st.info(ask_ai_assistant(question, context))
        # ################## 新增代码结束 ##################
        
            ai_feedback = ai_code_checker(6, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.subheader("恭喜！已用sklearn库完成葡萄酒数据集的KMeans聚类全流程")
                st.button("进入步骤7：总结与思考",
                         on_click=lambda: setattr(st.session_state, 'step', 7))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤7：总结与思考
def step7():
    st.header("步骤7：总结与思考")
    st.subheader("目标：梳理KMeans聚类流程并深入思考关键问题")
    
    # 检查前置条件
    if st.session_state.step < 6:
        st.warning("请先完成前面所有步骤再进行总结！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return

    st.info("""
    **任务说明**：  
    1. 回顾KMeans解决葡萄酒聚类问题的完整流程  
    2. 思考聚类算法的关键参数和适用场景  
    3. 分析聚类结果与原始标签的差异原因
    """)

    # 显示步骤6的评估指标
    st.subheader("📊 聚类评估结果回顾")
    metrics_available = all(key in st.session_state for key in ['silhouette', 'calinski_harabasz', 'cluster_labels'])
    if metrics_available:
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("轮廓系数（Silhouette Score）", f"{st.session_state.silhouette:.4f}")
        with metrics_cols[1]:
            st.metric("Calinski-Harabasz指数", f"{st.session_state.calinski_harabasz:.4f}")
        
        # 显示聚类分布与原始标签分布对比
        if 'cluster_labels' in st.session_state and 'true_labels' in st.session_state:
            st.write("### 聚类结果与原始标签分布对比")
            compare_df = pd.DataFrame({
                '原始标签': np.bincount(st.session_state.true_labels),
                '聚类结果': np.bincount(st.session_state.cluster_labels)
            }, index=[f'类别{i}' for i in range(3)])
            st.dataframe(compare_df, use_container_width=True)
    else:
        st.warning("未找到完整的聚类评估数据，请确保已完成步骤6")
    


    # 学生思考输入（关键问题）
    st.write("### 思考与分析")
    student_answer = st.text_area(
        """请思考以下问题：
            根据聚类评估数据分析，你觉得模型的聚类效果如何？""",
        height=100,
        key="student_answer"
    )
    
    # 调用AI评价
    if st.button("获取AI评价", key="get_ai_feedback"):
        if not student_answer.strip():
            st.warning("请先输入你的思考内容")
            return
        
        if not metrics_available:
            st.warning("未找到评估数据，请先完成步骤6的聚类评估")
            return
        
        # 构建包含评估指标和学生回答的上下文
        context = f"""
        聚类评估指标数据：
        - 轮廓系数（Silhouette Score）：{st.session_state.silhouette:.4f}
        - Calinski-Harabasz指数：{st.session_state.calinski_harabasz:.4f}
        
        聚类与原始标签分布对比：
        {pd.DataFrame({
            '原始标签': np.bincount(st.session_state.true_labels),
            '聚类结果': np.bincount(st.session_state.cluster_labels)
        }, index=[f'类别{i}' for i in range(3)]).to_string()}
        
        学生的分析回答：
        {student_answer}
        """
        
        # 步骤7中调用AI评价的代码
        response = ask_ai_assistant(
            question="请结合提供的聚类评估指标和学生的分析，先解读评估指标含义，重点评价学生分析的合理性，并补充专业建议。",
            context=context.strip()
        )
        
        # 显示AI评价
        st.write("### AI评价与分析")
        st.info(response)
    
    # 重新开始按钮
    st.button("重新开始全部流程", on_click=lambda: setattr(st.session_state, 'step', 0))


# 主程序
def main():
    st.title("📝 KMeans聚类分步编程训练")
    st.title("（葡萄酒数据集版）")
    st.write("基于葡萄酒数据集，用sklearn完成KMeans聚类全流程，AI辅助检查代码")
    
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据观察", "2. 特征准备", "3. 数据预处理",
        "4. 模型构建", "5. 训练聚类", "6. 结果评估", "7. 总结与思考"
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
