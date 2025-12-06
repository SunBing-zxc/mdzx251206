# C:\Users\孙冰\Desktop\AI助教
# streamlit run KMeans_demo.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import time
import io
import KMeans_step_by_step
from api_deepseek import client, ask_ai_assistant

# 设置页面
st.set_page_config(page_title="KMeans聚类交互式学习平台", layout="wide")
st.title("📊 KMeans聚类交互式学习平台")

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
        btn1 = st.button("什么是KMeans聚类?")
        btn2 = st.button("K值如何选择?")
    
    with col2:
        btn3 = st.button("KMeans的优缺点")
        btn4 = st.button("聚类与分类的区别")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是KMeans聚类?它的核心思想是什么?"
    elif btn2:
        question = "KMeans中的K值应该如何选择?有什么方法?"
    elif btn3:
        question = "KMeans算法有哪些优点和缺点?适用于什么场景?"
    elif btn4:
        question = "聚类和分类有什么本质区别?分别适用于什么情况?"
    
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
def generate_cluster_data(data_type, n_samples, n_centers, cluster_std, noise=0.05):
    """生成不同类型的聚类数据"""
    np.random.seed(42)
    
    if data_type == "球形聚类":
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=42
        )
    
    elif data_type == "半月形聚类":
        X = make_moons(n_samples=n_samples, noise=noise, random_state=42)[0]
        y_true = None  # 半月形数据没有真实的球形聚类标签
    
    elif data_type == "环形聚类":
        X = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)[0]
        y_true = None  # 环形数据没有真实的球形聚类标签
    
    elif data_type == "不均匀密度聚类":
        # 生成密度不同的聚类
        centers = [(-3, -3), (0, 0), (3, 3)]
        X = []
        y_true = []
        
        # 为每个中心生成不同数量的点（不同密度）
        sizes = [int(n_samples*0.6), int(n_samples*0.3), int(n_samples*0.1)]
        stds = [0.5, 1.0, 0.8]
        
        for i, (center, size, std) in enumerate(zip(centers, sizes, stds)):
            cluster = np.random.normal(loc=center, scale=std, size=(size, 2))
            X.append(cluster)
            y_true.extend([i]*size)
        
        X = np.vstack(X)
        y_true = np.array(y_true)
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y_true = y_true[indices]
    
    return X, y_true

# 绘制聚类数据
def plot_cluster_data(X, y=None, centers=None, title="聚类数据分布"):
    """绘制聚类数据散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if y is not None:
        # 如果有标签，使用不同颜色表示不同类别
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
        ax.legend(*scatter.legend_elements(), title="聚类")
    else:
        # 没有标签，使用单一颜色
        ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7, s=50)
    
    # 绘制中心点
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='中心点')
        ax.legend()
    
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# KMeans算法步骤可视化
def kmeans_step_visualization(X, n_clusters, max_iter=10):
    """可视化KMeans算法的每一步"""
    # 初始化中心点（随机选择样本作为初始中心）
    np.random.seed(42)
    indices = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[indices]
    
    steps = []
    steps.append((centers.copy(), np.zeros(len(X))))  # 记录初始状态
    
    for i in range(max_iter):
        # 步骤1: 分配每个点到最近的中心
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # 记录当前步骤
        steps.append((centers.copy(), labels.copy()))
        
        # 步骤2: 计算新的中心点
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # 如果中心点不再变化，提前结束
        if np.allclose(centers, new_centers):
            break
            
        centers = new_centers
    
    # 记录最终状态
    distances = np.sqrt(((X - centers[:, np.newaxis])** 2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    steps.append((centers.copy(), labels.copy()))
    
    return steps

# 绘制KMeans步骤
def plot_kmeans_steps(X, steps):
    """绘制KMeans算法的每一步"""
    figs = []
    
    for i, (centers, labels) in enumerate(steps):
        if i == 0:
            title = f"步骤 {i}: 初始化中心点"
        elif i == len(steps) - 1:
            title = f"步骤 {i}: 收敛完成"
        else:
            title = f"步骤 {i}: 迭代更新"
            
        fig = plot_cluster_data(X, labels, centers, title)
        figs.append(fig)
        
    return figs

# 绘制不同K值的聚类结果对比
def plot_k_comparison(X, k_values):
    """对比不同K值的聚类结果"""
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(5*n_k, 5))
    
    if n_k == 1:
        axes = [axes]
    
    for i, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        axes[i].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
        axes[i].set_title(f'K={k}, 惯性={kmeans.inertia_:.2f}')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# 绘制肘部法则图表
def plot_elbow_method(X, max_k=10):
    """绘制肘部法则图表帮助选择K值"""
    inertias = []
    k_range = range(1, max_k+1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertias, 'bo-')
    ax.set_xlabel('K值 (聚类数量)')
    ax.set_ylabel('惯性 (Inertia)')
    ax.set_title('肘部法则 (Elbow Method)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 标记可能的最佳K值点
    if max_k >= 3:
        ax.annotate('可能的最佳K值', xy=(3, inertias[2]), 
                   xytext=(4, inertias[2]+100),
                   arrowprops=dict(facecolor='black', shrink=0.05))
    
    return fig

# 绘制轮廓系数图表
def plot_silhouette_method(X, max_k=10):
    """绘制轮廓系数图表帮助选择K值"""
    silhouette_scores = []
    k_range = range(2, max_k+1)  # 轮廓系数不适用于k=1
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, 'go-')
    ax.set_xlabel('K值 (聚类数量)')
    ax.set_ylabel('平均轮廓系数')
    ax.set_title('轮廓系数法 (Silhouette Method)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 标记最佳K值点
    best_k = k_range[np.argmax(silhouette_scores)]
    ax.annotate(f'最佳K值={best_k}', 
               xy=(best_k, max(silhouette_scores)), 
               xytext=(best_k+1, max(silhouette_scores)-0.1),
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    return fig

# 数据生成与探索模块
def data_generation_section():
    st.header("📊 聚类数据生成与探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("选择数据类型", 
                               ["球形聚类", "半月形聚类", "环形聚类", "不均匀密度聚类"])
        n_samples = st.slider("样本数量", 100, 1000, 300)
        
        # 根据数据类型显示不同的参数
        if data_type == "球形聚类":
            n_centers = st.slider("聚类中心数量", 2, 6, 3)
            cluster_std = st.slider("聚类标准差（离散程度）", 0.3, 2.0, 0.8, 0.1)
            noise = 0.05
        elif data_type in ["半月形聚类", "环形聚类"]:
            n_centers = 2  # 这些类型数据固定为2个聚类
            cluster_std = 0.8
            noise = st.slider("噪声水平", 0.01, 0.3, 0.05, 0.01)
        else:  # 不均匀密度聚类
            n_centers = 3  # 固定为3个聚类
            cluster_std = 0.8
            noise = 0.05
        
        X, y_true = generate_cluster_data(data_type, n_samples, n_centers, cluster_std, noise)
        
        st.write(f"数据统计:")
        st.write(f"- 样本数量: {X.shape[0]}")
        st.write(f"- 特征数量: {X.shape[1]}")
        st.write(f"- 特征1均值: {np.mean(X[:, 0]):.2f}, 标准差: {np.std(X[:, 0]):.2f}")
        st.write(f"- 特征2均值: {np.mean(X[:, 1]):.2f}, 标准差: {np.std(X[:, 1]):.2f}")
    
    with col2:
        # 显示原始数据（不带聚类标签）
        fig_raw = plot_cluster_data(X, title=f'{data_type}原始数据分布')
        st.pyplot(fig_raw)
        
        # 如果有真实标签，显示带有标签的数据
        if y_true is not None and data_type != "半月形聚类" and data_type != "环形聚类":
            fig_labeled = plot_cluster_data(X, y_true, title=f'{data_type}真实聚类分布')
            st.pyplot(fig_labeled)
    
    st.info("""
    **聚类数据特点:**
    - 球形聚类: 数据自然形成球形簇，适合KMeans算法
    - 半月形/环形聚类: 非凸形状的聚类，KMeans效果较差
    - 不均匀密度聚类: 不同簇的密度差异大，对KMeans是挑战
    
    KMeans算法对球形、密度相近的聚类效果最好。
    """)
    
    # 存储数据供后续模块使用
    st.session_state.X = X
    st.session_state.data_type = data_type
    
    return f"数据生成模块: 创建了{data_type}数据，样本数={n_samples}"

# KMeans基本原理模块
def kmeans_basics_section():
    st.header("🔍 KMeans聚类基本原理")
    
    # 移除左右分栏，改为上下排版
    st.markdown("""
    **KMeans聚类核心思想:**
    KMeans是一种无监督学习算法，用于将数据自动分组为K个不同的簇。
    
    **算法步骤:**
    1. **初始化**: 选择K个初始中心点
    2. **分配**: 将每个数据点分配到最近的中心点所在的簇
    3. **更新**: 计算每个簇的平均值，作为新的中心点
    4. **重复**: 重复步骤2和3，直到中心点不再显著变化
    
    **数学表达:**
    目标是最小化所有数据点到其所属簇中心的距离平方和（惯性）:
    $$\\min \\sum_{k=1}^{K} \\sum_{x_i \\in C_k} ||x_i - \\mu_k||^2$$
    
    其中$C_k$是第k个簇，$\\mu_k$是第k个簇的中心。
    """)
    
    if 'X' not in st.session_state:
        st.session_state.X, _ = generate_cluster_data("球形聚类", 300, 3, 0.8)
    
    X = st.session_state.X
    
    # 展示KMeans的两个核心步骤
    st.subheader("核心步骤演示")
    k = st.slider("选择聚类数量K", 2, 5, 3)
    
    if st.button("演示KMeans核心步骤"):
        steps = kmeans_step_visualization(X, k, max_iter=5)
        figs = plot_kmeans_steps(X, steps)
        
        for fig in figs:
            st.pyplot(fig)
            time.sleep(1)
    
    # 以下内容原本在右侧列，现在移至下方
    st.subheader("KMeans聚类动画演示")
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif", 
             caption="KMeans聚类收敛过程动画")
    
    st.subheader("KMeans的几何解释")
    st.markdown("""
    - 每个簇由其中心点（质心）代表
    - 数据点根据距离最近的质心进行分组
    - 聚类边界是 Voronoi 图（垂直平分线）
    - 算法最终收敛到局部最优解
    
    ![Voronoi图](https://upload.wikimedia.org/wikipedia/commons/5/54/Euclidean_Voronoi_diagram.svg)
    """)
    
    return f"KMeans基本原理模块: 演示了K={k}时的聚类步骤"

# K值选择模块
def k_selection_section():
    st.header("🎯 K值选择方法")
    
    # 检查是否已有数据，没有则生成默认数据
    if 'X' not in st.session_state:
        st.session_state.X, _ = generate_cluster_data("球形聚类", 300, 3, 0.8)
    
    X = st.session_state.X
    
    # 移除左右分栏，改为上下排列
    st.subheader("肘部法则 (Elbow Method)")
    st.markdown("""
    肘部法则通过绘制不同K值对应的惯性（Inertia）来选择最佳K值：
    - 惯性：所有样本到其最近簇中心的距离平方和
    - 随着K增大，惯性会减小
    - 最佳K值出现在"肘部"位置，即惯性开始缓慢下降的点
    
    优点：计算简单快速
    
    缺点：主观性强，有时没有明显的肘部
    """)
    
    max_k_elbow = st.slider("最大K值（肘部法则）", 5, 15, 10)
    fig_elbow = plot_elbow_method(X, max_k_elbow)
    st.pyplot(fig_elbow)
    
    # 轮廓系数法部分移至下方
    st.subheader("轮廓系数法 (Silhouette Method)")
    st.markdown("""
    轮廓系数衡量每个样本与其自身簇内样本的相似度，以及与其他簇样本的不相似度：
    - 取值范围：[-1, 1]
    - 接近1：样本聚类合理
    - 接近0：样本位于两个簇的边界
    - 接近-1：样本可能被分到错误的簇
    
    优点：不需要知道真实标签，提供了聚类质量的量化评估
    
    缺点：计算成本高，对球形簇效果好但对非凸形状效果差
    """)
    
    max_k_silhouette = st.slider("最大K值（轮廓系数）", 5, 15, 10)
    fig_silhouette = plot_silhouette_method(X, max_k_silhouette)
    st.pyplot(fig_silhouette)
    
    # 不同K值对比部分保持不变
    st.subheader("不同K值聚类结果对比")
    k1 = st.slider("K1", 2, 6, 2)
    k2 = st.slider("K2", 2, 6, 3)
    k3 = st.slider("K3", 2, 6, 4)
    
    fig_compare = plot_k_comparison(X, [k1, k2, k3])
    st.pyplot(fig_compare)
    
    st.info("""
    **K值选择建议:**
    - 结合肘部法则和轮廓系数法进行判断
    - 考虑实际业务需求和解释性
    - 对于新数据，可以尝试多种K值并评估结果
    - 没有放之四海而皆准的最佳K值，需要根据具体情况选择
    """)
    
    return f"K值选择模块: 比较了K={k1}, {k2}, {k3}的聚类结果"

# KMeans局限性模块
def kmeans_limitations_section():
    st.header("⚠️ KMeans聚类的局限性")
    
    # 检查是否已有数据，没有则生成默认数据
    if 'X' not in st.session_state:
        st.session_state.X, _ = generate_cluster_data("球形聚类", 300, 3, 0.8)
    
    data_type = st.session_state.data_type if 'data_type' in st.session_state else "球形聚类"
    
    # 移除左右分栏，改为上下排列
    st.subheader("对非球形簇的处理")
    st.markdown("""
    KMeans假设聚类是凸形和球形的，对非球形簇效果较差：
    - 无法正确识别半月形、环形等复杂形状
    - 倾向于将数据分成大小相近的簇
    """)
    
    # 展示KMeans在半月形数据上的表现
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    kmeans_moons = KMeans(n_clusters=2, random_state=42)
    labels_moons = kmeans_moons.fit_predict(X_moons)
    
    fig_moons = plt.figure(figsize=(10, 6))
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1], 
               c='red', marker='X', s=200)
    plt.title('KMeans在半月形数据上的表现')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_moons)
    
    # 第二个局限性：对不同密度簇的处理
    st.subheader("对不同密度簇的处理")
    st.markdown("""
    KMeans对密度差异大的簇处理不佳：
    - 倾向于将高密度区域分割成多个簇
    - 低密度区域可能被合并成一个簇
    - 对异常值敏感
    """)
    
    # 生成密度不同的聚类数据
    X_density = np.vstack([
        np.random.normal(loc=(-3, -3), scale=0.5, size=(300, 2)),  # 高密度簇
        np.random.normal(loc=(0, 0), scale=1.2, size=(150, 2)),    # 中等密度簇
        np.random.normal(loc=(3, 3), scale=0.8, size=(50, 2))      # 低密度簇
    ])
    
    kmeans_density = KMeans(n_clusters=3, random_state=42)
    labels_density = kmeans_density.fit_predict(X_density)
    
    fig_density = plt.figure(figsize=(10, 6))
    plt.scatter(X_density[:, 0], X_density[:, 1], c=labels_density, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans_density.cluster_centers_[:, 0], kmeans_density.cluster_centers_[:, 1], 
               c='red', marker='X', s=200)
    plt.title('KMeans在不同密度簇上的表现')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_density)
    
    # 第三个局限性：初始中心点敏感性
    st.subheader("初始中心点敏感性")
    st.markdown("""
    KMeans的结果受初始中心点选择影响：
    - 不同的初始点可能导致不同的聚类结果
    - 可能收敛到局部最优而非全局最优
    """)
    
    # 展示不同初始点的影响
    X = st.session_state.X
    k = st.slider("聚类数量", 2, 5, 3)
    
    if st.button("展示初始点影响"):
        fig_initial, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, seed in enumerate([42, 100, 200]):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=1)  # n_init=1确保只运行一次
            labels = kmeans.fit_predict(X)
            
            axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
            axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                          c='red', marker='X', s=200)
            axes[i].set_title(f'随机种子={seed}, 惯性={kmeans.inertia_:.2f}')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig_initial)
    
    st.info("""
    **KMeans局限性总结:**
    1. 需要预先指定K值
    2. 对初始中心点敏感
    3. 只能发现凸形、球形簇
    4. 对噪声和异常值敏感
    5. 对不同大小和密度的簇处理不佳
    6. 不适合高维数据（维度灾难）
    
    **改进方法:**
    - 使用KMeans++初始化中心点
    - 多次运行取最优结果
    - 对高维数据先进行降维
    - 考虑使用DBSCAN等其他聚类算法处理非球形数据
    """)
    
    return f"KMeans局限性模块: 展示了K={k}时的初始点影响"

# 聚类评估指标模块
def evaluation_metrics_section():
    st.header("📈 聚类评估指标")
    
    # 生成有明确聚类的数据
    X, y_true = generate_cluster_data("球形聚类", 300, 3, 0.8)
    
    # 内部评估指标部分（上半部分）
    st.subheader("内部评估指标（无真实标签）")
    st.markdown("""
    当没有真实标签时，使用内部指标评估聚类质量：
    
    1. **惯性 (Inertia)**
       - 所有样本到其最近簇中心的距离平方和
       - 值越小表示聚类越紧凑
       - 缺点：随着K增大单调减小，无法确定最佳K值
    
    2. **轮廓系数 (Silhouette Score)**
       - 衡量样本与自身簇的相似度和与其他簇的差异性
       - 范围：[-1, 1]，越接近1越好
    
    3. **Calinski-Harabasz指数**
       - 簇间离散度与簇内离散度的比值
       - 值越大表示聚类质量越好
    
    4. **Davies-Bouldin指数**
       - 衡量簇之间的相似度
       - 值越小表示聚类质量越好
    """)
    
    k = st.slider("选择聚类数量K", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 计算评估指标
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    
    st.write("### 评估结果:")
    st.write(f"- 惯性: {inertia:.2f}")
    st.write(f"- 轮廓系数: {silhouette:.4f}")
    st.write(f"- Calinski-Harabasz指数: {calinski:.2f}")
    st.write(f"- Davies-Bouldin指数: {davies:.4f}")

    # 导入所需的评估指标函数
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score
    
    # 显示带真实标签的数据
    fig_true = plot_cluster_data(X, y_true, title="真实聚类分布")
    st.pyplot(fig_true)
    
    # 显示聚类结果
    fig_pred = plot_cluster_data(X, labels, kmeans.cluster_centers_, title=f"K={k}的聚类结果")
    st.pyplot(fig_pred)
    
    # 添加分隔线，增强视觉区分
    st.markdown("---")
    
    # 外部评估指标部分（下半部分）
    st.subheader("外部评估指标（有真实标签）")
    st.markdown("""
    当有真实标签时，使用外部指标评估聚类质量：
    
    1. **调整兰德指数 (ARI)**
       - 衡量聚类结果与真实标签的一致性
       - 范围：[-1, 1]，1表示完全一致
    
    2. **调整互信息 (AMI)**
       - 衡量两个聚类分布的一致性
       - 范围：[0, 1]，1表示完全一致
    
    3. **同质性 (Homogeneity)**
       - 每个簇是否只包含单一类别的样本
       - 范围：[0, 1]，1表示完全同质
    
    4. **完整性 (Completeness)**
       - 同一类别的样本是否被分配到同一个簇
       - 范围：[0, 1]，1表示完全完整
    """)
    
    # 计算外部评估指标
    ari = adjusted_rand_score(y_true, labels)
    ami = adjusted_mutual_info_score(y_true, labels)
    homogeneity = homogeneity_score(y_true, labels)
    completeness = completeness_score(y_true, labels)
    
    st.write("### 外部评估结果:")
    st.write(f"- 调整兰德指数: {ari:.4f}")
    st.write(f"- 调整互信息: {ami:.4f}")
    st.write(f"- 同质性: {homogeneity:.4f}")
    st.write(f"- 完整性: {completeness:.4f}")
    
    st.info("""
    **评估指标选择指南:**
    - 无真实标签: 主要使用轮廓系数和Calinski-Harabasz指数
    - 有真实标签: 优先使用调整兰德指数和调整互信息
    - 单一指标不足以评估聚类质量，应综合多个指标
    - 最重要的评估是聚类结果是否有实际业务意义
    """)
    
    return f"聚类评估模块: 评估了K={k}时的聚类结果"
    

# 概念测验模块
def quiz_section():
    st.header("🎯 KMeans聚类概念测验")
    
    question = st.selectbox(
        "选择一个问题测试你的理解:",
        [
            "KMeans中的K代表什么?",
            "KMeans的目标是什么?",
            "为什么KMeans对初始中心点敏感?",
            "KMeans适合处理什么样的数据?",
            "肘部法则的原理是什么?"
        ]
    )
    
    if question == "KMeans中的K代表什么?":
        answer = st.radio("选择正确答案:", [
            "迭代次数",
            "聚类的数量",
            "特征的维度"
        ], key="q1")
        if st.button("检查答案", key="b1"):
            if answer == "聚类的数量":
                st.success("✅ 正确! K代表我们希望将数据分成的聚类数量。")
            else:
                st.error("❌ 不正确，K代表的是聚类的数量，即我们希望将数据分成多少个组。")
    
    elif question == "KMeans的目标是什么?":
        answer = st.radio("选择正确答案:", [
            "最大化簇间距离，最小化簇内距离",
            "最小化所有数据点到其簇中心的距离平方和",
            "使每个簇的样本数量尽可能相等"
        ], key="q2")
        if st.button("检查答案", key="b2"):
            if answer == "最小化所有数据点到其簇中心的距离平方和":
                st.success("✅ 正确! KMeans的目标是最小化惯性（inertia），即所有样本到其最近簇中心的距离平方和。")
            else:
                st.error("❌ 不正确，KMeans的核心目标是最小化所有数据点到其所属簇中心的距离平方和。")
    
    elif question == "为什么KMeans对初始中心点敏感?":
        answer = st.radio("选择正确答案:", [
            "因为算法会收敛到局部最优而非全局最优",
            "因为初始点决定了特征权重",
            "因为计算精度有限"
        ], key="q3")
        if st.button("检查答案", key="b3"):
            if answer == "因为算法会收敛到局部最优而非全局最优":
                st.success("✅ 正确! KMeans使用贪婪算法，不同的初始点可能导致收敛到不同的局部最优解。")
            else:
                st.error("❌ 不正确，KMeans对初始点敏感是因为它会收敛到局部最优，而非全局最优解。")
    
    elif question == "KMeans适合处理什么样的数据?":
        answer = st.radio("选择正确答案:", [
            "高维稀疏数据",
            "非凸形状的聚类数据",
            "球形、密度相近的聚类数据"
        ], key="q4")
        if st.button("检查答案", key="b4"):
            if answer == "球形、密度相近的聚类数据":
                st.success("✅ 正确! KMeans对球形、凸形且密度相近的聚类数据效果最好。")
            else:
                st.error("❌ 不正确，KMeans最适合处理球形、密度相近的聚类数据，对非凸形状数据效果较差。")
    
    elif question == "肘部法则的原理是什么?":
        answer = st.radio("选择正确答案:", [
            "找到轮廓系数最大的K值",
            "找到惯性开始缓慢下降的K值点",
            "找到与真实标签最匹配的K值"
        ], key="q5")
        if st.button("检查答案", key="b5"):
            if answer == "找到惯性开始缓慢下降的K值点":
                st.success("✅ 正确! 肘部法则通过观察惯性随K值增加的变化，选择惯性开始缓慢下降的'肘部'位置作为最佳K值。")
            else:
                st.error("❌ 不正确，肘部法则的原理是找到惯性开始缓慢下降的K值点，这个点被称为'肘部'。")
    
    return f"概念测验模块: 当前问题='{question}'"

# 实际应用案例模块
def real_world_example_section():
    st.header("🌍 KMeans聚类实际应用案例")
    
    example = st.selectbox(
        "选择实际应用案例:",
        ["客户分群分析", "图像压缩", "异常检测", "文本聚类", "上传自己的数据"]
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
            
            if len(data.columns) < 2:
                st.error("数据至少需要包含两个特征列!")
                return
            
            # 标准化数据
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            
            analyze_custom_data(X, data.columns)
            return f"实际应用模块: 上传自定义数据"
    else:
        # 生成或加载示例数据
        X, feature_names, description = load_example_dataset(example)
        st.write(description)
        
        analyze_custom_data(X, feature_names)
        return f"实际应用模块: 使用{example}数据集"

# 加载示例数据集
def load_example_dataset(example_name):
    np.random.seed(42)
    
    if example_name == "客户分群分析":
        # 生成客户分群数据：RFM模型相关特征
        n_samples = 500
        
        # 特征：消费频率、平均消费金额、最近消费时间（天）
        freq = np.random.normal(15, 8, n_samples)
        amount = np.random.normal(500, 300, n_samples)
        recency = np.random.normal(30, 20, n_samples)
        
        # 确保值为正数
        freq = np.abs(freq)
        amount = np.abs(amount)
        recency = np.abs(recency)
        
        X = np.column_stack((freq, amount, recency))
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        feature_names = ["消费频率", "平均消费金额", "最近消费时间(天)"]
        description = "客户分群分析: 基于RFM模型的客户价值分析，帮助企业识别高价值客户群体"
        return X, feature_names, description
    
    elif example_name == "图像压缩":
        # 生成简单的图像数据（2D像素）
        from sklearn.datasets import load_sample_image
        
        # 加载示例图像并简化
        china = load_sample_image("china.jpg")
        # 缩小图像尺寸
        china = china[::10, ::10]
        # 转换为二维数组
        X = china.reshape(-1, 3)
        # 只取前5000个像素加速处理
        X = X[:5000]
        
        feature_names = ["R", "G", "B"]
        description = "图像压缩: 使用KMeans将图像颜色聚类，用较少的颜色表示图像，实现压缩效果"
        return X, feature_names, description
    
    elif example_name == "异常检测":
        # 生成正常数据和异常数据
        n_normal = 450
        n_anomalies = 50
        
        # 正常数据（三个簇）
        normal1 = np.random.normal(loc=(0, 0), scale=0.5, size=(n_normal//3, 2))
        normal2 = np.random.normal(loc=(3, 3), scale=0.7, size=(n_normal//3, 2))
        normal3 = np.random.normal(loc=(-3, 3), scale=0.6, size=(n_normal//3, 2))
        
        # 异常数据（远离正常簇）
        anomalies = np.random.uniform(low=-6, high=6, size=(n_anomalies, 2))
        # 过滤掉可能混入正常簇的异常点
        anomalies = anomalies[np.linalg.norm(anomalies, axis=1) > 4]
        
        X = np.vstack([normal1, normal2, normal3, anomalies])
        
        feature_names = ["特征1", "特征2"]
        description = "异常检测: 通过KMeans识别远离所有簇中心的点，这些点可能是异常值"
        return X, feature_names, description
    
    elif example_name == "文本聚类":
        # 生成文本数据（使用TF-IDF特征）
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 生成一些示例文本
        texts = [
            "机器学习是人工智能的一个分支",
            "深度学习是机器学习的一个子领域",
            "神经网络是深度学习的基础",
            "卷积神经网络适用于图像识别",
            "循环神经网络适用于序列数据",
            "支持向量机是一种分类算法",
            "决策树是一种简单的机器学习模型",
            "随机森林是多个决策树的集成",
            "聚类算法属于无监督学习",
            "KMeans是一种常用的聚类算法",
            "足球是世界上最受欢迎的运动",
            "篮球在美国非常流行",
            "网球是一项优雅的运动",
            "奥运会每四年举办一次",
            "世界杯是足球界的最高赛事",
            "Python是一种流行的编程语言",
            "Java是一种面向对象的编程语言",
            "C++运行速度很快",
            "JavaScript用于网页开发",
            "R语言常用于数据分析"
        ]
        
        # 重复文本以增加样本量
        texts = texts * 10
        
        # 提取TF-IDF特征
        vectorizer = TfidfVectorizer(max_features=10)
        X = vectorizer.fit_transform(texts).toarray()
        
        feature_names = vectorizer.get_feature_names_out()
        description = "文本聚类: 将文本转换为向量表示后使用KMeans进行聚类，识别主题相似的文本"
        return X, feature_names, description
    
    return None, None, ""

# 分析自定义数据
def analyze_custom_data(X, feature_names):
    if X.shape[0] < 10:
        st.error("数据点太少，至少需要10个样本!")
        return
    
    # 降维以便可视化（如果特征数大于2）
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        st.info(f"数据已通过PCA降维至2维以便可视化，保留了{sum(pca.explained_variance_ratio_)*100:.1f}%的方差")
    else:
        X_vis = X
    
    # 选择K值
    st.subheader("选择聚类数量K")
    k = st.slider("K值", 2, min(10, X.shape[0]//5), 3)
    
    # 运行KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 显示聚类结果
    st.subheader("聚类结果可视化")
    fig = plot_cluster_data(X_vis, labels, kmeans.cluster_centers_ if X.shape[1] <= 2 else pca.transform(kmeans.cluster_centers_))
    st.pyplot(fig)
    
    # 显示评估指标
    st.subheader("聚类评估指标")
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    
    st.write(f"- 轮廓系数: {silhouette:.4f}")
    st.write(f"- Calinski-Harabasz指数: {calinski:.2f}")
    st.write(f"- Davies-Bouldin指数: {davies:.4f}")
    
    # 显示簇中心特征（如果特征数较少）
    if X.shape[1] <= 10:
        st.subheader("各簇中心特征值")
        centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
        centers_df.index = [f'簇 {i}' for i in range(k)]
        st.dataframe(centers_df.style.highlight_max(axis=0))
        
        st.info("""
        **簇中心解释:**
        表格显示了每个簇在各个特征上的中心值，可用于解释不同簇的特征：
        - 数值较高的特征表示该簇在该特征上有明显倾向
        - 通过比较不同簇的中心值，可以发现簇之间的主要差异
        """)

# 主程序
def main():
    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "数据生成与探索"
    
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "数据生成与探索",
        "KMeans基本原理",
        "K值选择方法",
        "KMeans的局限性",
        "聚类评估指标",
        "概念测验",
        "实际应用案例",
        "编程实例（葡萄酒数据集）"
    ])
  
    # 更新会话状态
    st.session_state.section = section
    
    context = ""
    if section == "数据生成与探索":
        context = data_generation_section()
    elif section == "KMeans基本原理":
        context = kmeans_basics_section()
    elif section == "K值选择方法":
        context = k_selection_section()
    elif section == "KMeans的局限性":
        context = kmeans_limitations_section()
    elif section == "聚类评估指标":
        context = evaluation_metrics_section()
    elif section == "概念测验":
        context = quiz_section()
    elif section == "实际应用案例":
        context = real_world_example_section()
    elif section == "编程实例（葡萄酒数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        KMeans_step_by_step.main()
        context = "编程实例模块: 编程实例（葡萄酒数据集）分步编程训练"
    
    # 显示聊天界面
    display_chat_interface(context)
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **KMeans聚类交互式学习平台**
    
    设计用于机器学习教学，帮助学生理解:
    - KMeans聚类的基本原理与步骤
    - KMeans聚类的基本原理与步骤
    - K值选择的方法与技巧
    - 聚类结果的评估指标
    - KMeans算法的优缺点与适用场景
    """)


if __name__ == "__main__":
    main()
