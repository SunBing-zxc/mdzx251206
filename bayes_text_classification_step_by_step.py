# streamlit run bayes_text_classification_step_by_step.py
# 贝叶斯文本分类 - 完整流程

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from api_deepseek import ask_ai_assistant  # 导入复用的AI助手函数

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化会话状态
if 'step' not in st.session_state:
    st.session_state.step = 0  # 从0开始
if 'X_train_text' not in st.session_state:
    st.session_state.X_train_text = None  # 训练集文本
if 'X_test_text' not in st.session_state:
    st.session_state.X_test_text = None  # 测试集文本
if 'y_train' not in st.session_state:
    st.session_state.y_train = None  # 训练集标签
if 'y_test' not in st.session_state:
    st.session_state.y_test = None  # 测试集标签
if 'class_names' not in st.session_state:
    st.session_state.class_names = None  # 类别名称
if 'X_train_tfidf' not in st.session_state:
    st.session_state.X_train_tfidf = None  # 训练集TF-IDF特征
if 'X_test_tfidf' not in st.session_state:
    st.session_state.X_test_tfidf = None  # 测试集TF-IDF特征
if 'model' not in st.session_state:
    st.session_state.model = None  # 贝叶斯模型
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None  # 预测结果


# AI代码检查函数（适配贝叶斯文本分类）
def ai_code_checker(step, user_code):
    try:
        if step == 1:
            errors = []
            if 'fetch_20newsgroups' not in user_code:
                errors.append("❌ 请使用fetch_20newsgroups加载新闻数据集")
            if 'subset=\'train\'' not in user_code or 'subset=\'test\'' not in user_code:
                errors.append("❌ 请同时加载训练集(subset='train')和测试集(subset='test')")
            if 'remove=(' not in user_code:
                errors.append("❌ 请移除邮件头、签名和引用(remove=('headers', 'footers', 'quotes'))")
            return "✅ 步骤1通过！" if not errors else "\n".join(errors)
        
        elif step == 2:
            errors = []
            if 'X_train_text =' not in user_code or 'X_test_text =' not in user_code:
                errors.append("❌ 请正确提取训练集和测试集文本数据")
            if 'y_train =' not in user_code or 'y_test =' not in user_code:
                errors.append("❌ 请正确提取训练集和测试集标签")
            if 'Counter' not in user_code:
                errors.append("❌ 请使用Counter统计类别分布")
            return "✅ 步骤2通过！" if not errors else "\n".join(errors)
        
        elif step == 3:
            errors = []
            if 'TfidfVectorizer' not in user_code:
                errors.append("❌ 请使用TfidfVectorizer进行文本特征提取")
            if 'fit_transform' not in user_code:
                errors.append("❌ 训练集应使用fit_transform进行转换")
            if 'transform' not in user_code:
                errors.append("❌ 测试集应使用transform进行转换")
            return "✅ 步骤3通过！" if not errors else "\n".join(errors)
        
        elif step == 4:
            errors = []
            if 'MultinomialNB' not in user_code:
                errors.append("❌ 请导入并使用MultinomialNB模型")
            if 'model = MultinomialNB' not in user_code:
                errors.append("❌ 请正确实例化MultinomialNB模型")
            return "✅ 步骤4通过！" if not errors else "\n".join(errors)
        
        elif step == 5:
            errors = []
            if 'model.fit' not in user_code:
                errors.append("❌ 请使用model.fit训练模型")
            if 'X_train_tfidf' not in user_code or 'y_train' not in user_code:
                errors.append("❌ 训练时应使用X_train_tfidf和y_train")
            return "✅ 步骤5通过！" if not errors else "\n".join(errors)
        
        elif step == 6:
            errors = []
            if 'model.predict' not in user_code:
                errors.append("❌ 请使用model.predict进行预测")
            if 'accuracy_score' not in user_code:
                errors.append("❌ 请计算准确率(accuracy_score)")
            if 'classification_report' not in user_code:
                errors.append("❌ 请生成分类报告(classification_report)")
            return "✅ 步骤6通过！" if not errors else "\n".join(errors)
    
    except Exception as e:
        return f"⚠️ 代码错误：{str(e)}"


# 步骤0：项目说明与数据展示
def step0():
    st.header("步骤0：项目说明")
    st.subheader("朴素贝叶斯文本分类")
    
    # 项目目标
    st.info("""
    **数据集说明**：
    我们将使用20 Newsgroups数据集的一个子集，包含5个新闻主题类别：
    - rec.sport.baseball（棒球运动）
    - rec.motorcycles（摩托车）
    - sci.space（太空科学）
    - comp.graphics（计算机图形学）
    - talk.politics.misc（政治讨论）
    
    **项目目标**：  
    通过朴素贝叶斯算法对新闻文本进行分类，理解文本分类的完整流程，
    包括文本数据预处理、特征提取、模型训练与评估。
    """)
    
    # 数据集预览
    st.subheader("数据集预览")
    # 加载少量数据用于预览
    sample_data = fetch_20newsgroups(
        subset='train',
        categories=['rec.sport.baseball', 'sci.space'],
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )
    
    # 展示样本文本
    st.write("样本文本示例：")
    for i, text in enumerate(sample_data.data[:2]):
        st.text_area(f"样本 {i+1}（类别：{sample_data.target_names[sample_data.target[i]]}）", 
                    text[:300] + "...", height=280, disabled=False)
    
    st.button("进入步骤1：数据加载",
             on_click=lambda: setattr(st.session_state, 'step', 1))


# 步骤1：数据加载
def step1():
    st.header("步骤1：数据加载")
    st.subheader("目标：加载20 Newsgroups数据集的训练集和测试集")
    
    st.info("""
    **任务说明**：  
    1. 使用fetch_20newsgroups加载指定类别的新闻数据
    2. 分别加载训练集(subset='train')和测试集(subset='test')
    3. 移除邮件头、签名和引用内容，减少噪声
    """)
    
    # 代码骨架
    code_skeleton = """
# 导入数据集加载工具
from sklearn.datasets import fetch_20newsgroups

# 选择5个目标新闻主题
target_categories = [
    'rec.sport.baseball',   # 棒球运动
    'rec.motorcycles',      # 摩托车
    'sci.space',            # 太空科学
    'comp.graphics',        # 计算机图形学
    'talk.politics.misc'    # 政治讨论
]

# 加载训练集（用于模型学习）
newsgroups_train = fetch_20newsgroups(
    subset='train',          # 训练集
    categories=target_categories,
    remove=('headers', 'footers', 'quotes'),  # 移除噪声内容
    shuffle=True,            # 打乱数据
    random_state=42          # 固定随机种子，确保结果可复现
)

# 加载测试集（用于模型评估）
newsgroups_test = fetch_20newsgroups(
    subset='test',           # 测试集
    categories=target_categories,
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

# 查看数据集基本信息
print(f"训练集文本数：{len(newsgroups_train.data)}")
print(f"测试集文本数：{len(newsgroups_test.data)}")
print(f"新闻主题类别：{newsgroups_train.target_names}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=850, key="step1_code")
    
    if st.button("运行代码", key="run_step1"):
        try:
            locals_dict = {'fetch_20newsgroups': fetch_20newsgroups}
            exec(user_code, globals(), locals_dict)
            
            # 保存数据到会话状态
            st.session_state.X_train_text = locals_dict['newsgroups_train'].data
            st.session_state.X_test_text = locals_dict['newsgroups_test'].data
            st.session_state.y_train = locals_dict['newsgroups_train'].target
            st.session_state.y_test = locals_dict['newsgroups_test'].target
            st.session_state.class_names = locals_dict['newsgroups_train'].target_names
            
            # 展示结果
            st.success("代码运行成功！")
            with st.expander("查看输出"):
                st.write(f"训练集文本数：{len(st.session_state.X_train_text)}")
                st.write(f"测试集文本数：{len(st.session_state.X_test_text)}")
                st.write(f"新闻主题类别：{st.session_state.class_names}")
            
            # AI反馈
            ai_feedback = ai_code_checker(1, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤2：数据观察与理解",
                         on_click=lambda: setattr(st.session_state, 'step', 2))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤2：数据观察与理解
def step2():
    st.header("步骤2：数据观察与理解")
    st.subheader("目标：探索文本数据特征和类别分布")
    
    if st.session_state.X_train_text is None:
        st.warning("请先完成步骤1！")
        st.button("返回步骤1", on_click=lambda: setattr(st.session_state, 'step', 1))
        return
    
    st.info("""
    **任务说明**：  
    1. 提取文本特征和对应标签
    2. 分析训练集和测试集的类别分布
    3. 查看样本文本内容，了解数据特点
    """)
    
    # 代码骨架
    code_skeleton = """
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# 提取特征与标签
X_train_text = newsgroups_train.data  # 训练集文本
X_test_text = newsgroups_test.data    # 测试集文本
y_train = newsgroups_train.target     # 训练集标签
y_test = newsgroups_test.target       # 测试集标签
class_names = newsgroups_train.target_names  # 类别名称

# 查看样本文本
print("训练集第一个文本示例：")
print(X_train_text[0][:300] + "...")  # 显示前300个字符

# 统计各类别样本数量
train_class_count = Counter(y_train)
test_class_count = Counter(y_test)

print("训练集类别分布：")
for idx, count in train_class_count.items():
    print(f"{class_names[idx]}: {count}个样本")

print("测试集类别分布：")
for idx, count in test_class_count.items():
    print(f"{class_names[idx]}: {count}个样本")

# 绘制类别分布柱状图
plt.figure(figsize=(12, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题

# 训练集分布
plt.subplot(1, 2, 1)
plt.bar(train_class_count.keys(), train_class_count.values(), color='skyblue')
plt.title('训练集新闻主题分布')
plt.ylabel('样本数量')

# 测试集分布
plt.subplot(1, 2, 2)
plt.bar(test_class_count.keys(), test_class_count.values(), color='lightgreen')
plt.title('测试集新闻主题分布')
plt.ylabel('样本数量')

plt.tight_layout()
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=1100, key="step2_code")
    
    if st.button("运行代码", key="run_step2"):
        try:
            # 准备本地变量
            locals_dict = {
                'newsgroups_train': type('obj', (object,), {
                    'data': st.session_state.X_train_text,
                    'target': st.session_state.y_train,
                    'target_names': st.session_state.class_names
                }),
                'newsgroups_test': type('obj', (object,), {
                    'data': st.session_state.X_test_text,
                    'target': st.session_state.y_test
                }),
                'Counter': Counter,
                'plt': plt,
                'np': np
            }
            exec(user_code, globals(), locals_dict)
            
            st.success("数据观察完成！")
            
            # 显示样本文本
            st.subheader("样本文本示例：")
            st.text(st.session_state.X_train_text[0][:500] + "...")
            
            # 显示类别分布图表
            st.subheader("类别分布：")
            # 重新生成图表以显示在Streamlit中
            train_class_count = Counter(st.session_state.y_train)
            test_class_count = Counter(st.session_state.y_test)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 训练集分布
            ax1.bar(train_class_count.keys(), train_class_count.values(), color='skyblue')
            ax1.set_title('训练集新闻主题分布')
            ax1.set_xticks(list(train_class_count.keys()))
            ax1.set_xticklabels([st.session_state.class_names[i] for i in train_class_count.keys()], 
                               rotation=45, ha='right')
            ax1.set_ylabel('样本数量')
            
            # 测试集分布
            ax2.bar(test_class_count.keys(), test_class_count.values(), color='lightgreen')
            ax2.set_title('测试集新闻主题分布')
            ax2.set_xticks(list(test_class_count.keys()))
            ax2.set_xticklabels([st.session_state.class_names[i] for i in test_class_count.keys()], 
                               rotation=45, ha='right')
            ax2.set_ylabel('样本数量')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # AI反馈
            ai_feedback = ai_code_checker(2, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤3：文本特征提取",
                         on_click=lambda: setattr(st.session_state, 'step', 3))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤3：文本特征提取
def step3():
    st.header("步骤3：文本特征提取")
    st.subheader("目标：使用TF-IDF将文本转换为数值特征")
    
    if st.session_state.X_train_text is None:
        st.warning("请先完成步骤2！")
        st.button("返回步骤2", on_click=lambda: setattr(st.session_state, 'step', 2))
        return
    
    st.info("""
    **任务说明**：  
    1. 使用TF-IDF方法将文本转换为数值特征
    2. 训练集使用fit_transform，测试集使用transform
    3. 移除停用词并限制最大特征数量，优化特征质量
    """)
    
    # 代码骨架
    code_skeleton = """
# 导入TF-IDF特征提取工具
from sklearn.feature_extraction.text import TfidfVectorizer

# 初始化TF-IDF转换器
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',  # 移除英语停用词（如"the"、"and"等无实际语义的词）
    max_features=5000,     # 仅保留5000个最常见词，控制特征维度
    min_df=5               # 忽略在少于5篇文本中出现的词
)

# 对训练集文本进行"拟合+转换"
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
# 对测试集文本仅"转换"（使用训练集的词表规则）
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# 查看TF-IDF特征结构
print(f"训练集TF-IDF矩阵形状：{X_train_tfidf.shape}")  # (样本数, 特征数)
print(f"TF-IDF词表大小：{len(tfidf_vectorizer.vocabulary_)}")
print(f"前10个关键词示例：{list(tfidf_vectorizer.vocabulary_.keys())[:10]}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=500, key="step3_code")
    
    if st.button("运行代码", key="run_step3"):
        try:
            locals_dict = {
                'X_train_text': st.session_state.X_train_text,
                'X_test_text': st.session_state.X_test_text,
                'TfidfVectorizer': TfidfVectorizer
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存特征数据
            st.session_state.X_train_tfidf = locals_dict['X_train_tfidf']
            st.session_state.X_test_tfidf = locals_dict['X_test_tfidf']
            st.session_state.tfidf_vectorizer = locals_dict['tfidf_vectorizer']
            
            st.success("特征提取完成！")
            st.write(f"训练集TF-IDF矩阵形状：{st.session_state.X_train_tfidf.shape}")
            st.write(f"TF-IDF词表大小：{len(st.session_state.tfidf_vectorizer.vocabulary_)}")
            st.write(f"前10个关键词示例：{list(st.session_state.tfidf_vectorizer.vocabulary_.keys())[:10]}")
            
            # AI反馈
            ai_feedback = ai_code_checker(3, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤4：构建朴素贝叶斯模型",
                         on_click=lambda: setattr(st.session_state, 'step', 4))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤4：构建贝叶斯模型
def step4():
    st.header("步骤4：构建贝叶斯模型")
    st.subheader("目标：实例化多项式朴素贝叶斯分类模型")
    
    if st.session_state.X_train_tfidf is None:
        st.warning("请先完成步骤3！")
        st.button("返回步骤3", on_click=lambda: setattr(st.session_state, 'step', 3))
        return
    
    st.info("""
    **任务说明**：  
    1. 导入MultinomialNB模型
    2. 实例化多项式朴素贝叶斯模型
    3. 了解模型参数含义
    """)
    
    code_skeleton = """
# 导入多项式朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB

# 初始化模型（alpha为平滑系数，防止概率为0）
model = MultinomialNB(alpha=1.0)

# 查看模型参数
print("模型参数：", model.get_params())
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=250, key="step4_code")
    
    if st.button("运行代码", key="run_step4"):
        try:
            locals_dict = {'MultinomialNB': MultinomialNB}
            exec(user_code, globals(), locals_dict)
            st.session_state.model = locals_dict['model']
            
            st.success("模型构建成功！")
            st.write("模型参数：", st.session_state.model.get_params())
            
            # AI反馈
            ai_feedback = ai_code_checker(4, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤5：模型训练",
                         on_click=lambda: setattr(st.session_state, 'step', 5))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤5：模型训练
def step5():
    st.header("步骤5：模型训练")
    st.subheader("目标：用训练集数据训练朴素贝叶斯模型")
    
    if st.session_state.model is None:
        st.warning("请先完成步骤4！")
        st.button("返回步骤4", on_click=lambda: setattr(st.session_state, 'step', 4))
        return
    
    st.info("""
    **任务说明**：  
    1. 使用训练集的TF-IDF特征和标签训练模型
    2. 分析模型学到的主题-关键词关联
    """)
    
    code_skeleton = """
# 用训练集的TF-IDF特征与标签训练模型
model.fit(X_train_tfidf, y_train)

# 获取词表特征名称
feature_names = tfidf_vectorizer.get_feature_names_out()

# 查看模型学到的"主题-关键词"关联
print("各主题的核心关键词（前5个）：")
for class_idx, class_name in enumerate(class_names):
    # 提取该主题下概率最高的5个词的索引
    top_word_idx = model.feature_log_prob_[class_idx].argsort()[-5:]
    # 映射为词名
    top_words = []
    for idx in top_word_idx:
        top_words.append(feature_names[idx])
    print(f"{class_name}：{top_words}")
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=430, key="step5_code")
    
    if st.button("运行代码", key="run_step5"):
        try:
            locals_dict = {
                'model': st.session_state.model,
                'X_train_tfidf': st.session_state.X_train_tfidf,
                'y_train': st.session_state.y_train,
                'tfidf_vectorizer': st.session_state.tfidf_vectorizer,
                'class_names': st.session_state.class_names
            }
            exec(user_code, globals(), locals_dict)
            
            st.success("模型训练完成！")
            
            # 显示各主题的核心关键词
            feature_names = st.session_state.tfidf_vectorizer.get_feature_names_out()
            st.subheader("各主题的核心关键词：")
            for class_idx, class_name in enumerate(st.session_state.class_names):
                top_word_idx = st.session_state.model.feature_log_prob_[class_idx].argsort()[-5:]
                top_words = [feature_names[idx] for idx in top_word_idx]
                st.write(f"{class_name}：{top_words}")
            
            # AI反馈
            ai_feedback = ai_code_checker(5, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.button("进入步骤6：模型评估与可视化",
                         on_click=lambda: setattr(st.session_state, 'step', 6))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤6：模型评估与可视化
def step6():
    st.header("步骤6：模型评估与可视化")
    st.subheader("目标：评估模型性能并可视化关键结果")
    
    if st.session_state.model is None:
        st.warning("请先完成步骤5！")
        st.button("返回步骤5", on_click=lambda: setattr(st.session_state, 'step', 5))
        return
    
    st.info("""
    **任务说明**：  
    1. 用训练好的模型预测测试集文本类别
    2. 计算准确率和详细分类报告
    3. 可视化各主题的核心关键词
    """)
    
    code_skeleton = """
# 导入分类评估工具
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 用训练好的模型预测测试集文本类别
y_pred = model.predict(X_test_tfidf)

# 计算测试集准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集分类准确率：{accuracy:.4f}")

# 输出详细分类报告
print("文本分类详细报告：")
print(classification_report(
    y_test, y_pred,
    target_names=class_names
))

# 定义每个主题的关键词数量
n_top_words = 10

# 创建子图布局（2行3列，适配5个主题）
plt.figure(figsize=(15, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']

for i, class_name in enumerate(class_names):
    # 提取该主题下对数概率最高的10个词的索引
    top_word_idx = model.feature_log_prob_[i].argsort()[-n_top_words:]
    # 映射为词名和对应的概率值
    top_words = []
    for idx in top_word_idx:
        top_words.append(feature_names[idx])
    top_probs = []
    for idx in top_word_idx:
        top_probs.append(model.feature_log_prob_[i][idx])
    
    # 绘制水平条形图
    plt.subplot(2, 3, i+1)
    plt.barh(top_words, top_probs, color='salmon')
    plt.title(f'主题：{class_name}')
    plt.xlabel('关键词对数概率（值越高，主题代表性越强）')

plt.tight_layout()
plt.show()
    """.strip()
    
    user_code = st.text_area("请补充代码：", code_skeleton, height=1050, key="step6_code")
    
    if st.button("运行代码", key="run_step6"):
        try:
            locals_dict = {
                'model': st.session_state.model,
                'X_test_tfidf': st.session_state.X_test_tfidf,
                'y_test': st.session_state.y_test,
                'class_names': st.session_state.class_names,
                'feature_names': st.session_state.tfidf_vectorizer.get_feature_names_out(),
                'accuracy_score': accuracy_score,
                'classification_report': classification_report,
                'plt': plt
            }
            exec(user_code, globals(), locals_dict)
            
            # 保存预测结果
            st.session_state.y_pred = locals_dict['y_pred']
            st.session_state.accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            
            st.success("模型评估完成！")
            st.write(f"测试集分类准确率：{st.session_state.accuracy:.4f}")
            
            # 显示分类报告
            st.subheader("分类详细报告：")
            report = classification_report(
                st.session_state.y_test, 
                st.session_state.y_pred,
                target_names=st.session_state.class_names
            )
            st.text(report)
            
            # 显示关键词可视化
            st.subheader("各主题核心关键词：")
            n_top_words = 10
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.tight_layout(pad=5.0)
            
            for i, class_name in enumerate(st.session_state.class_names):
                row, col = i // 3, i % 3
                top_word_idx = st.session_state.model.feature_log_prob_[i].argsort()[-n_top_words:]
                top_words = [locals_dict['feature_names'][idx] for idx in top_word_idx]
                top_probs = [st.session_state.model.feature_log_prob_[i][idx] for idx in top_word_idx]
                
                axes[row, col].barh(top_words, top_probs, color='salmon')
                axes[row, col].set_title(f'主题：{class_name}')
                axes[row, col].set_xlabel('关键词对数概率')
            
            # 隐藏最后一个未使用的子图
            if len(st.session_state.class_names) < 6:
                axes[1, 2].axis('off')
                
            st.pyplot(fig)
            
            # AI分析评估结果
            context = f"这是贝叶斯文本分类模型的准确率：{st.session_state.accuracy}。"
            question = "请用300字简单且生动易懂的语言解释模型表现如何，哪些主题可能更容易分类？"
            with st.spinner("AI助教正在分析评估结果..."):
                st.info(ask_ai_assistant(question, context))
            
            # AI反馈
            ai_feedback = ai_code_checker(6, user_code)
            st.info(f"AI提示：{ai_feedback}")
            
            if "✅" in ai_feedback:
                st.subheader("恭喜！已用sklearn库完成朴素贝叶斯文本分类全流程")
                st.button("进入步骤7：总结与思考",
                         on_click=lambda: setattr(st.session_state, 'step', 7))
        
        except Exception as e:
            st.error(f"执行错误：{str(e)}")


# 步骤7：总结与思考
def step7():
    st.header("步骤7：总结与思考")
    st.subheader("目标：梳理朴素贝叶斯文本分类流程并深入思考关键问题")
    
    # 检查前置条件
    if st.session_state.step < 6:
        st.warning("请先完成前面所有步骤再进行总结！")
        st.button("返回步骤6", on_click=lambda: setattr(st.session_state, 'step', 6))
        return

    st.info("""
    **任务说明**：  
    1. 回顾朴素贝叶斯算法解决文本分类问题的完整流程  
    2. 思考文本分类的关键步骤和影响因素  
    3. 分析模型在不同类别上的表现差异原因
    """)

    # 显示步骤6的评估指标
    st.subheader("📊 分类评估结果回顾")
    if 'accuracy' in st.session_state:
        st.metric("测试集准确率", f"{st.session_state.accuracy:.4f}")
        
        # 显示分类报告
        st.subheader("分类详细报告：")
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred,
            target_names=st.session_state.class_names,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.warning("未找到完整的分类评估数据，请确保已完成步骤6")


    # 学生思考输入（关键问题）
    st.write("### 思考与分析")
    student_answer = st.text_area(
        """请思考以下问题：
            1. 朴素贝叶斯算法为什么适合文本分类任务？
            2. 从分类结果来看，哪些主题的分类效果较好，为什么？
            3. 如何进一步提高模型的分类性能？""",
        height=150,
        key="student_answer"
    )
    
    # 调用AI评价
    if st.button("获取AI评价", key="get_ai_feedback"):
        if not student_answer.strip():
            st.warning("请先输入你的思考内容")
            return
        
        if 'accuracy' not in st.session_state:
            st.warning("未找到评估数据，请先完成步骤6的模型评估")
            return
        
        # 构建包含评估指标和学生回答的上下文
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred,
            target_names=st.session_state.class_names,
            output_dict=True
        )
        context = f"""
        分类评估指标数据：
        - 准确率（Accuracy）：{st.session_state.accuracy:.4f}
        
        分类详细报告：
        {pd.DataFrame(report).transpose().to_string()}
        
        学生的分析回答：
        {student_answer}
        """
        
        # 获取AI评价
        response = ask_ai_assistant(
            question="请结合提供的分类评估指标和学生的分析，先解读评估指标含义，重点评价学生分析的合理性，并补充专业建议。",
            context=context.strip()
        )
        
        # 显示AI评价
        st.write("### AI评价与分析")
        st.info(response)
    
    # 重新开始按钮
    st.button("重新开始全部流程", on_click=lambda: setattr(st.session_state, 'step', 0))


# 主程序
def main():
    st.title("📝 朴素贝叶斯文本分类分步编程训练")
    st.write("基于20 Newsgroups数据集，用sklearn完成文本分类全流程，AI辅助检查代码")
    
    # 侧边栏步骤进度
    st.sidebar.title("步骤进度")
    steps = [
        "0. 项目说明",
        "1. 数据加载", "2. 数据观察", "3. 特征提取",
        "4. 模型构建", "5. 模型训练", "6. 结果评估", "7. 总结与思考"
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
