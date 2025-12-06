#  C:\Users\孙冰\Desktop\AI助教
#  streamlit run text_analysis_demo.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.datasets import fetch_20newsgroups
import jieba
import re
import time
import bayes_text_classification_step_by_step
from api_deepseek import client, ask_ai_assistant

# 页面设置
st.set_page_config(page_title="文本分析与分类学习平台", layout="wide")
st.title("📄 文本分析与分类交互式学习平台")

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

def display_chat_interface(context=""):
    """显示贝叶斯文本分类相关的聊天界面"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设贝叶斯文本分类相关的快捷问题
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是贝叶斯文本分类?")
        btn2 = st.button("贝叶斯分类的核心原理?")
    
    with col2:
        btn3 = st.button("TF-IDF的作用是什么?")
        btn4 = st.button("贝叶斯分类的优缺点?")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是贝叶斯文本分类?它适用于哪些场景?"
    elif btn2:
        question = "贝叶斯文本分类的核心原理是什么?基于哪些数学公式?"
    elif btn3:
        question = "在文本分类中，TF-IDF特征提取的作用是什么?如何计算?"
    elif btn4:
        question = "贝叶斯文本分类有哪些优点和缺点?与其他分类算法相比有何特点?"
    
    # 提问输入框
    user_input = st.sidebar.text_input("输入你的问题(关于贝叶斯文本分类):", key="question_input")
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

# 文本预处理函数
def preprocess_text(text, is_chinese=False):
    """文本预处理：清洗、分词"""
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\u4e00-\u9fa5]', ' ', text)
    # 转为小写
    text = text.lower() if not is_chinese else text
    # 分词
    if is_chinese:
        words = jieba.cut(text)
        return " ".join(words)
    else:
        return text

# 加载示例数据
@st.cache_data
def load_sample_data(dataset_name):
    """加载不同类型的文本数据集"""
    if dataset_name == "新闻主题分类":
        # 加载英文新闻数据集
        categories = ['rec.sport.hockey', 'sci.space', 'talk.politics.misc', 'comp.graphics']
        newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
        texts = [preprocess_text(text) for text in newsgroups.data[:500]]  # 取部分数据加快速度
        labels = newsgroups.target[:500]
        label_names = [newsgroups.target_names[i] for i in range(len(newsgroups.target_names))]
        return texts, labels, label_names, "英文"
    
    elif dataset_name == "中文情感分析":
        # 生成模拟中文情感数据（正面/负面）
        positive_samples = [
            "这个产品非常好，我很满意", "质量很棒，推荐购买", "体验超出预期，值得拥有",
            "服务态度很好，下次还会再来", "性价比高，非常划算", "物流很快，包装完好",
            "效果显著，确实有效", "外观设计很漂亮，很喜欢", "使用简单方便，操作流畅",
            "味道很好，家人都喜欢"
        ]
        negative_samples = [
            "质量太差了，完全不值这个价", "服务态度恶劣，非常失望", "一点用都没有，浪费钱",
            "物流太慢，包装破损", "体验很差，不会再买了", "味道很难闻，无法接受",
            "外观粗糙，有瑕疵", "操作复杂，一点都不方便", "效果很差，不如宣传的好",
            "性价比低，不推荐购买"
        ]
        # 扩展样本数量
        texts = positive_samples * 30 + negative_samples * 30
        labels = [1] * 300 + [0] * 300  # 1:正面, 0:负面
        label_names = ["负面", "正面"]
        # 中文预处理
        texts = [preprocess_text(text, is_chinese=True) for text in texts]
        return texts, labels, label_names, "中文"
    
    else:  # 自定义文本
        return [], [], [], "中文"

# 特征提取演示
def demo_feature_extraction(texts, lang):
    """演示词袋模型和TF-IDF"""
    st.subheader("文本向量化：从文字到数字")
    
    # 选择向量化方法
    vec_method = st.radio("选择向量化方法", ["词袋模型 (CountVectorizer)", "TF-IDF (TfidfVectorizer)"])
    
    # 设置参数
    max_features = st.slider("最大特征数", 10, 500, 100)
    ngram_range = st.slider("N-gram范围", 1, 3, 1)
    
    # 初始化向量器
    if vec_method.startswith("词袋"):
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(ngram_range, ngram_range),
            stop_words="english" if lang == "英文" else None
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(ngram_range, ngram_range),
            stop_words="english" if lang == "英文" else None
        )
    
    # 拟合并转换
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # 展示结果
    st.write(f"向量化后形状: {X.shape} (样本数 × 特征数)")
    
    # 显示前5个样本的特征
    if len(texts) >= 5:
        st.subheader("样本特征示例")
        df = pd.DataFrame(
            X[:5].toarray(), 
            columns=feature_names,
            index=[f"样本 {i+1}" for i in range(5)]
        )
        st.dataframe(df.style.highlight_max(axis=1))
    
    # 解释向量化原理
    st.info(f"""
    **{vec_method}原理**:
    - 将文本转换为数值特征向量，才能被机器学习模型处理
    - 词袋模型：统计每个词在文本中出现的频率
    - TF-IDF：同时考虑词在当前文本的频率和在所有文本中的分布
    - N-gram：考虑连续的n个词组合，如"机器学习"作为一个特征
    """)
    
    return X, vectorizer

# 模型训练与评估
def train_and_evaluate(X, y, label_names):
    """训练分类模型并评估"""
    st.subheader("模型训练与评估")
    
    # 划分训练集和测试集
    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 选择模型
    model_name = st.selectbox("选择分类模型", ["朴素贝叶斯 (MultinomialNB)", "逻辑回归 (LogisticRegression)"])
    
    # 初始化模型
    if model_name.startswith("朴素"):
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)
    
    # 训练模型
    with st.spinner("模型训练中..."):
        time.sleep(1)  # 模拟训练时间
        model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估指标
    st.subheader("模型评估结果")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"准确率 (Accuracy): {acc:.4f}")
    
    # 分类报告
    st.text("分类详细报告:")
    report = classification_report(y_test, y_pred, target_names=label_names)
    st.text(report)
    
    # 混淆矩阵
    st.subheader("混淆矩阵")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues")
    plt.title("混淆矩阵")
    st.pyplot(fig)
    
    return model

# 文本预测功能
def text_prediction_demo(model, vectorizer, label_names, lang):
    """演示文本预测"""
    st.subheader("文本预测演示")
    
    # 输入文本
    user_text = st.text_area("输入文本进行预测:", 
                           "这个产品很好，我非常满意" if lang == "中文" else 
                           "The team won the hockey game by two points")
    
    if st.button("预测"):
        # 预处理
        processed_text = preprocess_text(user_text, is_chinese=(lang == "中文"))
        # 向量化
        text_vec = vectorizer.transform([processed_text])
        # 预测
        pred = model.predict(text_vec)[0]
        pred_proba = model.predict_proba(text_vec)[0].max()
        
        st.success(f"预测结果: {label_names[pred]} (置信度: {pred_proba:.2f})")
        
        # 显示重要特征
        if hasattr(model, 'coef_'):
            st.subheader("关键特征分析")
            # 获取特征重要性
            coefs = model.coef_[0]
            feature_names = vectorizer.get_feature_names_out()
            
            # 排序并显示
            top_n = min(10, len(feature_names))
            indices = np.argsort(np.abs(coefs))[-top_n:]
            top_features = [feature_names[i] for i in indices]
            top_coefs = [coefs[i] for i in indices]
            
            # 可视化
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=top_coefs, y=top_features, ax=ax)
            ax.set_title("对预测影响最大的特征")
            st.pyplot(fig)

# 各模块实现
def text_introduction_section():
    """文本分析基础介绍"""
    st.header("📚 文本分析基础")
    
    st.markdown("""
    **什么是文本分析?**
    文本分析是从非结构化文本数据中提取有价值信息的过程，主要包括：
    - 文本分类：将文本划分到预定义类别（如新闻分类、垃圾邮件检测）
    - 情感分析：判断文本的情感倾向（如正面、负面评价）
    - 主题提取：识别文本中的主要主题
    - 命名实体识别：识别文本中的人名、地名、机构名等
    
    **文本数据的特点:**
    - 非结构化：没有固定格式
    - 高维度：词汇表可能非常大
    - 稀疏性：大多数词在大多数文本中不出现
    - 语义复杂性：一词多义、歧义等问题
    """)
    
    st.subheader("文本分析流程")
    st.image("https://miro.medium.com/max/1400/1*V86tTaEJ414VpNy4vDyQbg.png", 
             caption="文本分析基本流程", use_container_width=True)
    
    st.info("""
    **核心挑战:**
    机器学习模型只能处理数值数据，因此文本分析的关键是将文本转换为数值特征，这一过程称为"文本向量化"。
    """)

def text_preprocessing_section():
    """文本预处理模块"""
    st.header("✂️ 文本预处理")
    
    st.markdown("""
    **预处理的目的:**
    清洗文本数据，去除噪声，标准化格式，为后续向量化做准备。
    
    **基本步骤:**
    1. 去除特殊字符和无关符号
    2. 大小写转换（英文）
    3. 分词（将句子拆分为词语）
    4. 去除停用词（如"的"、"是"、"the"等无实际意义的词）
    5. 词形还原/词干提取（英文）
    """)
    
    # 演示预处理效果
    st.subheader("预处理效果演示")
    lang = st.radio("选择语言", ["中文", "英文"])
    
    if lang == "中文":
        raw_text = st.text_input("输入中文文本:", "大家好！今天天气真不错，我们去公园玩吧！")
        processed_text = preprocess_text(raw_text, is_chinese=True)
        st.write("原始文本:", raw_text)
        st.write("预处理后:", processed_text)
        st.write("分词结果:", "/".join(jieba.cut(raw_text)))
    else:
        raw_text = st.text_input("输入英文文本:", "Hello! Today is a beautiful day, let's go to the park!")
        processed_text = preprocess_text(raw_text)
        st.write("原始文本:", raw_text)
        st.write("预处理后:", processed_text)
    
    st.info("""
    **中文vs英文处理差异:**
    - 中文需要专门的分词工具（如jieba），英文可直接按空格分割
    - 英文有词形变化（复数、时态等），需要词干提取或词形还原
    - 中英文停用词表不同
    """)

def feature_extraction_section():
    """特征提取模块"""
    st.header("🔤 文本特征提取")
    
    # 选择数据集
    dataset_name = st.selectbox("选择演示数据集", ["新闻主题分类", "中文情感分析"])
    texts, labels, label_names, lang = load_sample_data(dataset_name)
    
    if texts:
        st.write(f"数据集信息: {len(texts)}个样本，{len(label_names)}个类别")
        st.write("类别:", label_names)
        
        # 显示样本
        st.subheader("样本示例")
        sample_idx = st.slider("选择样本", 0, min(10, len(texts)-1), 0)
        st.write(f"文本: {texts[sample_idx]}")
        st.write(f"标签: {label_names[labels[sample_idx]]}")
        
        # 特征提取演示
        X, vectorizer = demo_feature_extraction(texts, lang)
        st.session_state.X = X
        st.session_state.y = labels
        st.session_state.vectorizer = vectorizer
        st.session_state.label_names = label_names
        st.session_state.lang = lang

def model_training_section():
    """模型训练模块"""
    st.header("🧠 文本分类模型训练")
    
    if 'X' not in st.session_state:
        st.warning("请先在【文本特征提取】模块准备数据")
        return
    
    # 训练模型
    model = train_and_evaluate(
        st.session_state.X, 
        st.session_state.y, 
        st.session_state.label_names
    )
    
    # 保存模型供预测使用
    st.session_state.model = model
    
    # 预测演示
    text_prediction_demo(
        model, 
        st.session_state.vectorizer, 
        st.session_state.label_names,
        st.session_state.lang
    )

def sentiment_analysis_section():
    """情感分析专项模块"""
    st.header("😊 情感分析基础")
    
    st.markdown("""
    **什么是情感分析?**
    情感分析是文本分类的一种特殊形式，专注于识别文本中的主观情感倾向，主要包括：
    - 极性分析：正面、负面、中性
    - 情感强度分析：情感的强烈程度
    - 情感类型分析：喜悦、愤怒、悲伤等具体情感
    
    **应用场景:**
    - 产品评价分析
    - 社交媒体情感监测
    - 舆情分析
    - 客户反馈处理
    """)
    
    # 简单情感分析演示
    st.subheader("情感分析演示")
    texts, labels, label_names, _ = load_sample_data("中文情感分析")
    X, vectorizer = demo_feature_extraction(texts, "中文")
    model = train_and_evaluate(X, labels, label_names)
    
    # 实时预测
    st.subheader("实时情感预测")
    user_comment = st.text_area("输入商品评论:", "这个产品质量很好，价格也很实惠，非常推荐！")
    if st.button("分析情感"):
        processed = preprocess_text(user_comment, is_chinese=True)
        vec = vectorizer.transform([processed])
        pred = model.predict(vec)[0]
        st.success(f"情感预测: {label_names[pred]}")

def quiz_section():
    """概念测验模块"""
    st.header("❓ 文本分析概念测验")
    
    question = st.selectbox(
        "选择问题:",
        [
            "词袋模型和TF-IDF的主要区别是什么?",
            "为什么文本需要预处理?",
            "朴素贝叶斯为什么适合文本分类?",
            "情感分析与普通文本分类的区别是什么?",
            "什么是N-gram?"
        ]
    )
    
    if question == "词袋模型和TF-IDF的主要区别是什么?":
        ans = st.radio("选择答案:", [
            "词袋模型考虑词频，TF-IDF只考虑词是否出现",
            "TF-IDF同时考虑词在当前文本的频率和在所有文本中的分布",
            "词袋模型适用于中文，TF-IDF适用于英文"
        ])
        if st.button("检查答案", key="q1"):
            if ans == "TF-IDF同时考虑词在当前文本的频率和在所有文本中的分布":
                st.success("✅ 正确！TF-IDF通过逆文档频率惩罚常见词，突出重要词。")
            else:
                st.error("❌ 不正确。TF-IDF = 词频(TF) × 逆文档频率(IDF)，能更好地反映词的重要性。")
    
    elif question == "为什么文本需要预处理?":
        ans = st.radio("选择答案:", [
            "减少噪声，标准化数据，提高模型效果",
            "为了让文本看起来更整洁",
            "所有机器学习任务都必须进行预处理"
        ])
        if st.button("检查答案", key="q2"):
            if ans == "减少噪声，标准化数据，提高模型效果":
                st.success("✅ 正确！预处理可以去除无关信息，统一格式，帮助模型更好地学习。")
            else:
                st.error("❌ 不正确。预处理的主要目的是提高数据质量，从而提升模型性能。")
    
    # 其他问题类似...

# 主程序
def main():

    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "文本分析基础"
        
    # 侧边栏导航
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "文本分析基础",
        "文本预处理",
        "文本特征提取",
        "分类模型训练",
        "情感分析专项",
        "概念测验",
        "编程实例（新闻文本数据集）"
    ])
    
    # 显示对应模块编程实例模块: 贝叶斯文本分类分步编程训练"
    st.session_state.section = section    
    context = ""
    if section == "文本分析基础":
        text_introduction_section()
    elif section == "文本预处理":
        text_preprocessing_section()
    elif section == "文本特征提取":
        feature_extraction_section()
    elif section == "分类模型训练":
        model_training_section()
    elif section == "情感分析专项":
        sentiment_analysis_section()
    elif section == "概念测验":
        quiz_section()
    elif section == "编程实例（新闻文本数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        bayes_text_classification_step_by_step.main()
        context = "编程实例模块: 朴素贝叶斯文本分类分步编程训练"

    # 显示聊天界面
    display_chat_interface(context)
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    本平台帮助学习文本分析基础知识：
    - 文本预处理方法
    - 特征提取技术（词袋模型、TF-IDF）
    - 文本分类算法
    - 情感分析基础
    """)

if __name__ == "__main__":
    main()
