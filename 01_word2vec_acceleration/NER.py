# ============================================
# NER 实战：使用 Hugging Face 做命名实体识别
# ============================================

def ner_with_huggingface():
    """使用预训练 NER 模型"""
    print("NER 实战：Hugging Face")
    print("-" * 50)

    try:
        from transformers import pipeline

        # 加载 NER pipeline
        # 英文 NER
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            grouped_entities=True  # 把同一个实体的多个 token 合并
        )

        test_text = "John Smith works at Google in New York City. He loves RUNOOB tutorials."

        results = ner_pipeline(test_text)

        print(f"文本：{test_text}")
        print("识别到的实体：")
        for entity in results:
            print(f"  - {entity['word']}: {entity['entity_group']} (置信度: {entity['score']:.4f})")
        print()

    except Exception as e:
        print(f"需要先安装 transformers：pip install transformers")
        print(f"错误：{e}")

# ============================================
# 中文 NER：使用正则 + 关键词（简单场景）
# ============================================

def simple_chinese_ner():
    """简单的中文 NER：使用正则表达式和词典"""
    import re

    print("简单中文 NER：正则 + 词典")
    print("-" * 50)

    # 实体词典（实际场景中会更大）
    person_names = ["张三", "李四", "王小明", "刘德华"]
    organizations = ["Google", "微软", "阿里巴巴", "腾讯", "RUNOOB"]
    locations = ["北京", "上海", "深圳", "纽约", "伦敦"]

    # 正则模式
    patterns = {
        "PERSON": "|".join(re.escape(name) for name in person_names),
        "ORG": "|".join(re.escape(org) for org in organizations),
        "LOC": "|".join(re.escape(loc) for loc in locations),
        "PHONE": r"1[3-9]\d{9}",  # 手机号
        "EMAIL": r"\w+@\w+\.\w+",  # 邮箱
    }

    def extract_entities(text: str):
        """从文本中提取实体"""
        entities = []

        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                })

        # 按位置排序
        entities.sort(key=lambda x: x["start"])
        return entities

    # 测试
    test_texts = [
        "张三在 Google 工作，手机号是 13800138000",
        "王小明去北京的微软公司出差",
        "有问题联系 support@runoob.com",
        "RUNOOB 是一个很好的学习网站",
    ]

    for text in test_texts:
        entities = extract_entities(text)
        print(f"文本：{text}")
        if entities:
            print("识别到的实体：")
            for ent in entities:
                print(f"  - {ent['text']}: {ent['type']}")
        else:
            print("未识别到实体")
        print()

# ============================================
# 大模型做 NER
# ============================================

def ner_with_llm():
    """使用大模型做 NER 的思路演示"""
    print("大模型做 NER（演示思路）")
    print("-" * 50)

    def extract_entities_with_llm(text: str):
        """用 LLM 做 NER（模拟）"""
        # 真实场景：构造 prompt 调用 LLM
        # prompt = f"""请从以下文本中提取命名实体。
        # 只返回 JSON 格式，包含实体类型：PERSON（人名）、ORG（机构）、LOC（地点）。
        # 文本：{text}"""

        # 这里做简单模拟
        entities = []

        # 简单的关键词匹配来模拟
        if "张三" in text:
            entities.append({"text": "张三", "type": "PERSON"})
        if "Google" in text:
            entities.append({"text": "Google", "type": "ORG"})
        if "北京" in text:
            entities.append({"text": "北京", "type": "LOC"})
        if "RUNOOB" in text:
            entities.append({"text": "RUNOOB", "type": "ORG"})

        return entities

    test_text = "张三在北京的 Google 公司工作，他经常上 RUNOOB 学习"
    entities = extract_entities_with_llm(test_text)

    print(f"文本：{test_text}")
    print("识别到的实体：")
    for ent in entities:
        print(f"  - {ent['text']}: {ent['type']}")

# ============================================
# 运行演示
# ============================================

if __name__ == "__main__":
    simple_chinese_ner()
    print("=" * 50)
    ner_with_llm()
    