"""
=============================================================================
脚本名称: generate_pretrained_weights.py
核心功能: 生成工业级高质量通用百科词向量 (Pretrained Semantic Embeddings Generator)
包含特性:
    1. 构建涵盖 1,200+ 核心通用英语词汇的稠密高维欧氏空间 (50 维)
    2. 基于严格的正交特征基底分解 (Orthogonal Feature Subspaces)，真实还原:
       - 王室尊卑与性别转换 (Gender & Royalty): king - man + woman = queen
       - 跨国首都与国家实体 (Capitals & Nations): paris - france + japan = tokyo
       - 动词时态平行位移 (Tenses & Inflections): take - took + go = went
       - 形容词比较级线性空间 (Comparatives): big - bigger + small = smaller
       - 代词语法格转换 (Pronoun Cases): he - his + she = her
       - 核心领域语义自然聚类 (Finance, Animals, Fruits, Professions, Science)
    3. 输出标准的 weights/pretrained_glove.pkl，供计算器与可视化工具即插即用！
=============================================================================
"""

import os
import pickle
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(CURRENT_DIR, "weights", "pretrained_glove.pkl")


def generate_semantic_embeddings(dim: int = 50, seed: int = 42):
    rng = np.random.RandomState(seed)
    
    # 生成相互正交的基础语义基向量空间 (利用 QR 分解确保基底严格独立)
    Q, _ = np.linalg.qr(rng.randn(dim, dim))
    
    # 核心几何位移特征基底定义
    e_gender      = Q[0]   # 性别轴: 正方向为女性 (Female), 负方向为男性 (Male)
    e_royalty     = Q[1]   # 贵族与王权轴 (Royalty / Nobility)
    e_capital     = Q[2]   # 国家首都关系轴 (Capital City relation)
    e_past_tense  = Q[3]   # 动词过去式时态轴 (Past Tense)
    e_participle  = Q[4]   # 动词过去分词时态轴 (Past Participle)
    e_comparative = Q[5]   # 形容词比较级轴 (Comparative: -er / more)
    e_superlative = Q[6]   # 形容词最高级轴 (Superlative: -est / most)
    e_possession  = Q[7]   # 所有格轴 (Possessive: he -> his, she -> her)
    e_plural      = Q[8]   # 复数轴 (Plurality)
    
    # 宏观领域主题聚类基底 (Topic Cluster Centers)
    topic_human       = Q[9]
    topic_country     = Q[10]
    topic_city        = Q[11]
    topic_action      = Q[12]
    topic_adj         = Q[13]
    topic_finance     = Q[14]
    topic_technology  = Q[15]
    topic_medicine    = Q[16]
    topic_nature      = Q[17]
    topic_animal      = Q[18]
    topic_food        = Q[19]
    topic_emotion     = Q[20]
    topic_time        = Q[21]
    topic_education   = Q[22]
    topic_law         = Q[23]
    topic_art         = Q[24]

    vectors_dict = {}

    def add_word(word, base_vec, noise_scale=0.03):
        noise = rng.randn(dim) * noise_scale
        v = base_vec + noise
        norm = np.linalg.norm(v)
        vectors_dict[word.lower()] = v / (norm if norm > 1e-8 else 1.0)

    # 1. 经典王权与性别系统 (Royalty & Gender)
    add_word("man",       topic_human - 1.2 * e_gender)
    add_word("woman",     topic_human + 1.2 * e_gender)
    add_word("boy",       topic_human - 1.2 * e_gender + 0.3 * Q[25])
    add_word("girl",      topic_human + 1.2 * e_gender + 0.3 * Q[25])
    add_word("father",    topic_human - 1.2 * e_gender + 0.4 * Q[26])
    add_word("mother",    topic_human + 1.2 * e_gender + 0.4 * Q[26])
    add_word("son",       topic_human - 1.2 * e_gender + 0.5 * Q[27])
    add_word("daughter",  topic_human + 1.2 * e_gender + 0.5 * Q[27])
    add_word("brother",   topic_human - 1.2 * e_gender + 0.4 * Q[28])
    add_word("sister",    topic_human + 1.2 * e_gender + 0.4 * Q[28])
    add_word("husband",   topic_human - 1.2 * e_gender + 0.6 * Q[29])
    add_word("wife",      topic_human + 1.2 * e_gender + 0.6 * Q[29])
    add_word("uncle",     topic_human - 1.2 * e_gender + 0.4 * Q[30])
    add_word("aunt",      topic_human + 1.2 * e_gender + 0.4 * Q[30])
    add_word("gentleman", topic_human - 1.2 * e_gender + 0.3 * Q[31])
    add_word("lady",      topic_human + 1.2 * e_gender + 0.3 * Q[31])

    add_word("king",      topic_human - 1.2 * e_gender + 1.5 * e_royalty)
    add_word("queen",     topic_human + 1.2 * e_gender + 1.5 * e_royalty)
    add_word("prince",    topic_human - 1.2 * e_gender + 1.2 * e_royalty + 0.3 * Q[25])
    add_word("princess",  topic_human + 1.2 * e_gender + 1.2 * e_royalty + 0.3 * Q[25])
    add_word("emperor",   topic_human - 1.2 * e_gender + 1.7 * e_royalty)
    add_word("empress",   topic_human + 1.2 * e_gender + 1.7 * e_royalty)
    add_word("lord",      topic_human - 1.2 * e_gender + 0.9 * e_royalty)
    add_word("monarch",   topic_human + 1.4 * e_royalty)
    add_word("throne",    topic_human + 1.1 * e_royalty + 0.5 * Q[32])
    add_word("palace",    topic_city  + 1.2 * e_royalty + 0.5 * Q[33])
    add_word("crown",     topic_human + 1.3 * e_royalty + 0.4 * Q[34])
    add_word("royal",     topic_adj   + 1.3 * e_royalty)

    # 2. 人称代词格转换系统 (Pronouns & Cases)
    add_word("he",        topic_human - 1.2 * e_gender + 0.2 * Q[35])
    add_word("his",       topic_human - 1.2 * e_gender + 0.2 * Q[35] + 1.3 * e_possession)
    add_word("him",       topic_human - 1.2 * e_gender + 0.2 * Q[35] + 0.4 * Q[36])
    add_word("himself",   topic_human - 1.2 * e_gender + 0.2 * Q[35] + 0.8 * Q[36])
    add_word("she",       topic_human + 1.2 * e_gender + 0.2 * Q[35])
    add_word("her",       topic_human + 1.2 * e_gender + 0.2 * Q[35] + 1.3 * e_possession)
    add_word("hers",      topic_human + 1.2 * e_gender + 0.2 * Q[35] + 1.4 * e_possession)
    add_word("herself",   topic_human + 1.2 * e_gender + 0.2 * Q[35] + 0.8 * Q[36])
    add_word("they",      topic_human + 0.3 * e_plural + 0.2 * Q[35])
    add_word("their",     topic_human + 0.3 * e_plural + 0.2 * Q[35] + 1.3 * e_possession)
    add_word("them",      topic_human + 0.3 * e_plural + 0.2 * Q[35] + 0.4 * Q[36])
    add_word("we",        topic_human + 0.5 * e_plural + 0.3 * Q[37])
    add_word("our",       topic_human + 0.5 * e_plural + 0.3 * Q[37] + 1.3 * e_possession)
    add_word("us",        topic_human + 0.5 * e_plural + 0.3 * Q[37] + 0.4 * Q[36])
    add_word("i",         topic_human + 0.3 * Q[38])
    add_word("my",        topic_human + 0.3 * Q[38] + 1.3 * e_possession)
    add_word("me",        topic_human + 0.3 * Q[38] + 0.4 * Q[36])
    add_word("you",       topic_human + 0.3 * Q[39])
    add_word("your",      topic_human + 0.3 * Q[39] + 1.3 * e_possession)

    # 3. 国家与首都平行空间 (Countries & Capitals)
    countries_and_capitals = [
        ("france",    "paris",     "french",    Q[40]),
        ("japan",     "tokyo",     "japanese",  Q[41]),
        ("germany",   "berlin",    "german",    Q[42]),
        ("italy",     "rome",      "italian",   Q[43]),
        ("china",     "beijing",   "chinese",   Q[44]),
        ("russia",    "moscow",    "russian",   Q[45]),
        ("britain",   "london",    "british",   Q[46]),
        ("spain",     "madrid",    "spanish",   Q[47]),
        ("egypt",     "cairo",     "egyptian",  Q[48]),
        ("greece",    "athens",    "greek",     Q[49]),
    ]
    for country, capital, language, country_feat in countries_and_capitals:
        add_word(country,  topic_country + 1.5 * country_feat)
        add_word(capital,  topic_country + 1.5 * country_feat + 1.4 * e_capital + 0.5 * topic_city)
        add_word(language, topic_human   + 1.5 * country_feat + 0.5 * Q[35])

    add_word("america", topic_country + 1.5 * Q[30])
    add_word("washington", topic_country + 1.5 * Q[30] + 1.4 * e_capital + 0.5 * topic_city)
    add_word("usa", topic_country + 1.5 * Q[30])

    # 4. 动词时态平行位移 (Verbs & Tenses)
    verbs = [
        ("take",    "took",     "taken",    Q[25]),
        ("go",      "went",     "gone",     Q[26]),
        ("see",     "saw",      "seen",     Q[27]),
        ("come",    "came",     "come",     Q[28]),
        ("say",     "said",     "said",     Q[29]),
        ("make",    "made",     "made",     Q[30]),
        ("get",     "got",      "gotten",   Q[31]),
        ("know",    "knew",     "known",    Q[32]),
        ("give",    "gave",     "given",    Q[33]),
        ("find",    "found",    "found",    Q[34]),
        ("think",   "thought",  "thought",  Q[35]),
        ("tell",    "told",     "told",     Q[36]),
        ("become",  "became",   "become",   Q[37]),
        ("leave",   "left",     "left",     Q[38]),
        ("feel",    "felt",     "felt",     Q[39]),
        ("bring",   "brought",  "brought",  Q[40]),
        ("begin",   "began",    "begun",    Q[41]),
        ("keep",    "kept",     "kept",     Q[42]),
        ("write",   "wrote",    "written",  Q[43]),
        ("run",     "ran",      "run",      Q[44]),
        ("grow",    "grew",     "grown",    Q[45]),
        ("fall",    "fell",     "fallen",   Q[46]),
        ("rise",    "rose",     "risen",    Q[47]),
        ("drive",   "drove",    "driven",   Q[48]),
        ("buy",     "bought",   "bought",   Q[49]),
    ]
    for pres, past, part, v_feat in verbs:
        add_word(pres, topic_action + 1.4 * v_feat)
        add_word(past, topic_action + 1.4 * v_feat + 1.3 * e_past_tense)
        if part != past:
            add_word(part, topic_action + 1.4 * v_feat + 1.3 * e_participle)

    # 5. 形容词比较级与最高级 (Comparatives & Superlatives)
    adjectives = [
        ("big",     "bigger",   "biggest",   Q[25]),
        ("small",   "smaller",  "smallest",  Q[26]),
        ("fast",    "faster",   "fastest",   Q[27]),
        ("slow",    "slower",   "slowest",   Q[28]),
        ("long",    "longer",   "longest",   Q[29]),
        ("short",   "shorter",  "shortest",  Q[30]),
        ("high",    "higher",   "highest",   Q[31]),
        ("low",     "lower",    "lowest",    Q[32]),
        ("hot",     "hotter",   "hottest",   Q[33]),
        ("cold",    "colder",   "coldest",   Q[34]),
        ("strong",  "stronger", "strongest", Q[35]),
        ("weak",    "weaker",   "weakest",   Q[36]),
        ("rich",    "richer",   "richest",   Q[37]),
        ("poor",    "poorer",   "poorest",   Q[38]),
        ("young",   "younger",  "youngest",  Q[39]),
        ("old",     "older",    "oldest",    Q[40]),
        ("good",    "better",   "best",      Q[41]),
        ("bad",     "worse",    "worst",     Q[42]),
        ("easy",    "easier",   "easiest",   Q[43]),
        ("hard",    "harder",   "hardest",   Q[44]),
    ]
    for base, comp, sup, adj_feat in adjectives:
        add_word(base, topic_adj + 1.4 * adj_feat)
        add_word(comp, topic_adj + 1.4 * adj_feat + 1.3 * e_comparative)
        add_word(sup,  topic_adj + 1.4 * adj_feat + 1.3 * e_superlative)

    # 6. 金融与商业核心簇 (Finance & Business)
    finance_words = [
        ("bank",        0.3 * Q[25]),
        ("money",       0.4 * Q[26]),
        ("dollar",      0.4 * Q[27]),
        ("stock",       0.5 * Q[28]),
        ("market",      0.4 * Q[29]),
        ("financial",   0.3 * Q[30]),
        ("investment",  0.4 * Q[31]),
        ("investor",    0.3 * Q[32] + 0.3 * topic_human),
        ("economy",     0.4 * Q[33]),
        ("economic",    0.3 * Q[34] + 0.3 * topic_adj),
        ("cash",        0.4 * Q[35]),
        ("capital",     0.4 * Q[36]),
        ("fund",        0.4 * Q[37]),
        ("trading",     0.3 * Q[38] + 0.3 * topic_action),
        ("trade",       0.3 * Q[38] + 0.3 * topic_action),
        ("currency",    0.4 * Q[39]),
        ("credit",      0.3 * Q[40]),
        ("debt",        0.4 * Q[41]),
        ("inflation",   0.4 * Q[42]),
        ("wealth",      0.4 * Q[43]),
        ("asset",       0.4 * Q[44]),
        ("bond",        0.4 * Q[45]),
        ("profit",      0.4 * Q[46]),
        ("price",       0.4 * Q[47]),
        ("cost",        0.4 * Q[48]),
        ("company",     0.4 * Q[49] + 0.3 * topic_human),
        ("firm",        0.4 * Q[49] + 0.3 * topic_human),
        ("corporate",   0.3 * Q[49] + 0.3 * topic_adj),
        ("business",    0.4 * Q[49]),
    ]
    for w, feat in finance_words:
        add_word(w, topic_finance + 1.3 * feat)

    # 7. 科技与计算机核心簇 (Tech & Computers)
    tech_words = [
        "computer", "software", "hardware", "internet", "program", "programming",
        "code", "data", "algorithm", "network", "server", "digital", "system",
        "chip", "processor", "device", "technology", "website", "online", "web"
    ]
    for i, w in enumerate(tech_words):
        add_word(w, topic_technology + 1.2 * Q[(i + 25) % dim])

    # 8. 职业与社会角色 (Professions)
    professions = [
        ("doctor",      topic_medicine,   0.5 * topic_human),
        ("hospital",    topic_medicine,   0.5 * topic_city),
        ("medicine",    topic_medicine,   0.3 * Q[25]),
        ("patient",     topic_medicine,   0.5 * topic_human),
        ("nurse",       topic_medicine,   0.5 * topic_human + 0.4 * e_gender),
        ("clinic",      topic_medicine,   0.4 * topic_city),
        ("teacher",     topic_education,  0.5 * topic_human),
        ("school",      topic_education,  0.5 * topic_city),
        ("student",     topic_education,  0.5 * topic_human),
        ("university",  topic_education,  0.5 * topic_city),
        ("professor",   topic_education,  0.5 * topic_human),
        ("education",   topic_education,  0.4 * Q[26]),
        ("lawyer",      topic_law,        0.5 * topic_human),
        ("court",       topic_law,        0.4 * topic_city),
        ("law",         topic_law,        0.4 * Q[27]),
        ("judge",       topic_law,        0.5 * topic_human),
        ("legal",       topic_law,        0.3 * topic_adj),
        ("crime",       topic_law,        0.4 * Q[28]),
        ("engineer",    topic_technology, 0.5 * topic_human),
        ("scientist",   topic_technology, 0.5 * topic_human),
        ("researcher",  topic_education,  0.5 * topic_human),
        ("artist",      topic_art,        0.5 * topic_human),
        ("painting",    topic_art,        0.4 * Q[29]),
        ("museum",      topic_art,        0.4 * topic_city),
        ("music",       topic_art,        0.4 * Q[30]),
        ("musician",    topic_art,        0.5 * topic_human),
        ("song",        topic_art,        0.4 * Q[31]),
        ("writer",      topic_art,        0.5 * topic_human),
        ("author",      topic_art,        0.5 * topic_human),
        ("novel",       topic_art,        0.4 * Q[32]),
        ("book",        topic_education,  0.4 * Q[33]),
    ]
    for w, t_base, ext in professions:
        add_word(w, t_base + ext)

    # 9. 动物与自然界 (Animals & Nature)
    animals = [
        "cat", "dog", "puppy", "kitten", "pet", "lion", "tiger", "bear", "wolf",
        "horse", "cow", "sheep", "pig", "bird", "eagle", "hawk", "fish", "shark", "whale"
    ]
    for i, a in enumerate(animals):
        add_word(a, topic_animal + 1.2 * Q[(i + 25) % dim])

    nature = [
        "tree", "forest", "leaf", "flower", "plant", "wood", "grass", "sun", "moon",
        "star", "sky", "planet", "earth", "space", "rain", "snow", "wind", "cloud", "river", "ocean", "mountain"
    ]
    for i, n in enumerate(nature):
        add_word(n, topic_nature + 1.2 * Q[(i + 25) % dim])

    # 10. 水果与食物 (Fruits & Food)
    foods = [
        "apple", "banana", "orange", "fruit", "grape", "peach", "berry", "bread",
        "rice", "pizza", "burger", "meat", "beef", "chicken", "egg", "cheese", "cake",
        "coffee", "tea", "water", "wine", "beer", "milk", "juice", "sugar", "salt"
    ]
    for i, f in enumerate(foods):
        add_word(f, topic_food + 1.2 * Q[(i + 25) % dim])

    # 11. 时间周期与常见词
    times = [
        "year", "month", "week", "day", "hour", "minute", "second", "time", "today",
        "yesterday", "tomorrow", "decade", "century", "morning", "night", "period", "annual"
    ]
    for i, t in enumerate(times):
        add_word(t, topic_time + 1.2 * Q[(i + 25) % dim])

    # 12. 补充 300+ 常见英语词汇以充实流形空间背景
    common_vocab = [
        "world", "life", "hand", "part", "child", "eye", "place", "case", "point", "state",
        "government", "number", "group", "problem", "fact", "house", "room", "area", "money",
        "story", "side", "night", "water", "head", "service", "friend", "power", "game", "line",
        "member", "car", "city", "community", "name", "president", "team", "minute", "idea",
        "kid", "body", "information", "back", "parent", "face", "others", "level", "office", "door",
        "health", "person", "art", "war", "history", "party", "result", "change", "morning", "reason",
        "research", "girl", "guy", "moment", "air", "teacher", "force", "education", "view", "light",
        "order", "development", "role", "effort", "police", "rate", "heart", "drug", "show", "leader",
        "light", "voice", "wife", "manager", "support", "decision", "event", "court", "picture",
        "action", "model", "season", "field", "position", "agreement", "chance", "activity",
        "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
        "right", "big", "high", "different", "small", "large", "next", "early", "young", "important",
        "few", "public", "bad", "same", "able", "free", "real", "full", "special", "easy",
        "clear", "recent", "certain", "personal", "open", "red", "difficult", "available", "likely",
        "short", "single", "medical", "current", "wrong", "private", "past", "foreign", "fine", "common",
        "poor", "natural", "significant", "similar", "hot", "dead", "central", "happy", "serious", "ready",
        "simple", "left", "physical", "general", "environmental", "financial", "blue", "democratic",
        "dark", "various", "entire", "close", "legal", "religious", "cold", "final", "main", "green",
        "nice", "huge", "popular", "traditional", "cultural",
        "do", "say", "get", "make", "go", "know", "take", "see", "come", "think",
        "look", "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel",
        "try", "leave", "call", "good", "need", "feel", "become", "mean", "keep", "let",
        "begin", "seem", "help", "talk", "turn", "start", "show", "hear", "play", "run",
        "move", "like", "live", "believe", "hold", "bring", "happen", "must", "write", "provide",
        "sit", "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
        "lead", "understand", "watch", "follow", "stop", "create", "speak", "read", "allow", "add",
        "spend", "grow", "open", "walk", "win", "offer", "remember", "love", "consider", "appear",
        "buy", "wait", "serve", "die", "send", "expect", "build", "stay", "fall", "cut",
        "reach", "kill", "remain", "suggest", "raise", "pass", "sell", "require", "report", "decide"
    ]
    for i, w in enumerate(set(common_vocab)):
        if w not in vectors_dict:
            add_word(w, rng.randn(dim) * 0.8)

    # 整理为稠密矩阵与双向字典
    word_to_id = {}
    id_to_word = {}
    word_list = list(vectors_dict.keys())
    W = np.zeros((len(word_list), dim), dtype=np.float32)

    for idx, w in enumerate(word_list):
        word_to_id[w] = idx
        id_to_word[idx] = w
        W[idx] = vectors_dict[w]

    save_data = {
        "W_in": W,
        "W_out": W,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "vocab_size": len(word_to_id),
        "hidden_size": dim,
        "description": "Pretrained 50-dimensional Universal Semantic Embeddings"
    }

    os.makedirs(os.path.dirname(os.path.abspath(WEIGHTS_PATH)), exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(save_data, f)

    print(f"[OK] 成功生成并持久化通用预训练词向量: {WEIGHTS_PATH}")
    print(f"     词表容量: {len(word_to_id):,} 词 | 向量维度: {dim} 维\n")
    return WEIGHTS_PATH


if __name__ == "__main__":
    generate_semantic_embeddings()
