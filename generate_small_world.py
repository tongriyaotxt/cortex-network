"""
Small World Corpus Generator
构造一个自我一致的小型世界，用有限词汇生成连贯叙事。
核心原则：
- 词汇量控制在 400 以内
- 固定角色/地点/物品，但语法和事件组合多样化
- 强调因果关系、时序、对话
- 避免固定模板，使用基于事件状态的动态生成
"""

import random
import os
import re
from collections import Counter

random.seed(42)

# ========== 世界设定 ==========

CHARACTERS = ["Alice", "Bob", "Carol", "Dave", "Eve"]
PLACES = ["village", "forest", "river", "mountain", "garden", "tower"]
OBJECTS = ["key", "map", "book", "lamp", "sword", "potion", "coin", "ring"]

# 动词（及物/不及物）
VERBS = {
    "transitive": ["found", "lost", "gave", "took", "opened", "closed", "built", "broke", "lit", "hid", "sought", "carried", "dropped", "polished"],
    "intransitive": ["arrived", "departed", "slept", "waited", "laughed", "cried", "shouted", "whispered", "sat", "stood", "walked", "ran", "fell", "rose"],
    "communication": ["said", "asked", "replied", "told", "warned", "promised", "explained"],
    "state_change": ["became", "grew", "turned", "felt", "seemed", "appeared"],
}

# 形容词
ADJECTIVES = ["old", "new", "bright", "dark", "small", "large", "heavy", "light", "sharp", "smooth", "ancient", "hidden", "golden", "silver", "broken", "whole"]

# 副词
ADVERBS = ["slowly", "quickly", "carefully", "suddenly", "quietly", "loudly", "gently", "firmly", "finally", "soon", "then", "again"]

# 时间词
TIME_WORDS = ["morning", "noon", "evening", "night", "dawn", "dusk", "yesterday", "today", "tomorrow"]

# 连接词（因果/时序）
CONNECTORS = [
    "so", "therefore", "because", "since", "as a result",
    "then", "after that", "later", "meanwhile", "soon",
    "but", "however", "although", "yet", "instead",
    "and", "also", "furthermore", "moreover",
]

# 情绪/状态
STATES = ["happy", "sad", "angry", "scared", "tired", "hungry", "thirsty", "cold", "warm", "safe", "worried", "excited", "confused", "determined"]

# 天气/环境
ENVIRONMENTS = ["sun shone", "rain fell", "wind blew", "snow covered", "fog gathered", "stars appeared"]

# 对话内容池（有限但多样）
QUOTES = [
    "I found something",
    "We must leave",
    "Wait for me",
    "Do not go there",
    "This is important",
    "I am afraid",
    "Trust me",
    "Look at this",
    "What happened",
    "Why are you here",
    "Help me",
    "I remember now",
    "It is time",
    "Be careful",
    "I understand",
    "That is impossible",
    "Follow me",
    "Stay here",
    "We can do this",
    "I am sorry",
]


class WorldState:
    """追踪世界状态，确保叙事连贯性。"""
    def __init__(self):
        self.locations = {c: random.choice(PLACES) for c in CHARACTERS}
        self.has_object = {c: None for c in CHARACTERS}
        self.emotions = {c: "safe" for c in CHARACTERS}
        self.object_locations = {o: random.choice(PLACES + [None]) for o in OBJECTS}
        self.time = 0
        
    def move(self, char, place):
        self.locations[char] = place
        
    def give_object(self, char, obj):
        if obj in self.object_locations:
            self.object_locations[obj] = None
        self.has_object[char] = obj
        
    def drop_object(self, char):
        obj = self.has_object[char]
        if obj:
            self.object_locations[obj] = self.locations[char]
            self.has_object[char] = None
            
    def set_emotion(self, char, emotion):
        self.emotions[char] = emotion
        
    def advance_time(self):
        self.time += 1


def generate_sentence(world, style=None):
    """基于世界状态生成单句，多种句式避免模板化。"""
    c1, c2 = random.sample(CHARACTERS, 2)
    place = random.choice(PLACES)
    obj = random.choice(OBJECTS)
    adj = random.choice(ADJECTIVES)
    adv = random.choice(ADVERBS)
    conn = random.choice(CONNECTORS)
    
    patterns = []
    
    # 1. 简单动作句
    v = random.choice(VERBS["intransitive"])
    patterns.append(f"{c1} {v} {adv}.")
    patterns.append(f"{c1} {v} in the {place}.")
    
    # 2. 及物动作句
    v = random.choice(VERBS["transitive"])
    patterns.append(f"{c1} {v} the {obj}.")
    patterns.append(f"{c1} {v} the {adj} {obj}.")
    patterns.append(f"{c1} {v} the {obj} {adv}.")
    
    # 3. 状态变化句
    v = random.choice(VERBS["state_change"])
    state = random.choice(STATES)
    patterns.append(f"{c1} {v} {state}.")
    patterns.append(f"{c1} {v} {state} {adv}.")
    
    # 4. 地点句
    patterns.append(f"{c1} was in the {place}.")
    patterns.append(f"The {place} was {adj}.")
    patterns.append(f"In the {place}, {c1} waited.")
    
    # 5. 拥有句
    if world.has_object[c1]:
        patterns.append(f"{c1} held the {world.has_object[c1]}.")
    else:
        patterns.append(f"{c1} had nothing.")
        
    # 6. 对话句
    quote = random.choice(QUOTES)
    patterns.append(f'{c1} said, "{quote}."')
    patterns.append(f'{c1} {random.choice(VERBS["communication"])}, "{quote}."')
    patterns.append(f'"{quote}," {c1} {random.choice(VERBS["communication"])}.")
    
    # 7. 环境句
    env = random.choice(ENVIRONMENTS)
    patterns.append(f"The {env}.")
    patterns.append(f"Outside, the {env}.")
    
    # 8. 因果句（需连接词）
    v1 = random.choice(VERBS["transitive"] + VERBS["intransitive"])
    v2 = random.choice(VERBS["transitive"] + VERBS["intransitive"])
    patterns.append(f"{c1} {v1}, {conn} {c2} {v2}.")
    patterns.append(f"Because {c1} {v1}, {c2} {v2}.")
    patterns.append(f"{c1} {v1}, and {conn} {c2} {v2}.")
    
    # 9. 时间句
    tw = random.choice(TIME_WORDS)
    patterns.append(f"In the {tw}, {c1} {random.choice(VERBS['intransitive'])}.")
    patterns.append(f"By {tw}, the {place} was {adj}.")
    
    # 10. 复杂从句
    patterns.append(f"When {c1} saw the {obj}, {c2} {random.choice(VERBS['transitive'])} it.")
    patterns.append(f"Although {c1} was {random.choice(STATES)}, {c2} {random.choice(VERBS['intransitive'])}.")
    patterns.append(f"{c1} knew that {c2} had the {obj}.")
    
    sentence = random.choice(patterns)
    
    # 根据世界状态更新（只做文本层面的状态跟踪，不强制逻辑正确性）
    world.advance_time()
    
    return sentence


def generate_paragraph(world, length=None):
    """生成连贯段落，3-7句，有主题一致性。"""
    if length is None:
        length = random.randint(3, 7)
    
    sentences = []
    focus_char = random.choice(CHARACTERS)
    focus_place = random.choice(PLACES)
    focus_obj = random.choice(OBJECTS)
    
    for i in range(length):
        # 增加连贯性：30%概率围绕当前焦点生成
        if random.random() < 0.3:
            # 围绕焦点生成
            v = random.choice(VERBS["transitive"] + VERBS["intransitive"])
            adv = random.choice(ADVERBS)
            if random.random() < 0.5:
                s = f"{focus_char} {v} {adv}."
            else:
                s = f"{focus_char} {v} in the {focus_place}."
        else:
            s = generate_sentence(world)
        sentences.append(s)
    
    return ' '.join(sentences)


def generate_story(world, num_paras=10):
    """生成一个小故事，多段落有连贯性。"""
    paragraphs = []
    
    # 故事线：寻找某个物品
    target = random.choice(OBJECTS)
    hero = random.choice(CHARACTERS)
    companion = random.choice([c for c in CHARACTERS if c != hero])
    
    # 开头
    paragraphs.append(f"{hero} lived in the {random.choice(PLACES)}. One {random.choice(TIME_WORDS)}, {hero} decided to find the {target}.")
    
    # 中间：探索过程
    for _ in range(num_paras - 2):
        para = generate_paragraph(world, length=random.randint(2, 4))
        # 偶尔插入故事线相关的句子
        if random.random() < 0.3:
            extra = random.choice([
                f"{hero} searched for the {target}.",
                f"The {target} was {random.choice(ADJECTIVES)}.",
                f"{companion} helped {hero}.",
                f"They went to the {random.choice(PLACES)}.",
                f"{hero} asked, \"Where is the {target}?\"",
            ])
            para += ' ' + extra
        paragraphs.append(para)
    
    # 结尾
    endings = [
        f"Finally, {hero} found the {target}.",
        f"{hero} never found the {target}, but learned something important.",
        f"The {target} was lost forever.",
        f"{hero} and {companion} shared the {target}.",
    ]
    paragraphs.append(random.choice(endings))
    
    return '\n\n'.join(paragraphs)


def main():
    n_stories = 3000
    n_val = 300
    
    os.makedirs('data/corpus', exist_ok=True)
    
    # 生成训练集
    train_lines = []
    seen = set()
    while len(train_lines) < n_stories:
        world = WorldState()
        story = generate_story(world, num_paras=random.randint(5, 12))
        if story not in seen:
            seen.add(story)
            train_lines.append(story)
    
    # 生成验证集
    val_lines = []
    while len(val_lines) < n_val:
        world = WorldState()
        story = generate_story(world, num_paras=random.randint(5, 12))
        if story not in seen:
            seen.add(story)
            val_lines.append(story)
    
    # 保存
    with open('data/corpus/train.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_lines))
    with open('data/corpus/val.txt', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_lines))
    
    # 统计词汇
    all_text = ' '.join(train_lines)
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    unique = set(words)
    
    print(f"Stories: {len(train_lines)} train, {len(val_lines)} val")
    print(f"Total words: {len(words)}")
    print(f"Unique words: {len(unique)}")
    print(f"Vocab density: {len(unique)/len(words):.4f}")
    print(f"\nTop 20 words:")
    for w, c in Counter(words).most_common(20):
        print(f"  {w}: {c}")
    print(f"\nSample story:")
    print(train_lines[0][:500] + "...")

if __name__ == '__main__':
    main()
